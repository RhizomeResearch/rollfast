"""Small gradient transforms used by fine-tuning builders."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from rollfast.utils import (
    AxisName,
    _squared_magnitude_f32,
    astype_preserving_sharding,
    dist_reduce,
    resolve_partition_norm_axis_name,
    zeros_like_preserving_sharding,
)


class GlobalNormClipState(NamedTuple):
    """State for the global-norm clipping transform."""

    count: jax.Array


class AlwaysSkipNonFiniteState(NamedTuple):
    """State for a fail-closed nonfinite-update guard."""

    inner_state: optax.OptState
    consecutive_nonfinite: jax.Array
    total_nonfinite: jax.Array
    last_finite: jax.Array
    threshold_reached: jax.Array


def always_skip_nonfinite(
    inner: optax.GradientTransformation,
    *,
    alert_threshold: int,
) -> optax.GradientTransformationExtraArgs:
    """Reject every nonfinite update without advancing the inner transform."""

    if alert_threshold <= 0:
        raise ValueError("alert_threshold must be positive.")
    inner = optax.with_extra_args_support(inner)

    def init_fn(params):
        return AlwaysSkipNonFiniteState(
            inner_state=inner.init(params),
            consecutive_nonfinite=jnp.zeros([], dtype=jnp.int32),
            total_nonfinite=jnp.zeros([], dtype=jnp.int32),
            last_finite=jnp.ones([], dtype=jnp.bool_),
            threshold_reached=jnp.zeros([], dtype=jnp.bool_),
        )

    def update_fn(updates, state, params=None, **extra_args):
        finite = jnp.ones([], dtype=jnp.bool_)
        for leaf in jax.tree.leaves(updates, is_leaf=lambda x: x is None):
            if leaf is not None:
                finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf)))

        # Select per leaf instead of branching. A lax.cond that XLA captures
        # into a CUDA graph costs ~390 ms per update on H100 GPUs (driver
        # 595), and it copies donated state buffers. Zeros keep nonfinite values out of the
        # inner transform's numerics.
        new_updates, new_inner_state = inner.update(
            _zero_unless(finite, updates),
            state.inner_state,
            params,
            **extra_args,
        )
        guarded_updates = _zero_unless(finite, new_updates)
        inner_state = jax.tree.map(
            lambda old, new: jnp.where(finite, new, _preserved_state_leaf(old, new)),
            state.inner_state,
            new_inner_state,
        )
        consecutive_nonfinite = jnp.where(
            finite,
            jnp.zeros_like(state.consecutive_nonfinite),
            state.consecutive_nonfinite + 1,
        )
        total_nonfinite = state.total_nonfinite + jnp.logical_not(finite).astype(
            jnp.int32
        )
        return guarded_updates, AlwaysSkipNonFiniteState(
            inner_state=inner_state,
            consecutive_nonfinite=consecutive_nonfinite,
            total_nonfinite=total_nonfinite,
            last_finite=finite,
            threshold_reached=jnp.logical_and(
                jnp.logical_not(finite),
                consecutive_nonfinite >= alert_threshold,
            ),
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _zero_unless(condition, tree):
    return jax.tree.map(
        lambda leaf: (
            None
            if leaf is None
            else jnp.where(condition, leaf, zeros_like_preserving_sharding(leaf))
        ),
        tree,
        is_leaf=lambda x: x is None,
    )


def _preserved_state_leaf(old, new):
    """Return ``old`` in the dtype the inner transform gave its state leaf."""

    if not hasattr(old, "dtype") or not hasattr(new, "dtype"):
        return old
    if jnp.issubdtype(old.dtype, jnp.complexfloating) and not jnp.issubdtype(
        new.dtype, jnp.complexfloating
    ):
        return astype_preserving_sharding(jnp.real(old), new.dtype, old)
    return astype_preserving_sharding(old, new.dtype)


def clear_completed_accumulation(
    inner: optax.MultiSteps,
) -> optax.GradientTransformationExtraArgs:
    """Clear a completed MultiSteps window even when it contains NaNs."""

    def update_fn(updates, state, params=None, **extra_args):
        result, next_state = inner.update(updates, state, params, **extra_args)
        # MultiSteps clears with multiplication by zero, which retains NaNs.
        cleared = jax.tree.map(
            lambda grad: jnp.where(
                next_state.mini_step == 0, jnp.zeros_like(grad), grad
            ),
            next_state.acc_grads,
        )
        return result, next_state._replace(acc_grads=cleared)

    return optax.GradientTransformationExtraArgs(inner.init, update_fn)


def clip_by_global_norm(
    max_norm: float,
    *,
    axis_name: AxisName | None = None,
    partition_axis_names: AxisName | None = None,
    replicated_axis_names: AxisName | None = None,
) -> optax.GradientTransformation:
    """Clip by global norm over local leaves and parameter-partition axes."""

    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive.")

    def init_fn(params):
        del params
        return GlobalNormClipState(count=jnp.zeros([], dtype=jnp.int32))

    def update_fn(updates, state, params=None):
        del params
        norm = _global_norm(
            updates,
            axis_name=axis_name,
            partition_axis_names=partition_axis_names,
            replicated_axis_names=replicated_axis_names,
        )
        scale = jnp.minimum(
            1.0, jnp.asarray(max_norm, dtype=jnp.float32) / (norm + 1e-6)
        )
        clipped = jax.tree.map(
            lambda leaf: _scale_leaf(leaf, scale),
            updates,
            is_leaf=lambda x: x is None,
        )
        return clipped, GlobalNormClipState(count=state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


def _global_norm(
    tree,
    *,
    axis_name: AxisName | None,
    partition_axis_names: AxisName | None,
    replicated_axis_names: AxisName | None,
) -> jax.Array:
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in jax.tree.leaves(tree, is_leaf=lambda x: x is None):
        if leaf is None or not hasattr(leaf, "dtype"):
            continue
        total = total + jnp.sum(_squared_magnitude_f32(leaf))
    norm_axis_name = resolve_partition_norm_axis_name(
        axis_name=axis_name,
        partition_axis_names=partition_axis_names,
        replicated_axis_names=replicated_axis_names,
    )
    total = dist_reduce(total, norm_axis_name, "sum")
    return jnp.sqrt(total)


def _scale_leaf(leaf, scale):
    if leaf is None or not hasattr(leaf, "dtype"):
        return leaf
    # A low-precision scale can underflow or round away; scale in float32.
    working = leaf.astype(jnp.promote_types(leaf.dtype, jnp.float32))
    return (working * scale).astype(leaf.dtype)


__all__ = (
    "AlwaysSkipNonFiniteState",
    "always_skip_nonfinite",
    "clip_by_global_norm",
)
