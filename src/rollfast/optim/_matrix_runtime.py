"""Shared runtime plumbing for matrix optimizer transforms."""

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
from optax._src import base, numerics

from rollfast.optim.magma import apply_magma_internal
from rollfast.utils import (
    MomentumAccumulator,
    _apply_weight_decay_tree,
    _cast_state_tree,
    _has_nonzero_or_scheduled,
    _init_magma_state,
    _is_aux_leaf,
    _resolve_mask,
    _resolve_scalar,
    _tree_bias_correction_momentum,
    _tree_momentum_lookahead,
    _tree_stochastic_cast,
    _tree_update_moment_f32,
    _validate_grad_clip_max_amps,
    _validate_positive_static_scalar,
    _zeros_like_tree,
    add_tiny,
    dist_reduce,
)

GradClipMaxAmps = Optional[Union[float, Tuple[float, float]]]
MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]


class MatrixRuntimeStep(NamedTuple):
    """Prepared per-step state shared by matrix optimizer transforms."""

    count: jax.Array
    raw_gradients: base.Updates
    effective_updates: base.Updates
    mu_f32: base.Updates
    target_for_shape: base.Updates
    mu_cast: base.Updates
    should_skip: jax.Array
    next_key: jax.Array
    magma_key: Optional[jax.Array]


def _tree_cast_f32(tree: Any) -> Any:
    return _cast_state_tree(tree, jnp.float32)


def _tree_where_scalar(pred: jax.Array, a: Any, b: Any) -> Any:
    return jax.tree.map(
        lambda x, y: x if _is_aux_leaf(x) else jnp.where(pred, x, y),
        a,
        b,
        is_leaf=_is_aux_leaf,
    )


def _tree_global_norm(grads: Any, axis_name: Optional[str] = None) -> jax.Array:
    leaves = jax.tree.leaves(grads, is_leaf=_is_aux_leaf)
    sq_terms = [
        jnp.sum(
            numerics.abs_sq(
                x.astype(
                    jnp.complex64
                    if jnp.issubdtype(x.dtype, jnp.complexfloating)
                    else jnp.float32
                )
            )
        )
        for x in leaves
        if not _is_aux_leaf(x)
    ]
    local_sq = (
        sum(sq_terms, start=jnp.array(0.0, dtype=jnp.float32))
        if sq_terms
        else jnp.array(0.0, dtype=jnp.float32)
    )
    total_sq = dist_reduce(local_sq, axis_name, "sum")
    return jnp.sqrt(total_sq)


def _clip_per_tensor_rms(
    update: jax.Array,
    max_rms: float,
    max_val: float,
) -> jax.Array:
    rms = jnp.sqrt(jnp.mean(numerics.abs_sq(update)))
    scale_factor = jnp.minimum(1.0, max_rms / (rms + 1e-9))
    update = update * scale_factor
    if jnp.issubdtype(update.dtype, jnp.complexfloating):
        magnitude = jnp.abs(update)
        clamp_scale = jnp.minimum(1.0, max_val / add_tiny(magnitude))
        return update * clamp_scale
    return jnp.clip(update, -max_val, max_val)


def _split_runtime_keys(
    key: jax.Array,
    use_magma: bool,
) -> tuple[jax.Array, jax.Array, Optional[jax.Array]]:
    if use_magma:
        next_key, sr_key, magma_key = jax.random.split(key, 3)
        return next_key, sr_key, magma_key
    next_key, sr_key = jax.random.split(key, 2)
    return next_key, sr_key, None


def init_matrix_magma_state(params: base.Params, use_magma: bool) -> base.Params:
    """Initialize optional Magma state for matrix optimizer transforms."""
    return _init_magma_state(params) if use_magma else ()


def init_matrix_momentum_state(
    params: base.Params,
    dtype: jax.typing.DTypeLike,
) -> base.Params:
    """Initialize a masked-tree-safe first-moment state."""
    return _zeros_like_tree(params, dtype)


def prepare_matrix_runtime_step(
    updates: base.Updates,
    *,
    count: jax.Array,
    mu: base.Updates,
    key: jax.Array,
    beta: jax.typing.ArrayLike,
    nesterov: bool,
    shape_nesterov: bool,
    bias_correction: bool,
    momentum_accumulator: MomentumAccumulator,
    mu_dtype: jax.typing.DTypeLike,
    raw_global_grad_clip: Optional[float],
    permissive_spike_protection: bool,
    use_magma: bool,
    axis_name: Optional[str],
) -> MatrixRuntimeStep:
    """Prepare clipped gradients, momentum, and shaping targets for one step."""
    _validate_positive_static_scalar("raw_global_grad_clip", raw_global_grad_clip)
    next_key, sr_key, magma_key = _split_runtime_keys(key, use_magma)
    count_inc = cast(jax.Array, numerics.safe_increment(count))

    if raw_global_grad_clip is not None:
        grad_norm = _tree_global_norm(updates, axis_name=axis_name)
        is_spike = grad_norm > raw_global_grad_clip
        clip_scale = jnp.where(
            is_spike,
            raw_global_grad_clip / add_tiny(grad_norm),
            1.0,
        )
    else:
        is_spike = jnp.array(False, dtype=jnp.bool_)
        clip_scale = jnp.array(1.0, dtype=jnp.float32)

    should_skip = jnp.logical_and(
        is_spike,
        jnp.logical_not(permissive_spike_protection),
    )

    effective_updates = jax.tree.map(
        lambda g: (
            g
            if _is_aux_leaf(g)
            else jnp.where(should_skip, jnp.zeros_like(g), g * clip_scale)
        ),
        updates,
        is_leaf=_is_aux_leaf,
    )

    mu_candidate = _tree_update_moment_f32(
        effective_updates,
        mu,
        beta,
        momentum_accumulator=momentum_accumulator,
    )
    mu_f32 = _tree_where_scalar(should_skip, _tree_cast_f32(mu), mu_candidate)

    mu_target = mu_f32
    if nesterov:
        if bias_correction:
            mu_bc = _tree_bias_correction_momentum(
                mu_f32,
                beta,
                numerics.safe_increment(count_inc),
                momentum_accumulator=momentum_accumulator,
            )
            updates_bc = _tree_bias_correction_momentum(
                _tree_cast_f32(effective_updates),
                beta,
                count_inc,
                momentum_accumulator=momentum_accumulator,
            )
            mu_target = _tree_momentum_lookahead(
                mu_bc,
                updates_bc,
                beta,
                momentum_accumulator=momentum_accumulator,
            )
        else:
            mu_target = _tree_momentum_lookahead(
                mu_f32,
                effective_updates,
                beta,
                momentum_accumulator=momentum_accumulator,
            )
    elif bias_correction:
        mu_target = _tree_bias_correction_momentum(
            mu_f32,
            beta,
            count_inc,
            momentum_accumulator=momentum_accumulator,
        )

    if mu_dtype == jnp.bfloat16:
        mu_cast = _tree_stochastic_cast(mu_f32, mu_dtype, sr_key)
    else:
        mu_cast = _cast_state_tree(mu_f32, mu_dtype)

    target_for_shape = (
        mu_target
        if shape_nesterov
        else jax.tree.map(lambda _: None, effective_updates, is_leaf=_is_aux_leaf)
    )

    return MatrixRuntimeStep(
        count=count_inc,
        raw_gradients=updates,
        effective_updates=effective_updates,
        mu_f32=mu_f32,
        target_for_shape=target_for_shape,
        mu_cast=mu_cast,
        should_skip=should_skip,
        next_key=next_key,
        magma_key=magma_key,
    )


def apply_matrix_post_shape_lookahead(
    shaped_updates: base.Updates,
    runtime: MatrixRuntimeStep,
    *,
    beta: jax.typing.ArrayLike,
    nesterov: bool,
    shape_nesterov: bool,
    momentum_accumulator: MomentumAccumulator,
) -> base.Updates:
    """Apply post-shaping Nesterov lookahead when requested."""
    if nesterov and not shape_nesterov:
        return _tree_momentum_lookahead(
            shaped_updates,
            runtime.effective_updates,
            beta,
            momentum_accumulator=momentum_accumulator,
        )
    return shaped_updates


def finish_matrix_runtime_step(
    updates: base.Updates,
    runtime: MatrixRuntimeStep,
    *,
    params: Optional[base.Params],
    magma_s: base.Params,
    use_magma: bool,
    magma_p: float,
    magma_tau: float,
    weight_decay: base.ScalarOrSchedule,
    weight_decay_mask: MaskOrFn,
    grad_clip_max_amps: GradClipMaxAmps,
    axis_name: Optional[str],
    guard_fn: Optional[Callable[[jax.Array], jax.Array]] = None,
) -> tuple[base.Updates, base.Params]:
    """Apply shared post-processing after optimizer-specific matrix shaping."""
    _validate_grad_clip_max_amps(grad_clip_max_amps)
    new_updates = updates

    if guard_fn is not None:
        new_updates = jax.tree.map(
            lambda u: u if _is_aux_leaf(u) else guard_fn(u),
            new_updates,
            is_leaf=_is_aux_leaf,
        )

    if grad_clip_max_amps is not None:
        max_rms, max_val = (
            grad_clip_max_amps
            if isinstance(grad_clip_max_amps, tuple)
            else (grad_clip_max_amps, 10.0)
        )
        new_updates = jax.tree.map(
            lambda u: (
                u if _is_aux_leaf(u) else _clip_per_tensor_rms(u, max_rms, max_val)
            ),
            new_updates,
            is_leaf=_is_aux_leaf,
        )

    if _has_nonzero_or_scheduled(weight_decay) and params is None:
        raise ValueError(
            "`params` must be provided to matrix optimizer updates when "
            "`weight_decay` is nonzero or scheduled."
        )

    if _has_nonzero_or_scheduled(weight_decay):
        params = cast(base.Params, params)
        wd_step = _resolve_scalar(
            weight_decay,
            runtime.count - jnp.asarray(1, dtype=runtime.count.dtype),
        )
        new_updates = _apply_weight_decay_tree(
            new_updates,
            params,
            wd_step,
            _resolve_mask(weight_decay_mask, params),
        )

    new_updates = jax.tree.map(
        lambda u: (
            u
            if _is_aux_leaf(u)
            else jnp.where(runtime.should_skip, jnp.zeros_like(u), u)
        ),
        new_updates,
        is_leaf=_is_aux_leaf,
    )

    if use_magma:
        final_updates, new_magma_s = apply_magma_internal(
            raw_gradients=runtime.raw_gradients,
            first_moments=runtime.mu_f32,
            base_updates=new_updates,
            magma_s_prev=magma_s,
            key=runtime.magma_key,
            p=magma_p,
            tau=magma_tau,
            axis_name=axis_name,
        )
        new_magma_s = _tree_where_scalar(runtime.should_skip, magma_s, new_magma_s)
        return final_updates, new_magma_s

    return new_updates, magma_s


__all__ = [
    "GradClipMaxAmps",
    "MatrixRuntimeStep",
    "apply_matrix_post_shape_lookahead",
    "finish_matrix_runtime_step",
    "init_matrix_magma_state",
    "init_matrix_momentum_state",
    "prepare_matrix_runtime_step",
]
