"""Small gradient transforms used by fine-tuning builders."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from rollfast.utils import AxisName, dist_reduce, resolve_partition_norm_axis_name


class GlobalNormClipState(NamedTuple):
    """State for the global-norm clipping transform."""

    count: jax.Array


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
        arr = jnp.asarray(leaf, dtype=jnp.float32)
        total = total + jnp.sum(jnp.square(arr))
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
    return leaf * scale.astype(leaf.dtype)


__all__ = ("clip_by_global_norm",)
