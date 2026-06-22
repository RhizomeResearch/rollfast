"""Small gradient transforms used by fine-tuning builders."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from rollfast.utils import dist_reduce


class GlobalNormClipState(NamedTuple):
    count: jax.Array


def clip_by_global_norm(
    max_norm: float,
    *,
    axis_name: str | tuple[str, ...] | None = None,
) -> optax.GradientTransformation:
    """Clip by global norm, optionally reducing squared norm across devices."""

    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive.")

    def init_fn(params):
        del params
        return GlobalNormClipState(count=jnp.zeros([], dtype=jnp.int32))

    def update_fn(updates, state, params=None):
        del params
        norm = _global_norm(updates, axis_name=axis_name)
        scale = jnp.minimum(1.0, jnp.asarray(max_norm, dtype=jnp.float32) / (norm + 1e-6))
        clipped = jax.tree.map(
            lambda leaf: _scale_leaf(leaf, scale),
            updates,
            is_leaf=lambda x: x is None,
        )
        return clipped, GlobalNormClipState(count=state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


def _global_norm(tree, *, axis_name: str | tuple[str, ...] | None) -> jax.Array:
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in jax.tree.leaves(tree, is_leaf=lambda x: x is None):
        if leaf is None or not hasattr(leaf, "dtype"):
            continue
        arr = jnp.asarray(leaf, dtype=jnp.float32)
        total = total + jnp.sum(jnp.square(arr))
    total = dist_reduce(total, axis_name, "sum")
    return jnp.sqrt(total)


def _scale_leaf(leaf, scale):
    if leaf is None or not hasattr(leaf, "dtype"):
        return leaf
    return leaf * scale.astype(leaf.dtype)


__all__ = ("clip_by_global_norm",)
