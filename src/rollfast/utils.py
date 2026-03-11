from typing import Any, Optional

import jax
import jax.numpy as jnp
import optax
from optax._src import base
from optax.transforms import _masking


def add_tiny(x):
    """Add smallest normal number to avoid division by zero."""
    return x + jnp.finfo(x.dtype).tiny


def dist_reduce(x: jax.Array, axis_name: Optional[str], op: str = "mean") -> jax.Array:
    """Applies a distributed reduction (pmean, pmax, psum) if an axis name is provided.

    Args:
        x: The input array.
        axis_name: The name of the mapped axis (e.g., 'batch'). If None, returns x unchanged.
        op: The reduction operation ('mean', 'max', 'sum').

    Returns:
        The reduced array across devices.
    """
    if axis_name is None:
        return x
    if op == "mean":
        return jax.lax.pmean(x, axis_name=axis_name)
    elif op == "max":
        return jax.lax.pmax(x, axis_name=axis_name)
    elif op == "sum":
        return jax.lax.psum(x, axis_name=axis_name)
    return x


def _stochastic_round_bf16(x: jax.Array, key: jax.Array) -> jax.Array:
    """
    Bypasses XLA's deterministic RNE casting to preserve sub-representable gradient signals.
    IEEE 754 Infinities (0x7F800000 / 0xFF800000) are explicitly masked out.
    Adding integer noise to Infinity overflows into the NaN domain (0x7F800001+).
    Time: O(N), Space: O(N).
    """
    if x.dtype == jnp.bfloat16:
        return x

    x_f32 = x.astype(jnp.float32)
    noise = jax.random.randint(key, x.shape, minval=0, maxval=0x10000, dtype=jnp.uint32)
    x_u32 = jax.lax.bitcast_convert_type(x_f32, jnp.uint32)

    # Isolate non-finite values to prevent Inf -> NaN corruption
    is_finite = jnp.isfinite(x_f32)
    x_u32_noisy = jnp.where(is_finite, x_u32 + noise, x_u32)

    x_u32_rounded = jnp.bitwise_and(x_u32_noisy, 0xFFFF0000)
    return jax.lax.bitcast_convert_type(x_u32_rounded, jnp.float32).astype(jnp.bfloat16)


def _tree_stochastic_cast(
    tree: base.Params, target_dtype: Any, key: jax.Array
) -> base.Params:
    """
    Safely maps stochastic rounding across a PyTree while ignoring Partitioning masks.
    Time: O(N), Space: O(N).
    """
    if target_dtype != jnp.bfloat16:
        return optax.tree.cast(tree, target_dtype)

    leaves, treedef = jax.tree.flatten(tree)
    keys = jax.random.split(key, len(leaves))

    rounded_leaves = [
        _stochastic_round_bf16(leaf, k)
        if leaf is not None and not isinstance(leaf, _masking.MaskedNode)
        else leaf
        for leaf, k in zip(leaves, keys)
    ]
    return jax.tree.unflatten(treedef, rounded_leaves)


def _compute_ema_f32(m, u, b1):
    if m is None or u is None:
        return None
    return b1 * m.astype(jnp.float32) + (1.0 - b1) * u.astype(jnp.float32)
