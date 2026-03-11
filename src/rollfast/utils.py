from typing import Any, Optional

import jax
import jax.numpy as jnp
import optax
from optax._src import base, numerics
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


def _safe_bias_correction(tree, factor):
    """Safely bypasses MaskedNodes to prevent division TypeErrors in partitioned graphs."""
    return jax.tree.map(
        lambda t: (
            t
            if isinstance(t, _masking.MaskedNode)
            else (t / factor if t is not None else None)
        ),
        tree,
        is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
    )


def _stochastic_round_bf16(x: jax.Array, key: jax.Array) -> jax.Array:
    """
    Bypasses XLA's deterministic RNE casting to preserve sub-representable gradient signals.
    IEEE 754 Infinities (0x7F800000 / 0xFF800000) are explicitly masked out.
    Adding integer noise to Infinity overflows into the NaN domain (0x7F800001+).
    """
    if x.dtype == jnp.bfloat16:
        return x

    x_f32 = x.astype(jnp.float32)
    noise = jax.random.randint(key, x.shape, minval=0, maxval=0x10000, dtype=jnp.uint32)
    x_u32 = jax.lax.bitcast_convert_type(x_f32, jnp.uint32)

    # Isolate non-finite values to prevent Inf -> NaN corruption
    is_finite = jnp.isfinite(x_f32)
    x_u32_noisy = jnp.where(is_finite, x_u32 + noise, x_u32)

    MASK = jnp.array(0xFFFF0000, dtype=jnp.uint32)
    x_u32_rounded = jnp.bitwise_and(x_u32_noisy, MASK)

    return jax.lax.bitcast_convert_type(x_u32_rounded, jnp.float32).astype(jnp.bfloat16)


def _tree_stochastic_cast(
    tree: base.Params, target_dtype: Any, key: jax.Array
) -> base.Params:
    """
    Safely maps stochastic rounding across a PyTree while ignoring Partitioning masks
    and non-differentiable static leaves (e.g., Equinox Callables).
    """
    is_leaf_fn = lambda x: isinstance(x, _masking.MaskedNode) or x is None
    leaves, treedef = jax.tree.flatten(tree, is_leaf=is_leaf_fn)
    keys = jax.random.split(key, len(leaves))

    def _cast_leaf(leaf, k):
        # Short-circuit MaskedNodes and static Callables before ANY cast attempt.
        # This completely replaces the unsafe `optax.tree.cast` fast-path.
        if (
            leaf is None
            or isinstance(leaf, _masking.MaskedNode)
            or not hasattr(leaf, "dtype")
        ):
            return leaf

        if target_dtype == jnp.bfloat16:
            return _stochastic_round_bf16(leaf, k)
        return leaf.astype(target_dtype)

    rounded_leaves = [_cast_leaf(leaf, k) for leaf, k in zip(leaves, keys)]
    return jax.tree.unflatten(treedef, rounded_leaves)


def _tree_stochastic_cast_like(
    tree: base.Params, reference_tree: base.Params, key: jax.Array
) -> base.Params:
    """
    A stochastic equivalent to `optax.tree.cast_like`, strictly enforcing isomorphism.
    """
    is_leaf_fn = lambda x: isinstance(x, _masking.MaskedNode) or x is None
    leaves_x, treedef_x = jax.tree.flatten(tree, is_leaf=is_leaf_fn)
    leaves_ref, _ = jax.tree.flatten(reference_tree, is_leaf=is_leaf_fn)
    keys = jax.random.split(key, len(leaves_x))

    def _stochastic_cast_like_leaf(x_leaf, ref_leaf, k):
        if (
            x_leaf is None
            or ref_leaf is None
            or isinstance(x_leaf, _masking.MaskedNode)
            or not hasattr(x_leaf, "dtype")
        ):
            return x_leaf
        tgt_dtype = getattr(ref_leaf, "dtype", x_leaf.dtype)
        if tgt_dtype == jnp.bfloat16:
            return _stochastic_round_bf16(x_leaf, k)
        return x_leaf.astype(tgt_dtype)

    return jax.tree.unflatten(
        treedef_x,
        [
            _stochastic_cast_like_leaf(x, r, k)
            for x, r, k in zip(leaves_x, leaves_ref, keys)
        ],
    )


def _tree_update_moment_f32(
    updates: base.Updates, moments: base.Updates, decay: float
) -> base.Updates:
    """
    Prevents accumulator truncation. Optax internally casts to the state's dtype.
    This enforces strict FP32 calculation regardless of the state's storage precision.
    """

    def _update_moment_f32(g, t):
        if g is None or t is None or isinstance(g, _masking.MaskedNode):
            return g
        return (1.0 - decay) * g.astype(jnp.float32) + decay * t.astype(jnp.float32)

    return jax.tree.map(
        _update_moment_f32,
        updates,
        moments,
        is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
    )


def _tree_update_moment_sq_f32(
    updates: base.Updates, moments: base.Updates, decay: float
) -> base.Updates:
    """
    Prevents sub-normal variance from vanishing. Squaring an unscaled bf16 gradient
    instantly underflows if magnitude < 2^-64. Upcasting BEFORE squaring mitigates this.
    """

    def _update_moment_sq_f32(g, t):
        if g is None or t is None or isinstance(g, _masking.MaskedNode):
            return g
        g_f32 = g.astype(jnp.float32)
        sq_norm = jnp.square(g_f32) if jnp.isrealobj(g) else numerics.abs_sq(g_f32)
        return (1.0 - decay) * sq_norm + decay * t.astype(jnp.float32)

    return jax.tree.map(
        _update_moment_sq_f32,
        updates,
        moments,
        is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
    )


def _compute_ema_f32(m, u, b1):
    if m is None or u is None:
        return None
    return b1 * m.astype(jnp.float32) + (1.0 - b1) * u.astype(jnp.float32)


def stochastic_apply_updates(
    params: base.Params, updates: base.Updates, key: jax.Array
) -> base.Params:
    """Applies an update to the corresponding parameters with stochastic rounding for bf16.

    Why:
    1. Mirrors optax.apply_updates structural traversal and exact dtype enforcement.
    2. Overrides the deterministic RNE cast ONLY for bfloat16 parameters, preventing
       sub-representable update signals from collapsing to zero.
    """

    is_leaf_fn = lambda x: isinstance(x, _masking.MaskedNode) or x is None
    leaves, treedef = jax.tree.flatten(params, is_leaf=is_leaf_fn)
    keys = jax.random.split(key, len(leaves))
    keys_tree = jax.tree.unflatten(treedef, list(keys))

    def _apply_leaf(p, u, k):
        # Short-circuit when update is absent (Equinox static modules)
        # or when explicitly masked by Optax multi-transform behavior.
        if (
            p is None
            or u is None
            or isinstance(u, _masking.MaskedNode)
            or isinstance(p, _masking.MaskedNode)
        ):
            return p

        p_arr = jnp.asarray(p)
        u_arr = jnp.asarray(u)

        if p_arr.dtype == jnp.bfloat16:
            # Strictly FP32 addition BEFORE quantization
            sum_f32 = p_arr.astype(jnp.float32) + u_arr.astype(jnp.float32)
            return _stochastic_round_bf16(sum_f32, k)

        # Strict Optax parity: Force output to match the parameter's original dtype
        return jnp.asarray(p_arr + u_arr).astype(p_arr.dtype)

    return jax.tree.map(
        _apply_leaf,
        params,
        updates,
        keys_tree,
        is_leaf=is_leaf_fn,
    )
