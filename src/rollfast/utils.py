from typing import Any, Callable, Literal, Optional, TypeAlias, cast

import jax
import jax.numpy as jnp
from optax._src import base, numerics
from optax.transforms import _masking

MomentumAccumulator: TypeAlias = Literal["ema", "heavy_ball"]
MaskOrFn: TypeAlias = Optional[Any | Callable[[base.Params], Any]]


def _is_aux_leaf(x: Any) -> bool:
    return x is None or isinstance(x, _masking.MaskedNode)


def _zeros_like_tree(params: base.Params, dtype: jax.typing.DTypeLike) -> base.Params:
    target_dtype = jnp.dtype(dtype)

    def _zeros_leaf(x):
        if _is_aux_leaf(x):
            return x
        leaf_dtype = (
            jnp.complex64
            if jnp.issubdtype(x.dtype, jnp.complexfloating)
            and not jnp.issubdtype(target_dtype, jnp.complexfloating)
            else target_dtype
        )
        return jnp.zeros_like(x, dtype=leaf_dtype)

    return jax.tree.map(
        _zeros_leaf,
        params,
        is_leaf=_is_aux_leaf,
    )


def _cast_state_tree(tree: base.Params, dtype: jax.typing.DTypeLike) -> base.Params:
    target_dtype = jnp.dtype(dtype)

    def _cast_leaf(x):
        if _is_aux_leaf(x):
            return x
        if jnp.issubdtype(x.dtype, jnp.complexfloating) and not jnp.issubdtype(
            target_dtype, jnp.complexfloating
        ):
            return x.astype(jnp.complex64)
        return x.astype(target_dtype)

    return jax.tree.map(
        _cast_leaf,
        tree,
        is_leaf=_is_aux_leaf,
    )


def _init_magma_state(params: base.Params) -> base.Params:
    return jax.tree.map(
        lambda x: x if _is_aux_leaf(x) else jnp.array(0.5, dtype=jnp.float32),
        params,
        is_leaf=_is_aux_leaf,
    )


def _apply_weight_decay_leaf(
    update: jax.Array,
    param: jax.Array,
    weight_decay_step: jax.typing.ArrayLike,
    mask: Any = True,
) -> jax.Array:
    """Add decoupled weight decay to one update leaf with array-mask support."""
    if _is_aux_leaf(update) or _is_aux_leaf(param) or _is_aux_leaf(mask):
        return update

    decayed_update = update + weight_decay_step * param.astype(update.dtype)
    return jnp.where(jnp.asarray(mask, dtype=jnp.bool_), decayed_update, update)


def _resolve_scalar(
    value: base.ScalarOrSchedule,
    count: jax.Array,
) -> jax.typing.ArrayLike:
    """Resolve a scalar-or-schedule value at the given optimizer count."""
    if callable(value):
        return cast(Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike], value)(
            count
        )
    return value


def _has_nonzero_or_scheduled(value: base.ScalarOrSchedule) -> bool:
    """Return True when a scalar-or-schedule value may affect updates."""
    return not isinstance(value, (int, float)) or value > 0.0


def _is_mask_callable(mask: Any) -> bool:
    callable_leaves = jax.tree.leaves(jax.tree.map(callable, mask))
    return callable(mask) and len(callable_leaves) > 0 and all(callable_leaves)


def _resolve_mask(
    mask: MaskOrFn,
    params: base.Params,
    default_fn: Callable[[base.Params], base.Params] | None = None,
) -> base.Params | None:
    """Resolve a callable-or-tree mask, optionally using a default when unset."""
    if mask is None:
        return None if default_fn is None else default_fn(params)
    if _is_mask_callable(mask):
        return cast(Callable[[base.Params], Any], mask)(params)
    return cast(base.Params, mask)


def _apply_weight_decay_tree(
    updates: base.Updates,
    params: base.Params,
    weight_decay_step: jax.typing.ArrayLike,
    weight_decay_mask: base.Params | None = None,
    *,
    is_leaf: Callable[[Any], bool] = _is_aux_leaf,
) -> base.Updates:
    """Apply decoupled weight decay across a tree with optional array masks."""
    if weight_decay_mask is None:
        return jax.tree.map(
            lambda u, p: _apply_weight_decay_leaf(u, p, weight_decay_step),
            updates,
            params,
            is_leaf=is_leaf,
        )
    return jax.tree.map(
        lambda u, p, m: _apply_weight_decay_leaf(u, p, weight_decay_step, m),
        updates,
        params,
        weight_decay_mask,
        is_leaf=is_leaf,
    )


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
    if op not in ("mean", "max", "sum"):
        raise ValueError(f"op must be one of 'mean', 'max', or 'sum', got {op!r}.")
    if axis_name is None:
        return x
    if op == "mean":
        return jax.lax.pmean(x, axis_name=axis_name)
    if op == "max":
        return jax.lax.pmax(x, axis_name=axis_name)
    return jax.lax.psum(x, axis_name=axis_name)


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


def _momentum_grad_scale(
    decay: jax.typing.ArrayLike,
    momentum_accumulator: MomentumAccumulator,
) -> jax.Array:
    decay32 = jnp.asarray(decay, dtype=jnp.float32)
    if momentum_accumulator == "ema":
        return 1.0 - decay32
    if momentum_accumulator == "heavy_ball":
        return jnp.array(1.0, dtype=jnp.float32)
    raise ValueError(
        "momentum_accumulator must be 'ema' or 'heavy_ball', got "
        f"{momentum_accumulator!r}."
    )


def _stochastic_round_bf16(x: jax.Array, key: jax.Array) -> jax.Array:
    """Stochastic rounding from FP32 → BF16 via uint32 bit manipulation.

    For ±Inf (0x7F800000 / 0xFF800000), max noise 0xFFFF yields
    0x7F80FFFF / 0xFF80FFFF. The mask (& 0xFFFF0000) truncates
    these back to the original Inf encoding. No NaN can escape.

    randint() internally upcasts raw PRNG output to float32 for range
    scaling, producing float intermediates that break XLA's integer-only
    fusion chain and force HBM spills. bits() emits a single Threefry
    kernel in uint32, keeping the full pipeline integer-typed and fusible.

    Args:
        x: The input array to be rounded.
        key: A PRNG key for generating noise.

    Returns:
        The stochastically rounded array in bfloat16.
    """
    if x.dtype == jnp.bfloat16:
        return x

    x_f32 = x.astype(jnp.float32)
    noise = jax.random.bits(key, x_f32.shape, dtype=jnp.uint32) & jnp.uint32(0xFFFF)
    x_u32 = jax.lax.bitcast_convert_type(x_f32, jnp.uint32)
    abs_u32 = x_u32 & jnp.uint32(0x7FFFFFFF)
    is_special = abs_u32 >= jnp.uint32(0x7F800000)
    noise = jnp.where(is_special, jnp.uint32(0), noise)
    rounded = (x_u32 + noise) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(rounded, jnp.float32).astype(jnp.bfloat16)


def _tree_stochastic_cast(
    tree: base.Params, target_dtype: Any, key: jax.Array
) -> base.Params:
    """Safely maps stochastic rounding across a PyTree while ignoring Partitioning masks
    and non-differentiable static leaves (e.g., Equinox Callables).

    Args:
        tree: The input PyTree of parameters or updates.
        target_dtype: The target dtype for casting.
        key: A PRNG key for generating noise during rounding.

    Returns:
        A new PyTree with the casted and stochastically rounded leaves.
    """
    is_leaf_fn = lambda x: isinstance(x, _masking.MaskedNode) or x is None
    leaves, treedef = jax.tree.flatten(tree, is_leaf=is_leaf_fn)
    keys = jax.random.split(key, len(leaves))

    canonical_target_dtype = jnp.dtype(target_dtype)

    def _cast_leaf(leaf, k):
        # Short-circuit MaskedNodes and static Callables before ANY cast attempt.
        # This completely replaces the unsafe `optax.tree.cast` fast-path.
        if (
            leaf is None
            or isinstance(leaf, _masking.MaskedNode)
            or not hasattr(leaf, "dtype")
        ):
            return leaf

        if jnp.issubdtype(leaf.dtype, jnp.complexfloating) and not jnp.issubdtype(
            canonical_target_dtype, jnp.complexfloating
        ):
            return leaf.astype(jnp.complex64)

        if canonical_target_dtype == jnp.dtype(jnp.bfloat16):
            return _stochastic_round_bf16(leaf, k)
        return leaf.astype(canonical_target_dtype)

    rounded_leaves = [_cast_leaf(leaf, k) for leaf, k in zip(leaves, keys)]
    return jax.tree.unflatten(treedef, rounded_leaves)


def _tree_update_moment_f32(
    updates: base.Updates,
    moments: base.Updates,
    decay: jax.typing.ArrayLike,
    *,
    momentum_accumulator: MomentumAccumulator = "ema",
) -> base.Updates:
    """
    Prevents accumulator truncation. Optax internally casts to the state's dtype.
    This enforces strict FP32 calculation regardless of the state's storage precision.
    """
    decay32 = jnp.asarray(decay, dtype=jnp.float32)
    grad_scale = _momentum_grad_scale(decay32, momentum_accumulator)

    def _update_moment_f32(g, t):
        if (
            g is None
            or t is None
            or isinstance(g, _masking.MaskedNode)
            or isinstance(t, _masking.MaskedNode)
        ):
            return g
        compute_dtype = (
            jnp.complex64
            if jnp.issubdtype(g.dtype, jnp.complexfloating)
            or jnp.issubdtype(t.dtype, jnp.complexfloating)
            else jnp.float32
        )
        return grad_scale * g.astype(compute_dtype) + decay32 * t.astype(compute_dtype)

    return jax.tree.map(
        _update_moment_f32,
        updates,
        moments,
        is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
    )


def _tree_update_moment_sq_f32(
    updates: base.Updates, moments: base.Updates, decay: jax.typing.ArrayLike
) -> base.Updates:
    """
    Prevents sub-normal variance from vanishing. Squaring an unscaled bf16 gradient
    instantly underflows if magnitude < 2^-64. Upcasting BEFORE squaring mitigates this.
    """

    def _update_moment_sq_f32(g, t):
        if (
            g is None
            or t is None
            or isinstance(g, _masking.MaskedNode)
            or isinstance(t, _masking.MaskedNode)
        ):
            return g
        if jnp.issubdtype(g.dtype, jnp.complexfloating):
            sq_norm = numerics.abs_sq(g.astype(jnp.complex64))
        else:
            sq_norm = jnp.square(g.astype(jnp.float32))
        t_real = jnp.real(t).astype(jnp.float32)
        return (1.0 - decay) * sq_norm + decay * t_real

    return jax.tree.map(
        _update_moment_sq_f32,
        updates,
        moments,
        is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
    )


def _tree_bias_correction_momentum(
    tree: base.Updates,
    decay: jax.typing.ArrayLike,
    count: jax.typing.ArrayLike,
    *,
    momentum_accumulator: MomentumAccumulator,
) -> base.Updates:
    if momentum_accumulator == "heavy_ball":
        return _cast_state_tree(tree, jnp.float32)
    correction = 1.0 - jnp.power(jnp.asarray(decay, dtype=jnp.float32), count)
    return _safe_bias_correction(tree, correction)


def _tree_momentum_lookahead(
    moments: base.Updates,
    updates: base.Updates,
    decay: jax.typing.ArrayLike,
    *,
    momentum_accumulator: MomentumAccumulator,
) -> base.Updates:
    decay32 = jnp.asarray(decay, dtype=jnp.float32)
    grad_scale = _momentum_grad_scale(decay32, momentum_accumulator)

    def _lookahead_leaf(m, g):
        if _is_aux_leaf(m) or _is_aux_leaf(g):
            return m
        compute_dtype = (
            jnp.complex64
            if jnp.issubdtype(m.dtype, jnp.complexfloating)
            or jnp.issubdtype(g.dtype, jnp.complexfloating)
            else jnp.float32
        )
        return decay32 * m.astype(compute_dtype) + grad_scale * g.astype(compute_dtype)

    return jax.tree.map(_lookahead_leaf, moments, updates, is_leaf=_is_aux_leaf)


def apply_updates(
    params: base.Params,
    updates: base.Updates,
    key: jax.Array,
    stochastic: bool = True,
) -> base.Params:
    """Applies an update to the corresponding parameters with optional stochastic
    rounding for bf16.

    Why:
    1. Mirrors optax.apply_updates structural traversal and exact dtype enforcement.
    2. Overrides the deterministic RNE cast ONLY for bfloat16 parameters, preventing
       sub-representable update signals from collapsing to zero.

    Args:
        params: The current parameters.
        updates: The updates to apply.
        key: A PRNG key for stochastic rounding. Required if stochastic=True.
        stochastic: Whether to use stochastic rounding for bfloat16 params (default: True).

    Returns:
        The updated parameters.
    """
    if stochastic and key is None:
        raise ValueError(
            "apply_updates requires a PRNG `key` when stochastic=True. "
            "Pass stochastic=False for deterministic rounding."
        )

    is_leaf_fn = lambda x: isinstance(x, _masking.MaskedNode) or x is None
    leaves, treedef = jax.tree.flatten(params, is_leaf=is_leaf_fn)

    # Skip PRNG work entirely when deterministic — avoids materializing
    # len(leaves) unused uint32[2] arrays on device.
    if stochastic:
        keys = jax.random.split(key, len(leaves))
        keys_tree = jax.tree.unflatten(treedef, list(keys))
    else:
        keys_tree = jax.tree.unflatten(treedef, [None] * len(leaves))

    def _apply_leaf(p, u, k):
        if (
            p is None
            or u is None
            or isinstance(p, _masking.MaskedNode)
            or isinstance(u, _masking.MaskedNode)
        ):
            return p

        p_arr = jnp.asarray(p)
        u_arr = jnp.asarray(u)

        if p_arr.dtype == jnp.bfloat16:
            sum_f32 = p_arr.astype(jnp.float32) + u_arr.astype(jnp.float32)
            if stochastic:
                return _stochastic_round_bf16(sum_f32, k)
            return sum_f32.astype(jnp.bfloat16)

        return jnp.asarray(p_arr + u_arr).astype(p_arr.dtype)

    return jax.tree.map(
        _apply_leaf,
        params,
        updates,
        keys_tree,
        is_leaf=is_leaf_fn,
    )


def apply_updates_prefix(
    model: base.Params,
    updates: base.Updates,
    key: jax.Array,
    stochastic: bool = True,
) -> base.Params:
    """Equinox-compatible apply_updates: `updates` may be a prefix of `model`.

    Semantics:
    - None / MaskedNode update: corresponding model subtree returned unchanged.
    - Otherwise: model + update, with stochastic rounding for bfloat16 leaves.

    `updates` is the first tree passed to `jax.tree.map`, so its structure
    is the reference. `flatten_up_to` extracts entire model subtrees at
    positions where updates is a leaf (None), preserving non-differentiable
    Equinox leaves (Callables, static fields) without any cast attempt.

    Args:
        model: The current model parameters.
        updates: The updates to apply (may be a prefix tree of the model).
        key: A PRNG key for stochastic rounding. Required if stochastic=True.
        stochastic: Whether to use stochastic rounding for bfloat16 params (default: True).

    Returns:
        The updated model parameters.
    """
    if stochastic and key is None:
        raise ValueError(
            "apply_updates requires a PRNG `key` when stochastic=True. "
            "Pass stochastic=False for deterministic rounding."
        )

    is_leaf_fn = lambda x: isinstance(x, _masking.MaskedNode) or x is None

    update_leaves, update_treedef = jax.tree.flatten(updates, is_leaf=is_leaf_fn)

    if stochastic:
        keys = jax.random.split(key, len(update_leaves))
        keys_tree = jax.tree.unflatten(update_treedef, list(keys))
    else:
        keys_tree = jax.tree.unflatten(update_treedef, [None] * len(update_leaves))

    def _apply_update(u, p, k):
        if u is None or isinstance(u, _masking.MaskedNode):
            return p
        if p is None or isinstance(p, _masking.MaskedNode):
            return p
        # Non-array leaves (e.g. Equinox Callables) should never receive a
        # non-None update; surface the error immediately rather than masking it.
        p_arr = jnp.asarray(p)
        u_arr = jnp.asarray(u)

        if p_arr.dtype == jnp.bfloat16:
            sum_f32 = p_arr.astype(jnp.float32) + u_arr.astype(jnp.float32)
            if stochastic:
                return _stochastic_round_bf16(sum_f32, k)
            return sum_f32.astype(jnp.bfloat16)

        return jnp.asarray(p_arr + u_arr).astype(p_arr.dtype)

    return jax.tree.map(_apply_update, updates, model, keys_tree, is_leaf=is_leaf_fn)
