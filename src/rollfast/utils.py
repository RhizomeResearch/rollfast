from collections.abc import Callable
from typing import Any, Literal, NamedTuple, TypeAlias, cast

import jax
import jax.numpy as jnp
from optax._src import base, numerics
from optax.transforms import _masking

MomentumAccumulator: TypeAlias = Literal["ema", "heavy_ball"]
MaskOrFn: TypeAlias = Any | Callable[[base.Params], Any] | None


def _is_aux_leaf(x: Any) -> bool:
    return x is None or isinstance(x, _masking.MaskedNode)


def _map_non_aux(fn: Callable[[jax.Array], jax.Array], tree: Any) -> Any:
    return jax.tree.map(
        lambda x: x if _is_aux_leaf(x) else fn(x),
        tree,
        is_leaf=_is_aux_leaf,
    )


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
        return zeros_like_preserving_sharding(x, leaf_dtype)

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
            return astype_preserving_sharding(x, jnp.complex64)
        return astype_preserving_sharding(x, target_dtype)

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


def _fresh_prng_key(key: jax.Array | None, seed: int = 42) -> jax.Array:
    """Return a fresh PRNG key buffer for optimizer state initialization."""
    source = jax.random.PRNGKey(seed) if key is None else key
    return jnp.array(source, copy=True)


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
    if isinstance(value, (int, float)):
        if value < 0.0:
            raise ValueError("Scalar optimizer coefficients must be nonnegative.")
        return value != 0.0
    return True


def _validate_positive_static_scalar(
    name: str,
    value: jax.typing.ArrayLike | None,
) -> None:
    """Validate static Python scalar knobs that must be positive."""
    if isinstance(value, (int, float)) and value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _validate_nonnegative_static_scalar(
    name: str,
    value: jax.typing.ArrayLike | None,
) -> None:
    """Validate static Python scalar knobs that must be nonnegative."""
    if isinstance(value, (int, float)) and value < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}.")


def _validate_beta_static_scalar(
    name: str,
    value: jax.typing.ArrayLike | None,
) -> None:
    """Validate static Python scalar knobs that must be in [0, 1)."""
    if isinstance(value, (int, float)) and not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value!r}.")


def _validate_grad_clip_max_amps(
    grad_clip_max_amps: float | tuple[float, float] | None,
) -> None:
    """Validate static post-shaping clipping thresholds."""
    if grad_clip_max_amps is None:
        return
    if isinstance(grad_clip_max_amps, tuple):
        max_rms, max_val = grad_clip_max_amps
        _validate_positive_static_scalar("grad_clip_max_amps[0]", max_rms)
        _validate_positive_static_scalar("grad_clip_max_amps[1]", max_val)
        return
    _validate_positive_static_scalar("grad_clip_max_amps", grad_clip_max_amps)


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
    _validate_nonnegative_static_scalar("weight_decay", weight_decay_step)
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


AxisName = str | tuple[str, ...]


def with_reference_sharding(value: Any, reference: Any) -> Any:
    """Constrain ``value`` to the same JAX sharding as ``reference`` when possible."""

    if value is None or reference is None:
        return value
    sharding = getattr(reference, "sharding", None)
    if sharding is None or not hasattr(value, "shape"):
        return value
    if getattr(value, "sharding", None) == sharding:
        return value
    return jax.lax.with_sharding_constraint(value, sharding)


def astype_preserving_sharding(
    value: Any, dtype: Any, reference: Any | None = None
) -> Any:
    """Cast an array and keep its reference sharding explicit."""

    if value is None:
        return None
    cast = value.astype(dtype)
    return with_reference_sharding(cast, value if reference is None else reference)


def zeros_like_preserving_sharding(value: Any, dtype: Any | None = None) -> Any:
    """Create zeros that explicitly follow the input leaf's sharding."""

    if value is None:
        return None
    zeros = (
        jnp.zeros_like(value) if dtype is None else jnp.zeros_like(value, dtype=dtype)
    )
    return with_reference_sharding(zeros, value)


def dist_reduce(
    x: jax.Array,
    axis_name: AxisName | None,
    op: str = "mean",
) -> jax.Array:
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


def resolve_partition_norm_axis_name(
    *,
    axis_name: AxisName | None = None,
    partition_axis_names: AxisName | None = None,
    replicated_axis_names: AxisName | None = None,
) -> AxisName | None:
    """Resolve the collective axes that own parameter-value shards.

    ``axis_name`` is the legacy distributed-axis argument. When
    ``partition_axis_names`` is provided it wins, because replicated axes must
    not contribute to the squared parameter norm. When only legacy
    ``axis_name`` is present, explicit ``replicated_axis_names`` are filtered
    out.
    """

    partition_axes = _axis_name_tuple(partition_axis_names)
    replicated_axes = _axis_name_tuple(replicated_axis_names)
    overlap = set(partition_axes) & set(replicated_axes)
    if overlap:
        raise ValueError(
            "partition_axis_names and replicated_axis_names must be disjoint: "
            f"{tuple(sorted(overlap))!r}."
        )
    if partition_axis_names is not None:
        return _axis_name_or_none(partition_axes)

    axes = _axis_name_tuple(axis_name)
    if not axes:
        return None
    if replicated_axes:
        axes = tuple(axis for axis in axes if axis not in replicated_axes)
    return _axis_name_or_none(axes)


def _axis_name_tuple(axis_name: AxisName | None) -> tuple[str, ...]:
    if axis_name is None:
        return ()
    if isinstance(axis_name, str):
        return (axis_name,)
    return tuple(axis_name)


def _axis_name_or_none(axis_names: tuple[str, ...]) -> AxisName | None:
    if not axis_names:
        return None
    if len(axis_names) == 1:
        return axis_names[0]
    return axis_names


def _safe_bias_correction(tree, factor):
    """Safely bypasses MaskedNodes to prevent division TypeErrors in partitioned graphs."""
    return jax.tree.map(
        lambda t: t if _is_aux_leaf(t) else t / factor,
        tree,
        is_leaf=_is_aux_leaf,
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
    leaves, treedef = jax.tree.flatten(tree, is_leaf=_is_aux_leaf)
    keys = jax.random.split(key, len(leaves))

    canonical_target_dtype = jnp.dtype(target_dtype)

    def _cast_leaf(leaf, k):
        # Short-circuit MaskedNodes and static Callables before ANY cast attempt.
        # This completely replaces the unsafe `optax.tree.cast` fast-path.
        if _is_aux_leaf(leaf) or not hasattr(leaf, "dtype"):
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


def _store_moment_tree(
    tree: base.Params,
    dtype: jax.typing.DTypeLike,
    key: jax.Array | None,
    *,
    sr_key: jax.Array | None = None,
) -> tuple[base.Params, jax.Array | None]:
    """Store a moment tree, using stochastic rounding for BF16 state."""
    target_dtype = jnp.dtype(dtype)
    if target_dtype == jnp.dtype(jnp.bfloat16):
        if sr_key is None:
            key, sr_key = jax.random.split(cast(jax.Array, key), 2)
        return _tree_stochastic_cast(tree, target_dtype, cast(jax.Array, sr_key)), key
    return _cast_state_tree(tree, target_dtype), key


def _unzip_leaf_tuple_tree(tree: base.Params, width: int) -> tuple[base.Params, ...]:
    """Split a PyTree whose leaves are fixed-width tuples into tuple components."""
    is_tuple_leaf = lambda x: isinstance(x, tuple) and len(x) == width
    return tuple(
        jax.tree.map(lambda x, i=i: x[i], tree, is_leaf=is_tuple_leaf)
        for i in range(width)
    )


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
        if _is_aux_leaf(g) or _is_aux_leaf(t):
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
        is_leaf=_is_aux_leaf,
    )


def _tree_update_moment_sq_f32(
    updates: base.Updates, moments: base.Updates, decay: jax.typing.ArrayLike
) -> base.Updates:
    """
    Prevents sub-normal variance from vanishing. Squaring an unscaled bf16 gradient
    instantly underflows if magnitude < 2^-64. Upcasting BEFORE squaring mitigates this.
    """

    def _update_moment_sq_f32(g, t):
        if _is_aux_leaf(g) or _is_aux_leaf(t):
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
        is_leaf=_is_aux_leaf,
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


class FirstMomentRuntime(NamedTuple):
    """Prepared first-moment state for Muon-family matrix transforms."""

    count: jax.Array
    mu: base.Updates
    direction: base.Updates
    mu_stored: base.Updates
    key: jax.Array | None
    sr_key: jax.Array | None
    extra_keys: tuple[jax.Array, ...]


def _split_first_moment_keys(
    key: jax.Array | None,
    dtype: jax.typing.DTypeLike,
    *,
    reserve_sr_key: bool,
    extra_key_count: int,
) -> tuple[jax.Array | None, jax.Array | None, tuple[jax.Array, ...]]:
    target_dtype = jnp.dtype(dtype)
    needs_sr_key = target_dtype == jnp.dtype(jnp.bfloat16) or reserve_sr_key
    split_count = 1 + int(needs_sr_key) + extra_key_count
    if split_count == 1:
        return key, None, ()

    split_keys = jax.random.split(cast(jax.Array, key), split_count)
    next_key = split_keys[0]
    sr_key = split_keys[1] if needs_sr_key else None
    extra_start = 2 if needs_sr_key else 1
    return next_key, sr_key, tuple(split_keys[extra_start:])


def _prepare_first_moment_runtime(
    updates: base.Updates,
    moments: base.Updates,
    count: jax.Array,
    key: jax.Array | None,
    decay: jax.typing.ArrayLike,
    mu_dtype: jax.typing.DTypeLike,
    *,
    nesterov: bool,
    bias_correction: bool,
    momentum_accumulator: MomentumAccumulator,
    nesterov_moment_count_offset: int = 0,
    reserve_sr_key: bool = False,
    extra_key_count: int = 0,
) -> FirstMomentRuntime:
    """Update first moment, form the direction, store state, and advance keys."""
    count_inc = cast(jax.Array, numerics.safe_increment(count))
    mu = _tree_update_moment_f32(
        updates,
        moments,
        decay,
        momentum_accumulator=momentum_accumulator,
    )

    if bias_correction:
        moment_count = count_inc
        if nesterov:
            for _ in range(nesterov_moment_count_offset):
                moment_count = cast(jax.Array, numerics.safe_increment(moment_count))
        mu_target = _tree_bias_correction_momentum(
            mu,
            decay,
            moment_count,
            momentum_accumulator=momentum_accumulator,
        )
        updates_target = (
            _tree_bias_correction_momentum(
                updates,
                decay,
                count_inc,
                momentum_accumulator=momentum_accumulator,
            )
            if nesterov
            else updates
        )
    else:
        mu_target = mu
        updates_target = updates

    if nesterov:
        direction = _tree_momentum_lookahead(
            mu_target,
            updates_target,
            decay,
            momentum_accumulator=momentum_accumulator,
        )
    else:
        direction = mu_target

    next_key, sr_key, extra_keys = _split_first_moment_keys(
        key,
        mu_dtype,
        reserve_sr_key=reserve_sr_key,
        extra_key_count=extra_key_count,
    )
    mu_stored, next_key = _store_moment_tree(
        mu,
        mu_dtype,
        next_key,
        sr_key=sr_key,
    )

    return FirstMomentRuntime(
        count=count_inc,
        mu=mu,
        direction=direction,
        mu_stored=mu_stored,
        key=next_key,
        sr_key=sr_key,
        extra_keys=extra_keys,
    )


def apply_updates(
    params: base.Params,
    updates: base.Updates,
    key: jax.Array | None = None,
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

    leaves, treedef = jax.tree.flatten(params, is_leaf=_is_aux_leaf)

    # Skip PRNG work entirely when deterministic — avoids materializing
    # len(leaves) unused uint32[2] arrays on device.
    if stochastic:
        keys = jax.random.split(cast(jax.Array, key), len(leaves))
        keys_tree = jax.tree.unflatten(treedef, list(keys))
    else:
        keys_tree = jax.tree.unflatten(treedef, [None] * len(leaves))

    def _apply_leaf(p, u, k):
        if _is_aux_leaf(p) or _is_aux_leaf(u):
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
        is_leaf=_is_aux_leaf,
    )


def apply_updates_prefix(
    model: base.Params,
    updates: base.Updates,
    key: jax.Array | None = None,
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

    update_leaves, update_treedef = jax.tree.flatten(updates, is_leaf=_is_aux_leaf)

    if stochastic:
        keys = jax.random.split(cast(jax.Array, key), len(update_leaves))
        keys_tree = jax.tree.unflatten(update_treedef, list(keys))
    else:
        keys_tree = jax.tree.unflatten(update_treedef, [None] * len(update_leaves))

    def _apply_update(u, p, k):
        if _is_aux_leaf(u):
            return p
        if _is_aux_leaf(p):
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

    return jax.tree.map(_apply_update, updates, model, keys_tree, is_leaf=_is_aux_leaf)
