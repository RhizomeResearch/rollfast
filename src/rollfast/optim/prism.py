import math
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from optax._src import alias, base, combine, numerics, transform, utils
from optax.transforms import _masking

try:
    import equinox as eqx

    _EQUINOX_AVAILABLE = True
    _LINEAR_TYPES = (eqx.nn.Linear,)
    _CONV_TYPES = (
        eqx.nn.Conv1d,
        eqx.nn.Conv2d,
        eqx.nn.Conv3d,
        eqx.nn.ConvTranspose1d,
        eqx.nn.ConvTranspose2d,
        eqx.nn.ConvTranspose3d,
    )
except ImportError:
    eqx = None
    _EQUINOX_AVAILABLE = False
    _LINEAR_TYPES = ()
    _CONV_TYPES = ()

_DEFAULT_NS_COEFFS = (3.4445, -4.7750, 2.0315)


class PrismDimensionNumbers(NamedTuple):
    """Specification for which weight axes participate in PRISM's matrix optimization.

    PRISM primarily operates on 2D matrices. This specification defines how to reshape
    higher-rank tensors (like 4D convolution kernels) into a 2D matrix format
    (reduction_axis, output_axis) suitable for spectral processing.

    Attributes:
        reduction_axis: The axes that will be flattened into the first dimension (rows)
            of the matrix. These dimensions are "reduced" during the matrix product.
            Default is 0.
        output_axis: The axes that will be flattened into the second dimension (columns)
            of the matrix. Default is 1.

    Note:
        Any axes not specified in either `reduction_axis` or `output_axis` are treated
        as batch dimensions. The orthogonalization will be applied independently across
        these batch dimensions (via vmap).
    """

    reduction_axis: Sequence[int] | int = 0
    output_axis: Sequence[int] | int = 1


# Semantic alias for the PyTree of specs
DimNumsTree = base.Params
WeightDimNumOrFn = (
    PrismDimensionNumbers | DimNumsTree | Callable[[base.Params], DimNumsTree]
)
ReshapeFn = Callable[[jax.Array], jax.Array]

# Predicate for tree traversal that respects Partitioning masks
_is_prism_leaf = lambda x: (
    x is None
    or isinstance(x, PrismDimensionNumbers)
    or isinstance(x, _masking.MaskedNode)
)


def get_equinox_prism_spec(
    model: Any,
    skip_depthwise_conv: bool = True,
) -> WeightDimNumOrFn:
    """
    Generates a PRISM dimension spec tree for an Equinox model.

    Features:
    - Robust: Uses structural replacement (eqx.tree_at) instead of object identity.
    - Comprehensive: Handles Linear, Conv, and ConvTranspose.
    - Smart: Optionally skips Depthwise Convolutions (groups == in_channels).

    Args:
        model: The Equinox model to generate specs for.
        skip_depthwise_conv: If True (default), depthwise convolutions are handled
            by Adam instead of PRISM. Set to False to apply PRISM to all convolutions.
    """
    if not _EQUINOX_AVAILABLE:
        raise ImportError(
            "The function `get_equinox_prism_spec` requires the `equinox` library. "
            "Please install it via `pip install equinox`."
        )

    def _layer_to_spec(layer):
        target_spec = None

        if isinstance(layer, _LINEAR_TYPES):
            target_spec = PrismDimensionNumbers(reduction_axis=1, output_axis=0)

        elif isinstance(layer, _CONV_TYPES):
            groups = getattr(layer, "groups", 1)
            is_depthwise = (groups > 1) and (groups == layer.in_channels)

            if not (skip_depthwise_conv and is_depthwise):
                ndim = layer.weight.ndim
                target_spec = PrismDimensionNumbers(
                    reduction_axis=tuple(range(1, ndim)), output_axis=0
                )

        if target_spec is None:
            return jax.tree.map(lambda _: None, layer)

        specs = eqx.tree_at(lambda l: l.weight, layer, target_spec)

        return jax.tree.map(
            lambda x: x if isinstance(x, PrismDimensionNumbers) else None,
            specs,
            is_leaf=lambda x: isinstance(x, PrismDimensionNumbers),
        )

    return jax.tree.map(
        _layer_to_spec,
        model,
        is_leaf=lambda x: isinstance(x, _LINEAR_TYPES + _CONV_TYPES),
    )


def _get_dimension_numbers(
    weight_dimension_numbers: WeightDimNumOrFn | None, params: base.Params
) -> base.Params:
    """Returns a PyTree of PrismDimensionNumbers | None matching the params structure.

    This function resolves the `weight_dimension_numbers` argument into a concrete
    PyTree that aligns 1:1 with the provided `params`. It handles the default case
    (inferring 2D matrices) and ensures compatibility with `optax.combine.partition`
    by safely handling `MaskedNode` entries.

    Args:
        weight_dimension_numbers: The dimension specification. Can be None (auto-detect),
            a PyTree of specs, or a callable returning a PyTree.
        params: The parameters PyTree to align with.

    Returns:
        A PyTree with the same structure as `params` containing `PrismDimensionNumbers`
        for PRISM-optimized leaves and `None` for others.
    """
    if weight_dimension_numbers is None:

        def _get_default_spec(x):
            if isinstance(x, _masking.MaskedNode):
                return None
            return PrismDimensionNumbers() if x.ndim == 2 else None

        return jax.tree.map(
            _get_default_spec,
            params,
            is_leaf=lambda x: isinstance(x, _masking.MaskedNode),
        )

    if callable(weight_dimension_numbers):
        return weight_dimension_numbers(params)

    return weight_dimension_numbers


def _make_param_labels(dim_nums_tree: base.Params) -> base.Params:
    """Converts a dimension numbers tree into optimization labels.

    Args:
        dim_nums_tree: A PyTree of `PrismDimensionNumbers` or `None`.

    Returns:
        A PyTree of strings, where leaves are labeled 'prism' if a dimension spec
        exists, and 'adam' otherwise.
    """
    return jax.tree.map(
        lambda d: "prism" if d is not None else "adam",
        dim_nums_tree,
        is_leaf=_is_prism_leaf,
    )


def _mask_dimension_numbers(dim_nums_tree: base.Params) -> base.Params:
    """Replaces None entries with MaskedNode for partition compatibility.

    When using `optax.combine.partition`, the optimizer responsible for the 'prism'
    label must receive a parameter tree where non-prism parameters are masked out.
    This function prepares the dimension specification tree to match that masked structure.

    Args:
        dim_nums_tree: A PyTree of `PrismDimensionNumbers` or `None`.

    Returns:
        A PyTree where `None` leaves are replaced by `optax.transforms._masking.MaskedNode`.
    """
    return jax.tree.map(
        lambda d: d if d is not None else _masking.MaskedNode(),
        dim_nums_tree,
        is_leaf=_is_prism_leaf,
    )


def _is_standard_2d_spec(dim_nums: PrismDimensionNumbers) -> bool:
    """Checks if a spec represents the standard 2D matrix layout (reduction=0, output=1).

    This check is used to trigger the fast optimization path, avoiding expensive
    reshapes and vmaps for standard weight matrices. It robustly handles both
    integer and tuple scalar notations (e.g., `0` vs `(0,)`).

    Args:
        dim_nums: The dimension specification to check.

    Returns:
        True if the spec corresponds to a standard 2D matrix operation, False otherwise.
    """
    red = dim_nums.reduction_axis
    out = dim_nums.output_axis

    # Normalize to tuples for comparison
    red_norm = (red,) if isinstance(red, int) else tuple(red)
    out_norm = (out,) if isinstance(out, int) else tuple(out)

    return red_norm == (0,) and out_norm == (1,)


def _add_tiny(x):
    """Add smallest normal number to avoid division by zero."""
    return x + jnp.finfo(x.dtype).tiny


def _dist_reduce(x: jax.Array, axis_name: Optional[str], op: str = "mean") -> jax.Array:
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


def _prism_global_norm(grads: Any, axis_name: Optional[str] = None) -> jax.Array:
    """Computes the global L2 norm of gradients across all parameters and devices.

    Args:
        grads: The gradients PyTree.
        axis_name: The axis name for distributed reduction.

    Returns:
        A scalar float32 array representing the global L2 norm.
    """
    leaves = jax.tree.leaves(grads)
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    local_sq = sum(jnp.sum(numerics.abs_sq(x.astype(jnp.float32))) for x in leaves)
    total_sq = _dist_reduce(local_sq, axis_name, "sum")
    return jnp.sqrt(total_sq)


def _clip_per_tensor_rms(
    u: jax.Array, max_rms: float = 1.0, max_val: float = 10.0
) -> jax.Array:
    """Clips a tensor based on its Root Mean Square (RMS) and absolute value.

    This provides a dual-layer stability mechanism:
    1. Scales the tensor down if its RMS exceeds `max_rms`.
    2. Hard-clips values to the range `[-max_val, max_val]`.

    Args:
        u: The input tensor update.
        max_rms: The maximum allowed Root Mean Square value.
        max_val: The maximum absolute value for element-wise clipping.

    Returns:
        The clipped tensor.
    """
    rms = jnp.sqrt(jnp.mean(numerics.abs_sq(u)))
    scale_factor = jnp.minimum(1.0, max_rms / (rms + 1e-9))
    u = u * scale_factor
    return jnp.clip(u, -max_val, max_val)


def _normalize_axes(
    x: jax.Array, dim_nums: PrismDimensionNumbers
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Normalizes dimension specs to tuples of positive integers."""
    if isinstance(dim_nums.reduction_axis, int):
        dim_nums = dim_nums._replace(reduction_axis=(dim_nums.reduction_axis,))
    reduction_axes = tuple(ax % x.ndim for ax in dim_nums.reduction_axis)

    if isinstance(dim_nums.output_axis, int):
        dim_nums = dim_nums._replace(output_axis=(dim_nums.output_axis,))
    output_axes = tuple(ax % x.ndim for ax in dim_nums.output_axis)
    return reduction_axes, output_axes


def _compute_prism_reshape(
    x: jax.Array, dim_nums: PrismDimensionNumbers
) -> tuple[ReshapeFn, ReshapeFn]:
    """Computes reshape functions to transform a tensor into a PRISM-compatible format.

    PRISM operates on (Batch, Reduction, Output) structures. This function generates
    forward and inverse functions to map arbitrary tensors (e.g., 4D kernels) to this layout.

    Args:
        x: The template tensor (defining shape and rank).
        dim_nums: The specification of reduction and output axes.

    Returns:
        A tuple (reshape_fn, inverse_fn).
        - reshape_fn: Maps input -> (Batch_Dim, Reduction_Dim, Output_Dim).
        - inverse_fn: Maps (Batch_Dim, Reduction_Dim, Output_Dim) -> input shape.
    """
    if x.ndim < 2:
        raise ValueError(
            f"PRISM optimized parameters must have rank >= 2, got {x.ndim=}"
        )

    reduction_axes, output_axes = _normalize_axes(x, dim_nums)
    if set(reduction_axes) & set(output_axes):
        raise ValueError(
            f"Reduction and output axes must be disjoint. Got {reduction_axes} and {output_axes}."
        )

    batch_axes = tuple(
        sorted(set(range(x.ndim)) - set(reduction_axes) - set(output_axes))
    )
    transpose = batch_axes + reduction_axes + output_axes
    inv_transpose = tuple(sorted(range(x.ndim), key=lambda i: transpose[i]))

    axes2shape = lambda axes: tuple(x.shape[ax] for ax in axes)

    flat_shape = (
        math.prod(axes2shape(batch_axes)),
        math.prod(axes2shape(reduction_axes)),
        math.prod(axes2shape(output_axes)),
    )
    unflat_shape = (
        axes2shape(batch_axes) + axes2shape(reduction_axes) + axes2shape(output_axes)
    )

    reshape_fn = lambda x: x.transpose(transpose).reshape(flat_shape)
    inverse_fn = lambda x: x.reshape(unflat_shape).transpose(inv_transpose)
    return reshape_fn, inverse_fn


def _newton_schulz_iterator_muon(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    """Performs a single step of the Newton-Schulz iteration."""
    a = x @ x.T.conj()
    b = coeffs[1] * a + coeffs[2] * (a @ a)
    return coeffs[0] * x + (b @ x)


def _quintic_newton_schulz(
    x: jax.Array,
    iters: int = 5,
    eps: float = 1e-8,
    ns_coeffs: tuple[float, float, float] = _DEFAULT_NS_COEFFS,
) -> jax.Array:
    """Applies quintic Newton-Schulz orthogonalization to approximate the polar factor.

    This function orthogonalizes the input matrix `x` such that the result approximates
    `UV^T` (where `USV^T` is the SVD of `x`). It uses a specific set of coefficients
    tuned for the Muon/PRISM optimizer.

    Args:
        x: The input matrix of shape (..., Rows, Cols).
        iters: Number of iterations to run.
        eps: Epsilon for numerical stability during normalization.
        ns_coeffs: Coefficients for the quintic polynomial.

    Returns:
        The orthogonalized matrix.
    """
    x_f32 = x.astype(jnp.float32)

    # Standardize to (rows, cols) where rows >= cols for efficiency
    transposed = False
    if x_f32.shape[-2] > x_f32.shape[-1]:
        x_f32 = jnp.swapaxes(x_f32, -1, -2)
        transposed = True

    # Use vector norm on the last two dimensions to handle vmapped inputs safely
    norm = jnp.linalg.norm(x_f32, axis=(-2, -1), keepdims=True)
    x_f32 = x_f32 / (norm + eps)

    coeffs = jnp.asarray(ns_coeffs, dtype=x_f32.dtype)

    def body_fn(_, x_):
        return _newton_schulz_iterator_muon(x_, coeffs)

    x_f32 = jax.lax.fori_loop(0, iters, body_fn, x_f32, unroll=True)

    if transposed:
        x_f32 = jnp.swapaxes(x_f32, -1, -2)

    return x_f32.astype(x.dtype)


def _apply_prism_math(g, m_raw, m_target, gamma, ns_iters):
    """Core PRISM mathematical operation: Innovation -> Augmentation -> Orthogonalization.

    1. Computes Innovation: `D_t = G_t - M_raw`
    2. Augments Matrix: Concatenates `M_target` and `gamma * D_t`.
    3. Orthogonalizes: Applies Newton-Schulz to the augmented matrix.
    4. Slices: Returns the top block corresponding to the updated direction.

    Args:
        g: Current gradient.
        m_raw: Raw First Moment (without Nesterov).
        m_target: Target Momentum (Raw or Nesterov) to be shaped.
        gamma: Innovation damping coefficient.
        ns_iters: Number of Newton-Schulz iterations.

    Returns:
        The spectrally shaped update.
    """
    # Strict Innovation: D_t = G_t - M_t (raw)
    D_t = g - m_raw

    # Augment: [M_target; gamma * D_t]
    augmented_M = jnp.concatenate([m_target, gamma * D_t], axis=-2)

    # Orthogonalize
    augmented_O = _quintic_newton_schulz(augmented_M, iters=ns_iters)

    # Slice top block (split along the row dimension, which is -2)
    return augmented_O[..., : m_raw.shape[-2], :]


def _prism_ortho_step(
    updates: jax.Array,
    mu_raw: jax.Array,
    gamma: float,
    ns_iters: int,
    mu_nest: Optional[jax.Array] = None,
    dim_nums: Optional[PrismDimensionNumbers] = None,
) -> jax.Array:
    """Orchestrates the PRISM spectral shaping step, handling reshaping and batching.

    This function routes the execution through either:
    1. A passthrough (for masked nodes).
    2. A fast path (for standard 2D matrices).
    3. A general path (reshaping + vmapping for higher-order tensors).

    Args:
        updates: The gradient updates.
        mu_raw: The raw momentum accumulator state.
        gamma: The innovation damping coefficient.
        ns_iters: Number of Newton-Schulz iterations.
        mu_nest: The Nesterov-accelerated momentum (optional).
        dim_nums: The dimension reshaping specification.

    Returns:
        The PRISM-processed update tensor.
    """
    # Passthrough Case (Partitioning MaskedNode or None)
    if dim_nums is None or isinstance(dim_nums, _masking.MaskedNode):
        return mu_nest if mu_nest is not None else mu_raw

    m_target_eff = mu_nest if mu_nest is not None else mu_raw

    # Fast Path: Standard 2D Matrix
    # Avoids overhead of reshape + vmap for the most common case
    if updates.ndim == 2 and _is_standard_2d_spec(dim_nums):
        return _apply_prism_math(updates, mu_raw, m_target_eff, gamma, ns_iters)

    # General Path: Reshape -> Vmap -> Inverse
    reshape_fn, inverse_fn = _compute_prism_reshape(updates, dim_nums)

    # Reshape to (Batch, Reduction, Output)
    G_flat = reshape_fn(updates)
    M_raw_flat = reshape_fn(mu_raw)
    M_target_flat = reshape_fn(m_target_eff)

    # Vmap over batch axis (0)
    O_flat = jax.vmap(lambda g, m, t: _apply_prism_math(g, m, t, gamma, ns_iters))(
        G_flat, M_raw_flat, M_target_flat
    )

    return inverse_fn(O_flat)


class ScaleByPrismState(NamedTuple):
    """State for the PRISM gradient transformation."""

    count: jax.Array
    mu: base.Updates


def scale_by_prism(
    b1: float = 0.95,
    gamma: float = 1.0,
    ns_iters: int = 5,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    axis_name: Optional[str] = None,
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformation:
    """The core PRISM gradient transformation.

    Implements the core logic of momentum accumulation, innovation computation,
    and spectral shaping via Newton-Schulz orthogonalization.

    Args:
        b1: Exponential decay rate for the first moment (momentum).
        gamma: Damping coefficient for the innovation term. Controls the "anisotropy"
            of the spectral shaping.
        ns_iters: Number of Newton-Schulz iterations for orthogonalization.
        nesterov: Whether to use Nesterov momentum.
        shape_nesterov: If True, applies spectral shaping to the Nesterov-accelerated
            momentum. If False, shapes the raw momentum.
        bias_correction: Whether to apply bias correction to the momentum.
        mu_dtype: Data type for the momentum accumulator.
        raw_global_grad_clip: Threshold for global gradient norm clipping *before*
            momentum update.
        permissive_spike_protection: If False, completely skips updates when
            `raw_global_grad_clip` is triggered. If True, clips and proceeds.
        grad_clip_max_amps: Configuration for post-shaping clipping. Can be a float
            (max RMS) or a tuple (max RMS, max Abs).
        axis_name: Axis name for distributed (SPMD) global norm reduction.
        weight_dimension_numbers: Specification for reshaping tensors. If provided,
            `params` must be passed to `update`.

    Returns:
        An `optax.GradientTransformation`.
    """
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)
        return ScaleByPrismState(count=jnp.zeros([], jnp.int32), mu=mu)

    def update_fn(updates, state, params=None):
        # Strict requirement for params if dimension numbers are used
        if params is None:
            if weight_dimension_numbers is not None:
                raise ValueError(
                    "`params` must be provided to `scale_by_prism` when `weight_dimension_numbers` is set."
                )
            resolved_dim_nums = _get_dimension_numbers(None, updates)
        else:
            resolved_dim_nums = _get_dimension_numbers(weight_dimension_numbers, params)

        count_inc = numerics.safe_increment(state.count)

        # Pre-Preconditioning Clipping (Global)
        is_spike = jnp.array(False, dtype=jnp.bool_)
        if raw_global_grad_clip is not None:
            g_norm = _prism_global_norm(updates, axis_name=axis_name)
            is_spike = g_norm > raw_global_grad_clip

            clip_scale = jnp.where(
                g_norm > raw_global_grad_clip,
                raw_global_grad_clip / _add_tiny(g_norm),
                1.0,
            )
            updates = jax.tree.map(lambda g: g * clip_scale, updates)

        # Permissive Spike Protection Logic
        #   - If permissive=True: We clipped above, so we DO NOT skip.
        #   - If permissive=False: We Strict Skip on spikes.
        should_skip = jnp.logical_and(
            is_spike, jnp.logical_not(permissive_spike_protection)
        )

        # Update Momentum
        effective_updates = jax.lax.cond(
            should_skip,
            lambda u: jax.tree.map(
                lambda x: jnp.zeros_like(x) if x is not None else None, u
            ),
            lambda u: u,
            updates,
        )
        mu = optax.tree.update_moment(effective_updates, state.mu, b1, 1)

        # Nesterov / Bias Correction
        mu_nest = mu
        if nesterov:
            if bias_correction:
                mu_bc = optax.tree.bias_correction(
                    mu, b1, numerics.safe_increment(count_inc)
                )
                g_bc = optax.tree.bias_correction(effective_updates, b1, count_inc)
                mu_nest = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, mu_bc, g_bc)
            else:
                mu_nest = jax.tree.map(
                    lambda m, g: b1 * m + (1 - b1) * g, mu, effective_updates
                )
        elif bias_correction:
            mu_nest = optax.tree.bias_correction(mu, b1, count_inc)

        mu_cast = optax.tree.cast(mu, mu_dtype)
        mu_nest_cast = optax.tree.cast(mu_nest, mu_dtype)

        # Ensure target_for_ortho matches structure of updates
        if shape_nesterov:
            target_for_ortho = mu_nest_cast
        else:
            # Create a PyTree of None with the same structure as effective_updates
            target_for_ortho = jax.tree.map(lambda _: None, effective_updates)

        prism_out = jax.tree.map(
            lambda g, m, target, dims: _prism_ortho_step(
                g, m, gamma, ns_iters, mu_nest=target, dim_nums=dims
            ),
            effective_updates,
            mu_cast,
            target_for_ortho,
            resolved_dim_nums,
            is_leaf=_is_prism_leaf,
        )

        if nesterov and not shape_nesterov:
            # Strict Paper Mode: Update = beta * O_t + (1-beta) * G_t
            new_updates = jax.tree.map(
                lambda o, g: b1 * o + (1 - b1) * g, prism_out, effective_updates
            )
        else:
            new_updates = prism_out

        # Post-Preconditioning Clipping (Per-Tensor)
        if grad_clip_max_amps is not None:
            max_rms, max_val = (
                grad_clip_max_amps
                if isinstance(grad_clip_max_amps, tuple)
                else (grad_clip_max_amps, 10.0)
            )
            new_updates = jax.tree.map(
                lambda u: _clip_per_tensor_rms(u, max_rms, max_val), new_updates
            )

        # (Technically redundant if M_t froze and G_t is zero, but safe)
        new_updates = jax.lax.cond(
            should_skip,
            lambda u: jax.tree.map(jnp.zeros_like, u),
            lambda u: u,
            new_updates,
        )

        return new_updates, ScaleByPrismState(
            count=count_inc, mu=optax.tree.cast(mu, mu_dtype)
        )

    return base.GradientTransformation(init_fn, update_fn)


def prism(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.95,
    gamma: float = 1.0,
    weight_decay: float = 0.01,
    ns_iters: int = 5,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    axis_name: Optional[str] = None,
    # Partitioning Arguments
    adam_learning_rate: Optional[base.ScalarOrSchedule] = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    prism_weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformation:
    """PRISM Optimizer with automatic partitioning (Muon-Style).

    This function creates a composite optimizer that partitions parameters into two groups:
    1. 'prism': Matrices (or tensors with explicit specs) optimized via Spectral Shaping.
    2. 'adam': Vectors, Scalars, and other parameters optimized via standard AdamW.

    Args:
        learning_rate: The global learning rate for the PRISM branch.
        b1: Momentum coefficient for PRISM.
        gamma: Innovation damping coefficient for PRISM.
        weight_decay: Weight decay applied to both PRISM and Adam branches.
        ns_iters: Number of Newton-Schulz iterations for PRISM.
        nesterov: Whether to use Nesterov acceleration in PRISM.
        shape_nesterov: Whether to shape the Nesterov momentum or raw momentum.
        bias_correction: Whether to enable bias correction in PRISM.
        grad_clip_max_amps: Post-shaping clipping configuration (RMS, Abs).
        raw_global_grad_clip: Global gradient norm clipping threshold.
        permissive_spike_protection: Behavior when global clip is triggered (Clip vs Skip).
        mu_dtype: Dtype for momentum accumulators.
        axis_name: Axis name for distributed global norm reduction.
        adam_learning_rate: Learning rate for the Adam branch. Defaults to `learning_rate`.
        adam_b1: Beta1 for Adam.
        adam_b2: Beta2 for Adam.
        adam_eps: Epsilon for Adam.
        prism_weight_dimension_numbers: Optional PyTree of `PrismDimensionNumbers`.
            - If None (default): Automatically selects all 2D parameters for PRISM.
            - If provided: Uses the spec to select PRISM parameters. Leaves set to None
              fall back to Adam.

    Returns:
        A `optax.GradientTransformation` that handles the partitioned optimization.
    """
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    # Helper: Resolve specs from params
    def get_resolved_dim_nums(params):
        return _get_dimension_numbers(prism_weight_dimension_numbers, params)

    # Label generation
    def param_labels(params):
        dim_nums = get_resolved_dim_nums(params)
        return jax.tree.map(
            lambda d, p: None if p is None else ("prism" if d is not None else "adam"),
            dim_nums,
            params,
            is_leaf=_is_prism_leaf,
        )

    # Spec Masking for the Prism chain
    def prism_weight_dim_nums_fn(params):
        dim_nums = get_resolved_dim_nums(params)
        return _mask_dimension_numbers(dim_nums)

    return combine.partition(
        transforms={
            "prism": combine.chain(
                scale_by_prism(
                    b1=b1,
                    gamma=gamma,
                    ns_iters=ns_iters,
                    nesterov=nesterov,
                    shape_nesterov=shape_nesterov,
                    bias_correction=bias_correction,
                    mu_dtype=mu_dtype,
                    raw_global_grad_clip=raw_global_grad_clip,
                    permissive_spike_protection=permissive_spike_protection,
                    grad_clip_max_amps=grad_clip_max_amps,
                    axis_name=axis_name,
                    weight_dimension_numbers=prism_weight_dim_nums_fn,
                ),
                transform.add_decayed_weights(weight_decay),
                transform.scale_by_learning_rate(learning_rate),
            ),
            "adam": alias.adamw(
                learning_rate=adam_learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
                weight_decay=weight_decay,
                mu_dtype=mu_dtype,
            ),
        },
        param_labels=param_labels,
    )


class WeightingMode(str, Enum):
    """
    Determines how the iterate averaging parameter c_t is computed.

    Ref: 2511.07767v1 "Schedulers for Schedule-Free"
    """

    THEORETICAL = "theoretical"  # c_t = 1/t (w_t = 1)
    PRACTICAL = "practical"  # c_t = gamma_t^2 / sum(gamma^2) (w_t = gamma_t^2)
    SCHEDULET = "schedulet"  # c_t = gamma_t / sum(gamma) (w_t = gamma_t)


def wsd_schedule(
    peak_lr: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
) -> optax.Schedule:
    """Creates a Warmup-Stable-Decay (Trapezoidal) schedule.

    This schedule is recommended for Schedule-Free optimization. It consists of:
    1. Linear warmup.
    2. Constant (stable) phase.
    3. Linear decay (cooldown) phase.

    Args:
        peak_lr: The constant learning rate during the stable phase.
        total_steps: Total training steps T.
        warmup_fraction: Fraction of steps for linear warmup.
        decay_fraction: Fraction of steps for linear decay (cooldown).

    Returns:
        An optax schedule function.
    """
    warmup_steps = int(total_steps * warmup_fraction)
    decay_steps = int(total_steps * decay_fraction)

    T_w = warmup_steps
    T_c = total_steps - decay_steps
    T_final = total_steps - 1

    def schedule(count):
        # Case 1: 0 <= t <= T_w (Warmup)
        # Eq: (t + 1) / (T_w + 1)
        warmup_val = (count + 1.0) / (T_w + 1.0) * peak_lr

        # Case 2: T_w < t <= T_c (Stable)
        stable_val = peak_lr

        # Case 3: T_c < t <= T (Decay)
        # Eq: (T - t + 1) / (T - T_c + 1)
        decay_val = (T_final - count + 1.0) / (T_final - T_c + 1.0) * peak_lr

        is_warmup = count <= T_w
        is_decay = count > T_c

        val = jnp.where(is_warmup, warmup_val, stable_val)
        val = jnp.where(is_decay, decay_val, val)

        # Ensure we don't go negative or process past T_final roughly
        return jnp.maximum(0.0, val)

    return schedule


class ScheduleFreeState(NamedTuple):
    """State for the Schedule-Free wrapper."""

    b1: jax.Array
    weight_sum: base.Params
    step_count: jax.Array
    base_state: base.OptState
    z: base.Params


def schedule_free(
    base_optimizer: base.GradientTransformation,
    learning_rate: Union[base.ScalarOrSchedule, Callable[[int], base.Params]],
    b1: float = 0.9,
    weighting_mode: Union[str, WeightingMode] = WeightingMode.SCHEDULET,
    state_dtype: Optional[jax.typing.DTypeLike] = None,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free Wrapper supporting Schedulet, Practical, and Theoretical modes.

    Implements the Schedule-Free optimization wrapper. It maintains a primary
    sequence `z` (updated by the base optimizer) and an averaged sequence `x`
    (the parameters used for evaluation).

    Args:
        base_optimizer: The inner optimizer (e.g., PRISM). Must return the full
            update step (including LR scaling).
        learning_rate: The learning rate schedule function or scalar. Used to
            compute the weighting `c_t`.
        b1: The interpolation parameter beta (distinct from optimizer momentum).
            Controls the interpolation between `x` and `z`.
        weighting_mode: Strategy for averaging weights.
        state_dtype: Dtype for the z-sequence storage.

    Returns:
        A `GradientTransformationExtraArgs` wrapper.
    """
    if isinstance(weighting_mode, str):
        weighting_mode = WeightingMode(weighting_mode)

    def init_fn(params):
        dtype = (
            state_dtype
            if state_dtype is not None
            else optax.tree.dtype(params, "lowest")
        )
        z = optax.tree.cast(params, dtype=dtype)
        # Deep copy z to ensure distinct buffer
        z = jax.tree.map(lambda t: jnp.array(t, copy=True), z)

        # Initialize weight_sum as a PyTree of zeros matching params
        # This is O(N) memory but required for exact mathematical correctness
        # with partitioned schedules.
        # weight_sum = jax.tree.map(lambda x: jnp.zeros([], dtype=jnp.float32), params)
        weight_sum = jax.tree.map(
            lambda x: jnp.zeros([], dtype=jnp.float32) if x is not None else None,
            params,
        )

        return ScheduleFreeState(
            b1=jnp.array(b1, dtype=jnp.float32),
            weight_sum=weight_sum,
            # weight_sum=jnp.zeros([], dtype=jnp.float32),
            step_count=jnp.zeros([], dtype=jnp.int32),
            base_state=base_optimizer.init(params),
            z=z,
        )

    def update_fn(updates, state, params=None, **extra_args):
        if callable(learning_rate):
            try:
                lr_tree = learning_rate(state.step_count, params)
            except TypeError:
                # Fallback: The schedule does not accept params (standard optax schedule)
                lr_tree = learning_rate(state.step_count)
        else:
            lr_tree = learning_rate

        lr_tree = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), lr_tree)

        # Compute Base Optimizer Update (Preconditioned Gradient)
        # Note: base_optimizer should output the update step d = -lr * P * g
        # We must handle the LR scaling carefully. Standard SF assumes base_opt
        # returns (D^-1 g).
        base_updates, new_base_state = base_optimizer.update(
            updates, state.base_state, params, **extra_args
        )

        if weighting_mode == WeightingMode.SCHEDULET:
            weight_tree = lr_tree
        elif weighting_mode == WeightingMode.PRACTICAL:
            weight_tree = jax.tree.map(
                lambda x: jnp.square(x) if x is not None else None, lr_tree
            )
        else:  # THEORETICAL
            weight_tree = jax.tree.map(
                lambda x: jnp.ones_like(x) if x is not None else None, lr_tree
            )

        # Accumulate weights: W_t = W_{t-1} + w_t
        # new_weight_sum = jax.tree.map(
        #     lambda acc, w: acc + w, state.weight_sum, weight_tree
        # )
        new_weight_sum = jax.tree.map(
            lambda acc, w: (acc + w) if (acc is not None and w is not None) else None,
            state.weight_sum,
            weight_tree,
        )

        # c_t = w_t / W_t
        # Safety: avoid division by zero
        ck_tree = jax.tree.map(
            lambda w, sum_w: (
                jnp.where(sum_w > 0, w / sum_w, 0.0)
                if (w is not None and sum_w is not None)
                else None
            ),
            weight_tree,
            new_weight_sum,
        )

        # Protect against b1 -> 0
        b1_safe = jnp.maximum(state.b1, 1e-8)

        # Schedule-Free Update Dynamics
        # y_t = params (input)
        # z_t = z_{t-1} - gamma * (Base Update)
        # Note: 'base_updates' from chain usually includes LR scaling.
        # If base_optimizer includes scale_by_learning_rate, base_updates is actual step.
        z_next = jax.tree.map(lambda z, u: z + u, state.z, base_updates)

        # We recover x_t from y_t: x_t = (y_t - (1-b1)z_{t-1}) / b1
        x_curr = jax.tree.map(
            lambda y, z: (y - (1.0 - b1_safe) * z) / b1_safe, params, state.z
        )

        # x_{t+1} = (1 - c_{t+1}) x_t + c_{t+1} z_{t+1}
        x_next = jax.tree.map(
            lambda x, z, ck: (
                (1.0 - ck) * x + ck * z if ck is not None else x
            ),  # Fallback to x (param) if ck is None
            x_curr,
            z_next,
            ck_tree,
        )

        # y_{t+1} = (1-b1) z_{t+1} + b1 x_{t+1}
        y_next = jax.tree.map(
            lambda x, z: b1_safe * x + (1.0 - b1_safe) * z, x_next, z_next
        )

        # Final Update diff
        final_updates = jax.tree.map(lambda n, o: n - o, y_next, params)

        new_state = ScheduleFreeState(
            b1=state.b1,
            weight_sum=new_weight_sum,
            step_count=numerics.safe_increment(state.step_count),
            base_state=new_base_state,
            z=z_next,
        )

        return final_updates, new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def prism_schedule_free(
    learning_rate: float,
    total_steps: int,
    # Schedule Config
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    weighting_mode: Union[str, WeightingMode] = WeightingMode.SCHEDULET,
    # Schedule-Free Config
    sf_b1: float = 0.90,
    state_dtype: Optional[jax.typing.DTypeLike] = None,
    # PRISM Config
    prism_b1: float = 0.0,
    gamma: float = 1.0,
    ns_iters: int = 5,
    shape_nesterov: bool = True,
    weight_decay: float = 0.0,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    axis_name: Optional[str] = None,
    # Partitioning Arguments
    adam_learning_rate: Optional[float] = None,
    adam_b1: float = 0.0,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    prism_weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free PRISM Optimizer with Partitioning.

    Combines PRISM's spectral shaping and partitioning (matrices vs vectors)
    with the Schedule-Free optimization wrapper.

    Args:
        learning_rate: Peak learning rate for the PRISM branch.
        total_steps: Total training steps (required for WSD schedule).
        warmup_fraction: Fraction of steps for warmup.
        decay_fraction: Fraction of steps for decay.
        weighting_mode: Schedule-free weighting mode.
        sf_b1: Schedule-free interpolation parameter (distinct from momentum).
        state_dtype: Dtype for schedule-free z-sequence.
        prism_b1: Momentum coefficient for PRISM.
        gamma: Innovation damping coefficient for PRISM.
        ns_iters: Newton-Schulz iterations.
        shape_nesterov: Whether to shape Nesterov momentum.
        weight_decay: Weight decay.
        grad_clip_max_amps: Post-shaping clipping.
        raw_global_grad_clip: Pre-shaping global clipping.
        permissive_spike_protection: Clip vs Skip on spikes.
        mu_dtype: Momentum dtype.
        axis_name: Distributed axis name.
        adam_learning_rate: Peak learning rate for Adam branch. Defaults to `learning_rate`.
        adam_b1: Adam Beta1.
        adam_b2: Adam Beta2.
        adam_eps: Adam Epsilon.
        prism_weight_dimension_numbers: Spec for PRISM parameters.

    Returns:
        A Schedule-Free gradient transformation with partitioned inner optimizer.
    """
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    # We need separate schedules if the peak LRs differ, as the schedule outputs
    # the exact value to apply.
    prism_schedule = wsd_schedule(
        peak_lr=learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )

    if adam_learning_rate == learning_rate:
        adam_schedule = prism_schedule
    else:
        adam_schedule = wsd_schedule(
            peak_lr=adam_learning_rate,
            total_steps=total_steps,
            warmup_fraction=warmup_fraction,
            decay_fraction=decay_fraction,
        )

    def get_resolved_dim_nums(params):
        return _get_dimension_numbers(prism_weight_dimension_numbers, params)

    def param_labels(params):
        dim_nums = get_resolved_dim_nums(params)
        return jax.tree.map(
            lambda d, p: None if p is None else ("prism" if d is not None else "adam"),
            dim_nums,
            params,
            is_leaf=_is_prism_leaf,
        )
        # return _make_param_labels(dim_nums)

    def prism_weight_dim_nums_fn(params):
        dim_nums = get_resolved_dim_nums(params)
        return _mask_dimension_numbers(dim_nums)

    # Note: We must apply the schedule INSIDE the base optimizer branches
    # so that the updates passed to `schedule_free` are fully scaled.
    base_opt = combine.partition(
        transforms={
            "prism": combine.chain(
                scale_by_prism(
                    b1=prism_b1,
                    gamma=gamma,
                    ns_iters=ns_iters,
                    nesterov=False,
                    shape_nesterov=shape_nesterov,
                    # bias_correction=False,  # SF handles bias correction implicitly
                    mu_dtype=mu_dtype,
                    raw_global_grad_clip=raw_global_grad_clip,
                    permissive_spike_protection=permissive_spike_protection,
                    grad_clip_max_amps=grad_clip_max_amps,
                    axis_name=axis_name,
                    weight_dimension_numbers=prism_weight_dim_nums_fn,
                ),
                transform.add_decayed_weights(weight_decay),
                transform.scale_by_learning_rate(prism_schedule),
            ),
            "adam": alias.adamw(
                learning_rate=adam_schedule,
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
                weight_decay=weight_decay,
                mu_dtype=mu_dtype,
            ),
        },
        param_labels=param_labels,
    )

    # This function introspects params to decide which LR schedule to apply
    # for calculation of the schedule-free weighting c_t.
    def dual_schedule_fn(count, params):
        labels = param_labels(params)
        p_lr = prism_schedule(count)
        a_lr = adam_schedule(count)

        # return jax.tree.map(lambda l: p_lr if l == "prism" else a_lr, labels)
        return jax.tree.map(
            lambda l: None if l is None else (p_lr if l == "prism" else a_lr), labels
        )

    # We pass the prism_schedule to the wrapper solely for calculating `c_t`.
    # Since c_t depends on the *relative* progress of the schedule (and weighting mode),
    # using the prism schedule is sufficient even if Adam LR differs, provided the
    # shape (warmup/decay ratios) is the same.
    return schedule_free(
        base_optimizer=base_opt,
        learning_rate=dual_schedule_fn,
        b1=sf_b1,
        weighting_mode=weighting_mode,
        state_dtype=state_dtype,
    )


def schedule_free_eval_params(state: base.OptState, params: base.Params):
    """Params for evaluation of :func:`optax.contrib.schedule_free`.

    Args:
        state: The optimizer state (must be a ScheduleFreeState).
        params: The current parameters (the 'y' sequence).

    Returns:
        The parameters to use for evaluation (the 'x' sequence).
    """
    # Using ScheduleFreeState as a type hint above results in pytype errors in tests.
    b1 = getattr(state, "b1")
    z = getattr(state, "z")
    if b1 is None or z is None:
        raise ValueError(
            "schedule_free_eval_params requires a ScheduleFreeState as input."
        )
    b1_safe = jnp.maximum(b1, 1e-8)
    return jax.tree.map(lambda yi, zi: (yi - (1.0 - b1_safe) * zi) / b1_safe, params, z)
