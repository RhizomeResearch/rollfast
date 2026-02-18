import math
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from optax._src import alias, base, combine, numerics, transform, utils
from optax.transforms import _masking

from rollfast.utils import add_tiny, dist_reduce

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
    total_sq = dist_reduce(local_sq, axis_name, "sum")
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
                raw_global_grad_clip / add_tiny(g_norm),
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
