import math
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from optax._src import base, combine, numerics, transform, utils
from optax.transforms import _masking

from rollfast.optim.adam import adamw
from rollfast.optim.magma import apply_magma_internal
from rollfast.utils import (
    _safe_bias_correction,
    _tree_stochastic_cast,
    _tree_update_moment_f32,
    add_tiny,
    dist_reduce,
)

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
_INVROOT_COEFFS: dict[int, list[tuple[float, float, float]]] = {
    4: [  # P^{-1/4}  (used by Bidirectional-PRISM)
        (3.85003, -10.8539, 8.61893),
        (1.80992, -0.587778, 0.0647852),
        (1.50394, -0.594516, 0.121161),
        (45 / 32, -9 / 16, 5 / 32),
    ],
}


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

    return x_f32


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


def _invroot_coeffs_iter(r: int, steps: Optional[int] = None, scale: float = 1.0):
    """Yields (a, b, c) scaled for degree-r inverse root: a/s, b/s^{r+1}, c/s^{2r+1}.

    Cycles the final steady-state entry when `steps` > table length.
    """
    if r not in _INVROOT_COEFFS:
        raise ValueError(
            f"Inverse root degree r={r} unsupported. Choose from {set(_INVROOT_COEFFS)}."
        )
    w = _INVROOT_COEFFS[r]
    steps = steps or len(w)
    entries = list(w[:steps]) + [w[-1]] * max(steps - len(w), 0)
    for a, b, c in entries:
        yield (
            a / scale,
            b / scale ** (r + 1),
            c / scale ** (2 * r + 1),
        )


def _sym(M: jax.Array) -> jax.Array:
    """Symmetrize to counteract FP drift in SPD gram iterations."""
    return 0.5 * (M + M.mT)


def _safe_mm(
    A: jax.Array,
    B: jax.Array,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> jax.Array:
    """Computes matrix multiplication with specified precision.

    Bypasses XLA's default aggressive downcasting to BF16/TF32
    on modern hardware when HIGHEST is specified. For eigenvalue-sensitive
    polynomial iterations, maintaining a 23-bit mantissa is strictly required
    to preserve the Symmetric Positive Definite (SPD) property and remain
    within the convergence radius.
    """
    return jnp.matmul(A, B, precision=precision)


def _double_sided_matmul_invroot(
    Q: jax.Array,
    G: jax.Array,
    P: jax.Array,
    r: int,
    s: int = 1,
    steps: Optional[int] = None,
    eps: float = 1e-5,
    scale: float = 1.001,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> jax.Array:
    r"""Compute $Q^{-s/r} \, G \, P^{-s/r}$ via iterative polynomial approximation.

    Time:  O(steps * (m^3 + n^3 + m^2 n + m n^2))
    Space: O(m^2 + n^2) working memory beyond inputs.

    Safety features for numerical precision (especially in BF16/FP16):
    - Pre-normalizes inputs (`q_max`, `p_max`) to enforce absolute scale invariance
      and prevent overflow/underflow before computing the Frobenius norm.
    - Applies a 5% safety margin (`* 1.05`) to the Frobenius norm trace to strictly
      bound the eigenvalues of the scaled gram matrices strictly below 1.0, keeping
      the Newton-Schulz iterations well within their convergence radius.
    - Uses exact matmuls (`precision=jax.lax.Precision.HIGHEST` by default) during
      polynomial approximation to avoid asymmetric floating point drift that can break
      the Symmetric Positive Definite (SPD) property.
    """
    m, n = G.shape[-2], G.shape[-1]
    I_m = jnp.eye(m, dtype=Q.dtype)
    I_n = jnp.eye(n, dtype=P.dtype)

    q_max = jnp.maximum(jnp.max(jnp.abs(Q)), 1e-30)
    tQ = q_max * jnp.sqrt(jnp.sum(jnp.square(Q / q_max)))
    tQ_safe = jnp.maximum(tQ, 1e-30) * 1.05

    p_max = jnp.maximum(jnp.max(jnp.abs(P)), 1e-30)
    tP = p_max * jnp.sqrt(jnp.sum(jnp.square(P / p_max)))
    tP_safe = jnp.maximum(tP, 1e-30) * 1.05

    Q = Q / tQ_safe + eps * I_m
    P = P / tP_safe + eps * I_n

    for a, b, c in _invroot_coeffs_iter(r, steps, scale=scale):
        WQ = _sym(a * I_m + b * Q + c * _sym(_safe_mm(Q, Q, precision)))
        WP = _sym(a * I_n + b * P + c * _sym(_safe_mm(P, P, precision)))
        WQ1 = WQ if s == 1 else jnp.linalg.matrix_power(WQ, s)
        WP1 = WP if s == 1 else jnp.linalg.matrix_power(WP, s)
        # r=4 is the common path for bidirectional; explicit chain avoids
        # XLA's general eigendecomposition-based power lowering.
        if r == 1:
            WQ2, WP2 = WQ, WP
        elif r == 2:
            WQ2, WP2 = (
                _sym(_safe_mm(WQ, WQ, precision)),
                _sym(_safe_mm(WP, WP, precision)),
            )
        elif r == 4:
            WQ2_sq, WP2_sq = (
                _sym(_safe_mm(WQ, WQ, precision)),
                _sym(_safe_mm(WP, WP, precision)),
            )
            WQ2, WP2 = (
                _sym(_safe_mm(WQ2_sq, WQ2_sq, precision)),
                _sym(_safe_mm(WP2_sq, WP2_sq, precision)),
            )
        else:
            WQ2 = jnp.linalg.matrix_power(WQ, r)
            WP2 = jnp.linalg.matrix_power(WP, r)

        Q = _sym(_safe_mm(Q, WQ2, precision))
        G = _safe_mm(_safe_mm(WQ1, G, precision), WP1, precision)
        P = _sym(_safe_mm(P, WP2, precision))

    return G * tQ_safe ** (-s / r) * tP_safe ** (-s / r)


def _shampoo_prism_math(
    m_target: jax.Array,
    m_raw: jax.Array,
    g: jax.Array,
    gamma_l: float,
    gamma_r: float,
    inv_steps: int,
    inv_eps: float,
    inv_scale: float,
    eps_gram: float,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> jax.Array:
    r"""Bidirectional-PRISM: $\tilde{L}^{-1/4} \, M \, \tilde{R}^{-1/4}$.

    Shapes both output-feature (left) and input-feature (right) directions.
    The shaping coefficient $\rho_k^{bi} = \sqrt{\rho_k^{left} \cdot \rho_k^{right}}$
    is the geometric mean of one-sided PRISM coefficients, and lies in [0, 1],
    guaranteeing the spectral-norm trust-region constraint is satisfied.

    Safety features for numerical precision (especially in BF16/FP16):
    - Pre-normalizes inputs (`max_m`, `max_d`) to enforce absolute scale invariance
      and prevent overflow/underflow when computing the gram matrices.
    """
    m_target = m_target.astype(jnp.float32)
    m_raw = m_raw.astype(jnp.float32)
    g = g.astype(jnp.float32)

    D = g - m_raw
    m, n = m_target.shape[-2], m_target.shape[-1]

    gamma_max = max(gamma_l, gamma_r)
    max_m = jnp.max(jnp.abs(m_target), axis=(-2, -1), keepdims=True)
    max_d = jnp.max(jnp.abs(D), axis=(-2, -1), keepdims=True)
    max_val = jnp.maximum(jnp.maximum(max_m, gamma_max * max_d), 1e-30)

    norm = max_val * jnp.sqrt(
        jnp.sum(jnp.square(m_target / max_val), axis=(-2, -1), keepdims=True)
        + gamma_max**2 * jnp.sum(jnp.square(D / max_val), axis=(-2, -1), keepdims=True)
    )
    scale = jnp.maximum(norm, 1e-30)

    m_target_norm = m_target / scale
    D_norm = D / scale

    H_L = _sym(
        _safe_mm(m_target_norm, m_target_norm.mT, precision)
        + gamma_l**2 * _safe_mm(D_norm, D_norm.mT, precision)
    ) + eps_gram * jnp.eye(m, dtype=jnp.float32)

    H_R = _sym(
        _safe_mm(m_target_norm.mT, m_target_norm, precision)
        + gamma_r**2 * _safe_mm(D_norm.mT, D_norm, precision)
    ) + eps_gram * jnp.eye(n, dtype=jnp.float32)

    return _double_sided_matmul_invroot(
        H_L,
        m_target_norm,
        H_R,
        r=4,
        s=1,
        steps=inv_steps,
        eps=inv_eps,
        scale=inv_scale,
        precision=precision,
    )


def _prism_ortho_step(
    updates: jax.Array,
    mu_raw: jax.Array,
    gamma: float,
    ns_iters: int,
    mu_nest: Optional[jax.Array] = None,
    dim_nums: Optional[PrismDimensionNumbers] = None,
    mode: str = "original",
    inv_steps: int = 6,
    inv_eps: float = 1e-5,
    inv_scale: float = 1.001,
    eps_gram: float = 1e-6,
    gamma_l: Optional[float] = None,
    gamma_r: Optional[float] = None,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> jax.Array:
    """Orchestrates PRISM spectral shaping with mode selection.

    Modes:
        'original':      Newton-Schulz on augmented [M; γD].  Uses `ns_iters`.
        'bidirectional': H_L^{-1/4} M H_R^{-1/4} (Shampoo-style).  Uses `inv_steps`.
                         Shapes both left and right singular-vector spaces.
    """
    # Passthrough (partitioned-away or non-matrix leaves)
    if dim_nums is None or isinstance(dim_nums, _masking.MaskedNode):
        return mu_nest if mu_nest is not None else mu_raw

    m_target_eff = mu_nest if mu_nest is not None else mu_raw
    is_fast_2d = updates.ndim == 2 and _is_standard_2d_spec(dim_nums)

    # Original Newton-Schulz on augmented matrix
    if mode == "original":
        if is_fast_2d:
            return _apply_prism_math(updates, mu_raw, m_target_eff, gamma, ns_iters)
        reshape_fn, inverse_fn = _compute_prism_reshape(updates, dim_nums)
        O_flat = jax.vmap(lambda g, m, t: _apply_prism_math(g, m, t, gamma, ns_iters))(
            reshape_fn(updates), reshape_fn(mu_raw), reshape_fn(m_target_eff)
        )
        return inverse_fn(O_flat)

    # Bidirectional: Shampoo-style double-sided matmul-invroot
    if mode == "bidirectional":
        # Fall back to `gamma` when per-side gammas are not specified
        gl = gamma_l if gamma_l is not None else gamma
        gr = gamma_r if gamma_r is not None else gamma
        _fn = lambda g, m, t: _shampoo_prism_math(
            t,
            m,
            g,
            gl,
            gr,
            inv_steps,
            inv_eps,
            inv_scale,
            eps_gram,
            precision,
        )
        if is_fast_2d:
            return _fn(updates, mu_raw, m_target_eff)
        reshape_fn, inverse_fn = _compute_prism_reshape(updates, dim_nums)
        O_flat = jax.vmap(_fn)(
            reshape_fn(updates),
            reshape_fn(mu_raw),
            reshape_fn(m_target_eff),
        )
        return inverse_fn(O_flat)

    raise ValueError(
        f"Unknown PRISM mode: {mode!r}. Expected 'original' or 'bidirectional'."
    )


class ScaleByPrismState(NamedTuple):
    """State for the PRISM gradient transformation."""

    count: jax.Array
    mu: base.Updates
    magma_s: Any
    key: Optional[jax.Array]


def scale_by_prism(
    b1: float = 0.95,
    gamma: float = 1.0,
    ns_iters: int = 5,
    mode: str = "original",
    inv_steps: int = 6,
    inv_eps: float = 1e-5,
    inv_scale: float = 1.001,
    eps_gram: float = 1e-6,
    gamma_l: Optional[float] = None,
    gamma_r: Optional[float] = None,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    use_magma: bool = False,
    magma_tau: float = 2.0,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable]] = None,
    axis_name: Optional[str] = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """The core PRISM gradient transformation.

    Implements the core logic of momentum accumulation, innovation computation,
    and spectral shaping via Newton-Schulz orthogonalization.

    Args:
        b1: Exponential decay rate for the first moment (momentum).
        gamma: Damping coefficient for the innovation term. Controls the "anisotropy"
            of the spectral shaping.
        ns_iters: Number of Newton-Schulz iterations for orthogonalization.
        mode: Spectral shaping algorithm. 'original' uses Newton-Schulz on the
            augmented matrix. 'bidirectional' applies Shampoo-style bilateral shaping.
        inv_steps: Iteration count for the matmul-invroot polynomial (mode bidirectional).
        inv_eps: Regularization epsilon inside the iterative inverse root.
        inv_scale: Coefficient scaling factor (>1.0 for conservative convergence).
        eps_gram: Regularization added to gram matrices before inversion.
        gamma_l: Left-side innovation damping (bidirectional only). Defaults to `gamma`.
        gamma_r: Right-side innovation damping (bidirectional only). Defaults to `gamma`.
        precision: Numerical precision for bidirectional matmuls. Defaults to HIGHEST for
            eigenvalue-sensitive iterations.
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
        weight_dimension_numbers: Specification for reshaping tensors. If provided,
            `params` must be passed to `update`.
        use_magma: If True, applies Momentum-aligned gradient masking (Magma).
        magma_tau: Temperature parameter for the alignment sigmoid. Default is 2.0.
        weight_decay: Weight decay applied to PRISM.
        weight_decay_mask: A mask for weight decay.
        axis_name: Axis name for distributed (SPMD) global norm reduction.
        key: Initial PRNG key.

    Returns:
        An `optax.GradientTransformation`.

    References:
        Yang, Y. (2026). PRISM: Structured Optimization via Anisotropic Spectral Shaping.
        arXiv preprint arXiv:2602.03096.

        Cesista, F. L. (2026). Bidirectional-PRISM: Kronecker-Factored Optimization
        via Anisotropic Spectral Shaping.
        URL: https://leloykun.github.io/ponder/shampoo-prism/
    """
    if mu_dtype is None:
        mu_dtype = jnp.float32
    else:
        mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)

        if use_magma:

            def _init_s(x):
                if x is None:
                    return None
                if isinstance(x, _masking.MaskedNode):
                    return _masking.MaskedNode()
                return jnp.array(0.5, dtype=jnp.float32)

            magma_s = jax.tree.map(
                _init_s,
                params,
                is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
            )
        else:
            magma_s = ()

        return ScaleByPrismState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            magma_s=magma_s,
            key=key,
        )

    def update_fn(updates, state, params=None):
        raw_gradients = updates

        if use_magma:
            next_state_key, sr_key1, magma_key = jax.random.split(state.key, 3)
        else:
            next_state_key, sr_key1 = jax.random.split(state.key, 2)
            magma_key = None

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

        should_skip = jnp.logical_and(
            is_spike, jnp.logical_not(permissive_spike_protection)
        )

        effective_updates = jax.lax.cond(
            should_skip,
            lambda u: jax.tree.map(
                lambda x: jnp.zeros_like(x) if x is not None else None, u
            ),
            lambda u: u,
            updates,
        )

        # Enforce unified FP32 tracking to prevent accumulator vanishing
        mu_f32 = jax.lax.cond(
            should_skip,
            lambda: jax.tree.map(lambda m: m.astype(jnp.float32), state.mu),
            lambda: _tree_update_moment_f32(effective_updates, state.mu, b1),
        )

        # Nesterov / Bias Correction (Strictly FP32)
        mu_nest_f32 = mu_f32
        if nesterov:
            if bias_correction:
                mu_bc_factor = 1.0 - b1**count_inc
                mu_bc_factor_next = 1.0 - b1 ** numerics.safe_increment(count_inc)

                mu_bc_f32 = _safe_bias_correction(mu_f32, mu_bc_factor_next)

                # Explicitly bypass MaskedNodes for safe partitioned bias correction
                updates_f32 = jax.tree.map(
                    lambda x: (
                        x
                        if isinstance(x, _masking.MaskedNode)
                        else (x.astype(jnp.float32) if x is not None else None)
                    ),
                    effective_updates,
                    is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
                )
                g_bc_f32 = _safe_bias_correction(updates_f32, mu_bc_factor)

                mu_nest_f32 = jax.tree.map(
                    lambda m, g: (
                        m
                        if isinstance(m, _masking.MaskedNode)
                        else (b1 * m + (1.0 - b1) * g if m is not None else None)
                    ),
                    mu_bc_f32,
                    g_bc_f32,
                    is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
                )
            else:
                mu_nest_f32 = jax.tree.map(
                    lambda m, g: (
                        m
                        if isinstance(m, _masking.MaskedNode)
                        else (
                            b1 * m
                            + (1.0 - b1)
                            * (g.astype(jnp.float32) if g is not None else 0.0)
                            if m is not None
                            else None
                        )
                    ),
                    mu_f32,
                    effective_updates,
                    is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
                )
        elif bias_correction:
            mu_bc_factor = 1.0 - b1**count_inc
            mu_nest_f32 = _safe_bias_correction(mu_f32, mu_bc_factor)

        if mu_dtype == jnp.bfloat16:
            mu_cast = _tree_stochastic_cast(mu_f32, mu_dtype, sr_key1)
        else:
            mu_cast = optax.tree.cast(mu_f32, mu_dtype)

        if shape_nesterov:
            target_for_ortho_f32 = mu_nest_f32  # FP32, not mu_nest_cast
        else:
            target_for_ortho_f32 = jax.tree.map(lambda _: None, effective_updates)

        prism_out = jax.tree.map(
            lambda g, m, target, dims: _prism_ortho_step(
                g,
                m,
                gamma,
                ns_iters,
                mu_nest=target,
                dim_nums=dims,
                mode=mode,
                inv_steps=inv_steps,
                inv_eps=inv_eps,
                inv_scale=inv_scale,
                eps_gram=eps_gram,
                gamma_l=gamma_l,
                gamma_r=gamma_r,
                precision=precision,
            ),
            effective_updates,
            mu_f32,
            target_for_ortho_f32,
            resolved_dim_nums,
            is_leaf=_is_prism_leaf,
        )

        if nesterov and not shape_nesterov:
            new_updates = jax.tree.map(
                lambda o, g: b1 * o + (1 - b1) * g, prism_out, effective_updates
            )
        else:
            new_updates = prism_out

        if grad_clip_max_amps is not None:
            max_rms, max_val = (
                grad_clip_max_amps
                if isinstance(grad_clip_max_amps, tuple)
                else (grad_clip_max_amps, 10.0)
            )
            new_updates = jax.tree.map(
                lambda u: _clip_per_tensor_rms(u, max_rms, max_val), new_updates
            )

        _may_have_wd = not isinstance(weight_decay, (int, float)) or weight_decay > 0.0
        if _may_have_wd and params is not None:
            wd_step = (
                weight_decay(state.count) if callable(weight_decay) else weight_decay
            )
            _wd_mask = None
            if weight_decay_mask is not None:
                _wd_mask = (
                    weight_decay_mask(params)
                    if callable(weight_decay_mask)
                    else weight_decay_mask
                )

            def _add_wd(u, p, m=True):
                if _is_prism_leaf(u) or _is_prism_leaf(p):
                    return u
                if isinstance(m, _masking.MaskedNode) or m is None or not m:
                    return u
                return u + wd_step * p.astype(u.dtype)

            if _wd_mask is not None:
                new_updates = jax.tree.map(
                    _add_wd, new_updates, params, _wd_mask, is_leaf=_is_prism_leaf
                )
            else:
                new_updates = jax.tree.map(
                    lambda u, p: _add_wd(u, p),
                    new_updates,
                    params,
                    is_leaf=_is_prism_leaf,
                )

        new_updates = jax.lax.cond(
            should_skip,
            lambda u: jax.tree.map(jnp.zeros_like, u),
            lambda u: u,
            new_updates,
        )

        if use_magma:
            final_updates, new_magma_s = apply_magma_internal(
                raw_gradients=raw_gradients,
                first_moments=mu_f32,
                base_updates=new_updates,
                magma_s_prev=state.magma_s,
                key=magma_key,
                tau=magma_tau,
                axis_name=axis_name,
            )
        else:
            final_updates = new_updates
            new_magma_s = state.magma_s

        if use_magma:
            new_magma_s = jax.tree.map(
                lambda new_s, old_s: jnp.where(should_skip, old_s, new_s),
                new_magma_s,
                state.magma_s,
            )

        return final_updates, ScaleByPrismState(
            count=count_inc,
            mu=mu_cast,
            magma_s=new_magma_s,
            key=next_state_key,
        )

    return base.GradientTransformation(init_fn, update_fn)


def prism(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.95,
    gamma: float = 1.0,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    ns_iters: int = 5,
    mode: str = "original",
    inv_steps: int = 6,
    inv_eps: float = 1e-5,
    inv_scale: float = 1.001,
    eps_gram: float = 1e-6,
    gamma_l: Optional[float] = None,
    gamma_r: Optional[float] = None,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    axis_name: Optional[str] = None,
    use_magma: bool = False,
    magma_tau: float = 2.0,
    key: jax.Array = jax.random.PRNGKey(42),
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
        weight_decay_mask: Optional mask for weight decay.
        ns_iters: Number of Newton-Schulz iterations for PRISM.
        mode: Spectral shaping algorithm for the PRISM branch.
            'original': Newton-Schulz on augmented [M; γD] (default, uses `ns_iters`).
            'bidirectional': Shampoo-style bilateral shaping (uses `inv_steps`).
        inv_steps: Polynomial iterations for mode 'bidirectional'.
        inv_eps: Regularization for the iterative inverse root solver.
        inv_scale: Convergence scaling (>1.0). Default 1.001.
        eps_gram: Gram matrix regularization epsilon. Default 1e-6.
        gamma_l: Left innovation damping (bidirectional only). Defaults to `gamma`.
        gamma_r: Right innovation damping (bidirectional only). Defaults to `gamma`.
        precision: Numerical precision for bidirectional matmuls. Defaults to HIGHEST for
            eigenvalue-sensitive iterations.
        nesterov: Whether to use Nesterov acceleration in PRISM.
        shape_nesterov: Whether to shape the Nesterov momentum or raw momentum.
        bias_correction: Whether to enable bias correction in PRISM.
        grad_clip_max_amps: Post-shaping clipping configuration (RMS, Abs).
        raw_global_grad_clip: Global gradient norm clipping threshold.
        permissive_spike_protection: Behavior when global clip is triggered (Clip vs Skip).
        mu_dtype: Dtype for momentum accumulators.
        axis_name: Axis name for distributed (SPMD) global norm reduction.
        use_magma: If True, applies Momentum-aligned gradient masking (Magma).
            WARNING: Magma introduces intentional update bias (damping). At an
            equilibrium tau=2.0, non-masked steps scale updates by ~0.5, and
            50% of steps are masked. This yields an expected magnitude attenuation
            of ~0.25x. You may need to scale the global learning rate by ~4x to
            maintain the original update volume.
        magma_tau: Temperature parameter for the alignment sigmoid. Default is 2.0.
        key: Initial PRNG key for Magma's Bernoulli sampling.
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

    References:
        Yang, Y. (2026). PRISM: Structured Optimization via Anisotropic Spectral Shaping.
        arXiv preprint arXiv:2602.03096.

        Cesista, F. L. (2026). Bidirectional-PRISM: Kronecker-Factored Optimization
        via Anisotropic Spectral Shaping.
        URL: https://leloykun.github.io/ponder/shampoo-prism/
    """
    key_prism, key_adam = jax.random.split(key, 2)

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

    prism_components = [
        scale_by_prism(
            b1=b1,
            gamma=gamma,
            ns_iters=ns_iters,
            mode=mode,
            inv_steps=inv_steps,
            inv_eps=inv_eps,
            inv_scale=inv_scale,
            eps_gram=eps_gram,
            gamma_l=gamma_l,
            gamma_r=gamma_r,
            precision=precision,
            nesterov=nesterov,
            shape_nesterov=shape_nesterov,
            bias_correction=bias_correction,
            mu_dtype=mu_dtype,
            raw_global_grad_clip=raw_global_grad_clip,
            permissive_spike_protection=permissive_spike_protection,
            grad_clip_max_amps=grad_clip_max_amps,
            weight_dimension_numbers=prism_weight_dim_nums_fn,
            use_magma=use_magma,
            magma_tau=magma_tau,
            weight_decay=weight_decay if use_magma else 0.0,
            weight_decay_mask=weight_decay_mask if use_magma else None,
            axis_name=axis_name,
            key=key_prism,
        ),
    ]

    _wd_is_nonzero = (
        weight_decay > 0.0 if isinstance(weight_decay, (int, float)) else True
    )
    if _wd_is_nonzero and not use_magma:
        prism_components.append(
            transform.add_decayed_weights(weight_decay, weight_decay_mask)
        )

    prism_components.append(transform.scale_by_learning_rate(learning_rate))

    return combine.partition(
        transforms={
            "prism": combine.chain(*prism_components),
            "adam": adamw(
                learning_rate=adam_learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
                weight_decay=weight_decay,
                weight_decay_mask=weight_decay_mask,
                mu_dtype=mu_dtype,
                use_magma=use_magma,
                magma_tau=magma_tau,
                axis_name=axis_name,
                key=key_adam,
            ),
        },
        param_labels=param_labels,
    )
