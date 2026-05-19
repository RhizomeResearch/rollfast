"""PRISM matrix optimizers.

PRISM routes rank-2 leaves to the matrix branch by default and sends other
leaves to AdamW. ``mode="original"`` uses the shared Muon-family
Newton-Schulz coefficient plumbing; ``mode="bidirectional"`` keeps its separate
inverse-root coefficients.
"""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
from optax._src import base, combine, transform, utils
from optax.transforms import _masking

from rollfast.optim.adam import adamw
from rollfast.optim._matrix_runtime import (
    apply_matrix_post_shape_lookahead,
    finish_matrix_runtime_step,
    init_matrix_magma_state,
    init_matrix_momentum_state,
    prepare_matrix_runtime_step,
)
from rollfast.optim.dimension_numbers import (
    DimNumsTree,
    MatrixDimensionNumbers,
    WeightDimNumOrFn,
    _compute_matrix_reshape,
    _make_equinox_matrix_spec,
    _is_dimension_numbers_leaf,
    _is_standard_2d_spec,
    _make_matrix_partition_fns,
    _resolve_update_dimension_numbers,
    _validate_matrix_operand,
)
from rollfast.optim.magma import validate_magma_args
from rollfast.optim.orthogonalization import (
    MUON_NS_COEFFS,
    MuonNsCoeffs,
    MuonPreconditioning,
    quintic_newton_schulz,
    resolve_ns_coeffs,
)
from rollfast.utils import (
    MomentumAccumulator,
    _has_nonzero_or_scheduled,
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
    _EQUINOX_AVAILABLE = False
    _LINEAR_TYPES = ()
    _CONV_TYPES = ()

_DEFAULT_NS_COEFFS = MUON_NS_COEFFS
_INVROOT_COEFFS: dict[int, list[tuple[float, float, float]]] = {
    4: [  # P^{-1/4}  (used by Bidirectional-PRISM)
        (3.85003, -10.8539, 8.61893),
        (1.80992, -0.587778, 0.0647852),
        (1.50394, -0.594516, 0.121161),
        (45 / 32, -9 / 16, 5 / 32),
    ],
}


PrismDimensionNumbers = MatrixDimensionNumbers
_is_prism_leaf = _is_dimension_numbers_leaf
_compute_prism_reshape = _compute_matrix_reshape


def get_equinox_prism_spec(
    model: Any,
    skip_depthwise_conv: bool = True,
) -> DimNumsTree:
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
            'Please install it via `pip install "rollfast[equinox]"`.'
        )
    import equinox as eqx

    return _make_equinox_matrix_spec(
        model,
        eqx_module=eqx,
        linear_types=_LINEAR_TYPES,
        conv_types=_CONV_TYPES,
        dimension_numbers_type=PrismDimensionNumbers,
        skip_depthwise_conv=skip_depthwise_conv,
        strict_conv_in_channels=True,
    )


def _quintic_newton_schulz(
    x: jax.Array,
    iters: int = 5,
    eps: float = 1e-8,
    preconditioning: MuonPreconditioning = "frobenius",
    ns_coeffs: jax.Array | None = None,
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
    if ns_coeffs is None:
        ns_coeffs = resolve_ns_coeffs(_DEFAULT_NS_COEFFS, iters)
    return quintic_newton_schulz(
        x,
        ns_coeffs,
        ns_steps=iters,
        preconditioning=preconditioning,
        eps=eps,
    )


def _apply_prism_math(
    g,
    m_raw,
    m_target,
    gamma,
    ns_iters,
    ns_coeffs,
    preconditioning: MuonPreconditioning,
):
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
    augmented_O = _quintic_newton_schulz(
        augmented_M,
        iters=ns_iters,
        preconditioning=preconditioning,
        ns_coeffs=ns_coeffs,
    )

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
    ns_coeffs: jax.Array,
    mu_nest: Optional[jax.Array] = None,
    dim_nums: Optional[PrismDimensionNumbers] = None,
    mode: str = "original",
    preconditioning: MuonPreconditioning = "frobenius",
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
    _validate_matrix_operand(updates, dim_nums, "scale_by_prism")

    m_target_eff = mu_nest if mu_nest is not None else mu_raw
    is_fast_2d = updates.ndim == 2 and _is_standard_2d_spec(dim_nums)

    # Original Newton-Schulz on augmented matrix
    if mode == "original":
        if is_fast_2d:
            return _apply_prism_math(
                updates,
                mu_raw,
                m_target_eff,
                gamma,
                ns_iters,
                ns_coeffs,
                preconditioning,
            )
        reshape_fn, inverse_fn = _compute_prism_reshape(updates, dim_nums)
        O_flat = jax.vmap(
            lambda g, m, t: _apply_prism_math(
                g,
                m,
                t,
                gamma,
                ns_iters,
                ns_coeffs,
                preconditioning,
            )
        )(reshape_fn(updates), reshape_fn(mu_raw), reshape_fn(m_target_eff))
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
    ns_coeffs: MuonNsCoeffs = MUON_NS_COEFFS,
    mode: str = "original",
    preconditioning: MuonPreconditioning = "frobenius",
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
    momentum_accumulator: MomentumAccumulator = "ema",
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    use_magma: bool = False,
    magma_p: float = 0.5,
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
        ns_coeffs: Newton-Schulz coefficient preset or coefficients for
            ``mode='original'``. Supports ``"standard"``, ``"dion"``,
            ``"polar_express"``, a single ``(a, b, c)`` tuple, or an ordered
            per-step ``(n, 3)`` schedule.
        mode: Spectral shaping algorithm. 'original' uses Newton-Schulz on the
            augmented matrix. 'bidirectional' applies Shampoo-style bilateral shaping.
        preconditioning: Newton-Schulz input preconditioning for ``mode='original'``.
            Ignored by ``mode='bidirectional'``.
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
        momentum_accumulator: ``"ema"`` for exponential moving average momentum,
            or ``"heavy_ball"`` for heavy-ball accumulation.
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
        magma_p: Survival probability for the block-wise Bernoulli masking.
            Dictates the likelihood (0.0 <= p <= 1.0) that a parameter block's update
            survives at a given step. A value of 1.0 effectively bypasses stochastic
            dropping (though Magma EMA damping still applies). The default of 0.5 was
            empirically validated as optimal for transformer pre-training.
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
    if use_magma:
        validate_magma_args(magma_p, magma_tau)

    canonical_mu_dtype = cast(
        jax.typing.DTypeLike,
        jnp.float32 if mu_dtype is None else utils.canonicalize_dtype(mu_dtype),
    )

    def init_fn(params):
        return ScaleByPrismState(
            count=jnp.zeros([], jnp.int32),
            mu=init_matrix_momentum_state(params, canonical_mu_dtype),
            magma_s=init_matrix_magma_state(params, use_magma),
            key=key,
        )

    def update_fn(updates, state, params=None):
        resolved_dim_nums = _resolve_update_dimension_numbers(
            weight_dimension_numbers,
            params=params,
            updates=updates,
            transform_name="scale_by_prism",
        )
        jax.tree.map(
            lambda u, d: _validate_matrix_operand(u, d, "scale_by_prism"),
            updates,
            resolved_dim_nums,
            is_leaf=_is_prism_leaf,
        )

        runtime = prepare_matrix_runtime_step(
            updates,
            count=state.count,
            mu=state.mu,
            key=state.key,
            beta=b1,
            nesterov=nesterov,
            shape_nesterov=shape_nesterov,
            bias_correction=bias_correction,
            momentum_accumulator=momentum_accumulator,
            mu_dtype=canonical_mu_dtype,
            raw_global_grad_clip=raw_global_grad_clip,
            permissive_spike_protection=permissive_spike_protection,
            use_magma=use_magma,
            axis_name=axis_name,
        )

        resolved_ns_coeffs = resolve_ns_coeffs(ns_coeffs, ns_iters)
        prism_out = jax.tree.map(
            lambda g, m, target, dims: _prism_ortho_step(
                g,
                m,
                gamma,
                ns_iters,
                resolved_ns_coeffs,
                mu_nest=target,
                dim_nums=dims,
                mode=mode,
                preconditioning=preconditioning,
                inv_steps=inv_steps,
                inv_eps=inv_eps,
                inv_scale=inv_scale,
                eps_gram=eps_gram,
                gamma_l=gamma_l,
                gamma_r=gamma_r,
                precision=precision,
            ),
            runtime.effective_updates,
            runtime.mu_f32,
            runtime.target_for_shape,
            resolved_dim_nums,
            is_leaf=_is_prism_leaf,
        )
        prism_out = apply_matrix_post_shape_lookahead(
            prism_out,
            runtime,
            beta=b1,
            nesterov=nesterov,
            shape_nesterov=shape_nesterov,
            momentum_accumulator=momentum_accumulator,
        )
        final_updates, new_magma_s = finish_matrix_runtime_step(
            prism_out,
            runtime,
            params=params,
            magma_s=state.magma_s,
            use_magma=use_magma,
            magma_p=magma_p,
            magma_tau=magma_tau,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            grad_clip_max_amps=grad_clip_max_amps,
            axis_name=axis_name,
        )

        return final_updates, ScaleByPrismState(
            count=runtime.count,
            mu=runtime.mu_cast,
            magma_s=new_magma_s,
            key=runtime.next_key,
        )

    return base.GradientTransformation(init_fn, update_fn)


def _build_unscaled_prism_branch(
    *,
    b1: float,
    gamma: float,
    ns_iters: int,
    ns_coeffs: MuonNsCoeffs,
    mode: str,
    preconditioning: MuonPreconditioning,
    inv_steps: int,
    inv_eps: float,
    inv_scale: float,
    eps_gram: float,
    gamma_l: Optional[float],
    gamma_r: Optional[float],
    precision: jax.lax.PrecisionLike,
    nesterov: bool,
    shape_nesterov: bool,
    bias_correction: bool,
    momentum_accumulator: MomentumAccumulator,
    mu_dtype: Optional[jax.typing.DTypeLike],
    raw_global_grad_clip: Optional[float],
    permissive_spike_protection: bool,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]],
    weight_dimension_numbers: WeightDimNumOrFn | None,
    use_magma: bool,
    magma_p: float,
    magma_tau: float,
    weight_decay: base.ScalarOrSchedule,
    weight_decay_mask: Optional[Union[Any, Callable]],
    axis_name: Optional[str],
    key: jax.Array,
) -> base.GradientTransformation:
    """Build the unscaled PRISM direction branch shared by wrappers."""
    components = [
        scale_by_prism(
            b1=b1,
            gamma=gamma,
            ns_iters=ns_iters,
            ns_coeffs=ns_coeffs,
            mode=mode,
            preconditioning=preconditioning,
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
            momentum_accumulator=momentum_accumulator,
            mu_dtype=mu_dtype,
            raw_global_grad_clip=raw_global_grad_clip,
            permissive_spike_protection=permissive_spike_protection,
            grad_clip_max_amps=grad_clip_max_amps,
            weight_dimension_numbers=weight_dimension_numbers,
            use_magma=use_magma,
            magma_p=magma_p,
            magma_tau=magma_tau,
            weight_decay=weight_decay if use_magma else 0.0,
            weight_decay_mask=weight_decay_mask if use_magma else None,
            axis_name=axis_name,
            key=key,
        ),
    ]

    if _has_nonzero_or_scheduled(weight_decay) and not use_magma:
        components.append(
            transform.add_decayed_weights(weight_decay, weight_decay_mask)
        )

    return combine.chain(*components)


def prism(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.95,
    gamma: float = 1.0,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    ns_iters: int = 5,
    ns_coeffs: MuonNsCoeffs = MUON_NS_COEFFS,
    mode: str = "original",
    preconditioning: MuonPreconditioning = "frobenius",
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
    momentum_accumulator: MomentumAccumulator = "ema",
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    axis_name: Optional[str] = None,
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    key: jax.Array = jax.random.PRNGKey(42),
    # Partitioning Arguments
    adam_learning_rate: Optional[base.ScalarOrSchedule] = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    prism_weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformation:
    """PRISM optimizer with automatic matrix/AdamW partitioning.

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
        ns_coeffs: Newton-Schulz coefficient preset or coefficients for
            ``mode='original'``. Supports ``"standard"``, ``"dion"``,
            ``"polar_express"``, a single ``(a, b, c)`` tuple, or an ordered
            per-step ``(n, 3)`` schedule.
        mode: Spectral shaping algorithm for the PRISM branch.
            'original': Newton-Schulz on augmented [M; γD] (default, uses `ns_iters`).
            'bidirectional': Shampoo-style bilateral shaping (uses `inv_steps`).
        preconditioning: Newton-Schulz input preconditioning for original mode.
            Ignored by bidirectional mode.
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
        momentum_accumulator: ``"ema"`` for exponential moving average momentum,
            or ``"heavy_ball"`` for heavy-ball accumulation.
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
            maintain the undamped update volume.
        magma_p: Survival probability for the block-wise Bernoulli masking.
            Dictates the likelihood (0.0 <= p <= 1.0) that a parameter block's update
            survives at a given step. A value of 1.0 effectively bypasses stochastic
            dropping (though Magma EMA damping still applies). The default of 0.5 was
            empirically validated as optimal for transformer pre-training.
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

    partition = _make_matrix_partition_fns(prism_weight_dimension_numbers, "prism")

    prism_branch = _build_unscaled_prism_branch(
        b1=b1,
        gamma=gamma,
        ns_iters=ns_iters,
        ns_coeffs=ns_coeffs,
        mode=mode,
        preconditioning=preconditioning,
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
        momentum_accumulator=momentum_accumulator,
        mu_dtype=mu_dtype,
        raw_global_grad_clip=raw_global_grad_clip,
        permissive_spike_protection=permissive_spike_protection,
        grad_clip_max_amps=grad_clip_max_amps,
        weight_dimension_numbers=partition.masked_specs,
        use_magma=use_magma,
        magma_p=magma_p,
        magma_tau=magma_tau,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        axis_name=axis_name,
        key=key_prism,
    )

    return combine.partition(
        transforms={
            "prism": combine.chain(
                prism_branch,
                transform.scale_by_learning_rate(learning_rate),
            ),
            "adam": adamw(
                learning_rate=adam_learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
                weight_decay=weight_decay,
                weight_decay_mask=weight_decay_mask,
                mu_dtype=mu_dtype,
                nesterov=nesterov,
                use_magma=use_magma,
                magma_p=magma_p,
                magma_tau=magma_tau,
                axis_name=axis_name,
                key=key_adam,
            ),
        },
        param_labels=partition.labels,
    )
