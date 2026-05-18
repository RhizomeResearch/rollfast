"""Aurora optimizers for Rollfast.

This module implements the Aurora and Riemannian-Aurora matrix update rules from
Tilde Research's reference PyTorch release in Optax/Rollfast style.

The core transforms return positive, unscaled gradient-like updates. The public
`aurora` and `riemannian_aurora` helpers add decoupled weight decay and learning
rate scaling in the same style as `rollfast.optim.adam.adamw` and
`rollfast.optim.prism.prism`.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
import optax
from optax._src import base, combine, numerics, transform, utils
from optax.transforms import _masking

from rollfast.optim.adam import adamw
from rollfast.optim.dimension_numbers import (
    MatrixDimensionNumbers,
    WeightDimNumOrFn,
    _compute_matrix_reshape,
    _get_dimension_numbers,
    _is_dimension_numbers_leaf,
    _is_standard_2d_spec,
    _mask_dimension_numbers,
)
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

    _AURORA_EQUINOX_AVAILABLE = True
    _AURORA_LINEAR_TYPES = (eqx.nn.Linear,)
    _AURORA_CONV_TYPES = (
        eqx.nn.Conv1d,
        eqx.nn.Conv2d,
        eqx.nn.Conv3d,
        eqx.nn.ConvTranspose1d,
        eqx.nn.ConvTranspose2d,
        eqx.nn.ConvTranspose3d,
    )
except ImportError:
    _AURORA_EQUINOX_AVAILABLE = False
    _AURORA_LINEAR_TYPES = ()
    _AURORA_CONV_TYPES = ()

AuroraDimensionNumbers = MatrixDimensionNumbers
AuroraWeightDimNumOrFn = WeightDimNumOrFn

_SIMPLE_QUINTIC_COEFFS = (2.0, -1.5, 0.5)


def get_equinox_aurora_spec(
    model: Any,
    skip_depthwise_conv: bool = True,
) -> base.Params:
    """Generates an Aurora dimension spec tree for an Equinox model.

    This is the Aurora analogue of `get_equinox_prism_spec`.

    Features:
    - Uses structural replacement via `eqx.tree_at`, so it does not rely on
      object identity or fragile path matching.
    - Handles Equinox Linear, Conv, and ConvTranspose layers.
    - Optionally skips depthwise convolutions, routing them to the Adam branch.

    Args:
        model: Equinox model to generate specs for.
        skip_depthwise_conv: If True, depthwise convolutions are assigned `None`
            and therefore handled by the Adam fallback branch. Set to False to
            apply Aurora to all convolution kernels.

    Returns:
        A PyTree matching `model`, with `AuroraDimensionNumbers` on selected
        weight leaves and `None` elsewhere.
    """
    if not _AURORA_EQUINOX_AVAILABLE:
        raise ImportError(
            "The function `get_equinox_aurora_spec` requires the `equinox` "
            "library. Please install it via `pip install equinox`."
        )
    import equinox as eqx

    def _layer_to_spec(layer):
        target_spec = None

        if isinstance(layer, _AURORA_LINEAR_TYPES):
            # Equinox Linear weight layout is conventionally (out_features, in_features).
            # Aurora's reshape convention uses (reduction, output), so mirror PRISM:
            # reduction_axis=1, output_axis=0.
            target_spec = AuroraDimensionNumbers(
                reduction_axis=1,
                output_axis=0,
            )

        elif isinstance(layer, _AURORA_CONV_TYPES):
            groups = getattr(layer, "groups", 1)
            in_channels = getattr(layer, "in_channels", None)
            is_depthwise = (
                groups > 1 and in_channels is not None and groups == in_channels
            )

            if not (skip_depthwise_conv and is_depthwise):
                ndim = layer.weight.ndim
                target_spec = AuroraDimensionNumbers(
                    reduction_axis=tuple(range(1, ndim)),
                    output_axis=0,
                )

        if target_spec is None:
            return jax.tree.map(lambda _: None, layer)

        specs = eqx.tree_at(lambda l: l.weight, layer, target_spec)

        return jax.tree.map(
            lambda x: x if isinstance(x, AuroraDimensionNumbers) else None,
            specs,
            is_leaf=lambda x: isinstance(x, AuroraDimensionNumbers),
        )

    return jax.tree.map(
        _layer_to_spec,
        model,
        is_leaf=lambda x: isinstance(x, _AURORA_LINEAR_TYPES + _AURORA_CONV_TYPES),
    )


class ScaleByAuroraState(NamedTuple):
    """State for Aurora-style gradient transformations."""

    count: jax.Array
    mu: base.Updates
    magma_s: Any
    key: Optional[jax.Array]


def _is_aux_leaf(x: Any) -> bool:
    return x is None or isinstance(x, _masking.MaskedNode)


def _is_dim_leaf(x: Any) -> bool:
    return _is_dimension_numbers_leaf(x)


def _is_array_like(x: Any) -> bool:
    return hasattr(x, "dtype") and hasattr(x, "shape")


def _tree_cast_f32(tree: Any) -> Any:
    return jax.tree.map(
        lambda x: x if _is_aux_leaf(x) else x.astype(jnp.float32),
        tree,
        is_leaf=_is_aux_leaf,
    )


def _zero_like_tree(tree: Any) -> Any:
    return jax.tree.map(
        lambda x: x if _is_aux_leaf(x) else jnp.zeros_like(x),
        tree,
        is_leaf=_is_aux_leaf,
    )


def _tree_where_scalar(pred: jax.Array, a: Any, b: Any) -> Any:
    return jax.tree.map(
        lambda x, y: x if _is_aux_leaf(x) else jnp.where(pred, x, y),
        a,
        b,
        is_leaf=_is_aux_leaf,
    )


def _tree_global_norm(grads: Any, axis_name: Optional[str] = None) -> jax.Array:
    """Global L2 norm over a PyTree, robust to MaskedNode/None leaves."""
    leaves = jax.tree.leaves(grads, is_leaf=_is_aux_leaf)
    sq_terms = [
        jnp.sum(numerics.abs_sq(x.astype(jnp.float32)))
        for x in leaves
        if not _is_aux_leaf(x)
    ]
    if not sq_terms:
        local_sq = jnp.array(0.0, dtype=jnp.float32)
    else:
        local_sq = sum(sq_terms, start=jnp.array(0.0, dtype=jnp.float32))
    total_sq = dist_reduce(local_sq, axis_name, "sum")
    return jnp.sqrt(total_sq)


def _clip_per_tensor_rms(
    u: jax.Array, max_rms: float = 1.0, max_val: float = 10.0
) -> jax.Array:
    rms = jnp.sqrt(jnp.mean(numerics.abs_sq(u)))
    scale_factor = jnp.minimum(1.0, max_rms / (rms + 1e-9))
    return jnp.clip(u * scale_factor, -max_val, max_val)


def _guard_nonfinite_leaf(u: jax.Array) -> jax.Array:
    return jnp.where(jnp.all(jnp.isfinite(u)), u, jnp.zeros_like(u))


def _simple_quintic_polar(
    G: jax.Array,
    *,
    iters: int = 12,
    eps: float = 1e-7,
    compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    output_dtype: jax.typing.DTypeLike = jnp.float32,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> jax.Array:
    """Reference simple-quintic Newton-Schulz polar factor.

    This matches the reference Aurora `polar.py` algorithm: cast to bf16 by
    default, transpose tall matrices to make the small Gram matrix, normalize by
    Frobenius norm, and run 12 iterations of p(s)=2s-1.5s^3+0.5s^5.
    """
    if G.ndim < 2:
        raise ValueError(f"polar expects rank >= 2, got {G.ndim=}")

    X = G.astype(compute_dtype)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = jnp.swapaxes(X, -1, -2)

    norm = jnp.linalg.norm(X.astype(jnp.float32), axis=(-2, -1), keepdims=True)
    X = X / (norm.astype(X.dtype) + jnp.asarray(eps, dtype=X.dtype))

    a, b, c = (jnp.asarray(v, dtype=X.dtype) for v in _SIMPLE_QUINTIC_COEFFS)

    def body_fn(_, x):
        A = jnp.matmul(x, jnp.swapaxes(x, -1, -2), precision=precision)
        B = b * A + c * jnp.matmul(A, A, precision=precision)
        return a * x + jnp.matmul(B, x, precision=precision)

    X = jax.lax.fori_loop(0, iters, body_fn, X, unroll=True)

    if transposed:
        X = jnp.swapaxes(X, -1, -2)
    return X.astype(output_dtype)


def _aspect_scale_for_matrix(x: jax.Array) -> jax.Array:
    m, n = x.shape[-2], x.shape[-1]
    return jnp.asarray(max(1.0, float(m) / float(n)) ** 0.5, dtype=jnp.float32)


def _aurora_balanced_polar_matrix(
    update: jax.Array,
    *,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    eps: float = 1e-7,
    polar_ns_iters: int = 12,
    polar_compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    polar_output_dtype: jax.typing.DTypeLike = jnp.float32,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> jax.Array:
    """Aurora's practical diagonal-preconditioned polar update for one matrix."""
    if update.ndim != 2:
        raise ValueError(
            f"Aurora matrix update expects rank-2 leaves, got {update.ndim=}"
        )

    rows, cols = update.shape
    if rows == cols:
        U = _simple_quintic_polar(
            update,
            iters=polar_ns_iters,
            eps=eps,
            compute_dtype=polar_compute_dtype,
            output_dtype=polar_output_dtype,
            precision=precision,
        )
        return U.astype(jnp.float32) * _aspect_scale_for_matrix(update)

    transposed = rows < cols
    G = jnp.swapaxes(update, -1, -2) if transposed else update
    G32 = G.astype(jnp.float32)
    m, n = G32.shape
    target_row_sq = jnp.asarray(n / m, dtype=jnp.float32)

    row_norm = jnp.maximum(jnp.linalg.norm(G32, axis=-1, keepdims=True), eps)
    D = 1.0 / row_norm
    U = jnp.zeros_like(G32, dtype=jnp.float32)

    for k in range(pp_iterations):
        U = _simple_quintic_polar(
            D * G32,
            iters=polar_ns_iters,
            eps=eps,
            compute_dtype=polar_compute_dtype,
            output_dtype=polar_output_dtype,
            precision=precision,
        ).astype(jnp.float32)
        if k < pp_iterations - 1:
            row_sq = jnp.maximum(
                jnp.sum(jnp.square(U), axis=-1, keepdims=True), eps * eps
            )
            D = D * jnp.power(target_row_sq / row_sq, pp_beta)

    if transposed:
        U = jnp.swapaxes(U, -1, -2)
    return U.astype(jnp.float32) * _aspect_scale_for_matrix(update)


def _solve_row_norm_multipliers(
    U: jax.Array,
    r: jax.Array,
    b: jax.Array,
    *,
    max_iter: int = 20,
    eps: float = 1e-7,
) -> jax.Array:
    """Approximate solve of (r I - (P ∘ P)) λ = b, P=UU^T.

    The matvec avoids materializing P. This mirrors the reference CG solver but
    uses a fixed-length JAX loop with a carried convergence flag.
    """
    del eps  # Kept for API parity with the reference implementation.

    U = U.astype(jnp.float32)
    b = b.astype(jnp.float32)
    h_sq = jnp.square(jnp.sum(jnp.square(U), axis=-1))
    reg = jnp.maximum(jnp.max(h_sq) - r + 1e-3, 0.0)
    r_eff = r + reg

    def matvec(v):
        T = jnp.matmul(jnp.swapaxes(U, -1, -2), v[:, None] * U)
        return r_eff * v - jnp.sum(jnp.matmul(U, T) * U, axis=-1)

    x0 = jnp.zeros_like(b)
    res0 = b
    p0 = res0
    rs0 = jnp.sum(jnp.square(res0))
    b_norm = jnp.maximum(jnp.linalg.norm(b), 1e-12)
    done0 = jnp.array(False, dtype=jnp.bool_)

    def body_fn(_, state):
        x, res, p, rs_old, done = state
        Ap = matvec(p)
        denom = jnp.sum(p * Ap)
        active = jnp.logical_and(~done, denom >= 1e-30)
        alpha = rs_old / jnp.where(active, denom, 1.0)

        x_new = x + alpha * p
        res_new = res - alpha * Ap
        rs_new = jnp.sum(jnp.square(res_new))
        finite_rs = jnp.isfinite(rs_new)
        converged = jnp.sqrt(jnp.maximum(rs_new, 0.0)) < 1e-8 * b_norm
        done_new = done | (~active) | (~finite_rs) | converged

        beta = rs_new / jnp.maximum(rs_old, 1e-30)
        p_new = res_new + beta * p

        x = jnp.where(active, x_new, x)
        res = jnp.where(active, res_new, res)
        p = jnp.where(active & finite_rs & (~converged), p_new, p)
        rs_old = jnp.where(active & finite_rs, rs_new, rs_old)
        return x, res, p, rs_old, done_new

    x, _, _, _, _ = jax.lax.fori_loop(
        0, max_iter, body_fn, (x0, res0, p0, rs0, done0), unroll=True
    )
    return jnp.where(jnp.all(jnp.isfinite(x)), x, jnp.zeros_like(b))


def _riemannian_balanced_polar_matrix(
    G: jax.Array,
    *,
    outer_steps: int = 3,
    cg_steps: int = 20,
    riemannian_eta: float = 0.1,
    retraction_steps: int = 2,
    eps: float = 1e-7,
    polar_ns_iters: int = 12,
    polar_compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    polar_output_dtype: jax.typing.DTypeLike = jnp.float32,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> jax.Array:
    """Riemannian-Aurora balanced Stiefel update for one matrix."""
    if G.ndim != 2:
        raise ValueError(
            f"Riemannian-Aurora matrix update expects rank-2 leaves, got {G.ndim=}"
        )

    rows, cols = G.shape
    if rows == cols:
        U = _simple_quintic_polar(
            G,
            iters=polar_ns_iters,
            eps=eps,
            compute_dtype=polar_compute_dtype,
            output_dtype=polar_output_dtype,
            precision=precision,
        )
        return U.astype(jnp.float32) * _aspect_scale_for_matrix(G)

    transposed = rows < cols
    Gt = jnp.swapaxes(G, -1, -2) if transposed else G
    G32 = Gt.astype(jnp.float32)
    m, n = G32.shape
    r = jnp.asarray(n / m, dtype=jnp.float32)
    target_row_norm = jnp.sqrt(r)

    U0 = _simple_quintic_polar(
        G32,
        iters=polar_ns_iters,
        eps=eps,
        compute_dtype=polar_compute_dtype,
        output_dtype=polar_output_dtype,
        precision=precision,
    ).astype(jnp.float32)

    def retract(Y):
        for _ in range(retraction_steps):
            row_norm = jnp.maximum(jnp.linalg.norm(Y, axis=-1, keepdims=True), eps)
            Y = Y * (target_row_norm / row_norm)
            Y = _simple_quintic_polar(
                Y,
                iters=polar_ns_iters,
                eps=eps,
                compute_dtype=polar_compute_dtype,
                output_dtype=polar_output_dtype,
                precision=precision,
            ).astype(jnp.float32)
        return Y

    def outer_body(_, U):
        UtG = jnp.matmul(jnp.swapaxes(U, -1, -2), G32)
        B = 0.5 * (UtG + jnp.swapaxes(UtG, -1, -2))
        q = jnp.sum(G32 * U, axis=-1) - jnp.sum(jnp.matmul(U, B) * U, axis=-1)
        q = q - jnp.mean(q)

        lam = _solve_row_norm_multipliers(U, r, q, max_iter=cg_steps, eps=eps)
        lam = lam - jnp.mean(lam)

        S = B - jnp.matmul(jnp.swapaxes(U, -1, -2), lam[:, None] * U)
        Z = G32 - jnp.matmul(U, S) - lam[:, None] * U
        finite = jnp.all(jnp.isfinite(Z))
        Y = retract(U + riemannian_eta * Z)
        return jnp.where(finite, Y, U)

    U = jax.lax.fori_loop(0, outer_steps, outer_body, U0, unroll=True)
    if transposed:
        U = jnp.swapaxes(U, -1, -2)
    return U.astype(jnp.float32) * _aspect_scale_for_matrix(G)


def _apply_matrix_rule(
    updates: jax.Array,
    mu_raw: jax.Array,
    *,
    mu_nest: Optional[jax.Array],
    dim_nums: Optional[MatrixDimensionNumbers],
    riemannian: bool,
    pp_iterations: int,
    pp_beta: float,
    outer_steps: int,
    cg_steps: int,
    riemannian_eta: float,
    retraction_steps: int,
    eps: float,
    polar_ns_iters: int,
    polar_compute_dtype: jax.typing.DTypeLike,
    polar_output_dtype: jax.typing.DTypeLike,
    precision: jax.lax.PrecisionLike,
) -> jax.Array:
    """Apply Aurora/Riemannian-Aurora to a leaf with optional reshape spec."""
    if dim_nums is None or isinstance(dim_nums, _masking.MaskedNode):
        return mu_nest if mu_nest is not None else mu_raw

    target = mu_nest if mu_nest is not None else mu_raw

    def matrix_fn(x):
        if riemannian:
            return _riemannian_balanced_polar_matrix(
                x,
                outer_steps=outer_steps,
                cg_steps=cg_steps,
                riemannian_eta=riemannian_eta,
                retraction_steps=retraction_steps,
                eps=eps,
                polar_ns_iters=polar_ns_iters,
                polar_compute_dtype=polar_compute_dtype,
                polar_output_dtype=polar_output_dtype,
                precision=precision,
            )
        return _aurora_balanced_polar_matrix(
            x,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
            eps=eps,
            polar_ns_iters=polar_ns_iters,
            polar_compute_dtype=polar_compute_dtype,
            polar_output_dtype=polar_output_dtype,
            precision=precision,
        )

    if updates.ndim == 2 and _is_standard_2d_spec(dim_nums):
        return matrix_fn(target)

    reshape_fn, inverse_fn = _compute_matrix_reshape(updates, dim_nums)
    target_flat = reshape_fn(target)
    out_flat = jax.vmap(matrix_fn)(target_flat)
    return inverse_fn(out_flat)


def _scale_by_aurora_impl(
    *,
    riemannian: bool,
    b1: float = 0.95,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    outer_steps: int = 3,
    cg_steps: int = 20,
    riemannian_eta: float = 0.1,
    retraction_steps: int = 2,
    polar_ns_iters: int = 12,
    polar_compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    polar_output_dtype: jax.typing.DTypeLike = jnp.float32,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    eps: float = 1e-7,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    weight_dimension_numbers: AuroraWeightDimNumOrFn | None = None,
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable]] = None,
    axis_name: Optional[str] = None,
    guard_nonfinite: bool = True,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    if not (0.0 <= b1 < 1.0):
        raise ValueError(f"b1 must be in [0, 1), got {b1}")
    if pp_iterations < 1:
        raise ValueError(f"pp_iterations must be >= 1, got {pp_iterations}")
    if pp_beta <= 0.0:
        raise ValueError(f"pp_beta must be positive, got {pp_beta}")
    if outer_steps < 1:
        raise ValueError(f"outer_steps must be >= 1, got {outer_steps}")
    if cg_steps < 1:
        raise ValueError(f"cg_steps must be >= 1, got {cg_steps}")
    if retraction_steps < 1:
        raise ValueError(f"retraction_steps must be >= 1, got {retraction_steps}")
    if polar_ns_iters < 1:
        raise ValueError(f"polar_ns_iters must be >= 1, got {polar_ns_iters}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    if mu_dtype is None:
        mu_dtype = jnp.float32
    else:
        mu_dtype = utils.canonicalize_dtype(mu_dtype)
    polar_compute_dtype = cast(
        jax.typing.DTypeLike, utils.canonicalize_dtype(polar_compute_dtype)
    )
    polar_output_dtype = cast(
        jax.typing.DTypeLike, utils.canonicalize_dtype(polar_output_dtype)
    )

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)

        if use_magma:

            def _init_s(x):
                if x is None:
                    return None
                if isinstance(x, _masking.MaskedNode):
                    return _masking.MaskedNode()
                return jnp.array(0.5, dtype=jnp.float32)

            magma_s = jax.tree.map(_init_s, params, is_leaf=_is_aux_leaf)
        else:
            magma_s = ()

        return ScaleByAuroraState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            magma_s=magma_s,
            key=key,
        )

    def update_fn(updates, state, params=None):
        raw_gradients = updates

        if use_magma:
            next_state_key, sr_key, magma_key = jax.random.split(state.key, 3)
        else:
            next_state_key, sr_key = jax.random.split(state.key, 2)
            magma_key = None

        if params is None:
            if weight_dimension_numbers is not None:
                raise ValueError(
                    "`params` must be provided to Aurora when `weight_dimension_numbers` is set."
                )
            resolved_dim_nums = _get_dimension_numbers(None, updates)
        else:
            resolved_dim_nums = _get_dimension_numbers(weight_dimension_numbers, params)

        count_inc = cast(jax.Array, numerics.safe_increment(state.count))

        if raw_global_grad_clip is not None:
            g_norm = _tree_global_norm(updates, axis_name=axis_name)
            is_spike = g_norm > raw_global_grad_clip
            clip_scale = jnp.where(
                is_spike, raw_global_grad_clip / add_tiny(g_norm), 1.0
            )
        else:
            is_spike = jnp.array(False, dtype=jnp.bool_)
            clip_scale = jnp.array(1.0, dtype=jnp.float32)

        should_skip = jnp.logical_and(
            is_spike, jnp.logical_not(permissive_spike_protection)
        )

        effective_updates = jax.tree.map(
            lambda g: (
                g
                if _is_aux_leaf(g)
                else jnp.where(should_skip, jnp.zeros_like(g), g * clip_scale)
            ),
            updates,
            is_leaf=_is_aux_leaf,
        )

        mu_candidate = _tree_update_moment_f32(effective_updates, state.mu, b1)
        mu_old_f32 = _tree_cast_f32(state.mu)
        mu_f32 = _tree_where_scalar(should_skip, mu_old_f32, mu_candidate)

        mu_nest_f32 = mu_f32
        if nesterov:
            if bias_correction:
                mu_bc_factor = 1.0 - b1**count_inc
                mu_bc_factor_next = 1.0 - b1 ** numerics.safe_increment(count_inc)
                mu_bc_f32 = _safe_bias_correction(mu_f32, mu_bc_factor_next)
                updates_f32 = _tree_cast_f32(effective_updates)
                g_bc_f32 = _safe_bias_correction(updates_f32, mu_bc_factor)
                mu_nest_f32 = jax.tree.map(
                    lambda m, g: m if _is_aux_leaf(m) else b1 * m + (1.0 - b1) * g,
                    mu_bc_f32,
                    g_bc_f32,
                    is_leaf=_is_aux_leaf,
                )
            else:
                mu_nest_f32 = jax.tree.map(
                    lambda m, g: (
                        m
                        if _is_aux_leaf(m)
                        else b1 * m + (1.0 - b1) * g.astype(jnp.float32)
                    ),
                    mu_f32,
                    effective_updates,
                    is_leaf=_is_aux_leaf,
                )
        elif bias_correction:
            mu_bc_factor = 1.0 - b1**count_inc
            mu_nest_f32 = _safe_bias_correction(mu_f32, mu_bc_factor)

        if mu_dtype == jnp.bfloat16:
            mu_cast = _tree_stochastic_cast(mu_f32, mu_dtype, sr_key)
        else:
            mu_cast = optax.tree.cast(mu_f32, mu_dtype)

        target_for_shape = (
            mu_nest_f32
            if shape_nesterov
            else jax.tree.map(lambda _: None, effective_updates, is_leaf=_is_aux_leaf)
        )

        aurora_out = jax.tree.map(
            lambda g, m, target, dims: _apply_matrix_rule(
                g,
                m,
                mu_nest=target,
                dim_nums=dims,
                riemannian=riemannian,
                pp_iterations=pp_iterations,
                pp_beta=pp_beta,
                outer_steps=outer_steps,
                cg_steps=cg_steps,
                riemannian_eta=riemannian_eta,
                retraction_steps=retraction_steps,
                eps=eps,
                polar_ns_iters=polar_ns_iters,
                polar_compute_dtype=polar_compute_dtype,
                polar_output_dtype=polar_output_dtype,
                precision=precision,
            ),
            effective_updates,
            mu_f32,
            target_for_shape,
            resolved_dim_nums,
            is_leaf=_is_dim_leaf,
        )

        if nesterov and not shape_nesterov:
            new_updates = jax.tree.map(
                lambda shaped, g: (
                    shaped
                    if _is_aux_leaf(shaped)
                    else b1 * shaped + (1.0 - b1) * g.astype(jnp.float32)
                ),
                aurora_out,
                effective_updates,
                is_leaf=_is_aux_leaf,
            )
        else:
            new_updates = aurora_out

        if guard_nonfinite:
            new_updates = jax.tree.map(
                lambda u: u if _is_aux_leaf(u) else _guard_nonfinite_leaf(u),
                new_updates,
                is_leaf=_is_aux_leaf,
            )

        if grad_clip_max_amps is not None:
            max_rms, max_val = (
                grad_clip_max_amps
                if isinstance(grad_clip_max_amps, tuple)
                else (grad_clip_max_amps, 10.0)
            )
            new_updates = jax.tree.map(
                lambda u: (
                    u if _is_aux_leaf(u) else _clip_per_tensor_rms(u, max_rms, max_val)
                ),
                new_updates,
                is_leaf=_is_aux_leaf,
            )

        _may_have_wd = not isinstance(weight_decay, (int, float)) or weight_decay > 0.0
        if _may_have_wd and params is not None:
            wd_step = (
                cast(
                    Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike],
                    weight_decay,
                )(state.count)
                if callable(weight_decay)
                else weight_decay
            )
            _wd_mask = None
            if weight_decay_mask is not None:
                _wd_mask = (
                    weight_decay_mask(params)
                    if callable(weight_decay_mask)
                    else weight_decay_mask
                )

            def _add_wd(u, p, m=True):
                if _is_aux_leaf(u) or _is_aux_leaf(p):
                    return u
                if isinstance(m, _masking.MaskedNode) or m is None or not m:
                    return u
                return u + wd_step * p.astype(jnp.float32)

            if _wd_mask is not None:
                new_updates = jax.tree.map(
                    _add_wd, new_updates, params, _wd_mask, is_leaf=_is_aux_leaf
                )
            else:
                new_updates = jax.tree.map(
                    lambda u, p: _add_wd(u, p),
                    new_updates,
                    params,
                    is_leaf=_is_aux_leaf,
                )

        new_updates = jax.tree.map(
            lambda u: (
                u if _is_aux_leaf(u) else jnp.where(should_skip, jnp.zeros_like(u), u)
            ),
            new_updates,
            is_leaf=_is_aux_leaf,
        )

        if use_magma:
            final_updates, new_magma_s = apply_magma_internal(
                raw_gradients=raw_gradients,
                first_moments=mu_f32,
                base_updates=new_updates,
                magma_s_prev=state.magma_s,
                key=magma_key,
                p=magma_p,
                tau=magma_tau,
                axis_name=axis_name,
            )
            new_magma_s = jax.tree.map(
                lambda new_s, old_s: (
                    new_s
                    if _is_aux_leaf(new_s)
                    else jnp.where(should_skip, old_s, new_s)
                ),
                new_magma_s,
                state.magma_s,
                is_leaf=_is_aux_leaf,
            )
        else:
            final_updates = new_updates
            new_magma_s = state.magma_s

        return final_updates, ScaleByAuroraState(
            count=count_inc,
            mu=mu_cast,
            magma_s=new_magma_s,
            key=next_state_key,
        )

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_aurora(
    b1: float = 0.95,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    polar_ns_iters: int = 12,
    polar_compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    polar_output_dtype: jax.typing.DTypeLike = jnp.float32,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    eps: float = 1e-7,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    weight_dimension_numbers: AuroraWeightDimNumOrFn | None = None,
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable]] = None,
    axis_name: Optional[str] = None,
    guard_nonfinite: bool = True,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """Core practical Aurora transform.

    Returns unscaled, positive updates. Chain with
    `transform.scale_by_learning_rate` or use `aurora` below.
    """
    return _scale_by_aurora_impl(
        riemannian=False,
        b1=b1,
        pp_iterations=pp_iterations,
        pp_beta=pp_beta,
        polar_ns_iters=polar_ns_iters,
        polar_compute_dtype=polar_compute_dtype,
        polar_output_dtype=polar_output_dtype,
        precision=precision,
        eps=eps,
        nesterov=nesterov,
        shape_nesterov=shape_nesterov,
        bias_correction=bias_correction,
        mu_dtype=mu_dtype,
        raw_global_grad_clip=raw_global_grad_clip,
        permissive_spike_protection=permissive_spike_protection,
        grad_clip_max_amps=grad_clip_max_amps,
        weight_dimension_numbers=weight_dimension_numbers,
        use_magma=use_magma,
        magma_p=magma_p,
        magma_tau=magma_tau,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        axis_name=axis_name,
        guard_nonfinite=guard_nonfinite,
        key=key,
    )


def scale_by_riemannian_aurora(
    b1: float = 0.95,
    outer_steps: int = 3,
    cg_steps: int = 20,
    riemannian_eta: float = 0.1,
    retraction_steps: int = 2,
    polar_ns_iters: int = 12,
    polar_compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    polar_output_dtype: jax.typing.DTypeLike = jnp.float32,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    eps: float = 1e-7,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    weight_dimension_numbers: AuroraWeightDimNumOrFn | None = None,
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable]] = None,
    axis_name: Optional[str] = None,
    guard_nonfinite: bool = True,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """Core Riemannian-Aurora transform.

    This is much more expensive than practical Aurora and is mostly useful as a
    reference-quality balanced-Stiefel solver on moderate matrices.
    """
    return _scale_by_aurora_impl(
        riemannian=True,
        b1=b1,
        outer_steps=outer_steps,
        cg_steps=cg_steps,
        riemannian_eta=riemannian_eta,
        retraction_steps=retraction_steps,
        polar_ns_iters=polar_ns_iters,
        polar_compute_dtype=polar_compute_dtype,
        polar_output_dtype=polar_output_dtype,
        precision=precision,
        eps=eps,
        nesterov=nesterov,
        shape_nesterov=shape_nesterov,
        bias_correction=bias_correction,
        mu_dtype=mu_dtype,
        raw_global_grad_clip=raw_global_grad_clip,
        permissive_spike_protection=permissive_spike_protection,
        grad_clip_max_amps=grad_clip_max_amps,
        weight_dimension_numbers=weight_dimension_numbers,
        use_magma=use_magma,
        magma_p=magma_p,
        magma_tau=magma_tau,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        axis_name=axis_name,
        guard_nonfinite=guard_nonfinite,
        key=key,
    )


def _partitioned_aurora(
    *,
    riemannian: bool,
    learning_rate: base.ScalarOrSchedule,
    b1: float,
    weight_decay: base.ScalarOrSchedule,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]],
    pp_iterations: int,
    pp_beta: float,
    outer_steps: int,
    cg_steps: int,
    riemannian_eta: float,
    retraction_steps: int,
    polar_ns_iters: int,
    polar_compute_dtype: jax.typing.DTypeLike,
    polar_output_dtype: jax.typing.DTypeLike,
    precision: jax.lax.PrecisionLike,
    eps: float,
    nesterov: bool,
    shape_nesterov: bool,
    bias_correction: bool,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]],
    raw_global_grad_clip: Optional[float],
    permissive_spike_protection: bool,
    mu_dtype: Optional[jax.typing.DTypeLike],
    axis_name: Optional[str],
    use_magma: bool,
    magma_p: float,
    magma_tau: float,
    guard_nonfinite: bool,
    key: jax.Array,
    adam_learning_rate: Optional[base.ScalarOrSchedule],
    adam_b1: float,
    adam_b2: float,
    adam_eps: float,
    aurora_weight_dimension_numbers: AuroraWeightDimNumOrFn | None,
) -> base.GradientTransformation:
    key_aurora, key_adam = jax.random.split(key, 2)

    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    def get_resolved_dim_nums(params):
        return _get_dimension_numbers(aurora_weight_dimension_numbers, params)

    def param_labels(params):
        dim_nums = get_resolved_dim_nums(params)
        return jax.tree.map(
            lambda d, p: None if p is None else ("aurora" if d is not None else "adam"),
            dim_nums,
            params,
            is_leaf=_is_dim_leaf,
        )

    def aurora_weight_dim_nums_fn(params):
        dim_nums = get_resolved_dim_nums(params)
        return _mask_dimension_numbers(dim_nums)

    if riemannian:
        aurora_scale = scale_by_riemannian_aurora(
            b1=b1,
            outer_steps=outer_steps,
            cg_steps=cg_steps,
            riemannian_eta=riemannian_eta,
            retraction_steps=retraction_steps,
            polar_ns_iters=polar_ns_iters,
            polar_compute_dtype=polar_compute_dtype,
            polar_output_dtype=polar_output_dtype,
            precision=precision,
            eps=eps,
            nesterov=nesterov,
            shape_nesterov=shape_nesterov,
            bias_correction=bias_correction,
            mu_dtype=mu_dtype,
            raw_global_grad_clip=raw_global_grad_clip,
            permissive_spike_protection=permissive_spike_protection,
            grad_clip_max_amps=grad_clip_max_amps,
            weight_dimension_numbers=aurora_weight_dim_nums_fn,
            use_magma=use_magma,
            magma_p=magma_p,
            magma_tau=magma_tau,
            weight_decay=weight_decay if use_magma else 0.0,
            weight_decay_mask=weight_decay_mask if use_magma else None,
            axis_name=axis_name,
            guard_nonfinite=guard_nonfinite,
            key=key_aurora,
        )
    else:
        aurora_scale = scale_by_aurora(
            b1=b1,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
            polar_ns_iters=polar_ns_iters,
            polar_compute_dtype=polar_compute_dtype,
            polar_output_dtype=polar_output_dtype,
            precision=precision,
            eps=eps,
            nesterov=nesterov,
            shape_nesterov=shape_nesterov,
            bias_correction=bias_correction,
            mu_dtype=mu_dtype,
            raw_global_grad_clip=raw_global_grad_clip,
            permissive_spike_protection=permissive_spike_protection,
            grad_clip_max_amps=grad_clip_max_amps,
            weight_dimension_numbers=aurora_weight_dim_nums_fn,
            use_magma=use_magma,
            magma_p=magma_p,
            magma_tau=magma_tau,
            weight_decay=weight_decay if use_magma else 0.0,
            weight_decay_mask=weight_decay_mask if use_magma else None,
            axis_name=axis_name,
            guard_nonfinite=guard_nonfinite,
            key=key_aurora,
        )

    aurora_components = [aurora_scale]
    _wd_is_nonzero = (
        weight_decay > 0.0 if isinstance(weight_decay, (int, float)) else True
    )
    if _wd_is_nonzero and not use_magma:
        aurora_components.append(
            transform.add_decayed_weights(weight_decay, weight_decay_mask)
        )
    aurora_components.append(transform.scale_by_learning_rate(learning_rate))

    return combine.partition(
        transforms={
            "aurora": combine.chain(*aurora_components),
            "adam": adamw(
                learning_rate=adam_learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
                weight_decay=weight_decay,
                weight_decay_mask=weight_decay_mask,
                mu_dtype=mu_dtype,
                use_magma=use_magma,
                magma_p=magma_p,
                magma_tau=magma_tau,
                axis_name=axis_name,
                key=key_adam,
            ),
        },
        param_labels=param_labels,
    )


def aurora(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.95,
    weight_decay: base.ScalarOrSchedule = 0.025,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    polar_ns_iters: int = 12,
    polar_compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    polar_output_dtype: jax.typing.DTypeLike = jnp.float32,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    eps: float = 1e-7,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    axis_name: Optional[str] = None,
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    guard_nonfinite: bool = True,
    key: jax.Array = jax.random.PRNGKey(42),
    adam_learning_rate: Optional[base.ScalarOrSchedule] = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    aurora_weight_dimension_numbers: AuroraWeightDimNumOrFn | None = None,
) -> base.GradientTransformation:
    """Aurora optimizer with automatic matrix/Adam partitioning.

    Matrix leaves are optimized by Aurora. Non-matrix leaves are optimized by
    AdamW. By default, all 2D leaves are Aurora leaves; pass
    `aurora_weight_dimension_numbers` to opt in convolution kernels or to change
    which axes form the Aurora matrix.
    """
    return _partitioned_aurora(
        riemannian=False,
        learning_rate=learning_rate,
        b1=b1,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        pp_iterations=pp_iterations,
        pp_beta=pp_beta,
        outer_steps=3,
        cg_steps=20,
        riemannian_eta=0.1,
        retraction_steps=2,
        polar_ns_iters=polar_ns_iters,
        polar_compute_dtype=polar_compute_dtype,
        polar_output_dtype=polar_output_dtype,
        precision=precision,
        eps=eps,
        nesterov=nesterov,
        shape_nesterov=shape_nesterov,
        bias_correction=bias_correction,
        grad_clip_max_amps=grad_clip_max_amps,
        raw_global_grad_clip=raw_global_grad_clip,
        permissive_spike_protection=permissive_spike_protection,
        mu_dtype=mu_dtype,
        axis_name=axis_name,
        use_magma=use_magma,
        magma_p=magma_p,
        magma_tau=magma_tau,
        guard_nonfinite=guard_nonfinite,
        key=key,
        adam_learning_rate=adam_learning_rate,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps=adam_eps,
        aurora_weight_dimension_numbers=aurora_weight_dimension_numbers,
    )


def riemannian_aurora(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.95,
    weight_decay: base.ScalarOrSchedule = 0.025,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    outer_steps: int = 3,
    cg_steps: int = 20,
    riemannian_eta: float = 0.1,
    retraction_steps: int = 2,
    polar_ns_iters: int = 12,
    polar_compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    polar_output_dtype: jax.typing.DTypeLike = jnp.float32,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    eps: float = 1e-7,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    axis_name: Optional[str] = None,
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    guard_nonfinite: bool = True,
    key: jax.Array = jax.random.PRNGKey(42),
    adam_learning_rate: Optional[base.ScalarOrSchedule] = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    aurora_weight_dimension_numbers: AuroraWeightDimNumOrFn | None = None,
) -> base.GradientTransformation:
    """Riemannian-Aurora optimizer with automatic matrix/Adam partitioning."""
    return _partitioned_aurora(
        riemannian=True,
        learning_rate=learning_rate,
        b1=b1,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        pp_iterations=2,
        pp_beta=0.5,
        outer_steps=outer_steps,
        cg_steps=cg_steps,
        riemannian_eta=riemannian_eta,
        retraction_steps=retraction_steps,
        polar_ns_iters=polar_ns_iters,
        polar_compute_dtype=polar_compute_dtype,
        polar_output_dtype=polar_output_dtype,
        precision=precision,
        eps=eps,
        nesterov=nesterov,
        shape_nesterov=shape_nesterov,
        bias_correction=bias_correction,
        grad_clip_max_amps=grad_clip_max_amps,
        raw_global_grad_clip=raw_global_grad_clip,
        permissive_spike_protection=permissive_spike_protection,
        mu_dtype=mu_dtype,
        axis_name=axis_name,
        use_magma=use_magma,
        magma_p=magma_p,
        magma_tau=magma_tau,
        guard_nonfinite=guard_nonfinite,
        key=key,
        adam_learning_rate=adam_learning_rate,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps=adam_eps,
        aurora_weight_dimension_numbers=aurora_weight_dimension_numbers,
    )
