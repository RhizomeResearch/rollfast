import string
from enum import Enum
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from optax import tree_utils as otu
from optax._src import base, numerics, transform
from optax._src.combine import chain
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype

from rollfast.utils import add_tiny, dist_reduce


class PreconditionerMode(str, Enum):
    """Supported preconditioner update geometries.

    These modes determine how the Preconditioner Matrix Q is updated based on the
    computed covariance/Gram matrix G.

    Attributes:
        EQ: dQ = E * Q. The original PSGD update. Maintains triangular Q.
            Requires triangular solves.
        Q0P5EQ1P5: dQ = Q^0.5 * E * Q^1.5. Uses Procrustes rotation to
            update Q. Often more stable than EQ as it keeps Q closer to orthogonal.
        QUAD: Quadratic form update. Ensures Q stays SPD.
        NS: Newton-Schulz SPD projection. Iteratively projects Q onto the
            SPD manifold. Exact but more expensive.
        EXP: Matrix Exponential (Geodesic on SPD manifold).
        TAYLOR2: 2nd-order Taylor approximation of the Matrix Exponential.
        HYPER: Multiplicative Hyperbolic update.
    """

    EQ = "EQ"
    Q0P5EQ1P5 = "Q0.5EQ1.5"
    QUAD = "QUAD"
    NS = "NS"
    EXP = "EXP"
    TAYLOR2 = "TAYLOR2"
    HYPER = "HYPER"


class GradClipMode(str, Enum):
    """Supported gradient clipping modes."""

    PER_TENSOR_RMS = "per_tensor_rms"  # Current: clip each tensor by its RMS
    GLOBAL_RMS = "global_rms"  # Clip by global RMS across all tensors


class KronState(NamedTuple):
    """State for the Kronecker-factored preconditioner optimizer.

    Attributes:
        count: Step counter (int32).
        mu: Momentum accumulator (first moment). None if b1=0.
        Qs_preconditioners: PyTree of lists of preconditioner factors (Q matrices).
        Ls_lipschitz: PyTree of lists of Lipschitz constants (scalars) for Q factors.
        needs_scale_init: Boolean flag, True if Q needs to be scaled by first grad.
        key: PRNGKey for stochastic elements (Procrustes, norm estimation).
    """

    count: jax.Array
    mu: Optional[Any]
    Qs_preconditioners: Any
    Ls_lipschitz: Optional[Any]
    needs_scale_init: jax.Array
    magma_s: Optional[Any]
    key: jax.Array


def _compute_global_norm(
    grads: Any,
    axis_name: Optional[str] = None,
) -> jax.Array:
    """Computes the global L2 norm of gradients, handling distributed reduction.

    Used for detecting spikes in raw gradients before preconditioning.
    """
    leaves = jax.tree.leaves(grads)
    if not leaves:
        return jnp.array(0.0)

    # Calculate sum of squares locally
    # Cast to float32 to ensure stability if input is float16/bfloat16
    local_sq = sum(jnp.sum(numerics.abs_sq(x.astype(jnp.float32))) for x in leaves)

    # Sum across devices
    total_sq = dist_reduce(local_sq, axis_name, "sum")

    return jnp.sqrt(total_sq)


def _compute_global_rms_scale(
    precond_grads: List[jax.Array],
    max_rms: float,
    axis_name: Optional[str] = None,
) -> jax.Array:
    """Computes scaling factor for global RMS constraint.

    Calculates sqrt(sum(g^2) / total_numel) globally and determines scaling
    to keep it under max_rms. This is invariant to model size.

    Returns:
        Scalar scaling factor in (0, 1].
    """
    local_sq = sum(jnp.sum(numerics.abs_sq(pg)) for pg in precond_grads)
    local_numel = sum(pg.size for pg in precond_grads)

    # Sum squares and numel across all devices
    total_sq = dist_reduce(local_sq, axis_name, "sum")
    total_numel = dist_reduce(
        jnp.array(local_numel, dtype=jnp.float32), axis_name, "sum"
    )

    global_rms = jnp.sqrt(total_sq / total_numel)

    return jnp.minimum(
        1.0, max_rms / jnp.maximum(global_rms, jnp.finfo(global_rms.dtype).tiny)
    )


def _compute_init_scale_from_grads(
    grads: List[jax.Array],
    damping: float,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Computes whitening init scale from first gradients.

    Heuristic: scale = (max_leaf(mean(|g|^4)) + damping^4)^(-1/8).
    This roughly aligns the preconditioner magnitude with the gradient curvature,
    assuming a hyper-gaussian distribution of gradients.
    """

    def leaf_mean_fourth(g: jax.Array) -> jax.Array:
        return jnp.mean(jnp.power(jnp.abs(g), 4))

    mean_fourths = jnp.stack([leaf_mean_fourth(g) for g in grads])
    max_mean_fourth = jnp.max(mean_fourths)

    return jnp.power(max_mean_fourth + jnp.power(damping, 4), -1.0 / 8.0).astype(dtype)


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500
):
    """Anneals preconditioner update probability during beginning of training.

    Updates are expensive. It is often useful to update Q frequently early on
    to capture rapid curvature changes, then reduce frequency to save compute.
    """

    def _schedule(n):
        return jnp.minimum(
            jnp.maximum(max_prob * jnp.exp(-decay * (n - flat_start)), min_prob),
            max_prob,
        )

    return _schedule


def _norm_lower_bound(A: jax.Array) -> jax.Array:
    """Returns a cheap, deterministic lower bound for the spectral norm of A.

    Uses 1.5-step Power Iteration initialized with the heaviest row/column.
    Optimized to O(N) memory usage (avoids O(N^2) intermediate matrices).
    """
    max_abs = jnp.max(jnp.abs(A)) + jnp.finfo(A.dtype).tiny

    def calc(A_in):
        A_normed = A_in / max_abs
        A_conj = A_normed.conj()

        # Compute squared norms without full matrix multiply
        aa_sum0 = jnp.sum(numerics.abs_sq(A_normed), axis=0)  # Col norms
        aa_sum1 = jnp.sum(numerics.abs_sq(A_normed), axis=1)  # Row norms

        i = jnp.argmax(aa_sum0)
        j = jnp.argmax(aa_sum1)

        value0 = aa_sum0[i]
        value1 = aa_sum1[j]

        # Branch based on whether heaviest row or column dominates
        def gt_branch(_):
            x = jax.lax.dynamic_index_in_dim(A_normed, i, 1, keepdims=False)
            x = x.conj() @ A_normed
            norm_x = jnp.linalg.norm(x) + jnp.finfo(x.dtype).tiny
            x_normalized = x / norm_x
            final_vec = x_normalized @ A_conj.T
            return max_abs * jnp.linalg.norm(final_vec)

        def le_branch(_):
            x = jax.lax.dynamic_index_in_dim(A_normed, j, 0, keepdims=False)
            x = A_normed @ x.conj()
            norm_x = jnp.linalg.norm(x) + jnp.finfo(x.dtype).tiny
            x_normalized = x / norm_x
            final_vec = A_conj.T @ x_normalized
            return max_abs * jnp.linalg.norm(final_vec)

        return jax.lax.cond(value0 > value1, gt_branch, le_branch, operand=None)

    return jax.lax.cond(max_abs > jnp.finfo(A.dtype).tiny, calc, lambda _: max_abs, A)


def _norm_lower_bound_spd(
    A: jax.Array, key: jax.Array, k: int = 32, half_iters: int = 2
) -> jax.Array:
    """Returns a cheap lower bound for spectral norm of SPD matrix A.

    Uses subspace iteration (Power Method on multiple vectors) starting
    from random Gaussian noise + the heaviest row.
    """
    smallest_normal = jnp.finfo(A.dtype).smallest_normal
    normalizing_factor = jnp.max(jnp.real(jnp.diag(A))) + smallest_normal
    A = A / normalizing_factor

    # Heuristic initialization: pick row with max norm
    row_norms = jnp.linalg.norm(A, axis=1)
    j = jnp.argmax(row_norms)
    Aj = A[j]

    # Combine heuristic with random noise
    V = jax.random.normal(key, shape=(k, A.shape[1]), dtype=A.dtype)
    dots = jnp.sum(Aj * jnp.conj(V), axis=1, keepdims=True)
    V = Aj + jnp.sign(jnp.real(dots)) * V

    def iteration_body(V, _):
        V = V @ A
        V = V / (jnp.linalg.norm(V, axis=1, keepdims=True) + smallest_normal)
        V = V @ A
        return V, None

    V, _ = jax.lax.scan(iteration_body, V, None, length=half_iters)

    return normalizing_factor * jnp.max(jnp.linalg.norm(V, axis=1))


def _norm_lower_bound_skh(
    A: jax.Array, key: jax.Array, k: int = 32, half_iters: int = 2
) -> jax.Array:
    """Returns a cheap lower bound for spectral norm of Skew-Hermitian matrix A."""
    smallest_normal = jnp.finfo(A.dtype).smallest_normal
    normalizing_factor = jnp.max(jnp.abs(A)) + smallest_normal
    A = A / normalizing_factor

    row_norms = jnp.linalg.norm(A, axis=1)
    j = jnp.argmax(row_norms)
    Aj = A[j]

    V = jax.random.normal(key, shape=(k, A.shape[1]), dtype=A.dtype)

    dots = jnp.sum(Aj * jnp.conj(V), axis=1, keepdims=True)
    V = Aj + jnp.sign(jnp.real(dots)) * V

    def iteration_body(V, _):
        V = V @ A
        V = V / (jnp.linalg.norm(V, axis=1, keepdims=True) + smallest_normal)
        V = V @ A
        return V, None

    V, _ = jax.lax.scan(iteration_body, V, None, length=half_iters)

    return normalizing_factor * jnp.max(jnp.linalg.norm(V, axis=1))


def _procrustes_step2(
    Q: jax.Array, key: jax.Array, max_step_size: float = 1 / 8
) -> jax.Array:
    """Online solver for the orthogonal Procrustes problem via rotation.

    Used in Q0.5EQ1.5 mode to update Q while maintaining stability without
    expensive inversions.
    """
    R = jnp.conj(Q.T) - Q
    R_norm = _norm_lower_bound_skh(R, key) + jnp.finfo(R.dtype).smallest_normal
    R = R / R_norm

    RQ = R @ Q
    RRQ = R @ RQ

    tr_RQ = jnp.real(jnp.sum(jnp.diag(RQ)))
    tr_RRQ = jnp.real(jnp.sum(jnp.diag(RRQ)))

    a = jnp.where(
        tr_RRQ < 0, jnp.minimum(-tr_RQ / tr_RRQ, max_step_size), max_step_size
    )

    return Q + a * (RQ + 0.5 * a * RRQ)


def _newton_schulz_spd(
    Q: jax.Array, key: jax.Array, n_iters: int = 5, axis_name: Optional[str] = None
) -> jax.Array:
    """Projects matrix Q onto SPD manifold via Newton-Schultz polar decomposition.

    Target: P = (Q^H Q)^{1/2}.
    Iteration: X_{k+1} = 0.5 * X_k @ (3I - X_k^H X_k).
    Guarantees quadratic convergence if initialized within spectral radius bound.
    """
    dtype_storage = Q.dtype
    Q = Q.astype(jnp.float32)
    d = Q.shape[0]
    dtype = jnp.float32

    # top eigenvalue of Q^H Q (sigma_max(Q)^2)
    norm_sq = _norm_lower_bound_spd(jnp.conj(Q.T) @ Q, key, half_iters=2)
    norm_est = jnp.sqrt(norm_sq)
    norm_est = dist_reduce(norm_est, axis_name, "max")

    # scale to <= 1.0. This is well within the sqrt(3) limit.
    scale = norm_est + jnp.finfo(dtype).eps
    X = Q / scale
    I = jnp.eye(d, dtype=dtype)

    def ns_iteration(X, _):
        """Single Newton-Schulz iteration: X_{k+1} = 0.5 * X_k @ (3I - X_k^H X_k)"""
        XHX = jnp.conj(X.T) @ X
        X_new = 0.5 * X @ (3.0 * I - XHX)
        return X_new, None

    X, _ = jax.lax.scan(ns_iteration, X, None, length=n_iters)

    # X converges to Unitary factor U. P = U^H @ Q.
    P = jnp.conj(X.T) @ Q

    # Symmetrization to eliminate numerical drift
    P = (P + jnp.conj(P.T)) / 2.0

    return P.astype(dtype_storage)


def _init_Q_exprs(
    t,
    scale,
    max_size,
    max_skew,
    min_ndim_triangular,
    memory_save_mode,
    dtype,
    existing_Q=None,
) -> Tuple[List[jax.Array], Tuple[str, Tuple[str, ...], str]]:
    """Initialize preconditioner Q and einsum expressions for a tensor t.

    Splits a tensor shape into factors. Small dimensions become diagonal vectors;
    large dimensions are paired to form square matrices for Kronecker product.
    Strings are used to define Einsum ops dynamically (avoids JIT re-compilation).

    Returns:
        Qs: List of initialized matrices/vectors.
        Exprs: (exprA, exprGs, exprP) for Einstein summation.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    shape = t.shape
    if len(shape) == 0:
        Q = (
            [scale * jnp.ones_like(t, dtype=dtype)]
            if existing_Q is None
            else existing_Q
        )
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )

        scale = scale ** (1 / len(shape))
        total_numel = t.size

        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            raise ValueError(
                f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                "[None, 'one_diag', 'all_diag']"
            )

        Q = [] if existing_Q is None else existing_Q
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")

        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            is_diagonal = (
                size <= 1
                or size > max_size
                or size**2 > max_skew * total_numel
                or len(shape) < min_ndim_triangular
                or dim_d
            )

            if is_diagonal:
                if existing_Q is None:
                    Q.append(scale * jnp.ones(size, dtype=dtype))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                exprGs.append(piece1 + "," + piece1 + "->" + letters[i + 13])

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                if existing_Q is None:
                    Q.append(scale * jnp.eye(size, dtype=dtype))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (letters[i + 26] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                exprGs.append(
                    piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                )

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )

    exprGs = tuple(exprGs)
    if existing_Q is not None:
        return exprA, exprGs, exprP
    return [Q, (exprA, exprGs, exprP)]


def _solve_triangular_right(X: jax.Array, A: jax.Array) -> jax.Array:
    """Computes X @ inv(A) via triangular solve.

    A must be triangular. Equivalent to solving A^T Y = X^T for Y^T.
    """
    X_ndim = X.ndim
    if X_ndim < 2:
        X = X[None, :]

    dtype_in = jnp.promote_types(A.dtype, X.dtype)
    A, X = A.astype(dtype_in), X.astype(dtype_in)

    leading_dims = max(0, X.ndim - 2)
    solve_fn = partial(jax.lax.linalg.triangular_solve, left_side=False, lower=False)
    for _ in range(leading_dims):
        solve_fn = vmap(solve_fn, in_axes=(None, 0))
    solution = solve_fn(A, X)

    if X_ndim < 2:
        return solution[0]
    return solution


def _conjB(Q: List[jax.Array], G: jax.Array, V: jax.Array) -> jax.Array:
    """Computes conjB = V @ inv(Q) for EQ mode.

    Needed for the A^T A - B^T B update rule in EQ mode.
    """
    order = G.ndim
    p = list(range(order))
    conjB = jnp.transpose(V.conj(), p[1:] + p[:1])

    for i, q in enumerate(Q):
        if q.ndim < 2:
            conjB = conjB / q
        else:
            conjB = _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = jnp.swapaxes(conjB, i, order - 1)

    return conjB


def _update_precond_generic(
    Q: List[jax.Array],
    L: Optional[List[jax.Array]],
    X: jax.Array,
    Y: Optional[jax.Array],
    total_numel: int,
    exprs: Tuple[str, Tuple[str, ...], str],
    precond_lr: float,
    key: jax.Array,
    mode: str,
    conjB: Optional[jax.Array] = None,
    beta_l: float = 0.9,
    axis_name: Optional[str] = None,
    damping: float = 1e-9,
    ns_iters: int = 5,
) -> Tuple[List[jax.Array], Optional[List[jax.Array]]]:
    """Unified dense/triangular preconditioner update kernel.

    Args:
        Q: List of current preconditioner factors.
        L: List of current Lipschitz constants (or None).
        X: Input vector for covariance computation.
           - Dense Modes: X = Pg (Preconditioned Gradient, P*G)
           - EQ Mode: X = A (Whitened Gradient, Q*G)
        Y: Whitening source for EQ Mode (conjB), None for Dense.
        total_numel: Number of elements in the parameter tensor.
        exprs: Einsum expressions.
        precond_lr: Learning rate for Q.
        mode: Update mode (EQ, NS, QUAD, etc.).
        beta_l: EMA decay for Lipschitz tracking.
        ns_iters: Iterations for Newton-Schulz.
    """
    _, exprGs, _ = exprs

    X_f32 = X.astype(jnp.float32)
    X_conj = X_f32.conj()

    new_Qs = []
    new_Ls = [] if L is not None else None
    keys = jax.random.split(key, len(Q))

    for i, q in enumerate(Q):
        dtype_in = q.dtype

        l = L[i] if L is not None else None

        # Term 1 is always X @ X^H (Gram matrix of inputs)
        term1_local = jnp.einsum(exprGs[i], X_f32, X_conj)
        term1 = dist_reduce(term1_local, axis_name, "mean")

        # Term 2: Damping or Whitening subtraction term
        if Y is not None:
            # EQ Mode: term2 is derived from Y (conjB) to form (A^T A - B^T B)
            Y_f32 = Y.astype(jnp.float32)
            Y_conj = Y_f32.conj()
            term2_local = jnp.einsum(exprGs[i], Y_conj, Y_f32)
            term2 = dist_reduce(term2_local, axis_name, "mean")
            term1_reg = term1
        else:
            # Dense Modes: term2 is scalar identity (regularization)
            term1_reg = term1 + damping  # Apply LM Damping
            if q.ndim < 2:
                term2 = total_numel / q.size
            else:
                term2 = total_numel / q.shape[0]

        # Diagonal vs Matrix
        if q.ndim < 2:
            if mode == "EQ":
                # EQ Diagonal: dQ = (A^2 - B^2) Q
                diff = term1 - term2
                ell_local = jnp.max(jnp.real(term1 + term2))
            else:
                # Dense Diagonal: dQ = (G^2 - mean(G^2)) Q
                diff = term1_reg - term2
                ell_local = jnp.max(jnp.abs(term1_reg)) + term2

            ell = dist_reduce(ell_local, axis_name, "max")

            if l is not None:
                new_l = jnp.maximum(beta_l * l + (1 - beta_l) * ell, ell)
                step = precond_lr / new_l
            else:
                step = precond_lr / add_tiny(ell)
                new_l = None

            # Multiplicative update
            new_q = q * (1 - step * diff)
            new_q = jnp.abs(new_q)  # Maintain positivity

        else:
            d = q.shape[0]
            dtype = q.dtype

            # Lipschitz Estimation
            if mode == "EQ":
                ell_local = _norm_lower_bound_spd(term1 + term2, keys[i])
            else:
                # Spectral Norm: Precise O(k*N^2)
                norm_key, geom_key = jax.random.split(keys[i])
                ell_local = _norm_lower_bound_spd(term1_reg, norm_key) + term2

            ell = dist_reduce(ell_local, axis_name, "max")

            if l is not None:
                new_l = jnp.maximum(beta_l * l + (1 - beta_l) * ell, ell)
                lr_scaled = precond_lr / new_l
            else:
                lr_scaled = precond_lr / add_tiny(ell)
                new_l = None

            if mode == "EQ":
                # Triangular Update: triu(A A' - B B')
                grad_term = jnp.triu(term1 - term2)
                new_q = q - lr_scaled * (grad_term @ q)

            else:
                # Standard Dense Modes
                step = lr_scaled * (term1_reg @ q - term2 * q)

                if mode == "NS":
                    # SPD Projection
                    new_q = q - step
                    new_q = _newton_schulz_spd(
                        new_q, geom_key, n_iters=ns_iters, axis_name=axis_name
                    )
                elif mode == "QUAD":
                    # Quadratic form update
                    p = q - 0.5 * step
                    p = p - 0.5 * lr_scaled * (p @ term1_reg - p * term2)
                    new_q = (p + jnp.conj(p.T)) / 2.0
                elif mode == "EXP":
                    # Matrix Exponential
                    G_sym = (term1_reg + jnp.conj(term1_reg.T)) / 2.0 - term2 * jnp.eye(
                        d, dtype=jnp.float32
                    )
                    new_q = q @ jax.scipy.linalg.expm(-lr_scaled * G_sym)
                    new_q = (new_q + jnp.conj(new_q.T)) / 2.0
                elif mode == "TAYLOR2":
                    # Second-Order Taylor Expansion of Geodesic Flow
                    G_sym = (term1_reg + jnp.conj(term1_reg.T)) / 2.0 - term2 * jnp.eye(
                        d, dtype=jnp.float32
                    )
                    # Limit step size to trust region of Taylor expansion
                    safe_lr = jnp.minimum(lr_scaled, 0.5 / add_tiny(ell - term2))
                    H = safe_lr * G_sym
                    update_mat = jnp.eye(d, dtype=jnp.float32) - H @ (
                        jnp.eye(d, dtype=jnp.float32) - 0.5 * H
                    )
                    new_q = q @ update_mat
                    new_q = (new_q + jnp.conj(new_q.T)) / 2.0
                elif mode == "HYPER":
                    # Hyperbolic/Multiplicative
                    G_sym = (term1_reg + jnp.conj(term1_reg.T)) / 2.0 - term2 * jnp.eye(
                        d, dtype=jnp.float32
                    )
                    safe_lr = jnp.minimum(lr_scaled, 0.9 / add_tiny(ell - term2))
                    update_mat = jnp.eye(d, dtype=jnp.float32) - safe_lr * G_sym
                    new_q = q @ update_mat
                    new_q = (new_q + jnp.conj(new_q.T)) / 2.0
                else:  # Q0.5EQ1.5
                    new_q = q - step
                    new_q = _procrustes_step2(new_q, geom_key)

        new_Qs.append(new_q.astype(dtype_in))

        if new_Ls is not None:
            new_Ls.append(new_l)

    return new_Qs, new_Ls


def _precond_grad(
    Q: List[jax.Array],
    G: jax.Array,
    exprs: Tuple[str, Tuple[str, ...], str],
) -> jax.Array:
    """Preconditions gradient G with preconditioner Q: P @ G = Q^H @ Q @ G."""
    exprP = exprs[-1]
    return jnp.einsum(exprP, *[q.conj() for q in Q], *Q, G)


def _precond_grad_eq(
    Q: List[jax.Array],
    G: jax.Array,
    exprs: Tuple[str, Tuple[str, ...], str],
) -> jax.Array:
    """Compute whitened gradient for EQ mode: A = Q @ G."""
    exprA = exprs[0]
    return jnp.einsum(exprA, *Q, G)


def _balance_Q(Q: List[jax.Array], axis_name: Optional[str] = None) -> List[jax.Array]:
    """Balances the dynamic ranges of Q factors to avoid overflow/underflow.

    Rescales factors so their norms are approximately equal, without changing
    the product of the factors.
    """
    if len(Q) <= 1:
        return Q

    norms = jnp.array([jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32)
    norms = dist_reduce(norms, axis_name, op="max")

    gmean = jnp.prod(norms) ** (1.0 / len(norms))
    to_mul = gmean / norms

    return [q * x.astype(q.dtype) for q, x in zip(Q, to_mul)]


def scale_by_kron(
    b1: float = 0.9,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    max_skew_triangular: float = 1.0,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    whiten_grad: bool = True,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: Optional[float] = None,
    update_preconditioner_first: bool = True,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    preconditioner_mode: Union[str, PreconditionerMode] = PreconditionerMode.Q0P5EQ1P5,
    beta_lipschitz: float = 0.9,
    track_lipschitz: bool = True,
    damping: float = 1e-9,
    grad_clip_max_amps: float | Tuple[float, float] = (2.0, 10.0),
    grad_clip_mode: Union[str, GradClipMode] = GradClipMode.PER_TENSOR_RMS,
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    newton_schulz_iters: int = 5,
    use_magma: bool = False,
    magma_tau: float = 2.0,
    axis_name: Optional[str] = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """Implements PSGD Kron (Preconditioned SGD with Kronecker factorization).

    Args:
        b1: Momentum parameter.
        preconditioner_update_probability: Probability (or schedule) of updating
            the preconditioner matrix Q at each step.
        max_size_triangular: Max size for a dimension to be considered for
            dense/triangular preconditioning. Larger dims become diagonal.
        max_skew_triangular: Max aspect ratio skew for dense factors.
        min_ndim_triangular: Minimum tensor rank required for dense preconditioning.
        memory_save_mode: Strategy to force diagonal approximations to save RAM.
            Values: [None, 'one_diag', 'all_diag'].
        whiten_grad: If True, preconditioner applies to raw gradient.
            If False, applies to momentum (requires b1 > 0).
        preconditioner_lr: Learning rate for the preconditioner matrix Q.
        preconditioner_init_scale: Initial scale for Q. If None, computed on-the-fly.
        update_preconditioner_first: Update Q before applying it to the gradient.
        mu_dtype: Dtype for momentum accumulator.
        precond_dtype: Dtype for preconditioner storage (e.g. float32, bfloat16).
        precond_update_precision: JAX precision for Q update matmuls.
        precond_grads_precision: JAX precision for gradient application matmuls.
        scanned_layers: PyTree mask indicating layers that are vmapped/scanned.
        lax_map_scanned_layers: Use lax.map for scanning (saves memory vs vmap).
        lax_map_batch_size: Batch size for lax.map.
        preconditioner_mode: Update rule for Q. See PreconditionerMode enum.
        beta_lipschitz: EMA factor for Lipschitz constant estimation.
        track_lipschitz: Enable adaptive step size for Q based on Lipschitz.
        damping: Numerical damping for stability.
        grad_clip_max_amps: (max_rms, max_val) for gradient clipping.
        grad_clip_mode: Strategy for clipping ('per_tensor_rms' or 'global_rms').
        raw_global_grad_clip: Threshold for global gradient norm clipping (spike protection).
        permissive_spike_protection: If True, allows updates during spikes if prob=1.0.
        newton_schulz_iters: Iterations for NS mode (default 5).
        use_magma: If True, applies Momentum-aligned gradient masking (Magma).
            WARNING: Magma introduces intentional update bias (damping). At an
            equilibrium tau=2.0, non-masked steps scale updates by ~0.5, and
            50% of steps are masked. This yields an expected magnitude attenuation
            of ~0.25x. You may need to scale the global learning rate by ~4x to
            maintain the original update volume.
        magma_tau: Temperature parameter for the alignment sigmoid. Default is 2.0.
        axis_name: Axis name for distributed (SPMD) reduction.
        key: PRNG key for stochastic elements.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype)

    if not whiten_grad and b1 <= 0:
        raise ValueError(
            "Cannot whiten momentum (whiten_grad=False) when momentum is disabled (b1 <= 0)."
        )
    if not whiten_grad and jax.process_index() == 0:
        lr_reduction = int(((1 + b1) / (1 - b1)) ** 0.5)
        print(
            f"whiten_grad=False: Recommend reducing learning_rate by ~{lr_reduction}x"
        )

    # Normalize mode
    if isinstance(preconditioner_mode, str):
        mode_map = {
            "EQ": PreconditionerMode.EQ,
            "Q0.5EQ1.5": PreconditionerMode.Q0P5EQ1P5,
            "Q0p5EQ1p5": PreconditionerMode.Q0P5EQ1P5,
            "QUAD": PreconditionerMode.QUAD,
            "NS": PreconditionerMode.NS,
            "EXP": PreconditionerMode.EXP,
            "TAYLOR2": PreconditionerMode.TAYLOR2,
            "HYPER": PreconditionerMode.HYPER,
        }
        preconditioner_mode = mode_map.get(
            preconditioner_mode, PreconditionerMode.Q0P5EQ1P5
        )

    lazy_init = preconditioner_init_scale is None
    _init_scale = 1.0 if lazy_init else preconditioner_init_scale

    def map_fn(do_map, fn, *args):
        """Maybe map a fn along first axis."""
        if do_map:
            if lax_map_scanned_layers:
                return jax.lax.map(
                    lambda xs: fn(*xs),
                    xs=args,
                    batch_size=lax_map_batch_size if lax_map_batch_size > 1 else None,
                )
            else:
                return vmap(fn)(*args)
        else:
            return fn(*args)

    def init_fn(params):
        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, params)

        mu = None
        if b1 > 0:
            mu = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=mu_dtype), params)

        Qs = [
            _init_Q_exprs(
                t[0] if s else t,
                _init_scale,
                max_size_triangular,
                max_skew_triangular,
                min_ndim_triangular,
                memory_save_mode,
                precond_dtype,
            )[0]
            for t, s in zip(jax.tree.leaves(params), jax.tree.leaves(scanned_layers_))
        ]

        Qs = [
            (
                jax.tree.map(
                    lambda d: jnp.repeat(jnp.expand_dims(d, 0), t.shape[0], axis=0), q
                )
                if s
                else q
            )
            for q, t, s in zip(
                Qs, jax.tree.leaves(params), jax.tree.leaves(scanned_layers_)
            )
        ]

        Ls = None
        if track_lipschitz:
            Ls = [[jnp.zeros([], dtype=jnp.float32) for _ in q] for q in Qs]
            Ls = [
                (
                    [jnp.repeat(jnp.expand_dims(l, 0), t.shape[0], axis=0) for l in ls]
                    if s
                    else ls
                )
                for ls, t, s in zip(
                    Ls, jax.tree.leaves(params), jax.tree.leaves(scanned_layers_)
                )
            ]
            Ls = jax.tree.structure(params).unflatten(Ls)

        Qs = jax.tree.structure(params).unflatten(Qs)

        Qs_n_elements = sum([q.size for q in jax.tree.leaves(Qs)])
        Qs_size_MB = sum(
            [q.size * q.dtype.itemsize / (2**20) for q in jax.tree.leaves(Qs)]
        )
        if jax.process_index() == 0:
            init_msg = "on-the-fly" if lazy_init else f"{_init_scale}"
            mode_info = preconditioner_mode.value
            if preconditioner_mode == PreconditionerMode.NS:
                mode_info += f", {newton_schulz_iters} iters"
            print(
                f"PSGD Preconditioners ({mode_info}, init_scale={init_msg}): "
                f"{Qs_n_elements} elements, {Qs_size_MB:.2f} MB"
            )
        if mu is not None:
            mu_n_elements = sum([p.size for p in jax.tree.leaves(mu)])
            mu_size_MB = sum(
                [p.size * p.dtype.itemsize / (2**20) for p in jax.tree.leaves(mu)]
            )
            if jax.process_index() == 0:
                print(
                    f"PSGD Momentum size: {mu_n_elements} elements, {mu_size_MB:.2f} MB"
                )

        magma_s = (
            jax.tree.map(lambda x: jnp.zeros([], jnp.float32), params)
            if use_magma
            else None
        )

        return KronState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            Qs_preconditioners=Qs,
            Ls_lipschitz=Ls,
            needs_scale_init=jnp.array(lazy_init, dtype=jnp.bool_),
            magma_s=magma_s,
            key=key,
        )

    def update_fn(updates: base.Updates, state: KronState, params: base.Params = None):
        del params
        count_inc = safe_int32_increment(state.count)
        key, key_next = jax.random.split(state.key)

        is_spike = jnp.array(False, dtype=jnp.bool_)
        if raw_global_grad_clip is not None:
            g_norm = _compute_global_norm(updates, axis_name)
            is_spike = g_norm > raw_global_grad_clip
            clip_scale = jnp.where(
                is_spike, raw_global_grad_clip / add_tiny(g_norm), 1.0
            )
            updates = jax.tree.map(lambda g: g * clip_scale, updates)

        post_clip_raw_gradients = updates

        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, updates)

        update_prob_in = preconditioner_update_probability
        if isinstance(preconditioner_update_probability, Callable):
            update_prob_in = preconditioner_update_probability(count_inc)

        mu = None
        momentum_updates = updates
        if state.mu is not None:
            beta = jnp.minimum(
                state.count.astype(jnp.float32)
                / (1.0 + state.count.astype(jnp.float32)),
                b1,
            )
            mu = jax.tree.map(
                lambda m, g: beta * m + (1.0 - beta) * g,
                state.mu,
                updates,
            )
            momentum_updates = mu

        updates_flat, grads_structure = jax.tree.flatten(updates)
        momentum_updates_flat = grads_structure.flatten_up_to(momentum_updates)
        Qs_flat = grads_structure.flatten_up_to(state.Qs_preconditioners)
        scanned_layers_flat = grads_structure.flatten_up_to(scanned_layers_)

        Ls_flat = None
        if track_lipschitz and state.Ls_lipschitz is not None:
            Ls_flat = grads_structure.flatten_up_to(state.Ls_lipschitz)

        def apply_init_scale(Qs_in):
            grads_for_scale = [
                g[0] if s else g for g, s in zip(updates_flat, scanned_layers_flat)
            ]
            computed_scale = _compute_init_scale_from_grads(
                grads_for_scale, damping, precond_dtype or jnp.float32
            )
            computed_scale = dist_reduce(computed_scale, axis_name, "mean")

            def scale_q_list(q_list, grad, is_scanned):
                grad_shape = grad[0].shape if is_scanned else grad.shape
                ndim = len(grad_shape)
                if ndim == 0:
                    factor_scale = computed_scale
                else:
                    factor_scale = jnp.power(computed_scale, 1.0 / ndim)

                return [q * factor_scale.astype(q.dtype) for q in q_list]

            return [
                scale_q_list(q, g, s)
                for q, g, s in zip(Qs_in, updates_flat, scanned_layers_flat)
            ]

        Qs_flat = jax.lax.cond(
            state.needs_scale_init, apply_init_scale, lambda x: x, Qs_flat
        )
        needs_scale_init = jnp.array(False, dtype=jnp.bool_)

        expressions = [
            _init_Q_exprs(
                t[0] if s else t,
                _init_scale,
                max_size_triangular,
                max_skew_triangular,
                min_ndim_triangular,
                memory_save_mode,
                precond_dtype,
                existing_Q=jax.tree.map(lambda d: d[0], Q) if s else Q,
            )
            for t, s, Q in zip(updates_flat, scanned_layers_flat, Qs_flat)
        ]

        def update_preconditioner(key, Qs, Ls):
            with jax.default_matmul_precision(precond_update_precision):
                precond_updates_in = (
                    updates_flat if whiten_grad else momentum_updates_flat
                )

                key, key_noise = jax.random.split(key)
                Vs_keys = jax.random.split(key_noise, len(precond_updates_in))
                Vs = [
                    jax.random.normal(k, shape=g.shape, dtype=g.dtype)
                    for k, g in zip(Vs_keys, precond_updates_in)
                ]
                eps = jnp.finfo(precond_updates_in[0].dtype).eps
                precond_updates_in = [
                    g + (damping + eps * jnp.abs(g)) * v
                    for g, v in zip(precond_updates_in, Vs)
                ]

                key, key_updates = jax.random.split(key)
                layer_keys = jax.random.split(key_updates, len(Qs))

                if preconditioner_mode in [
                    PreconditionerMode.EQ,
                    PreconditionerMode.TAYLOR2,
                ]:
                    # For EQ mode: Vector = Q * G (Whitened Gradient, 'A')
                    Vectors = [
                        map_fn(s, partial(_precond_grad_eq, exprs=exprs), Q, g)
                        for s, exprs, Q, g in zip(
                            scanned_layers_flat, expressions, Qs, precond_updates_in
                        )
                    ]
                else:
                    # For Dense modes: Vector = Q^H * Q * G (Natural Gradient, 'Pg')
                    Vectors = [
                        map_fn(s, partial(_precond_grad, exprs=exprs), Q, g)
                        for s, exprs, Q, g in zip(
                            scanned_layers_flat, expressions, Qs, precond_updates_in
                        )
                    ]

                # Determine if we need conjB (Only EQ Mode uses it for A'A - B'B)
                if preconditioner_mode == PreconditionerMode.EQ:
                    conjBs = [
                        map_fn(s, _conjB, Q, g, v)
                        for s, Q, g, v in zip(
                            scanned_layers_flat, Qs, precond_updates_in, Vs
                        )
                    ]
                else:
                    # Dense modes do not use the auxiliary buffer Y
                    conjBs = [None] * len(Qs)

                mode_str = (
                    preconditioner_mode.value
                    if hasattr(preconditioner_mode, "value")
                    else preconditioner_mode
                )

                def _update_wrapper(Q, L, Vector, conjB, total_numel, exprs, layer_key):
                    return _update_precond_generic(
                        Q,
                        L,
                        Vector,
                        conjB,
                        total_numel,
                        exprs,
                        preconditioner_lr,
                        layer_key,
                        mode=mode_str,
                        beta_l=beta_lipschitz,
                        axis_name=axis_name,
                        damping=damping,
                        ns_iters=newton_schulz_iters,
                    )

                results = [
                    map_fn(
                        s,
                        partial(
                            _update_wrapper,
                            total_numel=g.size,
                            exprs=exprs,
                            layer_key=lk,
                        ),
                        Q,
                        L if Ls is not None else None,
                        Vector,
                        conjB,
                    )
                    for s, exprs, Q, L, g, Vector, conjB, lk in zip(
                        scanned_layers_flat,
                        expressions,
                        Qs,
                        Ls if Ls is not None else [None] * len(Qs),
                        precond_updates_in,
                        Vectors,
                        conjBs,
                        layer_keys,
                    )
                ]

                new_Qs = [r[0] for r in results]
                new_Ls = [r[1] for r in results] if track_lipschitz else None
                new_Qs = otu.tree_cast(new_Qs, precond_dtype)

                key, key_balance = jax.random.split(key)

                def balance_Qs(Qs_in):
                    return [
                        map_fn(s, partial(_balance_Q, axis_name=axis_name), Q)
                        if len(Q) > 1
                        else Q
                        for Q, s in zip(Qs_in, scanned_layers_flat)
                    ]

                do_balances = jax.random.uniform(key_balance) < 0.01
                new_Qs = jax.lax.cond(do_balances, balance_Qs, lambda qs: qs, new_Qs)

                return new_Qs, new_Ls

        key, key_dec, key_upd = jax.random.split(key, 3)
        do_update = jax.random.uniform(key_dec) < update_prob_in

        # Use >= 1.0 to be safe against tiny float precision issues, though == 1.0 usually works for manually set schedules.
        is_early_training = update_prob_in >= 1.0

        permissive = jnp.logical_and(permissive_spike_protection, is_early_training)

        # protect curvature state but still keep updates early in the training
        # Note: raw_global_grad_clip is static (Python variable), so 'is not None' is safe.
        skip_condition = jnp.logical_and(
            raw_global_grad_clip is not None, jnp.logical_not(permissive)
        )

        should_skip = jnp.logical_and(skip_condition, is_spike)

        # Only update if we were going to update AND we shouldn't skip
        do_update = jnp.logical_and(do_update, jnp.logical_not(should_skip))

        Qs_next, Ls_next = jax.lax.cond(
            do_update,
            update_preconditioner,
            lambda k, q, l: (q, l),
            key_upd,
            Qs_flat,
            Ls_flat,
        )

        use_new_Q = jnp.logical_and(do_update, update_preconditioner_first)
        Qs_for_grad = jax.tree.map(
            lambda n, o: jax.lax.select(use_new_Q, n, o), Qs_next, Qs_flat
        )

        with jax.default_matmul_precision(precond_grads_precision):
            precond_gs = [
                map_fn(s, partial(_precond_grad, exprs=exprs), Q, g)
                for s, exprs, Q, g in zip(
                    scanned_layers_flat, expressions, Qs_for_grad, momentum_updates_flat
                )
            ]

        # Clipping
        clip_mode = grad_clip_mode
        if isinstance(clip_mode, str):
            clip_mode = GradClipMode(clip_mode)

        if clip_mode == GradClipMode.GLOBAL_RMS:
            max_amp = (
                grad_clip_max_amps[0]
                if isinstance(grad_clip_max_amps, tuple)
                else grad_clip_max_amps
            )
            clip_scale = _compute_global_rms_scale(
                precond_gs, max_amp, axis_name=axis_name
            )
            precond_gs = [pg * clip_scale for pg in precond_gs]

        elif clip_mode == GradClipMode.PER_TENSOR_RMS:
            if isinstance(grad_clip_max_amps, (float, int)):
                max_rms_amp, max_element_amp = (grad_clip_max_amps, 10.0)
            else:
                max_rms_amp, max_element_amp = grad_clip_max_amps

            def _clip_fn(u):
                rms = jnp.sqrt(jnp.mean(numerics.abs_sq(u)))
                scale_factor = jnp.minimum(1.0, max_rms_amp / add_tiny(rms))
                u = u * scale_factor
                if jnp.iscomplexobj(u):
                    mag = jnp.abs(u)
                    clamp_scale = jnp.minimum(1.0, max_element_amp / add_tiny(mag))
                    u = u * clamp_scale
                else:
                    u = jnp.clip(u, -max_element_amp, max_element_amp)
                return u

            precond_gs = [_clip_fn(pg) for pg in precond_gs]

        else:
            raise ValueError(f"Unknown gradient clipping mode: {grad_clip_mode}")

        updates = grads_structure.unflatten(precond_gs)

        # NOTE: Magma masking is applied at the PyTree leaf level,
        # not the Kronecker sub-block level, treating each block as a distinct parameter unit.
        if use_magma:
            if state.mu is None:
                raise ValueError(
                    "Anti-Pattern: Magma strictly requires tracking momentum (b1 > 0)."
                )

            leaves_delta, treedef = jax.tree.flatten(updates)
            leaves_g = jax.tree.leaves(post_clip_raw_gradients)
            leaves_mu = jax.tree.leaves(mu)
            leaves_s = jax.tree.leaves(state.magma_s)

            # Advance the active PRNG stream; never re-split state.key to prevent
            # cryptographic correlation with preconditioner noise injections.
            magma_key, key_next = jax.random.split(key_next)
            subkeys = jax.random.split(magma_key, len(leaves_delta))

            new_leaves_delta, new_leaves_s = [], []
            for d, g, m_u, s_old, k_leaf in zip(
                leaves_delta, leaves_g, leaves_mu, leaves_s, subkeys
            ):
                g_f32 = g.astype(jnp.float32)
                m_f32 = m_u.astype(jnp.float32)

                dot = dist_reduce(jnp.sum(g_f32 * m_f32), axis_name, "sum")
                norm_g_sq = dist_reduce(jnp.sum(g_f32**2), axis_name, "sum")
                norm_mu_sq = dist_reduce(jnp.sum(m_f32**2), axis_name, "sum")

                cossim = dot / jnp.maximum(
                    jnp.sqrt(norm_g_sq) * jnp.sqrt(norm_mu_sq), 1e-9
                )
                s_tilde = jax.nn.sigmoid(cossim / magma_tau)
                s_new = 0.9 * s_old + 0.1 * s_tilde

                m_mask = jax.random.bernoulli(k_leaf, 0.5).astype(jnp.float32)

                new_leaves_delta.append(d * jnp.array(s_new * m_mask, dtype=d.dtype))
                new_leaves_s.append(s_new)

            updates = treedef.unflatten(new_leaves_delta)
            magma_s_to_save = treedef.unflatten(new_leaves_s)
        else:
            magma_s_to_save = state.magma_s

        Qs_to_save = grads_structure.unflatten(Qs_next)
        Qs_to_save = otu.tree_cast(Qs_to_save, precond_dtype)
        Ls_to_save = grads_structure.unflatten(Ls_next) if Ls_next else None
        mu = otu.tree_cast(mu, mu_dtype)

        new_state = KronState(
            count=count_inc,
            mu=mu,
            Qs_preconditioners=Qs_to_save,
            Ls_lipschitz=Ls_to_save,
            needs_scale_init=needs_scale_init,
            magma_s=magma_s_to_save,
            key=key_next,
        )

        return updates, new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def kron(
    learning_rate: Union[float, Callable[[int], float]] = 0.001,
    b1: float = 0.9,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    max_skew_triangular: float = 1.0,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    whiten_grad: bool = True,
    update_preconditioner_first: bool = True,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: Optional[float] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    preconditioner_mode: Union[str, PreconditionerMode] = PreconditionerMode.Q0P5EQ1P5,
    beta_lipschitz: float = 0.9,
    track_lipschitz: bool = True,
    damping: float = 1e-9,
    grad_clip_max_amps: float | Tuple[float, float] = (2.0, 10.0),
    grad_clip_mode: Union[str, GradClipMode] = GradClipMode.PER_TENSOR_RMS,
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    newton_schulz_iters: int = 5,
    use_magma: bool = False,
    magma_tau: float = 2.0,
    axis_name: Optional[str] = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    See `scale_by_kron` for detailed argument descriptions.
    This wrapper adds weight decay and learning rate scaling to the chain.
    """
    optimizer = [
        scale_by_kron(
            b1=b1,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            max_skew_triangular=max_skew_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            whiten_grad=whiten_grad,
            update_preconditioner_first=update_preconditioner_first,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            precond_update_precision=precond_update_precision,
            precond_grads_precision=precond_grads_precision,
            scanned_layers=scanned_layers,
            lax_map_scanned_layers=lax_map_scanned_layers,
            lax_map_batch_size=lax_map_batch_size,
            preconditioner_mode=preconditioner_mode,
            beta_lipschitz=beta_lipschitz,
            track_lipschitz=track_lipschitz,
            damping=damping,
            grad_clip_max_amps=grad_clip_max_amps,
            grad_clip_mode=grad_clip_mode,
            raw_global_grad_clip=raw_global_grad_clip,
            permissive_spike_protection=permissive_spike_protection,
            newton_schulz_iters=newton_schulz_iters,
            use_magma=use_magma,
            magma_tau=magma_tau,
            axis_name=axis_name,
            key=key,
        )
    ]
    if weight_decay > 0.0:
        optimizer.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    optimizer.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*optimizer)
