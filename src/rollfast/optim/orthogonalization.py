"""Shared Newton-Schulz coefficient and orthogonalization utilities.

Muon-family optimizers and PRISM ``mode="original"`` use the coefficient
presets and ordered schedules defined here. PRISM ``mode="bidirectional"``
uses separate inverse-root coefficients, while Aurora intentionally keeps its
reference simple-quintic polar implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from optax.transforms import _masking

from rollfast.optim.dimension_numbers import (
    MatrixDimensionNumbers,
    _compute_matrix_reshape,
)

MuonPreconditioning: TypeAlias = Literal["frobenius", "spectral", "aol", "schatten"]
NsCoeffTriple: TypeAlias = tuple[
    jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike
]
NsCoeffs: TypeAlias = NsCoeffTriple | tuple[NsCoeffTriple, ...] | str
MuonNsCoeffs: TypeAlias = NsCoeffs
OrthogonalizeFn: TypeAlias = Callable[
    [
        jax.Array,
        jax.Array,
        jax.typing.ArrayLike,
        MuonPreconditioning,
        jax.typing.ArrayLike,
        MatrixDimensionNumbers | None,
    ],
    jax.Array,
]

MUON_NS_COEFFS = (3.4445, -4.7750, 2.0315)
DION_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
NS_COEFFS_PRESETS = {
    "standard": MUON_NS_COEFFS,
    "dion": DION_NS_COEFFS,
}

_PRECONDITIONINGS = ("frobenius", "spectral", "aol", "schatten")


def _is_aux_leaf(x: Any) -> bool:
    return x is None or isinstance(x, _masking.MaskedNode)


def _optimal_quintic(l: float, u: float) -> tuple[float, float, float]:
    if not 0 < l <= u:
        raise ValueError(f"l must satisfy 0 < l <= u, got l={l}, u={u}.")
    if 1 - 1e-5 <= l / u:
        return (15 / 8) / u, (-10 / 8) / (u**3), (3 / 8) / (u**5)

    q = (3 * l + u) / 4
    r = (l + 3 * u) / 4
    err = np.inf
    old_err = None
    for _ in range(1000):
        if old_err is not None and abs(old_err - err) <= 1e-15:
            break
        old_err = err
        lhs = np.array(
            [
                [l, l**3, l**5, 1],
                [q, q**3, q**5, -1],
                [r, r**3, r**5, 1],
                [u, u**3, u**5, -1],
            ]
        )
        a, b, c, err = np.linalg.solve(lhs, np.ones(4))
        q, r = np.sqrt(
            (-3 * b + np.array([-1, 1]) * np.sqrt(9 * b**2 - 20 * a * c)) / (10 * c)
        )
    return float(a), float(b), float(c)


def polar_express_coeffs(
    l: float = 1e-3,
    num_iters: int = 8,
    *,
    safety_factor_eps: float = 2e-2,
    cushion: float = 0.02407327424182761,
) -> list[tuple[float, float, float]]:
    """Compute PolarExpress per-step quintic Newton-Schulz coefficients."""
    u = 1.0
    if not 0 < l <= u:
        raise ValueError(f"l must satisfy 0 < l <= 1, got {l}.")
    if num_iters < 1:
        raise ValueError(f"num_iters must be >= 1, got {num_iters}.")

    safety_factor = 1 + safety_factor_eps
    coefficients = []
    for i in range(num_iters):
        a, b, c = _optimal_quintic(max(l, cushion * u), u)
        if cushion * u > l:
            pl = a * l + b * l**3 + c * l**5
            pu = a * u + b * u**3 + c * u**5
            rescaler = 2 / (pl + pu)
            a *= rescaler
            b *= rescaler
            c *= rescaler
        if i < num_iters - 1:
            a /= safety_factor
            b /= safety_factor**3
            c /= safety_factor**5
        coefficients.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l
    return coefficients


def resolve_ns_coeffs(
    ns_coeffs: NsCoeffs,
    ns_steps: jax.typing.ArrayLike,
) -> jax.Array:
    """Resolve a preset, triple, or ordered schedule into a coefficient array."""
    ns_steps_int = int(cast(Any, ns_steps))
    if isinstance(ns_coeffs, str):
        if ns_coeffs == "polar_express":
            coeffs = polar_express_coeffs(num_iters=ns_steps_int)
        elif ns_coeffs in NS_COEFFS_PRESETS:
            coeffs = NS_COEFFS_PRESETS[ns_coeffs]
        else:
            raise ValueError(f"Unknown ns_coeff preset string: {ns_coeffs}")
    else:
        coeffs = ns_coeffs

    coeffs_array = jnp.asarray(coeffs)
    if coeffs_array.ndim > 2 or coeffs_array.shape[-1] != 3:
        raise ValueError(
            f"ns_coeffs must have shape (3,) or (n, 3), got {coeffs_array.shape}"
        )
    if coeffs_array.ndim == 2:
        if coeffs_array.shape[0] < ns_steps_int:
            raise ValueError(f"Not enough coeffs to perform {ns_steps} steps")
        coeffs_array = coeffs_array[:ns_steps_int]
    return coeffs_array


_resolve_ns_coeffs = resolve_ns_coeffs


def _aol_first_newton_schulz_iteration(
    x: jax.Array,
    coeffs: jax.Array,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
    a = x @ jnp.swapaxes(x, -1, -2).conj()
    rescaling = jnp.clip(jnp.abs(a).sum(axis=-1), min=eps)
    s = jnp.expand_dims(jax.lax.rsqrt(rescaling), -1)
    x, a = x * s, a * s * jnp.swapaxes(s, -1, -2)
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x


def _schatten_first_newton_schulz_iteration(
    x: jax.Array,
    coeffs: jax.Array,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
    a = x @ jnp.swapaxes(x, -1, -2).conj()
    rescaling = jnp.clip(jnp.linalg.norm(a, ord="fro", axis=(-2, -1)), min=eps)
    s = jnp.expand_dims(jax.lax.rsqrt(rescaling), (-2, -1))
    x, a = x * s, a * s**2
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x


def _base_newton_schulz_iteration(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    a = x @ jnp.swapaxes(x, -1, -2).conj()
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x


_newton_schulz_iterator = _base_newton_schulz_iteration


def _aol_ns_iterator(i, x, coeffs):
    return jax.lax.cond(
        i == 0,
        lambda x_: _aol_first_newton_schulz_iteration(x_, coeffs),
        lambda x_: _base_newton_schulz_iteration(x_, coeffs),
        x,
    )


def _schatten_ns_iterator(i, x, coeffs):
    return jax.lax.cond(
        i == 0,
        lambda x_: _schatten_first_newton_schulz_iteration(x_, coeffs),
        lambda x_: _base_newton_schulz_iteration(x_, coeffs),
        x,
    )


def _base_ns_iterator(i, x, coeffs):
    del i
    return _base_newton_schulz_iteration(x, coeffs)


def quintic_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike = 5,
    preconditioning: MuonPreconditioning = "frobenius",
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
    """Apply quintic Newton-Schulz orthogonalization over the last two axes."""
    if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
        raise ValueError(
            "Newton-Schulz coefficients must have shape (3,) or "
            f"(n, 3), got {ns_coeffs.shape}."
        )
    if x.ndim < 2:
        raise ValueError(f"Input must have rank >= 2, got shape={x.shape}.")

    matrix = x.astype(jnp.float32)
    transposed = False
    if matrix.shape[-2] > matrix.shape[-1]:
        matrix = jnp.swapaxes(matrix, -1, -2)
        transposed = True

    ns_iterators = {
        "frobenius": _base_ns_iterator,
        "spectral": _base_ns_iterator,
        "aol": _aol_ns_iterator,
        "schatten": _schatten_ns_iterator,
    }
    if preconditioning not in _PRECONDITIONINGS:
        raise ValueError(f"Unknown preconditioning {preconditioning!r}.")
    ns_iterator = ns_iterators[preconditioning]

    if preconditioning == "frobenius":
        matrix = matrix / (
            jnp.linalg.norm(matrix, ord="fro", axis=(-2, -1), keepdims=True) + eps
        )
    elif preconditioning == "spectral":
        matrix = matrix / (
            jnp.linalg.norm(matrix, ord=2, axis=(-2, -1), keepdims=True) + eps
        )

    coeffs = ns_coeffs.astype(matrix.dtype)
    if coeffs.ndim == 1:
        matrix = jax.lax.fori_loop(
            0,
            ns_steps,
            lambda i, matrix_: ns_iterator(i, matrix_, coeffs),
            matrix,
            unroll=True,
        )
    else:

        def scan_body(carry, coeffs_step):
            i, matrix_ = carry
            return (i + 1, ns_iterator(i, matrix_, coeffs_step)), None

        (_, matrix), _ = jax.lax.scan(
            scan_body, (jnp.asarray(0, dtype=jnp.int32), matrix), coeffs
        )

    if transposed:
        matrix = jnp.swapaxes(matrix, -1, -2)
    return matrix


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike = 5,
    preconditioning: MuonPreconditioning = "frobenius",
    eps: jax.typing.ArrayLike = 1e-8,
    dimension_numbers: MatrixDimensionNumbers | None = None,
) -> jax.Array:
    """Orthogonalize a matrix or matrix-shaped tensor via Newton-Schulz."""
    if _is_aux_leaf(x):
        return x
    if x.ndim != 2 and not isinstance(dimension_numbers, MatrixDimensionNumbers):
        raise ValueError(
            "Input must have shape (m, n) or weight dimension numbers must be "
            f"provided. Got shape={x.shape} and {dimension_numbers=}."
        )
    if x.ndim == 2:
        dimension_numbers = MatrixDimensionNumbers(reduction_axis=0, output_axis=1)
    assert dimension_numbers is not None

    reshape_fn, inverse_fn = _compute_matrix_reshape(x, dimension_numbers)
    matrix = reshape_fn(x)
    ortho = quintic_newton_schulz(
        matrix,
        ns_coeffs,
        ns_steps=ns_steps,
        preconditioning=preconditioning,
        eps=eps,
    )
    return inverse_fn(ortho)


__all__ = [
    "DION_NS_COEFFS",
    "MUON_NS_COEFFS",
    "MuonNsCoeffs",
    "MuonPreconditioning",
    "NS_COEFFS_PRESETS",
    "NsCoeffs",
    "NsCoeffTriple",
    "OrthogonalizeFn",
    "orthogonalize_via_newton_schulz",
    "polar_express_coeffs",
    "quintic_newton_schulz",
    "resolve_ns_coeffs",
]
