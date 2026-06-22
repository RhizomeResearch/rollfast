"""Sharpness-aware perturbation primitives."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def global_l2_norm(
    tree: Any,
    *,
    axis_name: str | tuple[str, ...] | None = None,
) -> jax.Array:
    """Return the global L2 norm of array leaves, optionally across devices."""

    sq_sum = sum(
        jnp.sum(jnp.square(leaf.astype(jnp.float32)))
        for leaf in jax.tree.leaves(tree, is_leaf=lambda x: x is None)
        if leaf is not None
    )
    sq_sum = jnp.asarray(sq_sum, dtype=jnp.float32)
    if axis_name is not None:
        sq_sum = jax.lax.psum(sq_sum, axis_name=axis_name)
    return jnp.sqrt(sq_sum)


def sam_perturbation(
    grads: Any,
    *,
    params: Any | None = None,
    rho: float = 0.05,
    adaptive: bool = False,
    eta: float = 0.0,
    eps: float = 1e-12,
    mask: Any | None = None,
    axis_name: str | tuple[str, ...] | None = None,
) -> tuple[Any, jax.Array]:
    """Construct the SAM/ASAM ascent perturbation for a trainable PyTree.

    ASAM uses the common diagonal normalization derived from the constraint
    ``||T_w^{-1} epsilon|| <= rho``: with ``T_w = abs(w) + eta``, the returned
    perturbation is proportional to ``T_w ** 2 * grad`` and normalized by
    ``||T_w * grad||``.
    """

    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if adaptive and params is None:
        raise ValueError("adaptive SAM requires params.")

    mask = (
        jax.tree.map(
            lambda g: None if g is None else True,
            grads,
            is_leaf=lambda x: x is None,
        )
        if mask is None
        else mask
    )

    direction = jax.tree.map(
        lambda g, p, m: _direction_leaf(
            g,
            p,
            m,
            adaptive=adaptive,
            eta=eta,
        ),
        grads,
        grads if params is None else params,
        mask,
        is_leaf=lambda x: x is None,
    )
    direction_norm = global_l2_norm(direction, axis_name=axis_name)
    scale = rho / (direction_norm + eps)
    perturbation = jax.tree.map(
        lambda d, p, m: _perturb_leaf(
            d,
            p,
            m,
            scale,
            adaptive=adaptive,
            eta=eta,
        ),
        direction,
        direction if params is None else params,
        mask,
        is_leaf=lambda x: x is None,
    )
    return perturbation, global_l2_norm(perturbation, axis_name=axis_name)


def add_perturbation(params: Any, perturbation: Any) -> Any:
    """Return ``params + perturbation`` while preserving absent leaves."""

    return jax.tree.map(
        lambda p, e: None if p is None else p + e,
        params,
        perturbation,
        is_leaf=lambda x: x is None,
    )


def _direction_leaf(
    grad: Any,
    param: Any,
    mask: bool | None,
    *,
    adaptive: bool,
    eta: float,
) -> Any:
    if grad is None:
        return None
    if not bool(mask):
        return jnp.zeros_like(grad, dtype=jnp.float32)
    grad = grad.astype(jnp.float32)
    if not adaptive:
        return grad
    return (jnp.abs(param).astype(jnp.float32) + eta) * grad


def _perturb_leaf(
    direction: Any,
    param: Any,
    mask: bool | None,
    scale: jax.Array,
    *,
    adaptive: bool,
    eta: float,
) -> Any:
    if direction is None:
        return None
    if not bool(mask):
        return jnp.zeros_like(direction, dtype=jnp.float32)
    if not adaptive:
        return direction * scale
    return (jnp.abs(param).astype(jnp.float32) + eta) * direction * scale


__all__ = ("add_perturbation", "global_l2_norm", "sam_perturbation")
