"""GaLore projected AdamW optimizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
from optax._src import base, numerics, utils
from optax.transforms import _masking

from rollfast.utils import _safe_bias_correction


Projection = Literal["auto", "left", "right", "two_sided"]
StateOnRefresh = Literal["reuse_coordinates", "reset", "transport"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GaLoreLeafState:
    """Per-leaf projected or fallback Adam state."""

    basis_left: jax.Array
    basis_right: jax.Array
    mu: jax.Array
    nu: jax.Array
    orientation: Literal["left", "right", "full"]
    projected: bool
    shape: tuple[int, ...]

    def tree_flatten(self):
        return (self.basis_left, self.basis_right, self.mu, self.nu), (
            self.orientation,
            self.projected,
            self.shape,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        orientation, projected, shape = aux_data
        basis_left, basis_right, mu, nu = children
        return cls(
            basis_left=basis_left,
            basis_right=basis_right,
            mu=mu,
            nu=nu,
            orientation=orientation,
            projected=projected,
            shape=shape,
        )


class ScaleByGaLoreState(NamedTuple):
    """State for GaLore projected Adam."""

    count: jax.Array
    leaves: Any


@dataclass(frozen=True)
class _LeafUpdateResult:
    update: Any
    state: Any


def galore_adamw(
    learning_rate: base.ScalarOrSchedule,
    *,
    rank: int,
    update_interval: int = 200,
    scale: float = 1.0,
    projection: Projection = "auto",
    basis_method: Literal["svd", "randomized_svd"] = "svd",
    basis_dtype: jax.typing.DTypeLike = jnp.float32,
    state_on_basis_refresh: StateOnRefresh = "reuse_coordinates",
    min_matrix_size: int = 4096,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: jax.typing.DTypeLike = jnp.float32,
    weight_decay: base.ScalarOrSchedule = 0.0,
) -> base.GradientTransformation:
    """AdamW with GaLore low-rank moment states for eligible matrices."""

    if rank < 1:
        raise ValueError("GaLore rank must be >= 1.")
    if update_interval < 1:
        raise ValueError("GaLore update_interval must be >= 1.")
    if scale <= 0.0:
        raise ValueError("GaLore scale must be positive.")
    if min_matrix_size < 0:
        raise ValueError("GaLore min_matrix_size must be non-negative.")
    if projection not in {"auto", "left", "right", "two_sided"}:
        raise ValueError(
            "GaLore projection must be 'auto', 'left', 'right', or 'two_sided'."
        )
    if basis_method != "svd":
        raise NotImplementedError("GaLore currently supports basis_method='svd'.")
    if state_on_basis_refresh not in {"reuse_coordinates", "reset", "transport"}:
        raise ValueError(
            "GaLore state_on_basis_refresh must be 'reuse_coordinates', "
            "'reset', or 'transport'."
        )

    basis_dtype = cast(jax.typing.DTypeLike, utils.canonicalize_dtype(basis_dtype))
    mu_dtype = cast(jax.typing.DTypeLike, utils.canonicalize_dtype(mu_dtype))

    def init_fn(params):
        leaves = jax.tree.map(
            lambda param: _init_leaf_state(
                param,
                rank=rank,
                projection=projection,
                basis_dtype=basis_dtype,
                mu_dtype=mu_dtype,
                min_matrix_size=min_matrix_size,
            ),
            params,
            is_leaf=_is_passthrough,
        )
        return ScaleByGaLoreState(count=jnp.zeros([], jnp.int32), leaves=leaves)

    def update_fn(updates, state, params=None):
        count_inc = cast(jax.Array, numerics.safe_increment(state.count))
        should_refresh = (state.count % update_interval) == 0
        wd_step = (
            cast(Callable[[jax.Array], Any], weight_decay)(state.count)
            if callable(weight_decay)
            else weight_decay
        )
        lr = (
            cast(Callable[[jax.Array], Any], learning_rate)(state.count)
            if callable(learning_rate)
            else learning_rate
        )

        results = jax.tree.map(
            lambda grad, leaf_state, param: _update_leaf(
                grad,
                leaf_state,
                param,
                count_inc=count_inc,
                should_refresh=should_refresh,
                state_on_basis_refresh=state_on_basis_refresh,
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                scale=scale,
                weight_decay=wd_step,
                learning_rate=lr,
                basis_dtype=basis_dtype,
                mu_dtype=mu_dtype,
            ),
            updates,
            state.leaves,
            updates if params is None else params,
            is_leaf=_is_state_or_passthrough,
        )
        new_updates = jax.tree.map(
            lambda result: result.update,
            results,
            is_leaf=lambda x: isinstance(x, _LeafUpdateResult),
        )
        new_leaves = jax.tree.map(
            lambda result: result.state,
            results,
            is_leaf=lambda x: isinstance(x, _LeafUpdateResult),
        )
        return new_updates, ScaleByGaLoreState(count=count_inc, leaves=new_leaves)

    return base.GradientTransformation(init_fn, update_fn)


def projected_state_nbytes(
    shape: tuple[int, ...],
    *,
    rank: int,
    projection: Projection = "auto",
    basis_dtype: jax.typing.DTypeLike = jnp.float32,
    moment_dtype: jax.typing.DTypeLike = jnp.float32,
    min_matrix_size: int = 4096,
) -> int:
    """Estimate GaLore basis plus two projected moment buffers for one leaf."""

    if len(shape) != 2 or _numel(shape) < min_matrix_size:
        return int(_numel(shape) * jnp.dtype(moment_dtype).itemsize * 2)
    rows, cols = shape
    rank = min(rank, rows, cols)
    orientation = _resolve_orientation(rows, cols, projection)
    basis_items = (
        (rows + cols) * rank
        if orientation == "full"
        else (rows if orientation == "left" else cols) * rank
    )
    coord_items = (
        rank * rank
        if orientation == "full"
        else (rank * cols if orientation == "left" else rows * rank)
    )
    return int(basis_items * jnp.dtype(basis_dtype).itemsize) + int(
        coord_items * jnp.dtype(moment_dtype).itemsize * 2
    )


def _init_leaf_state(
    param: Any,
    *,
    rank: int,
    projection: Projection,
    basis_dtype,
    mu_dtype,
    min_matrix_size: int,
) -> Any:
    if _is_passthrough(param):
        return param
    if not _eligible(param, min_matrix_size=min_matrix_size):
        zeros = jnp.zeros_like(param, dtype=mu_dtype)
        empty = jnp.zeros((0,), dtype=basis_dtype)
        return GaLoreLeafState(
            empty, empty, zeros, zeros, "full", False, tuple(param.shape)
        )
    rows, cols = param.shape
    rank = min(rank, rows, cols)
    orientation = _resolve_orientation(rows, cols, projection)
    if orientation == "left":
        basis_left = jnp.zeros((rows, rank), dtype=basis_dtype)
        basis_right = jnp.zeros((0,), dtype=basis_dtype)
        coord_shape = (rank, cols)
    elif orientation == "right":
        basis_left = jnp.zeros((0,), dtype=basis_dtype)
        basis_right = jnp.zeros((cols, rank), dtype=basis_dtype)
        coord_shape = (rows, rank)
    else:
        basis_left = jnp.zeros((rows, rank), dtype=basis_dtype)
        basis_right = jnp.zeros((cols, rank), dtype=basis_dtype)
        coord_shape = (rank, rank)
    zeros = jnp.zeros(coord_shape, dtype=mu_dtype)
    return GaLoreLeafState(
        basis_left,
        basis_right,
        zeros,
        zeros,
        orientation,
        True,
        tuple(param.shape),
    )


def _update_leaf(
    grad: Any,
    state: Any,
    param: Any,
    *,
    count_inc,
    should_refresh,
    state_on_basis_refresh: StateOnRefresh,
    b1,
    b2,
    eps,
    eps_root,
    scale: float,
    weight_decay,
    learning_rate,
    basis_dtype,
    mu_dtype,
) -> _LeafUpdateResult:
    if _is_passthrough(grad) or _is_passthrough(state):
        return _LeafUpdateResult(grad, state)
    if not isinstance(state, GaLoreLeafState) or not state.projected:
        return _full_adam_leaf(
            grad,
            state,
            param,
            count_inc=count_inc,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            mu_dtype=mu_dtype,
        )

    basis_left, basis_right = _refresh_basis(
        grad,
        state,
        should_refresh=should_refresh,
        basis_dtype=basis_dtype,
    )
    projected_grad = _project(grad, basis_left, basis_right, state.orientation)
    if state_on_basis_refresh == "reset":
        mu_prev = jnp.where(should_refresh, jnp.zeros_like(state.mu), state.mu)
        nu_prev = jnp.where(should_refresh, jnp.zeros_like(state.nu), state.nu)
    elif state_on_basis_refresh == "transport":
        mu_prev, nu_prev = _transport_projected_moments(
            state,
            basis_left,
            basis_right,
            should_refresh=should_refresh,
        )
    else:
        mu_prev = state.mu
        nu_prev = state.nu
    mu = (b1 * mu_prev + (1.0 - b1) * projected_grad).astype(mu_dtype)
    nu = (b2 * nu_prev + (1.0 - b2) * jnp.square(projected_grad)).astype(mu_dtype)
    mu_hat = _safe_bias_correction(mu.astype(jnp.float32), 1.0 - b1**count_inc)
    nu_hat = _safe_bias_correction(nu.astype(jnp.float32), 1.0 - b2**count_inc)
    projected_update = mu_hat / (jnp.sqrt(nu_hat + eps_root) + eps)
    update = _reconstruct(projected_update, basis_left, basis_right, state.orientation)
    update = update * scale
    if param is not None and weight_decay != 0.0:
        update = update + weight_decay * param.astype(jnp.float32)
    update = (-learning_rate * update).astype(grad.dtype)
    return _LeafUpdateResult(
        update,
        GaLoreLeafState(
            basis_left,
            basis_right,
            mu,
            nu,
            state.orientation,
            state.projected,
            state.shape,
        ),
    )


def _full_adam_leaf(
    grad,
    state,
    param,
    *,
    count_inc,
    b1,
    b2,
    eps,
    eps_root,
    weight_decay,
    learning_rate,
    mu_dtype,
) -> _LeafUpdateResult:
    mu = (b1 * state.mu + (1.0 - b1) * grad).astype(mu_dtype)
    nu = (b2 * state.nu + (1.0 - b2) * jnp.square(grad)).astype(mu_dtype)
    mu_hat = _safe_bias_correction(mu.astype(jnp.float32), 1.0 - b1**count_inc)
    nu_hat = _safe_bias_correction(nu.astype(jnp.float32), 1.0 - b2**count_inc)
    update = mu_hat / (jnp.sqrt(nu_hat + eps_root) + eps)
    if param is not None and weight_decay != 0.0:
        update = update + weight_decay * param.astype(jnp.float32)
    update = (-learning_rate * update).astype(grad.dtype)
    return _LeafUpdateResult(
        update,
        GaLoreLeafState(
            state.basis_left,
            state.basis_right,
            mu,
            nu,
            state.orientation,
            state.projected,
            state.shape,
        ),
    )


def _refresh_basis(
    grad: jax.Array,
    state: GaLoreLeafState,
    *,
    should_refresh,
    basis_dtype,
) -> tuple[jax.Array, jax.Array]:
    basis_left, basis_right = _basis_from_grad(grad, state.orientation, state.mu.shape)
    basis_left = basis_left.astype(basis_dtype)
    basis_right = basis_right.astype(basis_dtype)
    if state.orientation == "left":
        basis_left = jnp.where(should_refresh, basis_left, state.basis_left)
        basis_right = state.basis_right
    elif state.orientation == "right":
        basis_left = state.basis_left
        basis_right = jnp.where(should_refresh, basis_right, state.basis_right)
    else:
        basis_left = jnp.where(should_refresh, basis_left, state.basis_left)
        basis_right = jnp.where(should_refresh, basis_right, state.basis_right)
    return basis_left, basis_right


def _transport_projected_moments(
    state: GaLoreLeafState,
    basis_left: jax.Array,
    basis_right: jax.Array,
    *,
    should_refresh,
) -> tuple[jax.Array, jax.Array]:
    if state.orientation == "left":
        overlap = basis_left.T.astype(jnp.float32) @ state.basis_left.astype(
            jnp.float32
        )
        mu = overlap @ state.mu.astype(jnp.float32)
        nu = jnp.square(overlap) @ state.nu.astype(jnp.float32)
    elif state.orientation == "right":
        overlap = state.basis_right.T.astype(jnp.float32) @ basis_right.astype(
            jnp.float32
        )
        mu = state.mu.astype(jnp.float32) @ overlap
        nu = state.nu.astype(jnp.float32) @ jnp.square(overlap)
    else:
        left_overlap = basis_left.T.astype(jnp.float32) @ state.basis_left.astype(
            jnp.float32
        )
        right_overlap = state.basis_right.T.astype(jnp.float32) @ basis_right.astype(
            jnp.float32
        )
        mu = left_overlap @ state.mu.astype(jnp.float32) @ right_overlap
        nu = (
            jnp.square(left_overlap)
            @ state.nu.astype(jnp.float32)
            @ jnp.square(right_overlap)
        )
    return (
        jnp.where(should_refresh, mu.astype(state.mu.dtype), state.mu),
        jnp.where(should_refresh, nu.astype(state.nu.dtype), state.nu),
    )


def _basis_from_grad(
    grad: jax.Array,
    orientation: Literal["left", "right", "full"],
    coord_shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    grad32 = grad.astype(jnp.float32)
    u, _, vh = jnp.linalg.svd(grad32, full_matrices=False)
    if orientation == "left":
        rank = coord_shape[0]
        return _canonicalize_basis(u[:, :rank]), jnp.zeros((0,), dtype=grad32.dtype)
    if orientation == "right":
        rank = coord_shape[1]
        return jnp.zeros((0,), dtype=grad32.dtype), _canonicalize_basis(vh[:rank].T)
    rank = coord_shape[0]
    return _canonicalize_basis(u[:, :rank]), _canonicalize_basis(vh[:rank].T)


def _canonicalize_basis(basis: jax.Array) -> jax.Array:
    abs_basis = jnp.abs(basis)
    pivot = jnp.argmax(abs_basis, axis=0)
    signs = jnp.sign(jnp.take_along_axis(basis, pivot[None, :], axis=0)[0])
    signs = jnp.where(signs == 0, 1.0, signs)
    return basis * signs[None, :]


def _project(
    grad: jax.Array,
    basis_left: jax.Array,
    basis_right: jax.Array,
    orientation: Literal["left", "right", "full"],
) -> jax.Array:
    if orientation == "left":
        return basis_left.T.astype(grad.dtype) @ grad
    if orientation == "right":
        return grad @ basis_right.astype(grad.dtype)
    return basis_left.T.astype(grad.dtype) @ grad @ basis_right.astype(grad.dtype)


def _reconstruct(
    update: jax.Array,
    basis_left: jax.Array,
    basis_right: jax.Array,
    orientation: Literal["left", "right", "full"],
) -> jax.Array:
    if orientation == "left":
        return basis_left.astype(update.dtype) @ update
    if orientation == "right":
        return update @ basis_right.T.astype(update.dtype)
    return basis_left.astype(update.dtype) @ update @ basis_right.T.astype(update.dtype)


def _eligible(param: Any, *, min_matrix_size: int) -> bool:
    return (
        hasattr(param, "shape")
        and len(param.shape) == 2
        and _numel(param.shape) >= min_matrix_size
    )


def _resolve_orientation(
    rows: int,
    cols: int,
    projection: Projection,
) -> Literal["left", "right", "full"]:
    if projection == "auto":
        return "right" if rows <= cols else "left"
    if projection == "two_sided":
        return "full"
    return projection


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _is_passthrough(value: Any) -> bool:
    return value is None or isinstance(value, _masking.MaskedNode)


def _is_state_or_passthrough(value: Any) -> bool:
    return isinstance(value, GaLoreLeafState) or _is_passthrough(value)


__all__ = (
    "GaLoreLeafState",
    "ScaleByGaLoreState",
    "galore_adamw",
    "projected_state_nbytes",
)
