"""APOLLO projected-scaling AdamW optimizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
from optax._src import base, numerics, utils
from optax.transforms import _masking


Scaling = Literal["channel", "tensor"]
Orientation = Literal["left", "right", "full"]
REFERENCE_EPS = 1e-6
SCALING_FACTOR_EPS = 1e-8


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class APOLLOLeafState:
    """Per-leaf APOLLO or fallback Adam state."""

    projection: jax.Array
    mu: jax.Array
    nu: jax.Array
    prev_update_norm: jax.Array
    orientation: Orientation
    projected: bool
    shape: tuple[int, ...]
    leaf_index: int

    def tree_flatten(self):
        return (self.projection, self.mu, self.nu, self.prev_update_norm), (
            self.orientation,
            self.projected,
            self.shape,
            self.leaf_index,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        orientation, projected, shape, leaf_index = aux_data
        projection, mu, nu, prev_update_norm = children
        return cls(
            projection=projection,
            mu=mu,
            nu=nu,
            prev_update_norm=prev_update_norm,
            orientation=orientation,
            projected=projected,
            shape=shape,
            leaf_index=leaf_index,
        )


class ScaleByAPOLLOState(NamedTuple):
    """State for APOLLO projected-scaling Adam."""

    count: jax.Array
    leaves: Any


@dataclass(frozen=True)
class _LeafUpdateResult:
    update: Any
    state: Any


def apollo_adamw(
    learning_rate: base.ScalarOrSchedule,
    *,
    rank: int = 256,
    projection_seed: int = 0,
    projection_refresh_interval: int | None = None,
    scaling: Scaling = "channel",
    mini: bool = False,
    scale: float = 1.0,
    scale_front: bool = False,
    disable_norm_growth_limiter: bool = False,
    norm_growth_limiter: float = 1.01,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = REFERENCE_EPS,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: jax.typing.DTypeLike = jnp.float32,
    weight_decay: base.ScalarOrSchedule = 0.0,
) -> base.GradientTransformation:
    """AdamW using APOLLO low-rank projected gradient-scaling factors."""

    if rank < 1:
        raise ValueError("APOLLO rank must be >= 1.")
    if projection_refresh_interval is not None and projection_refresh_interval < 1:
        raise ValueError("projection_refresh_interval must be >= 1 when provided.")
    if scaling not in {"channel", "tensor"}:
        raise ValueError("APOLLO scaling must be 'channel' or 'tensor'.")
    if scale <= 0.0:
        raise ValueError("APOLLO scale must be positive.")
    if norm_growth_limiter <= 1.0:
        raise ValueError("norm_growth_limiter must be greater than 1.")

    resolved_rank = 1 if mini else rank
    resolved_scaling: Scaling = "tensor" if mini else scaling
    mu_dtype = cast(jax.typing.DTypeLike, utils.canonicalize_dtype(mu_dtype))

    def init_fn(params):
        leaves = _tree_map_with_index(
            lambda index, param: _init_leaf_state(
                index,
                param,
                rank=resolved_rank,
                projection_seed=projection_seed,
                mu_dtype=mu_dtype,
            ),
            params,
        )
        return ScaleByAPOLLOState(count=jnp.zeros([], jnp.int32), leaves=leaves)

    def update_fn(updates, state, params=None):
        count_inc = cast(jax.Array, numerics.safe_increment(state.count))
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
                count=state.count,
                count_inc=count_inc,
                rank=resolved_rank,
                projection_seed=projection_seed,
                projection_refresh_interval=projection_refresh_interval,
                scaling=resolved_scaling,
                scale=scale,
                scale_front=scale_front,
                disable_norm_growth_limiter=disable_norm_growth_limiter,
                norm_growth_limiter=norm_growth_limiter,
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                weight_decay=wd_step,
                learning_rate=lr,
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
        return new_updates, ScaleByAPOLLOState(count=count_inc, leaves=new_leaves)

    return base.GradientTransformation(init_fn, update_fn)


def apollo_state_nbytes(
    shape: tuple[int, ...],
    *,
    rank: int = 256,
    mini: bool = False,
    moment_dtype: jax.typing.DTypeLike = jnp.float32,
    projection_dtype: jax.typing.DTypeLike = jnp.float32,
) -> int:
    """Estimate APOLLO projection plus moment buffers for one leaf."""

    moment_itemsize = jnp.dtype(moment_dtype).itemsize
    projection_itemsize = jnp.dtype(projection_dtype).itemsize
    if len(shape) != 2:
        return int(_numel(shape) * moment_itemsize * 2)

    rows, cols = shape
    orientation = _resolve_orientation(rows, cols)
    projected_dim = rows if orientation == "left" else cols
    channel_dim = cols if orientation == "left" else rows
    resolved_rank = min(1 if mini else rank, projected_dim)
    projection_items = resolved_rank * projected_dim
    moment_items = resolved_rank * channel_dim
    return (
        int(projection_items * projection_itemsize)
        + int(moment_items * moment_itemsize * 2)
        + int(moment_itemsize)
    )


def _init_leaf_state(
    leaf_index: int,
    param: Any,
    *,
    rank: int,
    projection_seed: int,
    mu_dtype,
) -> Any:
    if _is_passthrough(param):
        return param
    if not _eligible(param):
        zeros = jnp.zeros_like(param, dtype=mu_dtype)
        empty = jnp.zeros((0,), dtype=jnp.float32)
        return APOLLOLeafState(
            empty,
            zeros,
            zeros,
            jnp.zeros([], dtype=mu_dtype),
            "full",
            False,
            tuple(param.shape),
            leaf_index,
        )

    rows, cols = param.shape
    orientation = _resolve_orientation(rows, cols)
    projected_dim = rows if orientation == "left" else cols
    channel_dim = cols if orientation == "left" else rows
    resolved_rank = min(rank, projected_dim)
    projection = _make_projection(
        projection_seed=projection_seed,
        leaf_index=leaf_index,
        step=jnp.array(0, dtype=jnp.int32),
        shape=(resolved_rank, projected_dim),
        dtype=jnp.float32,
    )
    zeros = jnp.zeros((resolved_rank, channel_dim), dtype=mu_dtype)
    return APOLLOLeafState(
        projection,
        zeros,
        zeros,
        jnp.zeros([], dtype=mu_dtype),
        orientation,
        True,
        tuple(param.shape),
        leaf_index,
    )


def _update_leaf(
    grad: Any,
    state: Any,
    param: Any,
    *,
    count,
    count_inc,
    rank: int,
    projection_seed: int,
    projection_refresh_interval: int | None,
    scaling: Scaling,
    scale: float,
    scale_front: bool,
    disable_norm_growth_limiter: bool,
    norm_growth_limiter: float,
    b1,
    b2,
    eps,
    eps_root,
    weight_decay,
    learning_rate,
    mu_dtype,
) -> _LeafUpdateResult:
    if _is_passthrough(grad) or _is_passthrough(state):
        return _LeafUpdateResult(grad, state)
    if not isinstance(state, APOLLOLeafState) or not state.projected:
        return _full_adam_leaf(
            grad,
            state,
            param,
            count_inc=count_inc,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            scale=scale,
            scale_front=scale_front,
            disable_norm_growth_limiter=disable_norm_growth_limiter,
            norm_growth_limiter=norm_growth_limiter,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            mu_dtype=mu_dtype,
        )

    projection = _refresh_projection(
        state,
        count=count,
        rank=rank,
        projection_seed=projection_seed,
        projection_refresh_interval=projection_refresh_interval,
    )
    projected_grad = _project(grad, projection, state.orientation)
    mu = (b1 * state.mu + (1.0 - b1) * projected_grad).astype(mu_dtype)
    nu = (b2 * state.nu + (1.0 - b2) * jnp.square(projected_grad)).astype(mu_dtype)
    step_size = _reference_step_size(learning_rate, count_inc, b1, b2)
    projected_update = mu.astype(jnp.float32) / (
        jnp.sqrt(nu.astype(jnp.float32) + eps_root) + eps
    )
    factors = _scaling_factors(
        projected_grad,
        projected_update,
        scaling=scaling,
        eps=SCALING_FACTOR_EPS,
    )
    update = _apply_scaling(grad.astype(jnp.float32), factors, state.orientation)
    update, update_norm = _apply_reference_scale_and_limiter(
        update,
        previous_norm=state.prev_update_norm,
        scale=scale,
        scale_front=scale_front,
        disable_norm_growth_limiter=disable_norm_growth_limiter,
        norm_growth_limiter=norm_growth_limiter,
    )
    update = -step_size * update
    if param is not None and weight_decay != 0.0:
        update = update - learning_rate * weight_decay * param.astype(jnp.float32)
    update = update.astype(grad.dtype)
    return _LeafUpdateResult(
        update,
        APOLLOLeafState(
            projection,
            mu,
            nu,
            update_norm.astype(mu_dtype),
            state.orientation,
            state.projected,
            state.shape,
            state.leaf_index,
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
    scale,
    scale_front,
    disable_norm_growth_limiter,
    norm_growth_limiter,
    weight_decay,
    learning_rate,
    mu_dtype,
) -> _LeafUpdateResult:
    mu = (b1 * state.mu + (1.0 - b1) * grad).astype(mu_dtype)
    nu = (b2 * state.nu + (1.0 - b2) * jnp.square(grad)).astype(mu_dtype)
    step_size = _reference_step_size(learning_rate, count_inc, b1, b2)
    update = mu.astype(jnp.float32) / (
        jnp.sqrt(nu.astype(jnp.float32) + eps_root) + eps
    )
    update, update_norm = _apply_reference_scale_and_limiter(
        update,
        previous_norm=state.prev_update_norm,
        scale=scale,
        scale_front=scale_front,
        disable_norm_growth_limiter=disable_norm_growth_limiter,
        norm_growth_limiter=norm_growth_limiter,
    )
    update = -step_size * update
    if param is not None and weight_decay != 0.0:
        update = update - learning_rate * weight_decay * param.astype(jnp.float32)
    update = update.astype(grad.dtype)
    return _LeafUpdateResult(
        update,
        APOLLOLeafState(
            state.projection,
            mu,
            nu,
            update_norm.astype(mu_dtype),
            state.orientation,
            state.projected,
            state.shape,
            state.leaf_index,
        ),
    )


def _reference_step_size(learning_rate, count_inc, b1, b2):
    bias_correction1 = 1.0 - b1**count_inc
    bias_correction2 = 1.0 - b2**count_inc
    return learning_rate * jnp.sqrt(bias_correction2) / bias_correction1


def _refresh_projection(
    state: APOLLOLeafState,
    *,
    count,
    rank: int,
    projection_seed: int,
    projection_refresh_interval: int | None,
) -> jax.Array:
    if projection_refresh_interval is None:
        return state.projection
    projected_dim = state.projection.shape[1]
    resolved_rank = min(rank, projected_dim)
    new_projection = _make_projection(
        projection_seed=projection_seed,
        leaf_index=state.leaf_index,
        step=count,
        shape=(resolved_rank, projected_dim),
        dtype=state.projection.dtype,
    )
    should_refresh = (count % projection_refresh_interval) == 0
    return jnp.where(should_refresh, new_projection, state.projection)


def _make_projection(
    *,
    projection_seed: int,
    leaf_index: int,
    step,
    shape: tuple[int, int],
    dtype,
) -> jax.Array:
    key = jax.random.PRNGKey(projection_seed)
    key = jax.random.fold_in(key, leaf_index)
    key = jax.random.fold_in(key, step)
    return jax.random.normal(key, shape=shape, dtype=dtype) / math.sqrt(shape[0])


def _project(
    grad: jax.Array,
    projection: jax.Array,
    orientation: Orientation,
) -> jax.Array:
    if orientation == "left":
        return projection.astype(grad.dtype) @ grad
    if orientation == "right":
        return (grad @ projection.T.astype(grad.dtype)).T
    raise ValueError("full fallback leaves are not projected.")


def _scaling_factors(
    projected_grad: jax.Array,
    projected_update: jax.Array,
    *,
    scaling: Scaling,
    eps,
) -> jax.Array:
    if scaling == "tensor":
        return jnp.linalg.norm(projected_update) / (
            jnp.linalg.norm(projected_grad) + eps
        )
    update_norm = jnp.linalg.norm(projected_update, axis=0)
    grad_norm = jnp.linalg.norm(projected_grad, axis=0)
    return update_norm / (grad_norm + eps)


def _apply_scaling(
    grad: jax.Array,
    factors: jax.Array,
    orientation: Orientation,
) -> jax.Array:
    if factors.ndim == 0:
        return grad * factors
    if orientation == "left":
        return grad * factors[None, :]
    if orientation == "right":
        return grad * factors[:, None]
    raise ValueError("full fallback leaves are not APOLLO-scaled.")


def _apply_reference_scale_and_limiter(
    update: jax.Array,
    *,
    previous_norm: jax.Array,
    scale: float,
    scale_front: bool,
    disable_norm_growth_limiter: bool,
    norm_growth_limiter: float,
) -> tuple[jax.Array, jax.Array]:
    scale_factor = math.sqrt(scale)
    update = update * scale_factor if scale_front else update
    if disable_norm_growth_limiter:
        tracked_norm = jnp.linalg.norm(update)
        return (update if scale_front else update * scale_factor), tracked_norm
    update, tracked_norm = _limit_norm_growth(
        update,
        previous_norm=previous_norm,
        norm_growth_limiter=norm_growth_limiter,
    )
    return (update if scale_front else update * scale_factor), tracked_norm


def _limit_norm_growth(
    update: jax.Array,
    *,
    previous_norm: jax.Array,
    norm_growth_limiter: float,
) -> tuple[jax.Array, jax.Array]:
    update_norm = jnp.linalg.norm(update)
    max_norm = previous_norm * norm_growth_limiter
    should_limit = (previous_norm > 0.0) & (update_norm > max_norm)
    limited = update / (update_norm + jnp.finfo(update.dtype).tiny) * max_norm
    limited_update = jnp.where(should_limit, limited, update)
    tracked_norm = jnp.where(should_limit, max_norm, update_norm)
    return limited_update, tracked_norm


def _eligible(param: Any) -> bool:
    return hasattr(param, "shape") and len(param.shape) == 2


def _resolve_orientation(rows: int, cols: int) -> Literal["left", "right"]:
    return "right" if rows >= cols else "left"


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _tree_map_with_index(fn, tree):
    leaves, treedef = jax.tree.flatten(tree, is_leaf=_is_passthrough)
    mapped = [fn(index, leaf) for index, leaf in enumerate(leaves)]
    return jax.tree.unflatten(treedef, mapped)


def _is_passthrough(value: Any) -> bool:
    return value is None or isinstance(value, _masking.MaskedNode)


def _is_state_or_passthrough(value: Any) -> bool:
    return isinstance(value, APOLLOLeafState) or _is_passthrough(value)


__all__ = (
    "APOLLOLeafState",
    "ScaleByAPOLLOState",
    "apollo_adamw",
    "apollo_state_nbytes",
)
