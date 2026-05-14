import math
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from optax._src import base, combine, numerics, transform, utils
from optax.transforms import _masking

from rollfast.optim.adam import adamw
from rollfast.optim.prism import (
    PrismDimensionNumbers,
    WeightDimNumOrFn,
    _compute_prism_reshape,
    _get_dimension_numbers,
    _is_prism_leaf,
    _mask_dimension_numbers,
    _normalize_axes,
)
from rollfast.utils import _tree_stochastic_cast

"""RMNP usage notes.

RMNP is a Muon-style matrix optimizer that replaces Newton-Schulz
orthogonalization with row-wise L2 normalization of the momentum matrix. This
makes the matrix branch linear in the number of parameters, but it also means
the row axis is part of the algorithm. For ordinary 2D leaves the default
``PrismDimensionNumbers(reduction_axis=0, output_axis=1)`` normalizes rows in
that flattened matrix layout. For Equinox ``Linear`` weights, convolution
kernels, transposed weights, or embeddings whose semantic axes differ from that
layout, pass explicit ``rmnp_weight_dimension_numbers``.

RMNP is an additive optimizer, unlike Pion, so it composes naturally with
ordinary Optax wrappers such as SODA and Hyperball. Large embedding tables and
LM heads can be routed to the AdamW fallback when row normalization is not the
intended geometry. As with Muon, learning rates for matrix leaves often need
their own tuning; the Adam fallback can use ``adam_learning_rate`` separately.
"""


class ScaleByRmnpState(NamedTuple):
    """State for RMNP's first-moment estimates."""

    count: jax.Array
    mu: base.Updates
    key: jax.Array | None


def _is_aux_leaf(x: Any) -> bool:
    return x is None or isinstance(x, _masking.MaskedNode)


def _zeros_like_tree(params: base.Params, dtype: jax.typing.DTypeLike) -> base.Params:
    return jax.tree.map(
        lambda x: x if _is_aux_leaf(x) else jnp.zeros_like(x, dtype=dtype),
        params,
        is_leaf=_is_aux_leaf,
    )


def _cast_state_tree(tree: base.Params, dtype: jax.typing.DTypeLike) -> base.Params:
    return jax.tree.map(
        lambda x: x if _is_aux_leaf(x) else x.astype(dtype),
        tree,
        is_leaf=_is_aux_leaf,
    )


def _bias_correction(moment: Any, beta: jax.typing.ArrayLike, count: jax.Array) -> Any:
    correction = 1.0 - jnp.power(jnp.asarray(beta, dtype=jnp.float32), count)
    return jax.tree.map(
        lambda m: m if _is_aux_leaf(m) else m.astype(jnp.float32) / correction,
        moment,
        is_leaf=_is_aux_leaf,
    )


def _update_moment(
    updates: base.Updates,
    moment: base.Updates,
    beta: jax.typing.ArrayLike,
) -> base.Updates:
    beta32 = jnp.asarray(beta, dtype=jnp.float32)
    return jax.tree.map(
        lambda g, m: (
            g
            if _is_aux_leaf(g) or _is_aux_leaf(m)
            else beta32 * m.astype(jnp.float32) + (1.0 - beta32) * g.astype(jnp.float32)
        ),
        updates,
        moment,
        is_leaf=_is_aux_leaf,
    )


def _row_normalize_matrix(
    x: jax.Array,
    dim_nums: PrismDimensionNumbers,
    eps: jax.typing.ArrayLike,
) -> jax.Array:
    reshape_fn, inverse_fn = _compute_prism_reshape(x, dim_nums)
    matrix = reshape_fn(x).astype(jnp.float32)
    eps32 = jnp.asarray(eps, dtype=jnp.float32)
    row_norm = jnp.linalg.norm(matrix, axis=-1, keepdims=True)
    normalized = matrix / jnp.maximum(row_norm, eps32)
    return inverse_fn(normalized).astype(x.dtype)


def _rmnp_leaf_update(
    update: Any,
    dim_nums: PrismDimensionNumbers | None,
    eps: jax.typing.ArrayLike,
) -> Any:
    if _is_aux_leaf(update) or dim_nums is None:
        return update
    if update.ndim < 2:
        raise ValueError(
            f"RMNP optimized parameters must have rank >= 2, got {update.ndim=}."
        )
    return _row_normalize_matrix(update, dim_nums, eps)


def _shape_scale_leaf(
    update: Any,
    dim_nums: PrismDimensionNumbers | None,
    consistent_rms: jax.typing.ArrayLike | None,
) -> Any:
    if _is_aux_leaf(update) or dim_nums is None:
        return update

    reduction_axes, output_axes = _normalize_axes(update, dim_nums)
    fan_in = math.prod(update.shape[axis] for axis in reduction_axes)
    fan_out = math.prod(update.shape[axis] for axis in output_axes)

    if consistent_rms is None:
        scale = jnp.sqrt(jnp.maximum(1.0, fan_out / fan_in))
    else:
        scale = (
            jnp.sqrt(jnp.asarray(max(fan_in, fan_out), dtype=jnp.float32))
            * consistent_rms
        )
    return update * scale


def _resolve_dimension_numbers(
    weight_dimension_numbers: WeightDimNumOrFn | None,
    params_or_updates: base.Params,
) -> base.Params:
    return _get_dimension_numbers(weight_dimension_numbers, params_or_updates)


def scale_by_rmnp(
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype: jax.typing.DTypeLike | None = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """Scale updates with Row-Momentum Normalized Preconditioning.

    This transform tracks a first moment, optionally forms a Nesterov lookahead
    moment, and normalizes each matrix row in the layout described by
    ``weight_dimension_numbers``. It returns a positive, unscaled descent
    direction; use ``rmnp`` for a full optimizer with learning-rate scaling and
    AdamW fallback.
    """
    if mu_dtype is None:
        mu_dtype = jnp.float32
    else:
        mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        return ScaleByRmnpState(
            count=jnp.zeros([], jnp.int32),
            mu=_zeros_like_tree(params, mu_dtype),
            key=key,
        )

    def update_fn(updates, state, params=None):
        dim_source = updates if params is None else params
        dim_nums = _resolve_dimension_numbers(weight_dimension_numbers, dim_source)

        mu = _update_moment(updates, state.mu, beta)
        count_inc = numerics.safe_increment(state.count)

        if nesterov:
            mu_bc = _bias_correction(mu, beta, numerics.safe_increment(count_inc))
            updates_bc = _bias_correction(updates, beta, count_inc)
            beta32 = jnp.asarray(beta, dtype=jnp.float32)
            direction = jax.tree.map(
                lambda m, g: (
                    m
                    if _is_aux_leaf(m) or _is_aux_leaf(g)
                    else beta32 * m + (1.0 - beta32) * g
                ),
                mu_bc,
                updates_bc,
                is_leaf=_is_aux_leaf,
            )
        else:
            direction = _bias_correction(mu, beta, count_inc)

        rmnp_updates = jax.tree.map(
            lambda u, d: _rmnp_leaf_update(u, d, eps),
            direction,
            dim_nums,
            is_leaf=_is_prism_leaf,
        )
        if adaptive:
            rmnp_updates = jax.tree.map(
                lambda x, y: (
                    x if _is_aux_leaf(x) or _is_aux_leaf(y) else jnp.sum(x * y) * y
                ),
                direction,
                rmnp_updates,
                is_leaf=_is_aux_leaf,
            )

        if mu_dtype == jnp.bfloat16:
            key, sr_key = jax.random.split(state.key, 2)
            mu = _tree_stochastic_cast(mu, mu_dtype, sr_key)
        else:
            key = state.key
            mu = _cast_state_tree(mu, mu_dtype)

        return rmnp_updates, ScaleByRmnpState(count=count_inc, mu=mu, key=key)

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_rmnp_shape(
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformation:
    """Scale RMNP matrix directions using Muon-style shape factors."""

    def update_fn(updates, state, params=None):
        dim_source = updates if params is None else params
        dim_nums = _resolve_dimension_numbers(weight_dimension_numbers, dim_source)
        scaled_updates = jax.tree.map(
            lambda u, d: _shape_scale_leaf(u, d, consistent_rms),
            updates,
            dim_nums,
            is_leaf=_is_prism_leaf,
        )
        return scaled_updates, state

    return base.GradientTransformation(base.init_empty_state, update_fn)


def rmnp(
    learning_rate: base.ScalarOrSchedule,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    mu_dtype: jax.typing.DTypeLike | None = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: base.ScalarOrSchedule = 0.0,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    rmnp_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """RMNP optimizer with AdamW fallback for non-matrix parameters."""
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    key_rmnp, key_adam = jax.random.split(key, 2)

    def get_resolved_dim_nums(params):
        return _get_dimension_numbers(rmnp_weight_dimension_numbers, params)

    def param_labels(params):
        dim_nums = get_resolved_dim_nums(params)
        return jax.tree.map(
            lambda d, p: None if p is None else ("rmnp" if d is not None else "adam"),
            dim_nums,
            params,
            is_leaf=_is_prism_leaf,
        )

    def rmnp_weight_dim_nums_fn(params):
        return _mask_dimension_numbers(get_resolved_dim_nums(params))

    return combine.partition(
        transforms={
            "rmnp": combine.chain(
                scale_by_rmnp(
                    beta=beta,
                    eps=eps,
                    mu_dtype=mu_dtype,
                    nesterov=nesterov,
                    adaptive=adaptive,
                    weight_dimension_numbers=rmnp_weight_dim_nums_fn,
                    key=key_rmnp,
                ),
                scale_by_rmnp_shape(
                    weight_dimension_numbers=rmnp_weight_dim_nums_fn,
                    consistent_rms=consistent_rms,
                ),
                transform.add_decayed_weights(weight_decay, weight_decay_mask),
                transform.scale_by_learning_rate(learning_rate),
            ),
            "adam": adamw(
                learning_rate=adam_learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=eps,
                eps_root=adam_eps_root,
                weight_decay=adam_weight_decay,
                weight_decay_mask=weight_decay_mask,
                mu_dtype=mu_dtype,
                nesterov=nesterov,
                key=key_adam,
            ),
        },
        param_labels=param_labels,
    )


__all__ = [
    "ScaleByRmnpState",
    "rmnp",
    "scale_by_rmnp",
    "scale_by_rmnp_shape",
]
