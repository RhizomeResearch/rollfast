"""RMNP usage notes.

RMNP is a Muon-style matrix optimizer that replaces Newton-Schulz
orthogonalization with row-wise L2 normalization of the momentum matrix. This
makes the matrix branch linear in the number of parameters, but it also means
the row axis is part of the algorithm. For ordinary 2D leaves the default
``MatrixDimensionNumbers(reduction_axis=0, output_axis=1)`` normalizes rows in
that flattened matrix layout. For Equinox ``Linear`` weights, convolution
kernels, transposed weights, or embeddings whose semantic axes differ from that
layout, pass explicit ``rmnp_weight_dimension_numbers``.

RMNP is an additive optimizer, unlike Pion, so it composes naturally with
ordinary Optax wrappers such as SODA and Hyperball. Large embedding tables and
LM heads can be routed to the AdamW fallback when row normalization is not the
intended geometry. As with Muon, learning rates for matrix leaves often need
their own tuning; the Adam fallback can use ``adam_learning_rate`` separately.
"""

import math
from typing import Any, Callable, NamedTuple, cast

import jax
import jax.numpy as jnp
from optax._src import base, combine, numerics, transform, utils

from rollfast.optim.adam import adamw
from rollfast.optim.dimension_numbers import (
    MatrixDimensionNumbers,
    WeightDimNumOrFn,
    _compute_matrix_reshape,
    _is_dimension_numbers_leaf,
    _make_matrix_partition_fns,
    _normalize_axes,
    _resolve_update_dimension_numbers,
)
from rollfast.utils import (
    MomentumAccumulator,
    _cast_state_tree,
    _has_nonzero_or_scheduled,
    _is_aux_leaf,
    _tree_bias_correction_momentum,
    _tree_momentum_lookahead,
    _tree_stochastic_cast,
    _tree_update_moment_f32,
    _zeros_like_tree,
)


class ScaleByRmnpState(NamedTuple):
    """State for RMNP's first-moment estimates."""

    count: jax.Array
    mu: base.Updates
    key: jax.Array | None


def _row_normalize_matrix(
    x: jax.Array,
    dim_nums: MatrixDimensionNumbers,
    eps: jax.typing.ArrayLike,
) -> jax.Array:
    reshape_fn, inverse_fn = _compute_matrix_reshape(x, dim_nums)
    matrix = reshape_fn(x).astype(jnp.float32)
    eps32 = jnp.asarray(eps, dtype=jnp.float32)
    row_norm = jnp.linalg.norm(matrix, axis=-1, keepdims=True)
    normalized = matrix / jnp.maximum(row_norm, eps32)
    return inverse_fn(normalized).astype(x.dtype)


def _rmnp_leaf_update(
    update: Any,
    dim_nums: MatrixDimensionNumbers | None,
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
    dim_nums: MatrixDimensionNumbers | None,
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
    *,
    params: base.Params | None,
    updates: base.Updates,
    transform_name: str,
) -> base.Params:
    return _resolve_update_dimension_numbers(
        weight_dimension_numbers,
        params=params,
        updates=updates,
        transform_name=transform_name,
    )


def scale_by_rmnp(
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype: jax.typing.DTypeLike | None = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    momentum_accumulator: MomentumAccumulator = "ema",
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
    canonical_mu_dtype = cast(
        jax.typing.DTypeLike,
        jnp.float32 if mu_dtype is None else utils.canonicalize_dtype(mu_dtype),
    )

    def init_fn(params):
        return ScaleByRmnpState(
            count=jnp.zeros([], jnp.int32),
            mu=_zeros_like_tree(params, canonical_mu_dtype),
            key=key,
        )

    def update_fn(updates, state, params=None):
        dim_nums = _resolve_dimension_numbers(
            weight_dimension_numbers,
            params=params,
            updates=updates,
            transform_name="scale_by_rmnp",
        )

        mu = _tree_update_moment_f32(
            updates,
            state.mu,
            beta,
            momentum_accumulator=momentum_accumulator,
        )
        count_inc = numerics.safe_increment(state.count)

        if nesterov:
            mu_bc = _tree_bias_correction_momentum(
                mu,
                beta,
                numerics.safe_increment(count_inc),
                momentum_accumulator=momentum_accumulator,
            )
            updates_bc = _tree_bias_correction_momentum(
                updates,
                beta,
                count_inc,
                momentum_accumulator=momentum_accumulator,
            )
            direction = _tree_momentum_lookahead(
                mu_bc,
                updates_bc,
                beta,
                momentum_accumulator=momentum_accumulator,
            )
        else:
            direction = _tree_bias_correction_momentum(
                mu,
                beta,
                count_inc,
                momentum_accumulator=momentum_accumulator,
            )

        rmnp_updates = jax.tree.map(
            lambda u, d: _rmnp_leaf_update(u, d, eps),
            direction,
            dim_nums,
            is_leaf=_is_dimension_numbers_leaf,
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

        if canonical_mu_dtype == jnp.bfloat16:
            key, sr_key = jax.random.split(cast(jax.Array, state.key), 2)
            mu = _tree_stochastic_cast(mu, canonical_mu_dtype, sr_key)
        else:
            key = state.key
            mu = _cast_state_tree(mu, canonical_mu_dtype)

        return rmnp_updates, ScaleByRmnpState(
            count=cast(jax.Array, count_inc), mu=mu, key=key
        )

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_rmnp_shape(
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformation:
    """Scale RMNP matrix directions using Muon-style shape factors."""

    def update_fn(updates, state, params=None):
        dim_nums = _resolve_dimension_numbers(
            weight_dimension_numbers,
            params=params,
            updates=updates,
            transform_name="scale_by_rmnp_shape",
        )
        scaled_updates = jax.tree.map(
            lambda u, d: _shape_scale_leaf(u, d, consistent_rms),
            updates,
            dim_nums,
            is_leaf=_is_dimension_numbers_leaf,
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
    momentum_accumulator: MomentumAccumulator = "ema",
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: base.ScalarOrSchedule | None = None,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    rmnp_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """RMNP optimizer with AdamW fallback for non-matrix parameters."""
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate
    effective_adam_weight_decay = (
        weight_decay if adam_weight_decay is None else adam_weight_decay
    )

    key_rmnp, key_adam = jax.random.split(key, 2)

    partition = _make_matrix_partition_fns(rmnp_weight_dimension_numbers, "rmnp")
    rmnp_components = [
        scale_by_rmnp(
            beta=beta,
            eps=eps,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
            adaptive=adaptive,
            momentum_accumulator=momentum_accumulator,
            weight_dimension_numbers=partition.masked_specs,
            key=key_rmnp,
        ),
        scale_by_rmnp_shape(
            weight_dimension_numbers=partition.masked_specs,
            consistent_rms=consistent_rms,
        ),
    ]
    if _has_nonzero_or_scheduled(weight_decay):
        rmnp_components.append(
            transform.add_decayed_weights(weight_decay, weight_decay_mask)
        )
    rmnp_components.append(transform.scale_by_learning_rate(learning_rate))

    return combine.partition(
        transforms={
            "rmnp": combine.chain(*rmnp_components),
            "adam": adamw(
                learning_rate=adam_learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=eps,
                eps_root=adam_eps_root,
                weight_decay=effective_adam_weight_decay,
                weight_decay_mask=weight_decay_mask,
                mu_dtype=mu_dtype,
                nesterov=nesterov,
                key=key_adam,
            ),
        },
        param_labels=partition.labels,
    )


__all__ = [
    "ScaleByRmnpState",
    "rmnp",
    "scale_by_rmnp",
    "scale_by_rmnp_shape",
]
