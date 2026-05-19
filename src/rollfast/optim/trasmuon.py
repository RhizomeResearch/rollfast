"""TrasMuon optimizer.

TrasMuon is a Muon-style matrix optimizer with explicit magnitude controls. It
orthogonalizes first momentum with Newton-Schulz, calibrates each matrix update
to a global RMS target, and damps high-energy columns with a trust-region clip.
Non-matrix leaves are routed to AdamW by the public ``trasmuon`` helper.
"""

from typing import Any, Callable, NamedTuple, cast

import jax
import jax.numpy as jnp
from optax._src import base, combine, numerics, transform, utils

from rollfast.optim.adam import adamw
from rollfast.optim.dimension_numbers import (
    MatrixDimensionNumbers,
    WeightDimNumOrFn,
    _compute_matrix_reshape,
    _get_dimension_numbers,
    _is_dimension_numbers_leaf,
    _make_matrix_partition_fns,
)
from rollfast.optim.muon import (
    MuonDimensionNumbers,
)
from rollfast.optim.orthogonalization import (
    MUON_NS_COEFFS,
    MuonNsCoeffs,
    MuonPreconditioning,
    orthogonalize_via_newton_schulz,
    resolve_ns_coeffs,
)
from rollfast.utils import (
    MomentumAccumulator,
    _cast_state_tree,
    _is_aux_leaf,
    _tree_stochastic_cast,
    _tree_update_moment_f32,
    _zeros_like_tree,
)


class ScaleByTrasMuonState(NamedTuple):
    """State for TrasMuon's matrix branch."""

    count: jax.Array
    mu: base.Updates
    v_row: base.Updates
    energy_ref: base.Updates
    clip_ema: base.Updates
    clip_last: base.Updates
    key: jax.Array | None


def _stat_shapes(
    param: Any,
    dim_nums: MatrixDimensionNumbers | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    if _is_aux_leaf(param) or dim_nums is None:
        return None
    reshape_fn, _ = _compute_matrix_reshape(param, dim_nums)
    matrix = reshape_fn(param)
    return (
        matrix.shape[:-1] + (1,),
        matrix.shape[:-2] + (1, 1),
        matrix.shape[:-2]
        + (
            1,
            matrix.shape[-1],
        ),
    )


def _zeros_for_v_row(
    param: Any,
    dim_nums: MatrixDimensionNumbers | None,
    dtype: jax.typing.DTypeLike,
) -> Any:
    shapes = _stat_shapes(param, dim_nums)
    if shapes is None:
        return param
    return jnp.zeros(shapes[0], dtype=dtype)


def _zeros_for_energy_ref(
    param: Any,
    dim_nums: MatrixDimensionNumbers | None,
    dtype: jax.typing.DTypeLike,
) -> Any:
    shapes = _stat_shapes(param, dim_nums)
    if shapes is None:
        return param
    return jnp.zeros(shapes[1], dtype=dtype)


def _ones_for_clip(
    param: Any,
    dim_nums: MatrixDimensionNumbers | None,
    dtype: jax.typing.DTypeLike,
) -> Any:
    shapes = _stat_shapes(param, dim_nums)
    if shapes is None:
        return param
    return jnp.ones(shapes[2], dtype=dtype)


def _orthogonalize_muon(
    x: jax.Array,
    ns_iters: int,
    eps: jax.typing.ArrayLike,
    ns_coeffs: jax.Array,
    preconditioning: MuonPreconditioning,
) -> jax.Array:
    return orthogonalize_via_newton_schulz(
        x,
        ns_coeffs,
        ns_steps=ns_iters,
        preconditioning=preconditioning,
        eps=eps,
        dimension_numbers=MuonDimensionNumbers(reduction_axis=-2, output_axis=-1),
    )


def _trasmuon_leaf_update(
    mu: Any,
    dim_nums: MatrixDimensionNumbers | None,
    v_row: Any,
    energy_ref: Any,
    clip_ema: Any,
    clip_last: Any,
    count: jax.Array,
    *,
    beta2: jax.typing.ArrayLike,
    eps: jax.typing.ArrayLike,
    ns_iters: int,
    ns_coeffs: jax.Array,
    preconditioning: MuonPreconditioning,
    clip_alpha: jax.typing.ArrayLike,
    clip_beta: jax.typing.ArrayLike,
    energy_beta: jax.typing.ArrayLike,
    clip_min: jax.typing.ArrayLike,
    trigger: jax.typing.ArrayLike,
    update_period: int,
    warmup_steps: int,
    mix: jax.typing.ArrayLike,
) -> tuple[Any, Any, Any, Any, Any]:
    if _is_aux_leaf(mu) or dim_nums is None:
        return mu, v_row, energy_ref, clip_ema, clip_last
    if mu.ndim < 2:
        raise ValueError(
            f"TrasMuon optimized parameters must have rank >= 2, got {mu.ndim=}."
        )

    reshape_fn, inverse_fn = _compute_matrix_reshape(mu, dim_nums)
    matrix = reshape_fn(mu).astype(jnp.float32)
    eps32 = jnp.asarray(eps, dtype=jnp.float32)

    ortho = _orthogonalize_muon(matrix, ns_iters, eps32, ns_coeffs, preconditioning)
    beta2_32 = jnp.asarray(beta2, dtype=jnp.float32)
    v_row_new = beta2_32 * v_row.astype(jnp.float32) + (1.0 - beta2_32) * jnp.mean(
        numerics.abs_sq(ortho), axis=-1, keepdims=True
    )
    base_direction = ortho / jnp.sqrt(v_row_new + eps32)

    rows = matrix.shape[-2]
    cols = matrix.shape[-1]
    target_norm = jnp.sqrt(jnp.asarray(rows * cols, dtype=jnp.float32))
    direction_norm = jnp.linalg.norm(base_direction, axis=(-2, -1), keepdims=True)
    calibrated_direction = base_direction * (target_norm / (direction_norm + eps32))

    energy = jnp.sum(numerics.abs_sq(matrix), axis=-2, keepdims=True)
    energy_cur = jnp.median(energy, axis=-1, keepdims=True)
    energy_beta32 = jnp.asarray(energy_beta, dtype=jnp.float32)
    energy_ref_new = jnp.where(
        count <= 1,
        energy_cur,
        energy_beta32 * energy_ref.astype(jnp.float32)
        + (1.0 - energy_beta32) * energy_cur,
    )

    trigger32 = jnp.asarray(trigger, dtype=jnp.float32)
    ratio = energy / (energy_ref_new + eps32)
    active_ratio = jnp.maximum(ratio / jnp.maximum(trigger32, eps32), 1.0)
    clip_raw = jnp.where(
        ratio <= trigger32,
        jnp.ones_like(ratio),
        1.0 / (1.0 + jnp.asarray(clip_alpha, jnp.float32) * jnp.log1p(active_ratio)),
    )
    clip_raw = jnp.clip(clip_raw, jnp.asarray(clip_min, jnp.float32), 1.0)

    clip_beta32 = jnp.asarray(clip_beta, dtype=jnp.float32)
    clip_ema_new = (
        clip_beta32 * clip_ema.astype(jnp.float32) + (1.0 - clip_beta32) * clip_raw
    )

    period = max(1, int(update_period))
    past_warmup = count > jnp.asarray(warmup_steps, dtype=count.dtype)
    period_step = jnp.equal(
        jnp.mod(count - jnp.asarray(warmup_steps, dtype=count.dtype), period), 0
    )
    update_clip = jnp.logical_and(past_warmup, period_step)
    clip_candidate = (1.0 - jnp.asarray(mix, dtype=jnp.float32)) * clip_last.astype(
        jnp.float32
    ) + jnp.asarray(mix, dtype=jnp.float32) * clip_ema_new
    clip_last_new = jnp.where(
        update_clip, clip_candidate, clip_last.astype(jnp.float32)
    )
    clipped_direction = calibrated_direction * clip_last_new

    return (
        inverse_fn(clipped_direction).astype(mu.dtype),
        v_row_new,
        energy_ref_new,
        clip_ema_new,
        clip_last_new,
    )


def scale_by_trasmuon(
    beta1: jax.typing.ArrayLike = 0.95,
    beta2: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    ns_iters: int = 5,
    ns_coeffs: MuonNsCoeffs = MUON_NS_COEFFS,
    *,
    clip_alpha: jax.typing.ArrayLike = 1.0,
    energy_beta: jax.typing.ArrayLike = 0.95,
    clip_beta: jax.typing.ArrayLike = 0.9,
    clip_min: jax.typing.ArrayLike = 0.1,
    trigger: jax.typing.ArrayLike = 1.0,
    update_period: int = 1,
    warmup_steps: int = 0,
    mix: jax.typing.ArrayLike = 1.0,
    preconditioning: MuonPreconditioning = "frobenius",
    momentum_accumulator: MomentumAccumulator = "ema",
    mu_dtype: jax.typing.DTypeLike | None = None,
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """Scale updates with TrasMuon's matrix direction.

    The returned updates are positive descent directions. Compose with
    ``optax.scale_by_learning_rate`` or use ``trasmuon`` for a complete
    optimizer with AdamW fallback.
    """
    if ns_iters < 1:
        raise ValueError(f"ns_iters must be >= 1, got {ns_iters}")
    if update_period < 1:
        raise ValueError(f"update_period must be >= 1, got {update_period}")

    canonical_mu_dtype = cast(
        jax.typing.DTypeLike,
        jnp.float32 if mu_dtype is None else utils.canonicalize_dtype(mu_dtype),
    )

    def resolve_dim_nums(params_or_updates):
        return _get_dimension_numbers(weight_dimension_numbers, params_or_updates)

    def init_fn(params):
        dim_nums = resolve_dim_nums(params)
        return ScaleByTrasMuonState(
            count=jnp.zeros([], jnp.int32),
            mu=_zeros_like_tree(params, canonical_mu_dtype),
            v_row=jax.tree.map(
                lambda p, d: _zeros_for_v_row(p, d, jnp.float32),
                params,
                dim_nums,
                is_leaf=_is_dimension_numbers_leaf,
            ),
            energy_ref=jax.tree.map(
                lambda p, d: _zeros_for_energy_ref(p, d, jnp.float32),
                params,
                dim_nums,
                is_leaf=_is_dimension_numbers_leaf,
            ),
            clip_ema=jax.tree.map(
                lambda p, d: _ones_for_clip(p, d, jnp.float32),
                params,
                dim_nums,
                is_leaf=_is_dimension_numbers_leaf,
            ),
            clip_last=jax.tree.map(
                lambda p, d: _ones_for_clip(p, d, jnp.float32),
                params,
                dim_nums,
                is_leaf=_is_dimension_numbers_leaf,
            ),
            key=key,
        )

    def update_fn(updates, state, params=None):
        dim_source = updates if params is None else params
        dim_nums = resolve_dim_nums(dim_source)

        mu = _tree_update_moment_f32(
            updates,
            state.mu,
            beta1,
            momentum_accumulator=momentum_accumulator,
        )
        count_inc = numerics.safe_increment(state.count)

        resolved_ns_coeffs = resolve_ns_coeffs(ns_coeffs, ns_iters)
        tras_result = jax.tree.map(
            lambda m, d, v, e, ce, cl: _trasmuon_leaf_update(
                m,
                d,
                v,
                e,
                ce,
                cl,
                cast(jax.Array, count_inc),
                beta2=beta2,
                eps=eps,
                ns_iters=ns_iters,
                ns_coeffs=resolved_ns_coeffs,
                preconditioning=preconditioning,
                clip_alpha=clip_alpha,
                clip_beta=clip_beta,
                energy_beta=energy_beta,
                clip_min=clip_min,
                trigger=trigger,
                update_period=update_period,
                warmup_steps=warmup_steps,
                mix=mix,
            ),
            mu,
            dim_nums,
            state.v_row,
            state.energy_ref,
            state.clip_ema,
            state.clip_last,
            is_leaf=_is_dimension_numbers_leaf,
        )
        result_is_leaf = lambda x: isinstance(x, tuple) and len(x) == 5
        tras_updates = jax.tree.map(lambda x: x[0], tras_result, is_leaf=result_is_leaf)
        v_row = jax.tree.map(lambda x: x[1], tras_result, is_leaf=result_is_leaf)
        energy_ref = jax.tree.map(lambda x: x[2], tras_result, is_leaf=result_is_leaf)
        clip_ema = jax.tree.map(lambda x: x[3], tras_result, is_leaf=result_is_leaf)
        clip_last = jax.tree.map(lambda x: x[4], tras_result, is_leaf=result_is_leaf)

        if canonical_mu_dtype == jnp.bfloat16:
            key, sr_key = jax.random.split(cast(jax.Array, state.key), 2)
            mu = _tree_stochastic_cast(mu, canonical_mu_dtype, sr_key)
        else:
            key = state.key
            mu = _cast_state_tree(mu, canonical_mu_dtype)

        return tras_updates, ScaleByTrasMuonState(
            count=cast(jax.Array, count_inc),
            mu=mu,
            v_row=v_row,
            energy_ref=energy_ref,
            clip_ema=clip_ema,
            clip_last=clip_last,
            key=key,
        )

    return base.GradientTransformation(init_fn, update_fn)


def trasmuon(
    learning_rate: base.ScalarOrSchedule,
    beta1: jax.typing.ArrayLike = 0.95,
    beta2: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    ns_iters: int = 5,
    ns_coeffs: MuonNsCoeffs = MUON_NS_COEFFS,
    *,
    clip_alpha: jax.typing.ArrayLike = 1.0,
    energy_beta: jax.typing.ArrayLike = 0.95,
    clip_beta: jax.typing.ArrayLike = 0.9,
    clip_min: jax.typing.ArrayLike = 0.1,
    trigger: jax.typing.ArrayLike = 1.0,
    update_period: int = 1,
    warmup_steps: int = 0,
    mix: jax.typing.ArrayLike = 1.0,
    preconditioning: MuonPreconditioning = "frobenius",
    momentum_accumulator: MomentumAccumulator = "ema",
    mu_dtype: jax.typing.DTypeLike | None = None,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: base.ScalarOrSchedule = 0.0,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    trasmuon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """TrasMuon optimizer with automatic matrix/AdamW partitioning."""
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    key_trasmuon, key_adam = jax.random.split(key, 2)

    partition = _make_matrix_partition_fns(
        trasmuon_weight_dimension_numbers,
        "trasmuon",
    )

    return combine.partition(
        transforms={
            "trasmuon": combine.chain(
                scale_by_trasmuon(
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    ns_iters=ns_iters,
                    ns_coeffs=ns_coeffs,
                    clip_alpha=clip_alpha,
                    energy_beta=energy_beta,
                    clip_beta=clip_beta,
                    clip_min=clip_min,
                    trigger=trigger,
                    update_period=update_period,
                    warmup_steps=warmup_steps,
                    mix=mix,
                    preconditioning=preconditioning,
                    momentum_accumulator=momentum_accumulator,
                    mu_dtype=mu_dtype,
                    weight_dimension_numbers=partition.masked_specs,
                    key=key_trasmuon,
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
                key=key_adam,
            ),
        },
        param_labels=partition.labels,
    )


__all__ = [
    "ScaleByTrasMuonState",
    "scale_by_trasmuon",
    "trasmuon",
]
