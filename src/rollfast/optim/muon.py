"""Rollfast Muon optimizer.

This module vendors Optax 0.2.8's Muon implementation and adds Rollfast's
optimizer extras: FP32 accumulator math, stochastic BF16 momentum storage,
masked-tree-safe routing, Rollfast AdamW fallback, optional Magma, PolarExpress
coefficients, and a swappable orthogonalization hook.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
from optax._src import base, combine, transform, utils

from rollfast.optim.adam import adamw
from rollfast.optim.dimension_numbers import (
    MatrixDimensionNumbers as MuonDimensionNumbers,
)
from rollfast.optim.dimension_numbers import (
    WeightDimNumOrFn,
    _get_dimension_numbers,
    _is_dimension_numbers_leaf,
    _make_matrix_partition_fns,
    _normalize_axes,
    _resolve_update_dimension_numbers,
    _validate_matrix_operand,
)
from rollfast.optim.magma import apply_magma_internal, validate_magma_args
from rollfast.optim.orthogonalization import (
    MUON_NS_COEFFS,
    MuonNsCoeffs,
    MuonPreconditioning,
    OrthogonalizeFn,
    orthogonalize_via_newton_schulz,
    polar_express_coeffs,
    resolve_ns_coeffs,
)
from rollfast.utils import (
    MomentumAccumulator,
    _apply_weight_decay_tree,
    _has_nonzero_or_scheduled,
    _init_magma_state,
    _is_aux_leaf,
    _prepare_first_moment_runtime,
    _resolve_mask,
    _resolve_scalar,
    _validate_beta_static_scalar,
    _validate_positive_static_scalar,
    _zeros_like_tree,
)

_DEFAULT_NS_COEFFS = MUON_NS_COEFFS


class MuonState(NamedTuple):
    """State for the Muon matrix branch."""

    count: jax.Array
    mu: base.Updates
    ns_coeffs: jax.Array
    magma_s: Any
    key: jax.Array | None


def _get_shape_products(
    x: jax.Array, dim_nums: MuonDimensionNumbers
) -> tuple[int, int]:
    reduction_axes, output_axes = _normalize_axes(x, dim_nums)
    fan_in = math.prod(x.shape[axis] for axis in reduction_axes)
    fan_out = math.prod(x.shape[axis] for axis in output_axes)
    return fan_in, fan_out


def _scale_update_for_width_transfer(
    update: Any, dim_nums: MuonDimensionNumbers | None
) -> Any:
    if _is_aux_leaf(update) or dim_nums is None:
        return update
    fan_in, fan_out = _get_shape_products(update, dim_nums)
    return update * jnp.sqrt(jnp.maximum(1.0, fan_out / fan_in))


def _scale_update_for_consistent_rms(
    update: Any,
    dim_nums: MuonDimensionNumbers | None,
    consistent_rms: jax.typing.ArrayLike,
) -> Any:
    if _is_aux_leaf(update) or dim_nums is None:
        return update
    fan_in, fan_out = _get_shape_products(update, dim_nums)
    return (
        update
        * jnp.sqrt(jnp.asarray(max(fan_in, fan_out), dtype=jnp.float32))
        * consistent_rms
    )


def _scale_muon_shape_tree(
    updates: base.Updates,
    dim_nums: base.Params,
    consistent_rms: jax.typing.ArrayLike | None,
) -> base.Updates:
    if consistent_rms is None:
        return jax.tree.map(
            _scale_update_for_width_transfer,
            updates,
            dim_nums,
            is_leaf=_is_dimension_numbers_leaf,
        )
    return jax.tree.map(
        lambda u, d: _scale_update_for_consistent_rms(u, d, consistent_rms),
        updates,
        dim_nums,
        is_leaf=_is_dimension_numbers_leaf,
    )


def scale_by_muon_shape(
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformation:
    """Scale Muon updates by width-transfer or consistent-RMS shape factors."""
    _validate_positive_static_scalar("consistent_rms", consistent_rms)

    def update_fn(updates, state, params=None):
        dim_nums = _resolve_update_dimension_numbers(
            weight_dimension_numbers,
            params=params,
            updates=updates,
            transform_name="scale_by_muon_shape",
        )
        return _scale_muon_shape_tree(updates, dim_nums, consistent_rms), state

    return base.GradientTransformation(base.init_empty_state, update_fn)


scale_by_shape = scale_by_muon_shape


_resolve_ns_coeffs = resolve_ns_coeffs


def _call_orthogonalize(
    orthogonalize_fn: OrthogonalizeFn,
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike,
    preconditioning: MuonPreconditioning,
    eps: jax.typing.ArrayLike,
    dim_nums: MuonDimensionNumbers | None,
) -> jax.Array:
    if _is_aux_leaf(x) or dim_nums is None:
        return x
    _validate_matrix_operand(x, dim_nums, "scale_by_muon")
    return orthogonalize_fn(x, ns_coeffs, ns_steps, preconditioning, eps, dim_nums)


def scale_by_muon(
    ns_coeffs: MuonNsCoeffs = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype: jax.typing.DTypeLike | None = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    preconditioning: MuonPreconditioning = "frobenius",
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    orthogonalize_fn: OrthogonalizeFn = orthogonalize_via_newton_schulz,
    momentum_accumulator: MomentumAccumulator = "ema",
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    shape_updates: bool = False,
    consistent_rms: jax.typing.ArrayLike | None = None,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    axis_name: str | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """Rescale updates according to the Muon algorithm.

    ``shape_updates`` and ``weight_decay`` are only used by the public Muon
    wrapper's Magma path. This keeps Magma aligned with AdamW/PRISM/Aurora: the
    complete pre-learning-rate base update, including shape scaling and decay,
    is passed through Magma. Plain Muon keeps those as separate transforms.
    """
    if isinstance(ns_steps, (int, float)) and ns_steps < 1:
        raise ValueError(f"ns_steps must be >= 1, got {ns_steps!r}.")
    _validate_beta_static_scalar("beta", beta)
    _validate_positive_static_scalar("eps", eps)
    if shape_updates:
        _validate_positive_static_scalar("consistent_rms", consistent_rms)
    if use_magma:
        validate_magma_args(magma_p, magma_tau)
    elif _has_nonzero_or_scheduled(weight_decay):
        raise ValueError(
            "`weight_decay` in `scale_by_muon` is only applied by the Magma path. "
            "Use `use_magma=True` or chain `optax.add_decayed_weights` separately."
        )

    canonical_mu_dtype = cast(
        jax.typing.DTypeLike,
        jnp.float32 if mu_dtype is None else utils.canonicalize_dtype(mu_dtype),
    )

    def resolve_dim_nums(params_or_updates):
        return _get_dimension_numbers(weight_dimension_numbers, params_or_updates)

    def init_fn(params):
        return MuonState(
            count=jnp.zeros([], jnp.int32),
            mu=_zeros_like_tree(params, canonical_mu_dtype),
            ns_coeffs=resolve_ns_coeffs(ns_coeffs, ns_steps),
            magma_s=_init_magma_state(params) if use_magma else (),
            key=key,
        )

    def update_fn(updates, state, params=None):
        raw_gradients = updates
        dim_nums = _resolve_update_dimension_numbers(
            weight_dimension_numbers,
            params=params,
            updates=updates,
            transform_name="scale_by_muon",
        )
        jax.tree.map(
            lambda u, d: _validate_matrix_operand(u, d, "scale_by_muon"),
            updates,
            dim_nums,
            is_leaf=_is_dimension_numbers_leaf,
        )

        runtime = _prepare_first_moment_runtime(
            updates,
            state.mu,
            state.count,
            state.key,
            beta,
            canonical_mu_dtype,
            nesterov=nesterov,
            bias_correction=True,
            momentum_accumulator=momentum_accumulator,
            nesterov_moment_count_offset=1,
            reserve_sr_key=use_magma,
            extra_key_count=1 if use_magma else 0,
        )
        mu = runtime.mu
        count_inc = runtime.count
        mu_hat = runtime.direction

        muon_updates = jax.tree.map(
            lambda x, dim_num: _call_orthogonalize(
                orthogonalize_fn,
                x,
                state.ns_coeffs,
                ns_steps,
                preconditioning,
                eps,
                dim_num,
            ),
            mu_hat,
            dim_nums,
            is_leaf=_is_dimension_numbers_leaf,
        )
        if adaptive:
            muon_updates = jax.tree.map(
                lambda x, y: (
                    y
                    if _is_aux_leaf(x) or _is_aux_leaf(y)
                    else jnp.sum(x.conj() * y) * y
                ),
                mu_hat,
                muon_updates,
                is_leaf=_is_aux_leaf,
            )

        if shape_updates:
            muon_updates = _scale_muon_shape_tree(
                muon_updates,
                dim_nums,
                consistent_rms,
            )

        if use_magma and _has_nonzero_or_scheduled(weight_decay) and params is None:
            raise ValueError(
                "`params` must be provided to `scale_by_muon.update` when "
                "`weight_decay` is nonzero or scheduled."
            )

        if use_magma and _has_nonzero_or_scheduled(weight_decay):
            params = cast(base.Params, params)
            wd_step = _resolve_scalar(weight_decay, state.count)
            muon_updates = _apply_weight_decay_tree(
                muon_updates,
                params,
                wd_step,
                _resolve_mask(weight_decay_mask, params),
            )

        next_key = runtime.key
        magma_key = runtime.extra_keys[0] if use_magma else None

        if use_magma:
            muon_updates, new_magma_s = apply_magma_internal(
                raw_gradients=raw_gradients,
                first_moments=mu,
                base_updates=muon_updates,
                magma_s_prev=state.magma_s,
                key=magma_key,
                p=magma_p,
                tau=magma_tau,
                axis_name=axis_name,
            )
        else:
            new_magma_s = state.magma_s

        return muon_updates, MuonState(
            count=count_inc,
            mu=runtime.mu_stored,
            ns_coeffs=state.ns_coeffs,
            magma_s=new_magma_s,
            key=next_key,
        )

    return base.GradientTransformation(init_fn, update_fn)


def _build_unscaled_muon_branch(
    *,
    ns_coeffs: MuonNsCoeffs,
    ns_steps: jax.typing.ArrayLike,
    beta: jax.typing.ArrayLike,
    eps: jax.typing.ArrayLike,
    mu_dtype: jax.typing.DTypeLike | None,
    nesterov: bool,
    adaptive: bool,
    preconditioning: MuonPreconditioning,
    weight_dimension_numbers: WeightDimNumOrFn | None,
    orthogonalize_fn: OrthogonalizeFn,
    momentum_accumulator: MomentumAccumulator,
    use_magma: bool,
    magma_p: float,
    magma_tau: float,
    consistent_rms: jax.typing.ArrayLike | None,
    weight_decay: base.ScalarOrSchedule,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None,
    axis_name: str | None,
    key: jax.Array,
) -> base.GradientTransformation:
    """Build the unscaled Muon direction branch shared by wrappers."""
    components: list[base.GradientTransformation] = [
        scale_by_muon(
            ns_coeffs=ns_coeffs,
            ns_steps=ns_steps,
            beta=beta,
            eps=eps,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
            adaptive=adaptive,
            preconditioning=preconditioning,
            weight_dimension_numbers=weight_dimension_numbers,
            orthogonalize_fn=orthogonalize_fn,
            momentum_accumulator=momentum_accumulator,
            use_magma=use_magma,
            magma_p=magma_p,
            magma_tau=magma_tau,
            shape_updates=use_magma,
            consistent_rms=consistent_rms,
            weight_decay=weight_decay if use_magma else 0.0,
            weight_decay_mask=weight_decay_mask if use_magma else None,
            axis_name=axis_name,
            key=key,
        ),
    ]

    if not use_magma:
        components.append(
            scale_by_shape(
                weight_dimension_numbers=weight_dimension_numbers,
                consistent_rms=consistent_rms,
            )
        )
        if _has_nonzero_or_scheduled(weight_decay):
            components.append(
                transform.add_decayed_weights(weight_decay, weight_decay_mask)
            )

    return combine.chain(*components)


def muon(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: MuonNsCoeffs = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    mu_dtype: jax.typing.DTypeLike | None = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    preconditioning: MuonPreconditioning = "frobenius",
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: base.ScalarOrSchedule | None = None,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    muon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    orthogonalize_fn: OrthogonalizeFn = orthogonalize_via_newton_schulz,
    momentum_accumulator: MomentumAccumulator = "ema",
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    axis_name: str | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """Muon optimizer with automatic Muon/AdamW partitioning."""
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate
    effective_adam_weight_decay = (
        weight_decay if adam_weight_decay is None else adam_weight_decay
    )

    key_muon, key_adam = jax.random.split(key, 2)

    partition = _make_matrix_partition_fns(muon_weight_dimension_numbers, "muon")

    muon_branch = _build_unscaled_muon_branch(
        ns_coeffs=ns_coeffs,
        ns_steps=ns_steps,
        beta=beta,
        eps=eps,
        mu_dtype=mu_dtype,
        nesterov=nesterov,
        adaptive=adaptive,
        preconditioning=preconditioning,
        weight_dimension_numbers=partition.masked_specs,
        orthogonalize_fn=orthogonalize_fn,
        momentum_accumulator=momentum_accumulator,
        use_magma=use_magma,
        magma_p=magma_p,
        magma_tau=magma_tau,
        consistent_rms=consistent_rms,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        axis_name=axis_name,
        key=key_muon,
    )

    return combine.partition(
        transforms={
            "muon": combine.chain(
                muon_branch,
                transform.scale_by_learning_rate(learning_rate),
            ),
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
                use_magma=use_magma,
                magma_p=magma_p,
                magma_tau=magma_tau,
                axis_name=axis_name,
                key=key_adam,
            ),
        },
        param_labels=partition.labels,
    )


__all__ = [
    "MomentumAccumulator",
    "MuonDimensionNumbers",
    "MuonNsCoeffs",
    "MuonState",
    "orthogonalize_via_newton_schulz",
    "polar_express_coeffs",
    "resolve_ns_coeffs",
    "scale_by_muon",
    "scale_by_muon_shape",
    "scale_by_shape",
    "muon",
]
