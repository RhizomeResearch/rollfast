"""Small functional update helpers for plan-aware optimizers."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax

from rollfast.optim.sam import add_perturbation, global_l2_norm, sam_perturbation
from rollfast.utils import (
    apply_updates,
    astype_preserving_sharding,
    resolve_partition_norm_axis_name,
    zeros_like_preserving_sharding,
)

from ._precision import master_leaf_dtype, uses_master_params
from ._protocols import CombinableFineTunePlanProtocol, FineTunePlanProtocol
from .config import (
    AccumulationConfig,
    AccumulationState,
    ASAMConfig,
    FineTuneStepState,
    LossBundle,
    LossScaleState,
    OptimizerBundle,
    PrecisionConfig,
    RNGStreams,
    SAMConfig,
    StepCounters,
)
from .update_status import UpdateStatus, _pending_microsteps, _update_status


class SAMStepInfo(NamedTuple):
    """Diagnostics returned by a two-pass SAM/ASAM update."""

    loss: Any
    perturbed_loss: Any
    aux: Any | None
    perturbed_aux: Any | None
    grad_norm: Any
    perturbation_norm: Any


class StatefulStepInfo(NamedTuple):
    """Diagnostics returned by a model-state-aware update.

    ``all_finite`` is true when the gradients and the candidate update were
    finite and committed. ``grads_finite`` reports gradient overflow alone.
    ``update_applied`` reports whether parameters changed: it is false while an
    accumulation window is incomplete and when an update is rejected.
    """

    value: Any
    proposed_model_state: Any | None
    all_finite: Any
    forward_key: Any | None
    grads_finite: Any = None
    update_applied: Any = None


class StatefulSAMStepInfo(NamedTuple):
    """Diagnostics returned by a stateful SAM/ASAM update."""

    loss: Any
    perturbed_loss: Any
    aux: Any | None
    perturbed_aux: Any | None
    proposed_model_state: Any | None
    perturbed_model_state: Any | None
    grad_norm: Any
    perturbation_norm: Any
    all_finite: Any
    failed_pass: Any
    sam_key: Any | None
    update_applied: Any = None


class AccumulatingLossBundleInfo(NamedTuple):
    """Diagnostics returned by an exact accumulated ``LossBundle`` step."""

    loss_bundle: LossBundle
    update_applied: Any
    all_finite: Any


ModelStateAggregator = Callable[[Any | None, Any | None], Any | None]

_RNG_STEPS = (
    "make_finetune_update_step, make_finetune_sam_step, "
    "make_stateful_loss_scaled_master_update_step, or make_stateful_sam_step"
)


def make_update_step(
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return a one-update function over a trainable PyTree.

    Updates apply directly to the supplied parameters with round-to-nearest.
    This helper keeps no master parameters and applies no loss scaling; use
    ``make_finetune_update_step`` to honor the bundle's precision policy.
    """

    _check_direct_precision(optimizer, "make_update_step")
    return _direct_update_step(loss_fn, optimizer, has_aux=has_aux)


def make_plan_update_step(
    plan: CombinableFineTunePlanProtocol,
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return a one-update function whose loss receives ``plan.combine`` output.

    Like ``make_update_step``, it updates the supplied parameters directly.
    """

    _check_direct_precision(optimizer, "make_plan_update_step")

    def trainable_loss(trainable, *args, **kwargs):
        return loss_fn(plan.combine(trainable), *args, **kwargs)

    return _direct_update_step(trainable_loss, optimizer, has_aux=has_aux)


def _direct_update_step(
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool,
) -> Callable[..., Any]:
    value_and_grad = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def step(trainable, opt_state, *args, **kwargs):
        value, grads = value_and_grad(trainable, *args, **kwargs)
        updates, opt_state = optimizer.update(grads, opt_state, trainable)
        trainable = apply_updates(trainable, updates, stochastic=False)
        return trainable, opt_state, value

    return step


def init_rng_streams(seed: int | Any) -> RNGStreams:
    """Derive named PRNG streams from a root seed/key."""

    root = jax.random.PRNGKey(seed) if isinstance(seed, int) else seed
    forward, sam, stochastic_rounding, quantization, controller = jax.random.split(
        root,
        5,
    )
    return RNGStreams(
        forward=forward,
        sam=sam,
        stochastic_rounding=stochastic_rounding,
        quantization=quantization,
        controller=controller,
    )


def init_step_counters() -> StepCounters:
    """Return zeroed canonical fine-tuning step counters."""

    zero = jnp.asarray(0, dtype=jnp.int32)
    return StepCounters(
        microstep=zero,
        attempted_update=zero,
        successful_update=zero,
        schedule_step=zero,
        rank_step=zero,
        average_step=zero,
        loss_scale_growth_step=zero,
    )


def init_accumulation_state(
    trainable: Any,
    accumulation: AccumulationConfig,
    *,
    pending_model_state: Any | None = None,
) -> AccumulationState:
    """Return an empty accumulation state matching a trainable PyTree."""

    zero = jnp.asarray(0, dtype=jnp.int32)
    return AccumulationState(
        grad_numerator=_zeros_like_accumulator(
            trainable,
            accumulation.accumulate_dtype,
        ),
        loss_sum=jnp.asarray(0, dtype=accumulation.accumulate_dtype),
        normalizer=jnp.asarray(0, dtype=accumulation.accumulate_dtype),
        metric_sums={},
        metric_normalizers={},
        aux_sums={},
        aux_normalizers={},
        microsteps_in_window=zero,
        all_finite=jnp.asarray(True),
        pending_model_state=pending_model_state,
        pending_model_state_valid=jnp.asarray(pending_model_state is not None),
    )


def init_finetune_step_state(
    bundle: OptimizerBundle,
    trainable: Any,
    *,
    model_state: Any | None = None,
    master_params: Any | None = None,
    optimizer_state: Any | None = None,
    accumulation: AccumulationState | None = None,
    loss_scale: LossScaleState | None = None,
    rng: RNGStreams | None = None,
    rng_seed: int | Any = 0,
    counters: StepCounters | None = None,
    ema_state: Any | None = None,
    swa_state: Any | None = None,
    adalora_state: Any | None = None,
    projection_state: Any | None = None,
    schedule_free_state: Any | None = None,
) -> FineTuneStepState:
    """Initialize the canonical composite fine-tuning step state.

    Pass ``master_params`` built from the original float32 weights, before
    casting them to the low-precision compute copy: masters built from
    ``trainable`` only upcast values that were already rounded. The optimizer
    state is initialized from the masters when they exist. The default
    ``accumulation`` record tracks the window position only; the optimizer
    state accumulates gradients.
    """

    if master_params is None:
        master_params = make_master_params(trainable, bundle.precision_config)
    else:
        _check_master_params(trainable, master_params, bundle.precision_config)
    optimizer_params = trainable if master_params is None else master_params
    if optimizer_state is None:
        optimizer_state = bundle.init(optimizer_params)
    if accumulation is None:
        accumulation = _window_accumulation_state(bundle.accumulation_config)
    if loss_scale is None:
        loss_scale = init_loss_scale_state(bundle.precision_config)
    return FineTuneStepState(
        optimizer_state=optimizer_state,
        model_state=model_state,
        master_params=master_params,
        accumulation=accumulation,
        loss_scale=loss_scale,
        ema_state=ema_state,
        swa_state=swa_state,
        adalora_state=adalora_state,
        projection_state=projection_state,
        schedule_free_state=schedule_free_state,
        counters=init_step_counters() if counters is None else counters,
        rng=init_rng_streams(rng_seed) if rng is None else rng,
    )


def make_master_params(
    trainable: Any,
    precision: PrecisionConfig,
) -> Any | None:
    """Return master params when the precision policy requires them.

    Each master leaf uses ``precision.master_param_dtype`` promoted with the
    leaf's own dtype, so wider leaves keep their precision and complex leaves
    keep complex masters.
    """

    if not uses_master_params(trainable, precision):
        return None
    return jax.tree.map(
        lambda leaf: (
            None
            if leaf is None
            else astype_preserving_sharding(
                leaf, master_leaf_dtype(leaf.dtype, precision.master_param_dtype)
            )
        ),
        trainable,
        is_leaf=lambda x: x is None,
    )


def init_loss_scale_state(precision: PrecisionConfig) -> LossScaleState | None:
    """Return the initial loss-scaling state for a precision policy."""

    if precision.loss_scale == "none":
        return None
    return LossScaleState(
        loss_scale=jnp.asarray(precision.static_loss_scale, dtype=jnp.float32),
        growth_tracker=jnp.asarray(0, dtype=jnp.int32),
    )


def make_master_update_step(
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return an update step that applies optimizer updates to master params.

    The step returns ``(visible, master, opt_state, value)``; ``master`` is
    ``None`` when the policy keeps no masters. Given ``master_params=None``
    under a policy that keeps masters, the step creates them, so pass masters
    from the first call to keep the argument structure stable under JIT. The
    step applies no finite-value or loss-scale handling of its own; use
    ``make_loss_scaled_master_update_step`` or ``make_finetune_update_step``
    for those.
    """

    _require_rng_free_rounding(optimizer, "make_master_update_step")
    value_and_grad = jax.value_and_grad(
        lambda master, visible_template, *args, **kwargs: loss_fn(
            _cast_tree_like(master, visible_template),
            *args,
            **kwargs,
        ),
        has_aux=has_aux,
    )

    def step(visible_params, master_params, opt_state, *args, **kwargs):
        precision = optimizer.precision_config
        master_params = _resolve_master_params(visible_params, master_params, precision)
        params = visible_params if master_params is None else master_params
        value, grads = value_and_grad(params, visible_params, *args, **kwargs)
        grads = _cast_gradient_tree(grads, precision.gradient_dtype)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = apply_updates(params, updates, stochastic=False)
        visible_params = _cast_tree_like(params, visible_params)
        master_params = None if master_params is None else params
        return visible_params, master_params, opt_state, value

    return step


class _Commit(NamedTuple):
    visible: Any
    master: Any | None
    opt_state: optax.OptState
    accepted: jax.Array
    status: UpdateStatus


def _commit_update(
    optimizer: OptimizerBundle,
    visible_params: Any,
    master_params: Any | None,
    opt_state: optax.OptState,
    grads: Any,
    grads_finite: jax.Array,
    *,
    sr_key: jax.Array | None,
) -> _Commit:
    """Commit parameters and optimizer state together only for a finite update.

    ``master_params`` is ``None`` for master-free storage. The returned status
    describes the optimizer invocation; ``update_applied`` also requires the
    result to have been committed.
    """

    params = visible_params if master_params is None else master_params
    accumulates = optimizer.accumulation_config.steps > 1
    if not accumulates:
        grads = _select_tree(grads_finite, grads, _tree_zeros_like(grads))

    def propose():
        updates, candidate_state = optimizer.update(grads, opt_state, params)
        candidate = _apply_parameter_updates(params, updates, optimizer, sr_key)
        candidate_visible = (
            candidate
            if master_params is None
            else _cast_tree_like(candidate, visible_params)
        )
        candidate_master = None if master_params is None else candidate
        return (
            (candidate_visible, candidate_master, candidate_state),
            _tree_all_finite((candidate_visible, candidate_master)),
            _update_status(opt_state, candidate_state),
        )

    # An accumulating optimizer must still receive a nonfinite microbatch so
    # its finite guard discards the window at the boundary.
    attempt = jnp.asarray(True) if accumulates else grads_finite
    (visible, master, state), accepted, status = _commit_optimizer_result(
        attempt,
        propose,
        (visible_params, master_params, opt_state),
        optimizer,
    )
    applied = jnp.logical_and(accepted, status.update_applied)
    status = UpdateStatus(
        accumulating=status.accumulating,
        update_applied=applied,
        window_discarded=jnp.logical_not(jnp.logical_or(status.accumulating, applied)),
        microsteps_in_window=_microsteps_or_zero(state),
    )
    return _Commit(visible, master, state, accepted, status)


def make_loss_scaled_master_update_step(
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return a master-parameter step with static/dynamic loss scaling.

    The step returns ``(visible, master, opt_state, loss_scale, value,
    all_finite)``. ``master`` is ``None`` when the policy keeps no masters.
    Masters and loss-scale state passed as ``None`` are created on that call,
    which changes the argument structure; pass them from the first call to
    avoid retracing under JIT. Nonfinite gradients and nonfinite candidate
    updates leave parameters and optimizer state unchanged; only gradient
    overflow lowers a dynamic loss scale. With accumulation, a nonfinite
    microbatch discards its window.
    """

    _require_rng_free_rounding(optimizer, "make_loss_scaled_master_update_step")
    _check_accumulation_guard(optimizer, "make_loss_scaled_master_update_step")
    scaled_value_and_grad = jax.value_and_grad(
        lambda master, visible_template, loss_scale, *args, **kwargs: _scale_loss_value(
            loss_fn(
                _cast_tree_like(master, visible_template),
                *args,
                **kwargs,
            ),
            loss_scale,
            has_aux=has_aux,
        ),
        has_aux=True,
    )

    def step(
        visible_params,
        master_params,
        opt_state,
        loss_scale_state,
        *args,
        **kwargs,
    ):
        precision = optimizer.precision_config
        master_params = _resolve_master_params(visible_params, master_params, precision)
        if loss_scale_state is None:
            loss_scale_state = init_loss_scale_state(precision)
        scale = _current_loss_scale(loss_scale_state, precision)
        (_, value), scaled_grads = scaled_value_and_grad(
            visible_params if master_params is None else master_params,
            visible_params,
            scale,
            *args,
            **kwargs,
        )
        grads, grads_finite = _unscale_gradients(
            optimizer, scaled_grads, scale, _primary_loss(value, has_aux=has_aux)
        )
        commit = _commit_update(
            optimizer,
            visible_params,
            master_params,
            opt_state,
            grads,
            grads_finite,
            sr_key=None,
        )
        loss_scale_state = _updated_loss_scale_state(
            loss_scale_state,
            grads_finite=grads_finite,
            accepted=commit.accepted,
            precision=precision,
        )
        return (
            commit.visible,
            commit.master,
            commit.opt_state,
            loss_scale_state,
            value,
            jnp.logical_and(grads_finite, commit.accepted),
        )

    return step


def make_stateful_loss_scaled_master_update_step(
    plan: CombinableFineTunePlanProtocol,
    loss_fn: Callable[..., tuple[Any, Any | None]],
    optimizer: OptimizerBundle,
    *,
    state_policy: Literal[
        "frozen",
        "microbatch_sequential",
        "optimizer_step_aggregate",
        "external",
    ] = "frozen",
    has_aux: bool = False,
    model_state_aggregator: ModelStateAggregator | None = None,
) -> Callable[..., Any]:
    """Return a model-state/RNG-aware master step with loss scaling.

    The step returns ``(visible, master, opt_state, model_state, loss_scale,
    rng, info)``; ``master`` is ``None`` when the policy keeps no masters.
    Masters and loss-scale state passed as ``None`` are created on that call;
    pass them from the first call to avoid retracing under JIT. With
    ``parameter_rounding="stochastic"``, rounding keys come from
    ``rng.stochastic_rounding``, which advances only when parameters change.
    """

    step = _make_stateful_master_step(
        plan,
        loss_fn,
        optimizer,
        state_policy=state_policy,
        has_aux=has_aux,
        model_state_aggregator=model_state_aggregator,
        helper="make_stateful_loss_scaled_master_update_step",
    )

    def public_step(*args, **kwargs):
        *results, _ = step(*args, **kwargs)
        return tuple(results)

    return public_step


def _make_stateful_master_step(
    plan: CombinableFineTunePlanProtocol,
    loss_fn: Callable[..., tuple[Any, Any | None]],
    optimizer: OptimizerBundle,
    *,
    state_policy: str,
    has_aux: bool,
    model_state_aggregator: ModelStateAggregator | None,
    helper: str,
) -> Callable[..., Any]:
    _check_state_policy(state_policy, model_state_aggregator)
    _check_accumulation_guard(optimizer, helper)

    def scaled_stateful_loss(
        master,
        visible_template,
        model_state,
        forward_key,
        loss_scale,
        *args,
        **kwargs,
    ):
        visible = _cast_tree_like(master, visible_template)
        return _scale_stateful_loss_value(
            loss_fn(
                plan.combine(visible),
                _clone_model_state(model_state),
                forward_key,
                *args,
                **kwargs,
            ),
            loss_scale,
            has_aux=has_aux,
        )

    scaled_value_and_grad = jax.value_and_grad(
        scaled_stateful_loss,
        has_aux=True,
    )

    def step(
        visible_params,
        master_params,
        opt_state,
        model_state,
        loss_scale_state,
        rng,
        *args,
        **kwargs,
    ):
        precision = optimizer.precision_config
        master_params = _resolve_master_params(visible_params, master_params, precision)
        if loss_scale_state is None:
            loss_scale_state = init_loss_scale_state(precision)
        candidate_rng, forward_key = _next_forward_key(rng)
        candidate_rng, sr_key = _next_rounding_key(candidate_rng, precision)
        scale = _current_loss_scale(loss_scale_state, precision)
        (_, (value, proposed_model_state)), scaled_grads = scaled_value_and_grad(
            visible_params if master_params is None else master_params,
            visible_params,
            model_state,
            forward_key,
            scale,
            *args,
            **kwargs,
        )
        grads, grads_finite = _unscale_gradients(
            optimizer, scaled_grads, scale, _primary_loss(value, has_aux=has_aux)
        )
        commit = _commit_update(
            optimizer,
            visible_params,
            master_params,
            opt_state,
            grads,
            grads_finite,
            sr_key=sr_key,
        )
        committed = jnp.logical_and(grads_finite, commit.accepted)
        model_state = _commit_model_state(
            state_policy,
            model_state,
            proposed_model_state,
            committed,
            model_state_aggregator=model_state_aggregator,
        )
        rng = _commit_rng(
            rng,
            candidate_rng,
            committed=committed,
            update_applied=commit.status.update_applied,
        )
        loss_scale_state = _updated_loss_scale_state(
            loss_scale_state,
            grads_finite=grads_finite,
            accepted=commit.accepted,
            precision=precision,
        )
        info = StatefulStepInfo(
            value=value,
            proposed_model_state=proposed_model_state,
            all_finite=committed,
            forward_key=forward_key,
            grads_finite=grads_finite,
            update_applied=commit.status.update_applied,
        )
        return (
            commit.visible,
            commit.master,
            commit.opt_state,
            model_state,
            loss_scale_state,
            rng,
            info,
            commit.status,
        )

    return step


def make_finetune_update_step(
    plan: CombinableFineTunePlanProtocol,
    loss_fn: Callable[..., tuple[Any, Any | None]],
    optimizer: OptimizerBundle,
    *,
    state_policy: Literal[
        "frozen",
        "microbatch_sequential",
        "optimizer_step_aggregate",
        "external",
    ] = "frozen",
    has_aux: bool = False,
    model_state_aggregator: ModelStateAggregator | None = None,
) -> Callable[..., Any]:
    """Return a step over visible params plus ``FineTuneStepState``.

    The step honors the bundle's precision policy: master parameters, loss
    scaling, and stochastic parameter rounding. With ``accumulation.steps > 1``
    each call is one microbatch and the optimizer state owns the window; a
    nonfinite microbatch discards its window. ``counters`` and
    ``info.update_applied`` follow applied updates. ``FineTuneStepState`` is a
    PyTree, so the step runs under ``jax.jit`` and ``jax.lax.scan``.
    """

    inner_step = _make_stateful_master_step(
        plan,
        loss_fn,
        optimizer,
        state_policy=state_policy,
        has_aux=has_aux,
        model_state_aggregator=model_state_aggregator,
        helper="make_finetune_update_step",
    )

    def step(visible_params, step_state: FineTuneStepState, *args, **kwargs):
        _check_step_state_masters(visible_params, step_state, optimizer)
        (
            visible_params,
            master_params,
            opt_state,
            model_state,
            loss_scale_state,
            rng,
            info,
            status,
        ) = inner_step(
            visible_params,
            step_state.master_params,
            step_state.optimizer_state,
            step_state.model_state,
            step_state.loss_scale,
            step_state.rng,
            *args,
            **kwargs,
        )
        step_state = _advance_step_state(
            step_state,
            optimizer,
            status=status,
            committed=info.all_finite,
            optimizer_state=opt_state,
            model_state=model_state,
            master_params=master_params,
            loss_scale=loss_scale_state,
            rng=rng,
        )
        return visible_params, step_state, info

    return step


def make_plan_loss_scaled_master_update_step(
    plan: CombinableFineTunePlanProtocol,
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return a loss-scaled master step whose loss receives ``plan.combine``."""

    def trainable_loss(trainable, *args, **kwargs):
        return loss_fn(plan.combine(trainable), *args, **kwargs)

    return make_loss_scaled_master_update_step(
        trainable_loss,
        optimizer,
        has_aux=has_aux,
    )


def make_plan_master_update_step(
    plan: CombinableFineTunePlanProtocol,
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return a master-parameter step whose loss receives ``plan.combine``."""

    def trainable_loss(trainable, *args, **kwargs):
        return loss_fn(plan.combine(trainable), *args, **kwargs)

    return make_master_update_step(trainable_loss, optimizer, has_aux=has_aux)


def make_loss_bundle_update_step(
    loss_fn: Callable[..., LossBundle],
    optimizer: OptimizerBundle,
    *,
    microbatch_axis: int | None = None,
    microbatch_count: int | None = None,
) -> Callable[..., Any]:
    """Return an update step for exact summed-loss/normalizer contracts.

    Microbatch sums accumulate in the bundle's accumulation dtype. Updates apply
    directly to the supplied parameters with round-to-nearest.
    """

    _check_direct_precision(optimizer, "make_loss_bundle_update_step")
    return _direct_loss_bundle_step(
        loss_fn,
        optimizer,
        microbatch_axis=microbatch_axis,
        microbatch_count=microbatch_count,
    )


def _direct_loss_bundle_step(
    loss_fn: Callable[..., LossBundle],
    optimizer: OptimizerBundle,
    *,
    microbatch_axis: int | None,
    microbatch_count: int | None,
) -> Callable[..., Any]:
    value_and_grad = jax.value_and_grad(_loss_bundle_sum, has_aux=True)

    def step(trainable, opt_state, *args, **kwargs):
        bundle, grads = _evaluate_loss_bundle_value_and_grad(
            value_and_grad,
            loss_fn,
            trainable,
            args,
            kwargs,
            microbatch_axis=microbatch_axis,
            microbatch_count=microbatch_count,
            accumulate_dtype=optimizer.accumulation_config.accumulate_dtype,
        )
        grads = _tree_divide(grads, bundle.normalizer)
        updates, opt_state = optimizer.update(grads, opt_state, trainable)
        trainable = apply_updates(trainable, updates, stochastic=False)
        return trainable, opt_state, _normalized_loss_bundle(bundle)

    return step


def make_accumulating_loss_bundle_update_step(
    loss_fn: Callable[..., LossBundle],
    optimizer: OptimizerBundle,
    *,
    accumulation: AccumulationConfig,
    model_state_aggregator: ModelStateAggregator | None = None,
) -> Callable[..., Any]:
    """Return a stateful exact numerator/normalizer accumulation step.

    Updates apply directly to the supplied parameters with round-to-nearest.
    """

    _check_direct_precision(optimizer, "make_accumulating_loss_bundle_update_step")
    if optimizer.accumulation_config.steps != 1:
        raise ValueError(
            "make_accumulating_loss_bundle_update_step requires an optimizer "
            "bundle compiled with accumulation steps=1."
        )
    if accumulation.steps <= 1:
        raise ValueError("stateful accumulation requires accumulation.steps > 1.")
    value_and_grad = jax.value_and_grad(_loss_bundle_sum, has_aux=True)

    def step(trainable, opt_state, accumulation_state, *args, **kwargs):
        (_, bundle), grads = value_and_grad(trainable, loss_fn, args, kwargs)
        if not isinstance(bundle, LossBundle):
            raise TypeError("loss_fn must return a LossBundle.")
        grads = _cast_gradient_tree(grads, accumulation.accumulate_dtype)
        all_finite = _tree_all_finite((bundle.loss_sum, bundle.normalizer, grads))
        safe_grads = _select_tree(all_finite, grads, _tree_zeros_like(grads))
        next_accumulation = _accumulate_loss_bundle_state(
            accumulation_state,
            bundle,
            safe_grads,
            all_finite=all_finite,
            accumulate_dtype=accumulation.accumulate_dtype,
            model_state_aggregator=model_state_aggregator,
        )
        accumulated_bundle = _loss_bundle_from_accumulation(next_accumulation)
        boundary = next_accumulation.microsteps_in_window >= accumulation.steps
        averaged_grads = _tree_divide(
            next_accumulation.grad_numerator,
            next_accumulation.normalizer,
        )
        window_finite = jnp.logical_and(
            next_accumulation.all_finite,
            _tree_all_finite(
                (
                    next_accumulation.grad_numerator,
                    next_accumulation.loss_sum,
                    next_accumulation.normalizer,
                    averaged_grads,
                )
            ),
        )
        window_finite = jnp.logical_and(window_finite, next_accumulation.normalizer > 0)
        attempt_update = jnp.logical_and(boundary, window_finite)

        def apply_accumulated_update(operand):
            current_trainable, current_opt_state = operand
            updates, next_opt_state = optimizer.update(
                averaged_grads,
                current_opt_state,
                current_trainable,
            )
            applied = _update_status(current_opt_state, next_opt_state).update_applied
            return (
                apply_updates(current_trainable, updates, stochastic=False),
                next_opt_state,
                applied,
            )

        trainable, opt_state, update_applied = jax.lax.cond(
            attempt_update,
            apply_accumulated_update,
            lambda operand: (*operand, jnp.asarray(False)),
            (trainable, opt_state),
        )
        reset_accumulation = init_accumulation_state(trainable, accumulation)
        accumulation_state = _select_accumulation_state(
            boundary,
            reset_accumulation,
            next_accumulation,
        )
        info = AccumulatingLossBundleInfo(
            loss_bundle=_normalized_loss_bundle(accumulated_bundle),
            update_applied=update_applied,
            all_finite=jnp.where(
                boundary,
                jnp.logical_and(window_finite, update_applied),
                next_accumulation.all_finite,
            ),
        )
        return trainable, opt_state, accumulation_state, info

    return step


def make_plan_loss_bundle_update_step(
    plan: CombinableFineTunePlanProtocol,
    loss_fn: Callable[..., LossBundle],
    optimizer: OptimizerBundle,
    *,
    microbatch_axis: int | None = None,
    microbatch_count: int | None = None,
) -> Callable[..., Any]:
    """Return a ``LossBundle`` update step whose loss receives ``plan.combine``."""

    _check_direct_precision(optimizer, "make_plan_loss_bundle_update_step")

    def trainable_loss(trainable, *args, **kwargs):
        return loss_fn(plan.combine(trainable), *args, **kwargs)

    return _direct_loss_bundle_step(
        trainable_loss,
        optimizer,
        microbatch_axis=microbatch_axis,
        microbatch_count=microbatch_count,
    )


def make_sam_step(
    *,
    plan: CombinableFineTunePlanProtocol,
    base_optimizer: OptimizerBundle,
    config: SAMConfig | ASAMConfig | None = None,
    loss_fn: Callable[..., Any],
    has_aux: bool = False,
    microbatch_axis: int | None = None,
    microbatch_count: int | None = None,
    microbatch_reduction: Literal["mean", "sum"] = "mean",
) -> Callable[..., Any]:
    """Return a functional two-pass SAM/ASAM update step.

    The base optimizer is applied to gradients from perturbed parameters, but
    the returned trainable tree descends from the original unperturbed params.
    Both passes evaluate the model at the trainable dtype, and microbatch sums
    accumulate in the bundle's accumulation dtype. Updates apply directly to
    the supplied parameters with round-to-nearest.
    """

    _check_direct_precision(base_optimizer, "make_sam_step")
    config = SAMConfig(enabled=True) if config is None else config
    adaptive = isinstance(config, ASAMConfig)
    eta = config.eta if adaptive else 0.0
    if base_optimizer.accumulation_config.steps != 1:
        raise ValueError(
            "SAM/ASAM steps require a base optimizer with accumulation_steps=1; "
            "use microbatch_axis on make_sam_step for exact SAM accumulation."
        )
    if microbatch_axis is not None and microbatch_axis < 0:
        raise ValueError("microbatch_axis must be non-negative when provided.")
    if microbatch_count is not None and microbatch_count <= 0:
        raise ValueError("microbatch_count must be positive when provided.")
    if microbatch_reduction not in {"mean", "sum"}:
        raise ValueError("microbatch_reduction must be 'mean' or 'sum'.")
    mask = _sam_perturb_mask(plan, config)
    accumulate_dtype = base_optimizer.accumulation_config.accumulate_dtype
    value_and_grad = jax.value_and_grad(
        lambda trainable, *args, **kwargs: loss_fn(
            plan.combine(trainable),
            *args,
            **kwargs,
        ),
        has_aux=has_aux,
    )

    def step(trainable, opt_state, *args, **kwargs):
        value, grads = _evaluate_sam_value_and_grad(
            value_and_grad,
            trainable,
            args,
            kwargs,
            microbatch_axis=microbatch_axis,
            microbatch_count=microbatch_count,
            microbatch_reduction=microbatch_reduction,
            accumulate_dtype=accumulate_dtype,
        )
        loss, aux = _split_value_aux(value, has_aux=has_aux)
        perturbation, perturbation_norm = sam_perturbation(
            grads,
            params=trainable,
            rho=config.rho,
            adaptive=adaptive,
            eta=eta,
            eps=config.eps,
            mask=mask,
            axis_name=config.axis_name,
            partition_axis_names=config.partition_axis_names,
            replicated_axis_names=config.replicated_axis_names,
        )
        perturbed = _cast_tree_like(
            add_perturbation(trainable, perturbation), trainable
        )
        perturbed_value, perturbed_grads = _evaluate_sam_value_and_grad(
            value_and_grad,
            perturbed,
            args,
            kwargs,
            microbatch_axis=microbatch_axis,
            microbatch_count=microbatch_count,
            microbatch_reduction=microbatch_reduction,
            accumulate_dtype=accumulate_dtype,
        )
        perturbed_loss, perturbed_aux = _split_value_aux(
            perturbed_value,
            has_aux=has_aux,
        )
        updates, opt_state = base_optimizer.update(
            perturbed_grads,
            opt_state,
            trainable,
        )
        trainable = apply_updates(trainable, updates, stochastic=False)
        info = SAMStepInfo(
            loss=loss,
            perturbed_loss=perturbed_loss,
            aux=aux,
            perturbed_aux=perturbed_aux,
            grad_norm=global_l2_norm(
                grads,
                axis_name=config.axis_name,
                partition_axis_names=config.partition_axis_names,
                replicated_axis_names=config.replicated_axis_names,
            ),
            perturbation_norm=perturbation_norm,
        )
        return trainable, opt_state, info

    return step


def make_stateful_sam_step(
    *,
    plan: CombinableFineTunePlanProtocol,
    base_optimizer: OptimizerBundle,
    config: SAMConfig | ASAMConfig | None = None,
    loss_fn: Callable[..., tuple[Any, Any | None]],
    state_policy: Literal[
        "frozen",
        "microbatch_sequential",
        "optimizer_step_aggregate",
        "external",
    ] = "microbatch_sequential",
    has_aux: bool = False,
    model_state_aggregator: ModelStateAggregator | None = None,
) -> Callable[..., Any]:
    """Return a state/RNG-aware SAM/ASAM update step.

    The same SAM PRNG key is replayed for both passes. Model-state updates from
    the first pass are committed only after a finite successful update; second
    pass state updates are always discarded. Updates apply directly to the
    supplied parameters, with ``parameter_rounding="stochastic"`` drawing keys
    from ``rng.stochastic_rounding``. Use ``make_finetune_sam_step`` for master
    parameters and loss scaling.
    """

    _check_direct_precision(
        base_optimizer, "make_stateful_sam_step", allows_stochastic_rounding=True
    )
    core = _make_sam_core(
        plan=plan,
        base_optimizer=base_optimizer,
        config=config,
        loss_fn=loss_fn,
        state_policy=state_policy,
        has_aux=has_aux,
        model_state_aggregator=model_state_aggregator,
        master_aware=False,
    )

    def step(trainable, opt_state, model_state, rng, *args, **kwargs):
        trainable, _, opt_state, model_state, _, rng, info, _ = core(
            trainable, None, opt_state, model_state, None, rng, *args, **kwargs
        )
        return trainable, opt_state, model_state, rng, info

    return step


def make_finetune_sam_step(
    *,
    plan: CombinableFineTunePlanProtocol,
    base_optimizer: OptimizerBundle,
    config: SAMConfig | ASAMConfig | None = None,
    loss_fn: Callable[..., tuple[Any, Any | None]],
    state_policy: Literal[
        "frozen",
        "microbatch_sequential",
        "optimizer_step_aggregate",
        "external",
    ] = "microbatch_sequential",
    has_aux: bool = False,
    model_state_aggregator: ModelStateAggregator | None = None,
) -> Callable[..., Any]:
    """Return a SAM/ASAM step over visible params plus ``FineTuneStepState``.

    The step honors the bundle's precision policy like
    ``make_finetune_update_step``. The perturbation is computed from and added
    to the optimization parameters (masters when present); both passes evaluate
    the model at the visible dtype and share one loss scale. A nonfinite pass
    rejects the update and lowers a dynamic loss scale.
    """

    core = _make_sam_core(
        plan=plan,
        base_optimizer=base_optimizer,
        config=config,
        loss_fn=loss_fn,
        state_policy=state_policy,
        has_aux=has_aux,
        model_state_aggregator=model_state_aggregator,
        master_aware=True,
    )

    def step(visible_params, step_state: FineTuneStepState, *args, **kwargs):
        _check_step_state_masters(visible_params, step_state, base_optimizer)
        (
            visible_params,
            master_params,
            opt_state,
            model_state,
            loss_scale_state,
            rng,
            info,
            status,
        ) = core(
            visible_params,
            step_state.master_params,
            step_state.optimizer_state,
            step_state.model_state,
            step_state.loss_scale,
            step_state.rng,
            *args,
            **kwargs,
        )
        step_state = _advance_step_state(
            step_state,
            base_optimizer,
            status=status,
            committed=info.all_finite,
            optimizer_state=opt_state,
            model_state=model_state,
            master_params=master_params,
            loss_scale=loss_scale_state,
            rng=rng,
        )
        return visible_params, step_state, info

    return step


def _make_sam_core(
    *,
    plan: CombinableFineTunePlanProtocol,
    base_optimizer: OptimizerBundle,
    config: SAMConfig | ASAMConfig | None,
    loss_fn: Callable[..., tuple[Any, Any | None]],
    state_policy: str,
    has_aux: bool,
    model_state_aggregator: ModelStateAggregator | None,
    master_aware: bool,
) -> Callable[..., Any]:
    config = SAMConfig(enabled=True) if config is None else config
    _check_state_policy(state_policy, model_state_aggregator)
    adaptive = isinstance(config, ASAMConfig)
    eta = config.eta if adaptive else 0.0
    if base_optimizer.accumulation_config.steps != 1:
        raise ValueError(
            "stateful SAM/ASAM steps require a base optimizer with accumulation_steps=1."
        )
    mask = _sam_perturb_mask(plan, config)
    precision = base_optimizer.precision_config

    def stateful_loss(
        params, visible_template, model_state, sam_key, scale, *args, **kwargs
    ):
        return _scale_stateful_loss_value(
            loss_fn(
                plan.combine(_cast_tree_like(params, visible_template)),
                _clone_model_state(model_state),
                sam_key,
                *args,
                **kwargs,
            ),
            scale,
            has_aux=has_aux,
        )

    value_and_grad = jax.value_and_grad(stateful_loss, has_aux=True)

    def step(
        visible_params,
        master_params,
        opt_state,
        model_state,
        loss_scale_state,
        rng,
        *args,
        **kwargs,
    ):
        if master_aware:
            master_params = _resolve_master_params(
                visible_params, master_params, precision
            )
            if loss_scale_state is None:
                loss_scale_state = init_loss_scale_state(precision)
        scale = (
            _current_loss_scale(loss_scale_state, precision)
            if master_aware
            else jnp.asarray(1.0, dtype=jnp.float32)
        )

        def gradients(scaled_grads, loss):
            if master_aware:
                return _unscale_gradients(base_optimizer, scaled_grads, scale, loss)
            finite = _tree_all_finite((loss, scaled_grads))
            return scaled_grads, _agree_across_axes(finite, base_optimizer)

        params = visible_params if master_params is None else master_params
        candidate_rng, sam_key = _next_sam_key(rng)
        candidate_rng, sr_key = _next_rounding_key(candidate_rng, precision)
        (_, (value, proposed_model_state)), scaled_grads = value_and_grad(
            params,
            visible_params,
            model_state,
            sam_key,
            scale,
            *args,
            **kwargs,
        )
        loss, aux = _split_value_aux(value, has_aux=has_aux)
        grads, first_finite = gradients(scaled_grads, loss)
        safe_grads = _select_tree(first_finite, grads, _tree_zeros_like(grads))
        perturbation, perturbation_norm = sam_perturbation(
            safe_grads,
            params=params,
            rho=config.rho,
            adaptive=adaptive,
            eta=eta,
            eps=config.eps,
            mask=mask,
            axis_name=config.axis_name,
            partition_axis_names=config.partition_axis_names,
            replicated_axis_names=config.replicated_axis_names,
        )
        perturbed = add_perturbation(params, perturbation)
        (_, (perturbed_value, perturbed_model_state)), scaled_perturbed_grads = (
            value_and_grad(
                perturbed,
                visible_params,
                model_state,
                sam_key,
                scale,
                *args,
                **kwargs,
            )
        )
        perturbed_loss, perturbed_aux = _split_value_aux(
            perturbed_value,
            has_aux=has_aux,
        )
        perturbed_grads, second_finite = gradients(
            scaled_perturbed_grads, perturbed_loss
        )
        grads_finite = jnp.logical_and(first_finite, second_finite)
        commit = _commit_update(
            base_optimizer,
            visible_params,
            master_params,
            opt_state,
            perturbed_grads,
            grads_finite,
            sr_key=sr_key,
        )
        committed = jnp.logical_and(grads_finite, commit.accepted)
        model_state = _commit_model_state(
            state_policy,
            model_state,
            proposed_model_state,
            committed,
            model_state_aggregator=model_state_aggregator,
        )
        rng = _commit_rng(
            rng,
            candidate_rng,
            committed=committed,
            update_applied=commit.status.update_applied,
        )
        if master_aware:
            loss_scale_state = _updated_loss_scale_state(
                loss_scale_state,
                grads_finite=grads_finite,
                accepted=commit.accepted,
                precision=precision,
            )
        info = StatefulSAMStepInfo(
            loss=loss,
            perturbed_loss=perturbed_loss,
            aux=aux,
            perturbed_aux=perturbed_aux,
            proposed_model_state=proposed_model_state,
            perturbed_model_state=perturbed_model_state,
            grad_norm=global_l2_norm(
                safe_grads,
                axis_name=config.axis_name,
                partition_axis_names=config.partition_axis_names,
                replicated_axis_names=config.replicated_axis_names,
            ),
            perturbation_norm=perturbation_norm,
            all_finite=committed,
            failed_pass=jnp.where(
                first_finite,
                jnp.where(second_finite, 0, 2),
                1,
            ),
            sam_key=sam_key,
            update_applied=commit.status.update_applied,
        )
        return (
            commit.visible,
            commit.master,
            commit.opt_state,
            model_state,
            loss_scale_state,
            rng,
            info,
            commit.status,
        )

    return step


def _evaluate_sam_value_and_grad(
    value_and_grad: Callable[..., Any],
    trainable: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    microbatch_axis: int | None,
    microbatch_count: int | None,
    microbatch_reduction: Literal["mean", "sum"],
    accumulate_dtype: Any,
) -> tuple[Any, Any]:
    if microbatch_axis is None:
        return value_and_grad(trainable, *args, **kwargs)

    count = _resolve_microbatch_count(
        args,
        kwargs,
        axis=microbatch_axis,
        count=microbatch_count,
    )

    def evaluate(index):
        micro_args = _slice_microbatch_tree(
            args,
            index=index,
            axis=microbatch_axis,
            count=count,
        )
        micro_kwargs = _slice_microbatch_tree(
            kwargs,
            index=index,
            axis=microbatch_axis,
            count=count,
        )
        return _promote_accumulator_tree(
            value_and_grad(trainable, *micro_args, **micro_kwargs),
            accumulate_dtype,
        )

    total_value, total_grads = evaluate(0) if count > 0 else (None, None)
    if count > 1:

        def accumulate(carry, index):
            value, grads = evaluate(index)
            return (_tree_add(carry[0], value), _tree_add(carry[1], grads)), None

        # Seed from the first microbatch to retain the original sum order.
        (total_value, total_grads), _ = jax.lax.scan(
            accumulate, (total_value, total_grads), jnp.arange(1, count)
        )

    if microbatch_reduction == "mean":
        scale = 1.0 / count
        total_value = _tree_scale(total_value, scale)
        total_grads = _tree_scale(total_grads, scale)
    return total_value, total_grads


def _loss_bundle_sum(
    trainable: Any,
    loss_fn: Callable[..., LossBundle],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, LossBundle]:
    bundle = loss_fn(trainable, *args, **kwargs)
    if not isinstance(bundle, LossBundle):
        raise TypeError("loss_fn must return a LossBundle.")
    return bundle.loss_sum, _stop_gradient_loss_bundle(bundle)


def _evaluate_loss_bundle_value_and_grad(
    value_and_grad: Callable[..., Any],
    loss_fn: Callable[..., LossBundle],
    trainable: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    microbatch_axis: int | None,
    microbatch_count: int | None,
    accumulate_dtype: Any,
) -> tuple[LossBundle, Any]:
    if microbatch_axis is None:
        (_, bundle), grads = value_and_grad(trainable, loss_fn, args, kwargs)
        return bundle, grads

    count = _resolve_microbatch_count(
        args,
        kwargs,
        axis=microbatch_axis,
        count=microbatch_count,
    )
    if count <= 0:
        raise ValueError("microbatch_count must be positive.")

    def evaluate(index):
        micro_args = _slice_microbatch_tree(
            args,
            index=index,
            axis=microbatch_axis,
            count=count,
        )
        micro_kwargs = _slice_microbatch_tree(
            kwargs,
            index=index,
            axis=microbatch_axis,
            count=count,
        )
        (_, bundle), grads = value_and_grad(
            trainable,
            loss_fn,
            micro_args,
            micro_kwargs,
        )
        return (
            _promote_loss_bundle(bundle, accumulate_dtype),
            _promote_accumulator_tree(grads, accumulate_dtype),
        )

    total_bundle, total_grads = evaluate(0)
    if count > 1:

        def accumulate(carry, index):
            bundle, grads = evaluate(index)
            return (
                _add_loss_bundles(carry[0], bundle),
                _tree_add(carry[1], grads),
            ), None

        (total_bundle, total_grads), _ = jax.lax.scan(
            accumulate, (total_bundle, total_grads), jnp.arange(1, count)
        )
    return total_bundle, total_grads


def _add_loss_bundles(left: LossBundle, right: LossBundle) -> LossBundle:
    return LossBundle(
        loss_sum=left.loss_sum + right.loss_sum,
        normalizer=left.normalizer + right.normalizer,
        metrics_sums=_add_mappings(left.metrics_sums, right.metrics_sums),
        metric_normalizers=_add_mappings(
            left.metric_normalizers,
            right.metric_normalizers,
        ),
        new_model_state=right.new_model_state,
        aux_sums=_add_mappings(left.aux_sums, right.aux_sums),
        aux_normalizers=_add_mappings(left.aux_normalizers, right.aux_normalizers),
    )


def _accumulate_loss_bundle_state(
    state: AccumulationState,
    bundle: LossBundle,
    grads: Any,
    *,
    all_finite: Any,
    accumulate_dtype: Any,
    model_state_aggregator: ModelStateAggregator | None = None,
) -> AccumulationState:
    bundle = _promote_loss_bundle(bundle, accumulate_dtype)
    return AccumulationState(
        grad_numerator=_tree_add(state.grad_numerator, grads),
        loss_sum=state.loss_sum + bundle.loss_sum,
        normalizer=state.normalizer + bundle.normalizer,
        metric_sums=_add_mappings(state.metric_sums, bundle.metrics_sums),
        metric_normalizers=_add_mappings(
            state.metric_normalizers,
            bundle.metric_normalizers,
        ),
        aux_sums=_add_mappings(state.aux_sums, bundle.aux_sums),
        aux_normalizers=_add_mappings(
            state.aux_normalizers,
            bundle.aux_normalizers,
        ),
        microsteps_in_window=state.microsteps_in_window
        + jnp.asarray(1, dtype=jnp.int32),
        all_finite=jnp.logical_and(state.all_finite, all_finite),
        pending_model_state=_merge_pending_model_state(
            state.pending_model_state,
            bundle.new_model_state,
            model_state_aggregator,
            current_state_valid=state.pending_model_state_valid,
        ),
        pending_model_state_valid=(
            state.pending_model_state_valid
            if bundle.new_model_state is None
            else jnp.asarray(True)
        ),
    )


def _loss_bundle_from_accumulation(state: AccumulationState) -> LossBundle:
    return LossBundle(
        loss_sum=state.loss_sum,
        normalizer=state.normalizer,
        metrics_sums=state.metric_sums,
        metric_normalizers=state.metric_normalizers,
        new_model_state=state.pending_model_state,
        aux_sums=state.aux_sums,
        aux_normalizers=state.aux_normalizers,
    )


def _merge_pending_model_state(
    current_state: Any | None,
    new_state: Any | None,
    model_state_aggregator: ModelStateAggregator | None,
    *,
    current_state_valid: Any,
) -> Any | None:
    if model_state_aggregator is None:
        return new_state
    if new_state is None:
        return current_state
    if current_state is None:
        return new_state
    merged_state = model_state_aggregator(current_state, new_state)
    return _select_tree(current_state_valid, merged_state, new_state)


def _select_accumulation_state(
    condition: Any,
    true_state: AccumulationState,
    false_state: AccumulationState,
) -> AccumulationState:
    return AccumulationState(
        grad_numerator=_select_tree(
            condition,
            true_state.grad_numerator,
            false_state.grad_numerator,
        ),
        loss_sum=jnp.where(condition, true_state.loss_sum, false_state.loss_sum),
        normalizer=jnp.where(condition, true_state.normalizer, false_state.normalizer),
        metric_sums=_select_mapping(
            condition,
            true_state.metric_sums,
            false_state.metric_sums,
        ),
        metric_normalizers=_select_mapping(
            condition,
            true_state.metric_normalizers,
            false_state.metric_normalizers,
        ),
        aux_sums=_select_mapping(
            condition,
            true_state.aux_sums,
            false_state.aux_sums,
        ),
        aux_normalizers=_select_mapping(
            condition,
            true_state.aux_normalizers,
            false_state.aux_normalizers,
        ),
        microsteps_in_window=jnp.where(
            condition,
            true_state.microsteps_in_window,
            false_state.microsteps_in_window,
        ),
        all_finite=jnp.where(condition, true_state.all_finite, false_state.all_finite),
        pending_model_state=false_state.pending_model_state,
        pending_model_state_valid=jnp.where(
            condition,
            true_state.pending_model_state_valid,
            false_state.pending_model_state_valid,
        ),
    )


def _normalized_loss_bundle(bundle: LossBundle) -> LossBundle:
    return LossBundle(
        loss_sum=bundle.loss_sum / bundle.normalizer,
        normalizer=jnp.asarray(1, dtype=jnp.asarray(bundle.normalizer).dtype),
        metrics_sums={
            key: value / bundle.metric_normalizers[key]
            for key, value in bundle.metrics_sums.items()
        },
        metric_normalizers={
            key: jnp.asarray(1, dtype=jnp.asarray(value).dtype)
            for key, value in bundle.metric_normalizers.items()
        },
        new_model_state=bundle.new_model_state,
        aux_sums={
            key: value / bundle.aux_normalizers[key]
            for key, value in bundle.aux_sums.items()
        },
        aux_normalizers={
            key: jnp.asarray(1, dtype=jnp.asarray(value).dtype)
            for key, value in bundle.aux_normalizers.items()
        },
    )


def _stop_gradient_loss_bundle(bundle: LossBundle) -> LossBundle:
    return LossBundle(
        loss_sum=jax.lax.stop_gradient(bundle.loss_sum),
        normalizer=jax.lax.stop_gradient(bundle.normalizer),
        metrics_sums={
            key: jax.lax.stop_gradient(value)
            for key, value in bundle.metrics_sums.items()
        },
        metric_normalizers={
            key: jax.lax.stop_gradient(value)
            for key, value in bundle.metric_normalizers.items()
        },
        new_model_state=bundle.new_model_state,
        aux_sums={
            key: jax.lax.stop_gradient(value) for key, value in bundle.aux_sums.items()
        },
        aux_normalizers={
            key: jax.lax.stop_gradient(value)
            for key, value in bundle.aux_normalizers.items()
        },
    )


def _add_mappings(
    left: dict[str, Any] | Any,
    right: dict[str, Any] | Any,
) -> dict[str, Any]:
    keys = set(left) | set(right)
    return {key: left.get(key, 0) + right.get(key, 0) for key in keys}


def _select_mapping(
    condition: Any,
    true: dict[str, Any],
    false: dict[str, Any],
) -> dict[str, Any]:
    keys = set(true) | set(false)
    return {
        key: jnp.where(condition, true.get(key, 0), false.get(key, 0)) for key in keys
    }


def _resolve_microbatch_count(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    axis: int,
    count: int | None,
) -> int:
    if count is not None:
        return count
    sizes = {
        int(leaf.shape[axis])
        for leaf in jax.tree.leaves((args, kwargs), is_leaf=lambda x: x is None)
        if leaf is not None and hasattr(leaf, "shape") and len(leaf.shape) > axis
    }
    if not sizes:
        raise ValueError(
            "microbatch_axis was provided but no array argument has that axis; "
            "pass microbatch_count for broadcast-only arguments."
        )
    if len(sizes) != 1:
        raise ValueError(
            "microbatch_axis inferred inconsistent leading sizes; pass "
            "microbatch_count to broadcast non-microbatched arrays explicitly."
        )
    return sizes.pop()


def _slice_microbatch_tree(
    tree: Any,
    *,
    index: int,
    axis: int,
    count: int,
) -> Any:
    return jax.tree.map(
        lambda leaf: _slice_microbatch_leaf(
            leaf,
            index=index,
            axis=axis,
            count=count,
        ),
        tree,
        is_leaf=lambda x: x is None,
    )


def _slice_microbatch_leaf(
    leaf: Any,
    *,
    index: int,
    axis: int,
    count: int,
) -> Any:
    if leaf is None or not hasattr(leaf, "shape") or len(leaf.shape) <= axis:
        return leaf
    if int(leaf.shape[axis]) != count:
        return leaf
    return jnp.take(leaf, index, axis=axis)


def _tree_add(left: Any, right: Any) -> Any:
    return jax.tree.map(
        lambda lhs, rhs: None if lhs is None else lhs + rhs,
        left,
        right,
        is_leaf=lambda x: x is None,
    )


def _tree_scale(tree: Any, scale: float) -> Any:
    return jax.tree.map(
        lambda leaf: None if leaf is None else leaf * scale,
        tree,
        is_leaf=lambda x: x is None,
    )


def _tree_divide(tree: Any, denominator: Any) -> Any:
    return jax.tree.map(
        lambda leaf: None if leaf is None else leaf / denominator,
        tree,
        is_leaf=lambda x: x is None,
    )


def _cast_tree_like(tree: Any, template: Any) -> Any:
    return jax.tree.map(
        lambda leaf, target: (
            None
            if leaf is None
            else astype_preserving_sharding(leaf, target.dtype, target)
            if hasattr(target, "dtype")
            else leaf
        ),
        tree,
        template,
        is_leaf=lambda x: x is None,
    )


def _cast_gradient_tree(tree: Any, dtype: Any) -> Any:
    return jax.tree.map(
        lambda leaf: (
            None
            if leaf is None
            else astype_preserving_sharding(
                leaf,
                leaf.dtype
                if jnp.issubdtype(leaf.dtype, jnp.complexfloating)
                and not jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating)
                else dtype,
            )
        ),
        tree,
        is_leaf=lambda x: x is None,
    )


def _zeros_like_accumulator(tree: Any, dtype: Any) -> Any:
    def zeros_leaf(leaf):
        if leaf is None:
            return None
        leaf_dtype = (
            leaf.dtype
            if jnp.issubdtype(leaf.dtype, jnp.complexfloating)
            and not jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating)
            else dtype
        )
        return zeros_like_preserving_sharding(leaf, leaf_dtype)

    return jax.tree.map(
        zeros_leaf,
        tree,
        is_leaf=lambda x: x is None,
    )


def _advance_step_counters(
    counters: StepCounters,
    *,
    status: UpdateStatus,
    committed: Any,
    schedule_counter: Literal["optimizer", "micro"],
    dynamic_loss_scale: bool,
) -> StepCounters:
    one = jnp.asarray(1, dtype=jnp.int32)
    applied = jnp.asarray(status.update_applied, dtype=jnp.int32)
    attempted = jnp.asarray(jnp.logical_not(status.accumulating), dtype=jnp.int32)
    schedule_inc = one if schedule_counter == "micro" else applied
    loss_scale_inc = (
        jnp.asarray(committed, dtype=jnp.int32)
        if dynamic_loss_scale
        else jnp.asarray(0, dtype=jnp.int32)
    )
    return StepCounters(
        microstep=counters.microstep + one,
        attempted_update=counters.attempted_update + attempted,
        successful_update=counters.successful_update + applied,
        schedule_step=counters.schedule_step + schedule_inc,
        rank_step=counters.rank_step + applied,
        average_step=counters.average_step + applied,
        loss_scale_growth_step=counters.loss_scale_growth_step + loss_scale_inc,
    )


def _scale_loss_value(value: Any, loss_scale: Any, *, has_aux: bool) -> tuple[Any, Any]:
    if has_aux:
        loss, _ = value
        return loss * loss_scale, value
    return value * loss_scale, value


def _clone_model_state(model_state: Any | None) -> Any | None:
    # A stateful forward may consume its container; retain the original for rollback.
    # Rebuild only the PyTree structure, sharing the array leaves.
    return jax.tree.map(lambda leaf: leaf, model_state)


def _scale_stateful_loss_value(
    value_and_state: tuple[Any, Any | None],
    loss_scale: Any,
    *,
    has_aux: bool,
) -> tuple[Any, tuple[Any, Any | None]]:
    value, new_model_state = value_and_state
    return _primary_loss(value, has_aux=has_aux) * loss_scale, (
        value,
        new_model_state,
    )


def _primary_loss(value: Any, *, has_aux: bool) -> Any:
    if has_aux:
        loss, _ = value
        return loss
    return value


def _current_loss_scale(
    state: LossScaleState | None,
    precision: PrecisionConfig,
) -> Any:
    if precision.loss_scale == "none":
        return jnp.asarray(1.0, dtype=jnp.float32)
    if state is None:
        return jnp.asarray(precision.static_loss_scale, dtype=jnp.float32)
    return state.loss_scale


def _updated_loss_scale_state(
    state: LossScaleState | None,
    *,
    grads_finite: Any,
    accepted: Any,
    precision: PrecisionConfig,
) -> LossScaleState | None:
    if precision.loss_scale == "none":
        return None
    if state is None:
        state = cast(LossScaleState, init_loss_scale_state(precision))
    if precision.loss_scale == "static":
        return state

    success_tracker = state.growth_tracker + jnp.asarray(1, dtype=jnp.int32)
    should_grow = success_tracker >= precision.growth_interval
    grown_scale = jnp.where(
        should_grow,
        jnp.minimum(
            state.loss_scale * precision.growth_factor, precision.max_loss_scale
        ),
        state.loss_scale,
    )
    grown_tracker = jnp.where(
        should_grow,
        jnp.asarray(0, dtype=jnp.int32),
        success_tracker,
    )
    backed_off_scale = jnp.maximum(
        state.loss_scale * precision.backoff_factor, precision.min_loss_scale
    )
    # Only gradient overflow lowers the scale. A rejected update from finite
    # gradients is an optimizer failure that a smaller scale would not fix.
    succeeded = jnp.logical_and(grads_finite, accepted)
    return LossScaleState(
        loss_scale=jnp.where(
            succeeded,
            grown_scale,
            jnp.where(grads_finite, state.loss_scale, backed_off_scale),
        ),
        growth_tracker=jnp.where(
            succeeded,
            grown_tracker,
            jnp.where(
                grads_finite, state.growth_tracker, jnp.asarray(0, dtype=jnp.int32)
            ),
        ),
    )


def _tree_all_finite(tree: Any) -> Any:
    leaves = [
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree.leaves(tree, is_leaf=lambda x: x is None)
        if leaf is not None and hasattr(leaf, "dtype")
    ]
    if not leaves:
        return jnp.asarray(True)
    finite = leaves[0]
    for leaf_finite in leaves[1:]:
        finite = jnp.logical_and(finite, leaf_finite)
    return finite


def _tree_zeros_like(tree: Any) -> Any:
    return jax.tree.map(
        zeros_like_preserving_sharding,
        tree,
        is_leaf=lambda x: x is None,
    )


def _select_tree(condition: Any, true_tree: Any, false_tree: Any) -> Any:
    return jax.tree.map(
        lambda true_leaf, false_leaf: (
            None if true_leaf is None else jnp.where(condition, true_leaf, false_leaf)
        ),
        true_tree,
        false_tree,
        is_leaf=lambda x: x is None,
    )


def _commit_optimizer_result(
    attempt: jax.Array,
    propose: Callable[[], tuple[Any, jax.Array, UpdateStatus]],
    current: Any,
    optimizer: OptimizerBundle,
) -> tuple[Any, jax.Array, UpdateStatus]:
    """Return the proposed candidate when attempted and finite, else ``current``.

    ``propose`` returns ``(candidate, candidate_finite, status)``.
    """

    if _has_declared_collectives(optimizer):
        # Replica-local flags must not change participation in collectives,
        # and every partition must reach the same decision.
        candidate, finite, status = propose()
        accepted = _agree_across_axes(jnp.logical_and(attempt, finite), optimizer)
        return _select_tree(accepted, candidate, current), accepted, status

    def accept():
        candidate, finite, status = propose()
        # Retain where's dtype promotion for transforms that canonicalize state
        # on their first update, including restored mixed-precision state.
        return _select_tree(finite, candidate, current), finite, status

    proposed_jaxpr, result_shape = jax.make_jaxpr(accept, return_shape=True)()
    if proposed_jaxpr.effects:
        # Custom transformations can introduce collectives or callbacks that
        # are not described by the bundle's policies. Preserve those effects.
        candidate, finite, status = propose()
        accepted = jnp.logical_and(attempt, finite)
        return _select_tree(accepted, candidate, current), accepted, status

    def reject():
        rejected_status = UpdateStatus(
            accumulating=jnp.asarray(False),
            update_applied=jnp.asarray(False),
            window_discarded=jnp.asarray(True),
            microsteps_in_window=jnp.asarray(0, dtype=jnp.int32),
        )
        return _cast_tree_like(
            (current, jnp.asarray(False), rejected_status), result_shape
        )

    return jax.lax.cond(attempt, accept, reject)


def _has_declared_collectives(optimizer: OptimizerBundle) -> bool:
    policy = optimizer.gradient_policy
    return (
        policy.axis_name is not None
        or policy.partition_axis_names is not None
        or policy.replicated_axis_names is not None
        or bool(optimizer.sharding_policy.mesh_axes)
    )


def _agree_across_axes(flag: Any, optimizer: OptimizerBundle) -> Any:
    """Return whether ``flag`` holds on every parameter-partition participant.

    The axes are the ones global-norm clipping reduces over, so they must be
    bound by ``shard_map``, ``pmap``, or ``vmap`` whenever they are declared.
    """

    policy = optimizer.gradient_policy
    axis_name = resolve_partition_norm_axis_name(
        axis_name=policy.axis_name,
        partition_axis_names=policy.partition_axis_names,
        replicated_axis_names=policy.replicated_axis_names,
    )
    if axis_name is None:
        return flag
    failures = jax.lax.psum(
        jnp.asarray(jnp.logical_not(flag), dtype=jnp.int32), axis_name=axis_name
    )
    return failures == 0


def _apply_parameter_updates(
    params: Any,
    updates: Any,
    optimizer: OptimizerBundle,
    key: jax.Array | None,
) -> Any:
    if optimizer.precision_config.parameter_rounding != "stochastic":
        return apply_updates(params, updates, stochastic=False)
    partition_axes = optimizer.gradient_policy.partition_axis_names
    if partition_axes is not None:
        # Parameter shards need independent noise; replicas share the key so
        # replicated parameters stay identical.
        key = jax.random.fold_in(
            cast(jax.Array, key), jax.lax.axis_index(partition_axes)
        )
    return apply_updates(params, updates, key, stochastic=True)


def _unscale_gradients(
    optimizer: OptimizerBundle,
    scaled_grads: Any,
    scale: Any,
    loss: Any,
) -> tuple[Any, Any]:
    """Return unscaled gradients in the gradient dtype and their finiteness."""

    grads = _tree_divide(scaled_grads, scale)
    grads = _cast_gradient_tree(grads, optimizer.precision_config.gradient_dtype)
    return grads, _agree_across_axes(_tree_all_finite((loss, grads)), optimizer)


def _resolve_master_params(
    visible_params: Any,
    master_params: Any | None,
    precision: PrecisionConfig,
) -> Any | None:
    if master_params is None:
        return make_master_params(visible_params, precision)
    if precision.parameter_rounding == "stochastic":
        raise ValueError(
            "parameter_rounding='stochastic' updates master-free parameters; "
            "pass master_params=None."
        )
    _check_master_params(visible_params, master_params, precision)
    return master_params


def _check_master_params(
    visible_params: Any,
    master_params: Any,
    precision: PrecisionConfig,
) -> None:
    """Reject caller-supplied masters that cannot represent the visible params."""

    visible_paths = jax.tree_util.tree_leaves_with_path(
        visible_params, is_leaf=lambda x: x is None
    )
    master_leaves = jax.tree_util.tree_leaves(
        master_params, is_leaf=lambda x: x is None
    )
    if jax.tree.structure(
        visible_params, is_leaf=lambda x: x is None
    ) != jax.tree.structure(master_params, is_leaf=lambda x: x is None):
        raise ValueError("master_params must have the visible parameters' structure.")
    for (path, visible), master in zip(visible_paths, master_leaves, strict=True):
        if visible is None or master is None:
            if (visible is None) != (master is None):
                raise ValueError(
                    "master_params and visible parameters disagree on missing "
                    f"leaves at {jax.tree_util.keystr(path)}."
                )
            continue
        if tuple(master.shape) != tuple(visible.shape):
            raise ValueError(
                f"master parameter at {jax.tree_util.keystr(path)} has shape "
                f"{tuple(master.shape)}; expected {tuple(visible.shape)}."
            )
        expected = master_leaf_dtype(visible.dtype, precision.master_param_dtype)
        if jnp.promote_types(master.dtype, expected) != jnp.dtype(master.dtype):
            raise ValueError(
                f"master parameter at {jax.tree_util.keystr(path)} has dtype "
                f"{jnp.dtype(master.dtype).name}; expected at least {expected.name}."
            )


def _check_step_state_masters(
    visible_params: Any,
    step_state: FineTuneStepState,
    optimizer: OptimizerBundle,
) -> None:
    if step_state.master_params is None and uses_master_params(
        visible_params, optimizer.precision_config
    ):
        # Creating masters here would change the state structure between calls.
        raise ValueError(
            "FineTuneStepState.master_params is None but the precision policy "
            "keeps master parameters; build the state with init_finetune_step_state."
        )


def _advance_step_state(
    step_state: FineTuneStepState,
    optimizer: OptimizerBundle,
    *,
    status: UpdateStatus,
    committed: Any,
    optimizer_state: Any,
    model_state: Any | None,
    master_params: Any | None,
    loss_scale: LossScaleState | None,
    rng: RNGStreams | None,
) -> FineTuneStepState:
    return replace(
        step_state,
        optimizer_state=optimizer_state,
        model_state=model_state,
        master_params=master_params,
        accumulation=replace(
            step_state.accumulation,
            microsteps_in_window=jnp.asarray(
                status.microsteps_in_window,
                dtype=jnp.asarray(step_state.accumulation.microsteps_in_window).dtype,
            ),
        ),
        loss_scale=loss_scale,
        counters=_advance_step_counters(
            step_state.counters,
            status=status,
            committed=committed,
            schedule_counter=optimizer.schedule_config.step_counter,
            dynamic_loss_scale=optimizer.precision_config.loss_scale == "dynamic",
        ),
        rng=rng,
    )


def _window_accumulation_state(accumulation: AccumulationConfig) -> AccumulationState:
    """Return an accumulation record that tracks the window position only.

    Composite steps accumulate gradients in the optimizer state, so the record
    holds no gradient-sized buffer.
    """

    zero = jnp.asarray(0, dtype=accumulation.accumulate_dtype)
    return AccumulationState(
        grad_numerator=None,
        loss_sum=zero,
        normalizer=zero,
        metric_sums={},
        metric_normalizers={},
        aux_sums={},
        aux_normalizers={},
        microsteps_in_window=jnp.asarray(0, dtype=jnp.int32),
        all_finite=jnp.asarray(True),
        pending_model_state=None,
        pending_model_state_valid=jnp.asarray(False),
    )


def _microsteps_or_zero(opt_state: Any) -> jax.Array:
    pending = _pending_microsteps(opt_state)
    if pending is None:
        return jnp.asarray(0, dtype=jnp.int32)
    return jnp.asarray(pending, dtype=jnp.int32)


def _promote_accumulator(value: Any, dtype: Any) -> Any:
    """Widen a floating value to the accumulation dtype without narrowing it."""

    if (
        value is None
        or not hasattr(value, "dtype")
        or not jnp.issubdtype(value.dtype, jnp.inexact)
    ):
        return value
    return astype_preserving_sharding(value, jnp.promote_types(value.dtype, dtype))


def _promote_accumulator_tree(tree: Any, dtype: Any) -> Any:
    return jax.tree.map(
        lambda leaf: _promote_accumulator(leaf, dtype),
        tree,
        is_leaf=lambda x: x is None,
    )


def _promote_loss_bundle(bundle: LossBundle, dtype: Any) -> LossBundle:
    """Promote a bundle's sums and normalizers, leaving its model state alone."""

    def promote(values: Any) -> dict[str, Any]:
        return {
            key: _promote_accumulator(value, dtype) for key, value in values.items()
        }

    return replace(
        bundle,
        loss_sum=_promote_accumulator(bundle.loss_sum, dtype),
        normalizer=_promote_accumulator(bundle.normalizer, dtype),
        metrics_sums=promote(bundle.metrics_sums),
        metric_normalizers=promote(bundle.metric_normalizers),
        aux_sums=promote(bundle.aux_sums),
        aux_normalizers=promote(bundle.aux_normalizers),
    )


def _require_rng_free_rounding(optimizer: OptimizerBundle, helper: str) -> None:
    if optimizer.precision_config.parameter_rounding == "stochastic":
        raise ValueError(
            f"{helper} carries no RNG, so it cannot apply "
            f"parameter_rounding='stochastic'; use {_RNG_STEPS}."
        )


def _check_accumulation_guard(optimizer: OptimizerBundle, helper: str) -> None:
    if (
        optimizer.accumulation_config.steps > 1
        and optimizer.gradient_policy.nonfinite != "skip"
    ):
        raise NotImplementedError(
            f"{helper} discards nonfinite accumulation windows through the "
            "bundle's finite guard; accumulation requires "
            "GradientPolicy(nonfinite='skip')."
        )


def _check_direct_precision(
    optimizer: OptimizerBundle,
    helper: str,
    *,
    allows_stochastic_rounding: bool = False,
) -> None:
    """Warn when a direct-parameter helper ignores parts of the precision policy."""

    if not allows_stochastic_rounding:
        _require_rng_free_rounding(optimizer, helper)
    ignored = []
    report = optimizer.report.precision
    if report is not None and report.master_params:
        ignored.append("master parameters")
    if optimizer.precision_config.loss_scale != "none":
        ignored.append("loss scaling")
    if ignored:
        warnings.warn(
            f"{helper} updates the supplied parameters directly and ignores the "
            f"bundle's {' and '.join(ignored)}; use make_finetune_update_step or "
            "make_finetune_sam_step to honor them.",
            UserWarning,
            stacklevel=3,
        )


def _check_state_policy(
    state_policy: str,
    model_state_aggregator: ModelStateAggregator | None = None,
) -> None:
    if state_policy == "optimizer_step_aggregate" and model_state_aggregator is None:
        raise NotImplementedError(
            "optimizer_step_aggregate requires a model-provided state aggregation hook."
        )
    if state_policy not in {
        "frozen",
        "microbatch_sequential",
        "optimizer_step_aggregate",
        "external",
    }:
        raise ValueError(f"unknown model state policy: {state_policy!r}.")


def _next_forward_key(rng: RNGStreams | None) -> tuple[RNGStreams | None, Any | None]:
    if rng is None:
        return None, None
    forward, forward_key = jax.random.split(rng.forward)
    return replace(rng, forward=forward), forward_key


def _next_sam_key(rng: RNGStreams | None) -> tuple[RNGStreams | None, Any | None]:
    if rng is None:
        return None, None
    sam, sam_key = jax.random.split(rng.sam)
    return replace(rng, sam=sam), sam_key


def _commit_model_state(
    state_policy: str,
    old_state: Any | None,
    proposed_state: Any | None,
    all_finite: Any,
    *,
    model_state_aggregator: ModelStateAggregator | None = None,
) -> Any | None:
    if state_policy in {"frozen", "external"} or proposed_state is None:
        return old_state
    if old_state is None:
        return None
    if state_policy == "optimizer_step_aggregate":
        if model_state_aggregator is None:
            raise NotImplementedError(
                "optimizer_step_aggregate requires a model-provided state "
                "aggregation hook."
            )
        proposed_state = model_state_aggregator(old_state, proposed_state)
    return _select_tree(all_finite, proposed_state, old_state)


def _next_rounding_key(
    rng: RNGStreams | None,
    precision: PrecisionConfig,
) -> tuple[RNGStreams | None, Any | None]:
    if precision.parameter_rounding != "stochastic":
        return rng, None
    if rng is None:
        raise ValueError(
            "parameter_rounding='stochastic' draws keys from "
            "RNGStreams.stochastic_rounding; pass rng from init_rng_streams."
        )
    stream, key = jax.random.split(rng.stochastic_rounding)
    return replace(rng, stochastic_rounding=stream), key


def _commit_rng(
    current: RNGStreams | None,
    candidate: RNGStreams | None,
    *,
    committed: Any,
    update_applied: Any,
) -> RNGStreams | None:
    """Advance consumed streams: rounding keys follow applied updates only."""

    if current is None or candidate is None:
        return current

    def pick(condition, name):
        return jnp.where(condition, getattr(candidate, name), getattr(current, name))

    return RNGStreams(
        forward=pick(committed, "forward"),
        sam=pick(committed, "sam"),
        stochastic_rounding=pick(update_applied, "stochastic_rounding"),
        quantization=pick(committed, "quantization"),
        controller=pick(committed, "controller"),
    )


def sam_cost_report(
    plan: FineTunePlanProtocol,
    config: SAMConfig | ASAMConfig | None = None,
) -> dict[str, Any]:
    """Return static cost metadata for a SAM/ASAM fine-tuning step."""

    config = SAMConfig(enabled=True) if config is None else config
    adaptive = isinstance(config, ASAMConfig)
    mask = _sam_perturb_mask(plan, config)
    perturbation_bytes = 0
    for param, include in zip(
        jax.tree.leaves(plan.trainable, is_leaf=lambda x: x is None),
        jax.tree.leaves(mask, is_leaf=lambda x: x is None),
        strict=True,
    ):
        if param is not None and include:
            perturbation_bytes += int(param.size * param.dtype.itemsize)
    return {
        "method": "ASAM" if adaptive else "SAM",
        "forward_backward_evaluations": 2,
        "perturbation_bytes": perturbation_bytes,
        "rho": config.rho,
        "adaptive": adaptive,
        "eta": config.eta if adaptive else 0.0,
        "norm": config.norm,
    }


def _split_value_aux(value: Any, *, has_aux: bool) -> tuple[Any, Any | None]:
    if not has_aux:
        return value, None
    loss, aux = value
    return loss, aux


def _sam_perturb_mask(
    plan: FineTunePlanProtocol,
    config: SAMConfig | ASAMConfig,
) -> Any:
    group_specs = getattr(plan, "group_specs", {})
    return jax.tree.map(
        lambda label: (
            None
            if label is None
            else _perturb_label(str(label), group_specs.get(str(label)), config)
        ),
        plan.labels,
        is_leaf=lambda x: x is None,
    )


def _perturb_label(label: str, group: Any, config: SAMConfig | ASAMConfig) -> bool:
    terms = {label.lower()}
    if group is not None:
        terms.add(str(getattr(group, "role", "")).lower())
        terms.update(str(tag).lower() for tag in getattr(group, "tags", ()))
    if not config.perturb_bias and any("bias" in term for term in terms):
        return False
    return not (not config.perturb_norm and any("norm" in term for term in terms))


__all__ = (
    "AccumulatingLossBundleInfo",
    "ModelStateAggregator",
    "SAMStepInfo",
    "StatefulSAMStepInfo",
    "StatefulStepInfo",
    "init_accumulation_state",
    "init_finetune_step_state",
    "init_rng_streams",
    "init_step_counters",
    "make_accumulating_loss_bundle_update_step",
    "make_finetune_sam_step",
    "make_finetune_update_step",
    "make_loss_bundle_update_step",
    "make_loss_scaled_master_update_step",
    "make_master_params",
    "make_master_update_step",
    "make_plan_loss_bundle_update_step",
    "make_plan_loss_scaled_master_update_step",
    "make_plan_master_update_step",
    "make_plan_update_step",
    "make_sam_step",
    "make_stateful_loss_scaled_master_update_step",
    "make_stateful_sam_step",
    "make_update_step",
    "sam_cost_report",
)
