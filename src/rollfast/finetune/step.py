"""Small functional update helpers for plan-aware optimizers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax

from rollfast.optim.sam import add_perturbation, global_l2_norm, sam_perturbation
from rollfast.utils import astype_preserving_sharding, zeros_like_preserving_sharding

from ._protocols import FineTunePlanProtocol
from .config import (
    ASAMConfig,
    AccumulationConfig,
    AccumulationState,
    FineTuneStepState,
    LossBundle,
    LossScaleState,
    OptimizerBundle,
    PrecisionConfig,
    RNGStreams,
    SAMConfig,
    StepCounters,
)


class SAMStepInfo(NamedTuple):
    """Diagnostics returned by a two-pass SAM/ASAM update."""

    loss: Any
    perturbed_loss: Any
    aux: Any | None
    perturbed_aux: Any | None
    grad_norm: Any
    perturbation_norm: Any


class StatefulStepInfo(NamedTuple):
    """Diagnostics returned by a model-state-aware update."""

    value: Any
    proposed_model_state: Any | None
    all_finite: Any
    forward_key: Any | None


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


class AccumulatingLossBundleInfo(NamedTuple):
    """Diagnostics returned by an exact accumulated ``LossBundle`` step."""

    loss_bundle: LossBundle
    update_applied: Any
    all_finite: Any


ModelStateAggregator = Callable[[Any | None, Any | None], Any | None]


def make_update_step(
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return a one-update function over a trainable PyTree."""

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def step(trainable, opt_state, *args, **kwargs):
        value, grads = value_and_grad(trainable, *args, **kwargs)
        updates, opt_state = optimizer.update(grads, opt_state, trainable)
        trainable = optax.apply_updates(trainable, updates)
        return trainable, opt_state, value

    return step


def make_plan_update_step(
    plan: FineTunePlanProtocol,
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return a one-update function whose loss receives ``plan.combine`` output."""

    def trainable_loss(trainable, *args, **kwargs):
        return loss_fn(plan.combine(trainable), *args, **kwargs)

    return make_update_step(trainable_loss, optimizer, has_aux=has_aux)


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
        normalizer=jnp.asarray(0, dtype=accumulation.accumulate_dtype),
        metric_sums={},
        metric_normalizers={},
        microsteps_in_window=zero,
        all_finite=jnp.asarray(True),
        pending_model_state=pending_model_state,
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
    """Initialize the canonical composite fine-tuning step state."""

    if master_params is None:
        master_params = make_master_params(trainable, bundle.precision_config)
    optimizer_params = trainable if master_params is None else master_params
    if optimizer_state is None:
        optimizer_state = bundle.init(optimizer_params)
    if accumulation is None:
        accumulation = init_accumulation_state(trainable, bundle.accumulation_config)
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
    """Return fp32-style master params when the precision policy requires them."""

    if not _uses_master_params(trainable, precision):
        return None
    return jax.tree.map(
        lambda leaf: (
            None
            if leaf is None
            else astype_preserving_sharding(leaf, precision.master_param_dtype)
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
    """Return an update step that applies optimizer updates to master params."""

    value_and_grad = jax.value_and_grad(
        lambda master, visible_template, *args, **kwargs: loss_fn(
            _cast_tree_like(master, visible_template),
            *args,
            **kwargs,
        ),
        has_aux=has_aux,
    )

    def step(visible_params, master_params, opt_state, *args, **kwargs):
        if master_params is None:
            master_params = make_master_params(
                visible_params,
                optimizer.precision_config,
            )
        if master_params is None:
            master_params = visible_params
        value, grads = value_and_grad(master_params, visible_params, *args, **kwargs)
        grads = _cast_gradient_tree(grads, optimizer.precision_config.gradient_dtype)
        updates, opt_state = optimizer.update(grads, opt_state, master_params)
        master_params = optax.apply_updates(master_params, updates)
        visible_params = _cast_tree_like(master_params, visible_params)
        return visible_params, master_params, opt_state, value

    return step


def make_loss_scaled_master_update_step(
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return a master-parameter step with static/dynamic loss scaling."""

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
        if master_params is None:
            master_params = make_master_params(visible_params, precision)
        if master_params is None:
            master_params = visible_params
        if loss_scale_state is None:
            loss_scale_state = init_loss_scale_state(precision)
        scale = _current_loss_scale(loss_scale_state, precision)
        (_, value), scaled_grads = scaled_value_and_grad(
            master_params,
            visible_params,
            scale,
            *args,
            **kwargs,
        )
        grads = _tree_divide(scaled_grads, scale)
        all_finite = _tree_all_finite((_primary_loss(value, has_aux=has_aux), grads))
        grads = _select_tree(all_finite, grads, _tree_zeros_like(grads))
        grads = _cast_gradient_tree(grads, precision.gradient_dtype)
        updates, candidate_opt_state = optimizer.update(grads, opt_state, master_params)
        candidate_master = optax.apply_updates(master_params, updates)
        candidate_visible = _cast_tree_like(candidate_master, visible_params)
        visible_params = _select_tree(
            all_finite,
            candidate_visible,
            visible_params,
        )
        master_params = _select_tree(all_finite, candidate_master, master_params)
        opt_state = _select_tree(all_finite, candidate_opt_state, opt_state)
        loss_scale_state = _updated_loss_scale_state(
            loss_scale_state,
            all_finite,
            precision,
        )
        return (
            visible_params,
            master_params,
            opt_state,
            loss_scale_state,
            value,
            all_finite,
        )

    return step


def make_stateful_loss_scaled_master_update_step(
    plan: FineTunePlanProtocol,
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
    """Return a model-state/RNG-aware master step with loss scaling."""

    _check_state_policy(state_policy, model_state_aggregator)

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
                model_state,
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
        if master_params is None:
            master_params = make_master_params(visible_params, precision)
        if master_params is None:
            master_params = visible_params
        if loss_scale_state is None:
            loss_scale_state = init_loss_scale_state(precision)
        candidate_rng, forward_key = _next_forward_key(rng)
        scale = _current_loss_scale(loss_scale_state, precision)
        (_, (value, proposed_model_state)), scaled_grads = scaled_value_and_grad(
            master_params,
            visible_params,
            model_state,
            forward_key,
            scale,
            *args,
            **kwargs,
        )
        grads = _tree_divide(scaled_grads, scale)
        all_finite = _tree_all_finite((_primary_loss(value, has_aux=has_aux), grads))
        grads = _select_tree(all_finite, grads, _tree_zeros_like(grads))
        grads = _cast_gradient_tree(grads, precision.gradient_dtype)
        updates, candidate_opt_state = optimizer.update(grads, opt_state, master_params)
        candidate_master = optax.apply_updates(master_params, updates)
        candidate_visible = _cast_tree_like(candidate_master, visible_params)
        visible_params = _select_tree(
            all_finite,
            candidate_visible,
            visible_params,
        )
        master_params = _select_tree(all_finite, candidate_master, master_params)
        opt_state = _select_tree(all_finite, candidate_opt_state, opt_state)
        model_state = _commit_model_state(
            state_policy,
            model_state,
            proposed_model_state,
            all_finite,
            model_state_aggregator=model_state_aggregator,
        )
        rng = _select_rng(all_finite, candidate_rng, rng)
        loss_scale_state = _updated_loss_scale_state(
            loss_scale_state,
            all_finite,
            precision,
        )
        info = StatefulStepInfo(
            value=value,
            proposed_model_state=proposed_model_state,
            all_finite=all_finite,
            forward_key=forward_key,
        )
        return (
            visible_params,
            master_params,
            opt_state,
            model_state,
            loss_scale_state,
            rng,
            info,
        )

    return step


def make_finetune_update_step(
    plan: FineTunePlanProtocol,
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
    """Return a step over visible params plus ``FineTuneStepState``."""

    if optimizer.accumulation_config.steps != 1:
        raise NotImplementedError(
            "FineTuneStepState update currently requires accumulation steps=1."
        )
    inner_step = make_stateful_loss_scaled_master_update_step(
        plan,
        loss_fn,
        optimizer,
        state_policy=state_policy,
        has_aux=has_aux,
        model_state_aggregator=model_state_aggregator,
    )

    def step(visible_params, step_state: FineTuneStepState, *args, **kwargs):
        (
            visible_params,
            master_params,
            opt_state,
            model_state,
            loss_scale_state,
            rng,
            info,
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
        step_state = replace(
            step_state,
            optimizer_state=opt_state,
            model_state=model_state,
            master_params=master_params,
            accumulation=init_accumulation_state(
                visible_params,
                optimizer.accumulation_config,
            ),
            loss_scale=loss_scale_state,
            counters=_advance_step_counters(
                step_state.counters,
                all_finite=info.all_finite,
                schedule_counter=optimizer.schedule_config.step_counter,
                dynamic_loss_scale=optimizer.precision_config.loss_scale == "dynamic",
            ),
            rng=rng,
        )
        return visible_params, step_state, info

    return step


def make_plan_loss_scaled_master_update_step(
    plan: FineTunePlanProtocol,
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
    plan: FineTunePlanProtocol,
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
    """Return an update step for exact summed-loss/normalizer contracts."""

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
        )
        grads = _tree_divide(grads, bundle.normalizer)
        updates, opt_state = optimizer.update(grads, opt_state, trainable)
        trainable = optax.apply_updates(trainable, updates)
        return trainable, opt_state, _normalized_loss_bundle(bundle)

    return step


def make_accumulating_loss_bundle_update_step(
    loss_fn: Callable[..., LossBundle],
    optimizer: OptimizerBundle,
    *,
    accumulation: AccumulationConfig,
    model_state_aggregator: ModelStateAggregator | None = None,
) -> Callable[..., Any]:
    """Return a stateful exact numerator/normalizer accumulation step."""

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
            model_state_aggregator=model_state_aggregator,
        )
        accumulated_bundle = _loss_bundle_from_accumulation(next_accumulation)
        averaged_grads = _tree_divide(
            next_accumulation.grad_numerator,
            next_accumulation.normalizer,
        )
        boundary = next_accumulation.microsteps_in_window >= accumulation.steps
        update_applied = jnp.logical_and(boundary, next_accumulation.all_finite)
        updates, candidate_opt_state = optimizer.update(
            averaged_grads,
            opt_state,
            trainable,
        )
        candidate_trainable = optax.apply_updates(trainable, updates)
        trainable = _select_tree(update_applied, candidate_trainable, trainable)
        opt_state = _select_tree(update_applied, candidate_opt_state, opt_state)
        reset_accumulation = init_accumulation_state(trainable, accumulation)
        accumulation_state = _select_accumulation_state(
            boundary,
            reset_accumulation,
            next_accumulation,
        )
        info = AccumulatingLossBundleInfo(
            loss_bundle=_normalized_loss_bundle(accumulated_bundle),
            update_applied=update_applied,
            all_finite=next_accumulation.all_finite,
        )
        return trainable, opt_state, accumulation_state, info

    return step


def make_plan_loss_bundle_update_step(
    plan: FineTunePlanProtocol,
    loss_fn: Callable[..., LossBundle],
    optimizer: OptimizerBundle,
    *,
    microbatch_axis: int | None = None,
    microbatch_count: int | None = None,
) -> Callable[..., Any]:
    """Return a ``LossBundle`` update step whose loss receives ``plan.combine``."""

    def trainable_loss(trainable, *args, **kwargs):
        return loss_fn(plan.combine(trainable), *args, **kwargs)

    return make_loss_bundle_update_step(
        trainable_loss,
        optimizer,
        microbatch_axis=microbatch_axis,
        microbatch_count=microbatch_count,
    )


def make_sam_step(
    *,
    plan: FineTunePlanProtocol,
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
    """

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
        perturbed = add_perturbation(trainable, perturbation)
        perturbed_value, perturbed_grads = _evaluate_sam_value_and_grad(
            value_and_grad,
            perturbed,
            args,
            kwargs,
            microbatch_axis=microbatch_axis,
            microbatch_count=microbatch_count,
            microbatch_reduction=microbatch_reduction,
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
        trainable = optax.apply_updates(trainable, updates)
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
    plan: FineTunePlanProtocol,
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
    pass state updates are always discarded.
    """

    config = SAMConfig(enabled=True) if config is None else config
    _check_state_policy(state_policy, model_state_aggregator)
    adaptive = isinstance(config, ASAMConfig)
    eta = config.eta if adaptive else 0.0
    if base_optimizer.accumulation_config.steps != 1:
        raise ValueError(
            "stateful SAM/ASAM steps require a base optimizer with accumulation_steps=1."
        )
    mask = _sam_perturb_mask(plan, config)

    def stateful_loss(trainable, model_state, sam_key, *args, **kwargs):
        return _scale_stateful_loss_value(
            loss_fn(
                plan.combine(trainable),
                model_state,
                sam_key,
                *args,
                **kwargs,
            ),
            jnp.asarray(1.0, dtype=jnp.float32),
            has_aux=has_aux,
        )

    value_and_grad = jax.value_and_grad(stateful_loss, has_aux=True)

    def step(trainable, opt_state, model_state, rng, *args, **kwargs):
        candidate_rng, sam_key = _next_sam_key(rng)
        (_, (value, proposed_model_state)), grads = value_and_grad(
            trainable,
            model_state,
            sam_key,
            *args,
            **kwargs,
        )
        loss, aux = _split_value_aux(value, has_aux=has_aux)
        first_finite = _tree_all_finite((loss, grads))
        safe_grads = _select_tree(first_finite, grads, _tree_zeros_like(grads))
        perturbation, perturbation_norm = sam_perturbation(
            safe_grads,
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
        perturbed = add_perturbation(trainable, perturbation)
        (_, (perturbed_value, perturbed_model_state)), perturbed_grads = value_and_grad(
            perturbed,
            model_state,
            sam_key,
            *args,
            **kwargs,
        )
        perturbed_loss, perturbed_aux = _split_value_aux(
            perturbed_value,
            has_aux=has_aux,
        )
        second_finite = _tree_all_finite((perturbed_loss, perturbed_grads))
        all_finite = jnp.logical_and(first_finite, second_finite)
        safe_perturbed_grads = _select_tree(
            all_finite,
            perturbed_grads,
            _tree_zeros_like(perturbed_grads),
        )
        updates, candidate_opt_state = base_optimizer.update(
            safe_perturbed_grads,
            opt_state,
            trainable,
        )
        candidate_trainable = optax.apply_updates(trainable, updates)
        trainable = _select_tree(all_finite, candidate_trainable, trainable)
        opt_state = _select_tree(all_finite, candidate_opt_state, opt_state)
        model_state = _commit_model_state(
            state_policy,
            model_state,
            proposed_model_state,
            all_finite,
            model_state_aggregator=model_state_aggregator,
        )
        rng = _select_rng(all_finite, candidate_rng, rng)
        failed_pass = jnp.where(
            first_finite,
            jnp.where(second_finite, 0, 2),
            1,
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
            all_finite=all_finite,
            failed_pass=failed_pass,
            sam_key=sam_key,
        )
        return trainable, opt_state, model_state, rng, info

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
) -> tuple[Any, Any]:
    if microbatch_axis is None:
        return value_and_grad(trainable, *args, **kwargs)

    count = _resolve_microbatch_count(
        args,
        kwargs,
        axis=microbatch_axis,
        count=microbatch_count,
    )
    total_value = None
    total_grads = None
    for index in range(count):
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
        value, grads = value_and_grad(trainable, *micro_args, **micro_kwargs)
        total_value = value if total_value is None else _tree_add(total_value, value)
        total_grads = grads if total_grads is None else _tree_add(total_grads, grads)

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
    total_bundle = None
    total_grads = None
    for index in range(count):
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
        total_bundle = (
            bundle if total_bundle is None else _add_loss_bundles(total_bundle, bundle)
        )
        total_grads = grads if total_grads is None else _tree_add(total_grads, grads)
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
    model_state_aggregator: ModelStateAggregator | None = None,
) -> AccumulationState:
    return AccumulationState(
        grad_numerator=_tree_add(state.grad_numerator, grads),
        normalizer=state.normalizer + bundle.normalizer,
        metric_sums=_add_mappings(state.metric_sums, bundle.metrics_sums),
        metric_normalizers=_add_mappings(
            state.metric_normalizers,
            bundle.metric_normalizers,
        ),
        microsteps_in_window=state.microsteps_in_window
        + jnp.asarray(1, dtype=jnp.int32),
        all_finite=jnp.logical_and(state.all_finite, all_finite),
        pending_model_state=_merge_pending_model_state(
            state.pending_model_state,
            bundle.new_model_state,
            model_state_aggregator,
        ),
    )


def _loss_bundle_from_accumulation(state: AccumulationState) -> LossBundle:
    return LossBundle(
        loss_sum=state.normalizer * jnp.asarray(0, dtype=state.normalizer.dtype),
        normalizer=state.normalizer,
        metrics_sums=state.metric_sums,
        metric_normalizers=state.metric_normalizers,
        new_model_state=state.pending_model_state,
    )


def _merge_pending_model_state(
    current_state: Any | None,
    new_state: Any | None,
    model_state_aggregator: ModelStateAggregator | None,
) -> Any | None:
    if model_state_aggregator is None:
        return new_state
    if new_state is None:
        return current_state
    if current_state is None:
        return new_state
    return model_state_aggregator(current_state, new_state)


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
        microsteps_in_window=jnp.where(
            condition,
            true_state.microsteps_in_window,
            false_state.microsteps_in_window,
        ),
        all_finite=jnp.where(condition, true_state.all_finite, false_state.all_finite),
        pending_model_state=false_state.pending_model_state,
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


def _uses_master_params(trainable: Any, precision: PrecisionConfig) -> bool:
    if precision.master_params == "always":
        return True
    if precision.master_params == "never":
        return False
    for leaf in jax.tree.leaves(trainable, is_leaf=lambda x: x is None):
        if leaf is not None and jnp.dtype(leaf.dtype) in {
            jnp.dtype(jnp.float16),
            jnp.dtype(jnp.bfloat16),
        }:
            return True
    return False


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
        lambda leaf: None if leaf is None else astype_preserving_sharding(leaf, dtype),
        tree,
        is_leaf=lambda x: x is None,
    )


def _zeros_like_accumulator(tree: Any, dtype: Any) -> Any:
    return jax.tree.map(
        lambda leaf: zeros_like_preserving_sharding(leaf, dtype),
        tree,
        is_leaf=lambda x: x is None,
    )


def _advance_step_counters(
    counters: StepCounters,
    *,
    all_finite: Any,
    schedule_counter: Literal["optimizer", "micro"],
    dynamic_loss_scale: bool,
) -> StepCounters:
    one = jnp.asarray(1, dtype=jnp.int32)
    success = jnp.asarray(all_finite, dtype=jnp.int32)
    schedule_inc = one if schedule_counter == "micro" else success
    loss_scale_inc = success if dynamic_loss_scale else jnp.asarray(0, dtype=jnp.int32)
    return StepCounters(
        microstep=counters.microstep + one,
        attempted_update=counters.attempted_update + one,
        successful_update=counters.successful_update + success,
        schedule_step=counters.schedule_step + schedule_inc,
        rank_step=counters.rank_step + success,
        average_step=counters.average_step + success,
        loss_scale_growth_step=counters.loss_scale_growth_step + loss_scale_inc,
    )


def _scale_loss_value(value: Any, loss_scale: Any, *, has_aux: bool) -> tuple[Any, Any]:
    if has_aux:
        loss, _ = value
        return loss * loss_scale, value
    return value * loss_scale, value


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
    all_finite: Any,
    precision: PrecisionConfig,
) -> LossScaleState | None:
    if precision.loss_scale == "none":
        return None
    if state is None:
        state = init_loss_scale_state(precision)
    if precision.loss_scale == "static":
        return state

    success_tracker = state.growth_tracker + jnp.asarray(1, dtype=jnp.int32)
    should_grow = success_tracker >= precision.growth_interval
    grown_scale = jnp.where(
        should_grow,
        state.loss_scale * precision.growth_factor,
        state.loss_scale,
    )
    grown_tracker = jnp.where(
        should_grow,
        jnp.asarray(0, dtype=jnp.int32),
        success_tracker,
    )
    return LossScaleState(
        loss_scale=jnp.where(
            all_finite,
            grown_scale,
            state.loss_scale * precision.backoff_factor,
        ),
        growth_tracker=jnp.where(
            all_finite,
            grown_tracker,
            jnp.asarray(0, dtype=jnp.int32),
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


def _select_rng(
    condition: Any,
    true_rng: RNGStreams | None,
    false_rng: RNGStreams | None,
) -> RNGStreams | None:
    if true_rng is None or false_rng is None:
        return false_rng
    return RNGStreams(
        forward=jnp.where(condition, true_rng.forward, false_rng.forward),
        sam=jnp.where(condition, true_rng.sam, false_rng.sam),
        stochastic_rounding=jnp.where(
            condition,
            true_rng.stochastic_rounding,
            false_rng.stochastic_rounding,
        ),
        quantization=jnp.where(
            condition,
            true_rng.quantization,
            false_rng.quantization,
        ),
        controller=jnp.where(condition, true_rng.controller, false_rng.controller),
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
    if not config.perturb_norm and any("norm" in term for term in terms):
        return False
    return True


__all__ = (
    "SAMStepInfo",
    "AccumulatingLossBundleInfo",
    "ModelStateAggregator",
    "StatefulSAMStepInfo",
    "StatefulStepInfo",
    "init_accumulation_state",
    "init_finetune_step_state",
    "init_rng_streams",
    "init_step_counters",
    "make_finetune_update_step",
    "make_accumulating_loss_bundle_update_step",
    "make_loss_bundle_update_step",
    "make_loss_scaled_master_update_step",
    "make_master_params",
    "make_master_update_step",
    "make_plan_loss_scaled_master_update_step",
    "make_plan_update_step",
    "make_plan_loss_bundle_update_step",
    "make_plan_master_update_step",
    "make_sam_step",
    "make_stateful_sam_step",
    "make_stateful_loss_scaled_master_update_step",
    "make_update_step",
    "sam_cost_report",
)
