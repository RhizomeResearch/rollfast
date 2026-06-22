"""Optimizer builders for structural fine-tuning plans."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from rollfast.optim.adam import adamw, scale_by_adam
from rollfast.optim.adam8 import adamw8, estimate_quantized_moment_bytes
from rollfast.optim.apollo import apollo_adamw, apollo_state_nbytes
from rollfast.optim.aurora import aurora
from rollfast.optim.galore import galore_adamw, projected_state_nbytes
from rollfast.optim.prism import prism
from rollfast.optim.psgd import kron
from rollfast.schedules.schedulefree import (
    WeightingMode,
    schedule_free,
    schedule_free_eval_params,
)

from ._protocols import FineTunePlanProtocol
from .averaging import (
    default_eval_view,
    eval_views,
    make_averaging_eval_fn,
    wrap_with_averaging,
)
from .config import (
    AccumulationConfig,
    APOLLOConfig,
    CompiledGroup,
    EMAConfig,
    GaLoreConfig,
    GradientPolicy,
    GroupRule,
    OptimizerBundle,
    OptimizerConfig,
    OptimizerReport,
    MuonConfig,
    PrecisionConfig,
    SWAConfig,
    ScheduleConfig,
    ShardingPolicy,
    SOAPConfig,
    StateQuantizationConfig,
)
from .groups import compile_groups
from .schedules import build_schedule, preview_schedule
from .transforms import clip_by_global_norm
from .validation import validate_plan


class _FactorizedAdamWState(NamedTuple):
    count: jax.Array
    inner_state: optax.OptState


def compile_optimizer(
    plan: FineTunePlanProtocol,
    *,
    recipe: Any | None = None,
    optimizer: OptimizerConfig | None = None,
    schedule: ScheduleConfig | None = None,
    gradient_policy: GradientPolicy | None = None,
    accumulation: AccumulationConfig | None = None,
    precision: PrecisionConfig | None = None,
    state_quantization: StateQuantizationConfig | None = None,
    sharding: ShardingPolicy | None = None,
    ema: EMAConfig | None = None,
    swa: SWAConfig | None = None,
    group_rules: Iterable[GroupRule] = (),
    total_steps: int | None = None,
    allow_empty_groups: bool = False,
) -> OptimizerBundle:
    """Compile a structural fine-tuning plan into an Optax transformation."""

    group_rules = tuple(group_rules)
    if recipe is not None:
        optimizer = recipe.optimizer if optimizer is None else optimizer
        schedule = recipe.schedule if schedule is None else schedule
        gradient_policy = (
            recipe.gradient_policy if gradient_policy is None else gradient_policy
        )
        accumulation = recipe.accumulation if accumulation is None else accumulation
        precision = recipe.precision if precision is None else precision
        sharding = getattr(recipe, "sharding", sharding)
        ema = getattr(recipe, "ema", ema)
        swa = getattr(recipe, "swa", swa)
        if not group_rules:
            group_rules = recipe.group_rules

    optimizer = OptimizerConfig() if optimizer is None else optimizer
    schedule = ScheduleConfig() if schedule is None else schedule
    gradient_policy = GradientPolicy() if gradient_policy is None else gradient_policy
    accumulation = AccumulationConfig() if accumulation is None else accumulation
    precision = PrecisionConfig() if precision is None else precision
    sharding = ShardingPolicy() if sharding is None else sharding
    gradient_policy = _with_sharding_norm_axes(gradient_policy, sharding)
    ema = EMAConfig() if ema is None else ema
    swa = SWAConfig() if swa is None else swa
    state_quantization = (
        StateQuantizationConfig() if state_quantization is None else state_quantization
    )
    if state_quantization.enabled:
        if optimizer.name not in ("adamw", "adamw8"):
            raise NotImplementedError(
                "8-bit optimizer-state quantization currently supports AdamW only."
            )
        optimizer = replace(optimizer, name="adamw8")

    normalized = validate_plan(plan, allow_empty_groups=allow_empty_groups)
    schedule = schedule.resolved(total_steps)
    compiled_groups = compile_groups(normalized.groups, optimizer, group_rules)
    tx = _build_grouped_transform(
        labels=normalized.labels,
        groups=compiled_groups,
        optimizer=optimizer,
        schedule=schedule,
        gradient_policy=gradient_policy,
        accumulation=accumulation,
        precision=precision,
        state_quantization=state_quantization,
    )
    tx = wrap_with_averaging(
        tx,
        ema=ema,
        swa=swa,
        total_steps=schedule.total_steps,
        labels=normalized.labels,
        groups=compiled_groups,
    )
    eval_fn = make_averaging_eval_fn(ema=ema, swa=swa)
    views = eval_views(ema=ema, swa=swa)
    default_view = default_eval_view(ema=ema, swa=swa)
    report = _make_report(
        normalized.fingerprint,
        normalized.logical_id_table_hash,
        normalized.model_state_structure_hash,
        compiled_groups,
        schedule,
        accumulation,
        precision,
        state_quantization=state_quantization,
        warnings=_report_warnings(normalized.warnings, normalized.trainable, precision),
    )
    return OptimizerBundle(
        tx=tx,
        report=report,
        optimizer_config=optimizer,
        schedule_config=schedule,
        gradient_policy=gradient_policy,
        accumulation_config=accumulation,
        precision_config=precision,
        quantization_config=state_quantization,
        sharding_policy=sharding,
        ema_config=ema,
        swa_config=swa,
        eval_fn=eval_fn,
        eval_params_kind="averaging" if eval_fn is not None else "identity",
        eval_views=views,
        default_eval_view=default_view,
    )


def optimizer_from_plan(
    plan: FineTunePlanProtocol,
    **kwargs: Any,
) -> OptimizerBundle:
    """Alias for ``compile_optimizer``."""

    return compile_optimizer(plan, **kwargs)


def _with_sharding_norm_axes(
    gradient_policy: GradientPolicy,
    sharding: ShardingPolicy,
) -> GradientPolicy:
    if not sharding.mesh_axes:
        return gradient_policy
    if (
        gradient_policy.partition_axis_names is not None
        or gradient_policy.replicated_axis_names is not None
    ):
        return gradient_policy
    return replace(
        gradient_policy,
        partition_axis_names=sharding.parameter_axes or None,
        replicated_axis_names=sharding.data_axes or None,
    )


def _clip_by_policy(gradient_policy: GradientPolicy) -> optax.GradientTransformation:
    if gradient_policy.clip_global_norm is None:
        raise ValueError("clip_global_norm is disabled.")
    return clip_by_global_norm(
        gradient_policy.clip_global_norm,
        axis_name=gradient_policy.axis_name,
        partition_axis_names=gradient_policy.partition_axis_names,
        replicated_axis_names=gradient_policy.replicated_axis_names,
    )


def adamw_from_plan(
    plan: FineTunePlanProtocol,
    *,
    total_steps: int | None = None,
    base_lr: float = 5e-4,
    schedule: str | ScheduleConfig = "warmup_cosine",
    weight_decay: float = 0.05,
    clip_global_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    moment_dtype: Any = jnp.float32,
    lora_b_lr_ratio: float | None = None,
    axis_name: str | tuple[str, ...] | None = None,
    group_rules: Iterable[GroupRule] = (),
    sharding: ShardingPolicy | None = None,
    ema: EMAConfig | None = None,
    swa: SWAConfig | None = None,
    **kwargs: Any,
) -> OptimizerBundle:
    """Build grouped AdamW from a fine-tuning plan."""

    optimizer = OptimizerConfig(
        name="adamw",
        base_lr=base_lr,
        weight_decay=weight_decay,
        lora_b_lr_ratio=lora_b_lr_ratio,
        **kwargs,
    )
    schedule_config = (
        ScheduleConfig(kind=schedule, total_steps=total_steps)
        if isinstance(schedule, str)
        else schedule.resolved(total_steps)
    )
    return compile_optimizer(
        plan,
        optimizer=optimizer,
        schedule=schedule_config,
        gradient_policy=GradientPolicy(
            clip_global_norm=clip_global_norm,
            nonfinite="skip",
            axis_name=axis_name,
        ),
        accumulation=AccumulationConfig(steps=accumulation_steps),
        precision=PrecisionConfig(moment_dtype=moment_dtype),
        sharding=sharding,
        group_rules=group_rules,
        ema=ema,
        swa=swa,
        total_steps=total_steps,
    )


def galore_adamw_from_plan(
    plan: FineTunePlanProtocol,
    *,
    galore: GaLoreConfig,
    total_steps: int | None = None,
    base_lr: float = 5e-4,
    schedule: str | ScheduleConfig = "warmup_cosine",
    weight_decay: float = 0.05,
    clip_global_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    moment_dtype: Any = jnp.float32,
    axis_name: str | tuple[str, ...] | None = None,
    group_rules: Iterable[GroupRule] = (),
    sharding: ShardingPolicy | None = None,
    ema: EMAConfig | None = None,
    swa: SWAConfig | None = None,
    **kwargs: Any,
) -> OptimizerBundle:
    """Build grouped GaLore AdamW from a fine-tuning plan."""

    optimizer = OptimizerConfig(
        name="galore_adamw",
        base_lr=base_lr,
        weight_decay=weight_decay,
        **kwargs,
    )
    schedule_config = (
        ScheduleConfig(kind=schedule, total_steps=total_steps)
        if isinstance(schedule, str)
        else schedule.resolved(total_steps)
    )
    gradient_policy = GradientPolicy(
        clip_global_norm=clip_global_norm,
        nonfinite="skip",
        axis_name=axis_name,
    )
    accumulation = AccumulationConfig(steps=accumulation_steps)
    precision = PrecisionConfig(moment_dtype=moment_dtype)
    sharding = ShardingPolicy() if sharding is None else sharding
    gradient_policy = _with_sharding_norm_axes(gradient_policy, sharding)
    ema = EMAConfig() if ema is None else ema
    swa = SWAConfig() if swa is None else swa

    normalized = validate_plan(plan)
    compiled_groups = compile_groups(normalized.groups, optimizer, tuple(group_rules))
    refresh_communication_bytes = _estimate_galore_refresh_communication_bytes(
        normalized.trainable,
        normalized.labels,
        galore,
    )
    _check_galore_sharding_policy(
        sharding,
        refresh_communication_bytes=refresh_communication_bytes,
    )
    tx = _build_grouped_galore_transform(
        labels=normalized.labels,
        groups=compiled_groups,
        optimizer=optimizer,
        schedule=schedule_config,
        gradient_policy=gradient_policy,
        accumulation=accumulation,
        precision=precision,
        galore=galore,
    )
    tx = wrap_with_averaging(
        tx,
        ema=ema,
        swa=swa,
        total_steps=schedule_config.total_steps,
        labels=normalized.labels,
        groups=compiled_groups,
    )
    eval_fn = make_averaging_eval_fn(ema=ema, swa=swa)
    views = eval_views(ema=ema, swa=swa)
    default_view = default_eval_view(ema=ema, swa=swa)
    report = _make_report(
        normalized.fingerprint,
        normalized.logical_id_table_hash,
        normalized.model_state_structure_hash,
        compiled_groups,
        schedule_config,
        accumulation,
        precision,
        warnings=(
            *_report_warnings(normalized.warnings, normalized.trainable, precision),
            *_galore_profile_warnings(galore),
            *_galore_sharding_warnings(
                sharding,
                refresh_communication_bytes=refresh_communication_bytes,
            ),
        ),
    )
    report = replace(
        report,
        estimated_state_bytes=_estimate_galore_state_bytes(
            normalized.trainable,
            normalized.labels,
            galore,
            precision,
        ),
        state_policies={
            group.source_label: "galore_projected_moments"
            for group in compiled_groups
        },
    )
    return OptimizerBundle(
        tx=tx,
        report=report,
        optimizer_config=optimizer,
        schedule_config=schedule_config,
        gradient_policy=gradient_policy,
        accumulation_config=accumulation,
        precision_config=precision,
        quantization_config=StateQuantizationConfig(),
        sharding_policy=sharding,
        ema_config=ema,
        swa_config=swa,
        method_config=_galore_method_config(
            galore,
            sharding,
            refresh_communication_bytes=refresh_communication_bytes,
        ),
        eval_fn=eval_fn,
        eval_params_kind="averaging" if eval_fn is not None else "identity",
        eval_views=views,
        default_eval_view=default_view,
    )


def apollo_adamw_from_plan(
    plan: FineTunePlanProtocol,
    *,
    apollo: APOLLOConfig | None = None,
    total_steps: int | None = None,
    base_lr: float = 5e-4,
    schedule: str | ScheduleConfig = "warmup_cosine",
    weight_decay: float | None = None,
    clip_global_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    moment_dtype: Any = jnp.float32,
    axis_name: str | tuple[str, ...] | None = None,
    group_rules: Iterable[GroupRule] = (),
    sharding: ShardingPolicy | None = None,
    ema: EMAConfig | None = None,
    swa: SWAConfig | None = None,
    **kwargs: Any,
) -> OptimizerBundle:
    """Build grouped APOLLO AdamW from a fine-tuning plan."""

    apollo = APOLLOConfig() if apollo is None else apollo
    resolved_weight_decay = apollo.weight_decay if weight_decay is None else weight_decay
    optimizer = OptimizerConfig(
        name="apollo_adamw",
        base_lr=base_lr,
        weight_decay=resolved_weight_decay,
        eps=apollo.eps,
        **kwargs,
    )
    schedule_config = (
        ScheduleConfig(kind=schedule, total_steps=total_steps)
        if isinstance(schedule, str)
        else schedule.resolved(total_steps)
    )
    gradient_policy = GradientPolicy(
        clip_global_norm=clip_global_norm,
        nonfinite="skip",
        axis_name=axis_name,
    )
    accumulation = AccumulationConfig(steps=accumulation_steps)
    precision = PrecisionConfig(moment_dtype=moment_dtype)
    sharding = ShardingPolicy() if sharding is None else sharding
    gradient_policy = _with_sharding_norm_axes(gradient_policy, sharding)
    ema = EMAConfig() if ema is None else ema
    swa = SWAConfig() if swa is None else swa

    normalized = validate_plan(plan)
    compiled_groups = compile_groups(normalized.groups, optimizer, tuple(group_rules))
    tx = _build_grouped_apollo_transform(
        labels=normalized.labels,
        groups=compiled_groups,
        optimizer=optimizer,
        schedule=schedule_config,
        gradient_policy=gradient_policy,
        accumulation=accumulation,
        precision=precision,
        apollo=apollo,
    )
    tx = wrap_with_averaging(
        tx,
        ema=ema,
        swa=swa,
        total_steps=schedule_config.total_steps,
        labels=normalized.labels,
        groups=compiled_groups,
    )
    eval_fn = make_averaging_eval_fn(ema=ema, swa=swa)
    views = eval_views(ema=ema, swa=swa)
    default_view = default_eval_view(ema=ema, swa=swa)
    report = _make_report(
        normalized.fingerprint,
        normalized.logical_id_table_hash,
        normalized.model_state_structure_hash,
        compiled_groups,
        schedule_config,
        accumulation,
        precision,
        warnings=(
            *_report_warnings(normalized.warnings, normalized.trainable, precision),
            *_apollo_profile_warnings(),
        ),
    )
    policy = "apollo_mini_tensor_scaling" if apollo.mini else f"apollo_{apollo.scaling}_scaling"
    report = replace(
        report,
        estimated_state_bytes=_estimate_apollo_state_bytes(
            normalized.trainable,
            apollo,
            precision,
        ),
        state_policies={group.source_label: policy for group in compiled_groups},
    )
    return OptimizerBundle(
        tx=tx,
        report=report,
        optimizer_config=optimizer,
        schedule_config=schedule_config,
        gradient_policy=gradient_policy,
        accumulation_config=accumulation,
        precision_config=precision,
        quantization_config=StateQuantizationConfig(),
        sharding_policy=sharding,
        ema_config=ema,
        swa_config=swa,
        method_config=_apollo_method_config(apollo),
        eval_fn=eval_fn,
        eval_params_kind="averaging" if eval_fn is not None else "identity",
        eval_views=views,
        default_eval_view=default_view,
    )


def schedule_free_adam_from_plan(
    plan: FineTunePlanProtocol,
    *,
    total_steps: int,
    base_lr: float = 5e-4,
    schedule: str | ScheduleConfig = "wsd",
    weight_decay: float = 0.0,
    clip_global_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    moment_dtype: Any = jnp.float32,
    state_dtype: Any | None = None,
    lora_b_lr_ratio: float | None = None,
    axis_name: str | tuple[str, ...] | None = None,
    group_rules: Iterable[GroupRule] = (),
    ema: EMAConfig | None = None,
    swa: SWAConfig | None = None,
    weighting_mode: str | WeightingMode = WeightingMode.PRACTICAL,
    sf_b1: float = 0.9,
    b1: float | None = None,
    b2: float | None = None,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    schedule_free_plus: bool = False,
    sf_r: float = 0.0,
    sf_c_warmup: int = 0,
    sf_weight_lr_power: float = 2.0,
    sf_use_lr_max: bool | None = None,
    sf_b1_anneal_steps: int = 0,
    sf_b1_max: float | None = 0.965,
    polyak: bool | None = None,
    use_adamc: bool | None = None,
    polyak_beta: float = 0.0,
    polyak_f_star: float = 0.0,
    polyak_axis_name: str | tuple[str, ...] | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> OptimizerBundle:
    """Build grouped Schedule-Free Adam from a fine-tuning plan."""

    group_rules = tuple(group_rules)
    ema = EMAConfig() if ema is None else ema
    swa = SWAConfig() if swa is None else swa
    optimizer = OptimizerConfig(
        name="schedule_free_adam",
        base_lr=base_lr,
        weight_decay=weight_decay,
        b1=(0.9 if schedule_free_plus else 0.0) if b1 is None else b1,
        b2=(0.95 if schedule_free_plus else 0.999) if b2 is None else b2,
        eps=eps,
        eps_root=eps_root,
        lora_b_lr_ratio=lora_b_lr_ratio,
    )
    schedule_config = (
        ScheduleConfig(kind=schedule, total_steps=total_steps)
        if isinstance(schedule, str)
        else schedule.resolved(total_steps)
    )
    if schedule_config.kind == "warmup_cosine":
        # Schedule-Free defaults to WSD because the inner optimizer still needs
        # a stabilizing schedule while the wrapper maintains evaluation params.
        schedule_config = replace(schedule_config, kind="wsd")

    gradient_policy = GradientPolicy(
        clip_global_norm=clip_global_norm,
        nonfinite="skip",
        axis_name=axis_name,
    )
    accumulation = AccumulationConfig(steps=accumulation_steps)
    precision = PrecisionConfig(moment_dtype=moment_dtype)

    normalized = validate_plan(plan)
    compiled_groups = compile_groups(normalized.groups, optimizer, group_rules)
    tx = _build_grouped_schedule_free_transform(
        labels=normalized.labels,
        groups=compiled_groups,
        optimizer=optimizer,
        schedule=schedule_config,
        gradient_policy=gradient_policy,
        accumulation=accumulation,
        precision=precision,
        weighting_mode=weighting_mode,
        sf_b1=sf_b1,
        state_dtype=state_dtype,
        key=key,
        schedule_free_plus=schedule_free_plus,
        sf_r=sf_r,
        sf_c_warmup=sf_c_warmup,
        sf_weight_lr_power=sf_weight_lr_power,
        sf_use_lr_max=sf_use_lr_max,
        sf_b1_anneal_steps=sf_b1_anneal_steps,
        sf_b1_max=sf_b1_max,
        polyak=polyak,
        use_adamc=use_adamc,
        polyak_beta=polyak_beta,
        polyak_f_star=polyak_f_star,
        polyak_axis_name=polyak_axis_name,
    )
    tx = wrap_with_averaging(
        tx,
        ema=ema,
        swa=swa,
        total_steps=schedule_config.total_steps,
        labels=normalized.labels,
        groups=compiled_groups,
    )
    eval_fn = make_averaging_eval_fn(
        ema=ema,
        swa=swa,
        inner_eval_fn=_schedule_free_eval_params,
    )
    views = eval_views(ema=ema, swa=swa, inner_views=("optimizer", "schedule_free"))
    default_view = default_eval_view(
        ema=ema,
        swa=swa,
        inner_default="schedule_free",
    )
    report = _make_report(
        normalized.fingerprint,
        normalized.logical_id_table_hash,
        normalized.model_state_structure_hash,
        compiled_groups,
        schedule_config,
        accumulation,
        precision,
        warnings=_report_warnings(normalized.warnings, normalized.trainable, precision),
    )
    return OptimizerBundle(
        tx=tx,
        report=report,
        optimizer_config=optimizer,
        schedule_config=schedule_config,
        gradient_policy=gradient_policy,
        accumulation_config=accumulation,
        precision_config=precision,
        quantization_config=StateQuantizationConfig(),
        ema_config=ema,
        swa_config=swa,
        eval_fn=eval_fn,
        eval_params_kind="averaging_schedule_free"
        if ema.enabled or swa.enabled
        else "schedule_free",
        eval_views=views,
        default_eval_view=default_view,
    )


def adamw8_from_plan(
    plan: FineTunePlanProtocol,
    *,
    total_steps: int | None = None,
    base_lr: float = 5e-4,
    schedule: str | ScheduleConfig = "warmup_cosine",
    weight_decay: float = 0.05,
    clip_global_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    lora_b_lr_ratio: float | None = None,
    axis_name: str | tuple[str, ...] | None = None,
    group_rules: Iterable[GroupRule] = (),
    sharding: ShardingPolicy | None = None,
    ema: EMAConfig | None = None,
    swa: SWAConfig | None = None,
    state_quantization: StateQuantizationConfig | None = None,
    **kwargs: Any,
) -> OptimizerBundle:
    """Build grouped AdamW with blockwise 8-bit moment state."""

    quantization = (
        StateQuantizationConfig(enabled=True)
        if state_quantization is None
        else state_quantization
    )
    if not quantization.enabled:
        quantization = replace(quantization, enabled=True)
    optimizer = OptimizerConfig(
        name="adamw8",
        base_lr=base_lr,
        weight_decay=weight_decay,
        lora_b_lr_ratio=lora_b_lr_ratio,
        **kwargs,
    )
    schedule_config = (
        ScheduleConfig(kind=schedule, total_steps=total_steps)
        if isinstance(schedule, str)
        else schedule.resolved(total_steps)
    )
    return compile_optimizer(
        plan,
        optimizer=optimizer,
        schedule=schedule_config,
        gradient_policy=GradientPolicy(
            clip_global_norm=clip_global_norm,
            nonfinite="skip",
            axis_name=axis_name,
        ),
        accumulation=AccumulationConfig(steps=accumulation_steps),
        precision=PrecisionConfig(moment_dtype=quantization.fallback_dtype),
        state_quantization=quantization,
        sharding=sharding,
        group_rules=group_rules,
        ema=ema,
        swa=swa,
        total_steps=total_steps,
    )


def hybrid_aurora_adam_from_plan(
    plan: FineTunePlanProtocol,
    *,
    total_steps: int | None = None,
    base_lr: float = 5e-4,
    schedule: str | ScheduleConfig = "warmup_cosine",
    weight_decay: float = 0.05,
    clip_global_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    moment_dtype: Any = jnp.float32,
    axis_name: str | tuple[str, ...] | None = None,
    group_rules: Iterable[GroupRule] = (),
    ema: EMAConfig | None = None,
    swa: SWAConfig | None = None,
    b1: float = 0.95,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    polar_ns_iters: int = 12,
    key: jax.Array = jax.random.PRNGKey(42),
) -> OptimizerBundle:
    """Build grouped Aurora/Adam from a fine-tuning plan."""

    return _hybrid_optimizer_from_plan(
        plan,
        optimizer_name="aurora_adam",
        family="aurora",
        total_steps=total_steps,
        base_lr=base_lr,
        schedule=schedule,
        weight_decay=weight_decay,
        clip_global_norm=clip_global_norm,
        accumulation_steps=accumulation_steps,
        moment_dtype=moment_dtype,
        axis_name=axis_name,
        group_rules=group_rules,
        ema=ema,
        swa=swa,
        b1=b1,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps=adam_eps,
        key=key,
        family_kwargs={"polar_ns_iters": polar_ns_iters},
    )


def hybrid_prism_adam_from_plan(
    plan: FineTunePlanProtocol,
    *,
    total_steps: int | None = None,
    base_lr: float = 5e-4,
    schedule: str | ScheduleConfig = "warmup_cosine",
    weight_decay: float = 0.05,
    clip_global_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    moment_dtype: Any = jnp.float32,
    axis_name: str | tuple[str, ...] | None = None,
    group_rules: Iterable[GroupRule] = (),
    ema: EMAConfig | None = None,
    swa: SWAConfig | None = None,
    b1: float = 0.95,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    ns_iters: int = 5,
    key: jax.Array = jax.random.PRNGKey(42),
) -> OptimizerBundle:
    """Build grouped PRISM/Adam from a fine-tuning plan."""

    return _hybrid_optimizer_from_plan(
        plan,
        optimizer_name="prism_adam",
        family="prism",
        total_steps=total_steps,
        base_lr=base_lr,
        schedule=schedule,
        weight_decay=weight_decay,
        clip_global_norm=clip_global_norm,
        accumulation_steps=accumulation_steps,
        moment_dtype=moment_dtype,
        axis_name=axis_name,
        group_rules=group_rules,
        ema=ema,
        swa=swa,
        b1=b1,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps=adam_eps,
        key=key,
        family_kwargs={"ns_iters": ns_iters},
    )


def hybrid_kron_adam_from_plan(
    plan: FineTunePlanProtocol,
    *,
    total_steps: int | None = None,
    base_lr: float = 5e-4,
    schedule: str | ScheduleConfig = "warmup_cosine",
    weight_decay: float = 0.05,
    clip_global_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    moment_dtype: Any = jnp.float32,
    preconditioner_dtype: Any | None = None,
    axis_name: str | tuple[str, ...] | None = None,
    group_rules: Iterable[GroupRule] = (),
    ema: EMAConfig | None = None,
    swa: SWAConfig | None = None,
    b1: float = 0.9,
    preconditioner_update_probability: float = 1.0,
    key: jax.Array = jax.random.PRNGKey(42),
) -> OptimizerBundle:
    """Build grouped PSGD/Kron from a fine-tuning plan."""

    return _hybrid_optimizer_from_plan(
        plan,
        optimizer_name="kron_adam",
        family="kron",
        total_steps=total_steps,
        base_lr=base_lr,
        schedule=schedule,
        weight_decay=weight_decay,
        clip_global_norm=clip_global_norm,
        accumulation_steps=accumulation_steps,
        moment_dtype=moment_dtype,
        axis_name=axis_name,
        group_rules=group_rules,
        ema=ema,
        swa=swa,
        b1=b1,
        adam_b1=0.9,
        adam_b2=0.999,
        adam_eps=1e-8,
        key=key,
        family_kwargs={
            "preconditioner_update_probability": preconditioner_update_probability,
            "precond_dtype": preconditioner_dtype,
        },
    )


def muon_adam_from_plan(
    plan: FineTunePlanProtocol,
    *,
    muon: MuonConfig | None = None,
    total_steps: int | None = None,
    base_lr: float = 5e-4,
    schedule: str | ScheduleConfig = "warmup_cosine",
    weight_decay: float = 0.05,
    clip_global_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    moment_dtype: Any = jnp.float32,
    axis_name: str | tuple[str, ...] | None = None,
    group_rules: Iterable[GroupRule] = (),
    ema: EMAConfig | None = None,
    swa: SWAConfig | None = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    key: jax.Array = jax.random.PRNGKey(42),
) -> OptimizerBundle:
    """Build grouped Muon/AdamW from a fine-tuning plan."""

    muon = MuonConfig() if muon is None else muon
    return _hybrid_optimizer_from_plan(
        plan,
        optimizer_name="muon_adam",
        family="muon",
        total_steps=total_steps,
        base_lr=base_lr,
        schedule=schedule,
        weight_decay=weight_decay,
        clip_global_norm=clip_global_norm,
        accumulation_steps=accumulation_steps,
        moment_dtype=moment_dtype,
        axis_name=axis_name,
        group_rules=group_rules,
        ema=ema,
        swa=swa,
        b1=muon.beta,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps=adam_eps,
        key=key,
        family_kwargs={
            "config": muon.to_dict(),
            "ns_steps": muon.ns_steps,
            "eps": muon.eps,
            "preconditioning": muon.preconditioning,
            "nesterov": muon.nesterov,
            "adaptive": muon.adaptive,
            "consistent_rms": muon.consistent_rms,
        },
    )


def soap_adam_from_plan(
    plan: FineTunePlanProtocol,
    *,
    soap: SOAPConfig | None = None,
    total_steps: int | None = None,
    base_lr: float = 5e-4,
    schedule: str | ScheduleConfig = "warmup_cosine",
    weight_decay: float = 0.05,
    clip_global_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    moment_dtype: Any = jnp.float32,
    axis_name: str | tuple[str, ...] | None = None,
    group_rules: Iterable[GroupRule] = (),
    ema: EMAConfig | None = None,
    swa: SWAConfig | None = None,
    b1: float = 0.9,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    key: jax.Array = jax.random.PRNGKey(42),
) -> OptimizerBundle:
    """Build experimental SOAP-style matrix preconditioning with AdamW fallback."""

    soap = SOAPConfig() if soap is None else soap
    return _hybrid_optimizer_from_plan(
        plan,
        optimizer_name="soap_adam",
        family="soap",
        total_steps=total_steps,
        base_lr=base_lr,
        schedule=schedule,
        weight_decay=weight_decay,
        clip_global_norm=clip_global_norm,
        accumulation_steps=accumulation_steps,
        moment_dtype=moment_dtype,
        axis_name=axis_name,
        group_rules=group_rules,
        ema=ema,
        swa=swa,
        b1=b1,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps=adam_eps,
        key=key,
        family_kwargs={
            "config": soap.to_dict(),
            "preconditioner_update_interval": soap.preconditioner_update_interval,
            "inv_steps": soap.inv_steps,
            "eps_gram": soap.eps_gram,
        },
    )


def _hybrid_optimizer_from_plan(
    plan: FineTunePlanProtocol,
    *,
    optimizer_name: str,
    family: str,
    total_steps: int | None,
    base_lr: float,
    schedule: str | ScheduleConfig,
    weight_decay: float,
    clip_global_norm: float | None,
    accumulation_steps: int,
    moment_dtype: Any,
    axis_name: str | tuple[str, ...] | None,
    group_rules: Iterable[GroupRule],
    ema: EMAConfig | None,
    swa: SWAConfig | None,
    b1: float,
    adam_b1: float,
    adam_b2: float,
    adam_eps: float,
    key: jax.Array,
    family_kwargs: dict[str, Any],
) -> OptimizerBundle:
    group_rules = tuple(group_rules)
    ema = EMAConfig() if ema is None else ema
    swa = SWAConfig() if swa is None else swa
    optimizer = OptimizerConfig(
        name=optimizer_name,
        base_lr=base_lr,
        weight_decay=weight_decay,
        b1=b1,
        b2=adam_b2,
        eps=adam_eps,
    )
    schedule_config = (
        ScheduleConfig(kind=schedule, total_steps=total_steps)
        if isinstance(schedule, str)
        else schedule.resolved(total_steps)
    )
    gradient_policy = GradientPolicy(
        clip_global_norm=clip_global_norm,
        nonfinite="skip",
        axis_name=axis_name,
    )
    accumulation = AccumulationConfig(steps=accumulation_steps)
    precision = PrecisionConfig(moment_dtype=moment_dtype)
    normalized = validate_plan(plan)
    compiled_groups = compile_groups(normalized.groups, optimizer, group_rules)
    tx = _build_grouped_hybrid_transform(
        labels=normalized.labels,
        groups=compiled_groups,
        optimizer=optimizer,
        schedule=schedule_config,
        gradient_policy=gradient_policy,
        accumulation=accumulation,
        precision=precision,
        family=family,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps=adam_eps,
        key=key,
        family_kwargs=family_kwargs,
    )
    tx = wrap_with_averaging(
        tx,
        ema=ema,
        swa=swa,
        total_steps=schedule_config.total_steps,
        labels=normalized.labels,
        groups=compiled_groups,
    )
    eval_fn = make_averaging_eval_fn(ema=ema, swa=swa)
    views = eval_views(ema=ema, swa=swa)
    default_view = default_eval_view(ema=ema, swa=swa)
    report = _make_report(
        normalized.fingerprint,
        normalized.logical_id_table_hash,
        normalized.model_state_structure_hash,
        compiled_groups,
        schedule_config,
        accumulation,
        precision,
        warnings=(
            *_report_warnings(normalized.warnings, normalized.trainable, precision),
            *_hybrid_profile_warnings(family),
        ),
    )
    return OptimizerBundle(
        tx=tx,
        report=report,
        optimizer_config=optimizer,
        schedule_config=schedule_config,
        gradient_policy=gradient_policy,
        accumulation_config=accumulation,
        precision_config=precision,
        quantization_config=StateQuantizationConfig(),
        ema_config=ema,
        swa_config=swa,
        method_config=_hybrid_method_config(family, family_kwargs),
        eval_fn=eval_fn,
        eval_params_kind="averaging" if eval_fn is not None else "identity",
        eval_views=views,
        default_eval_view=default_view,
    )


def estimate_optimizer_state(
    groups: tuple[CompiledGroup, ...],
    precision: PrecisionConfig | None = None,
    state_quantization: StateQuantizationConfig | None = None,
) -> int:
    """Estimate first- and second-moment state bytes for compiled groups."""

    precision = PrecisionConfig() if precision is None else precision
    state_quantization = (
        StateQuantizationConfig()
        if state_quantization is None
        else state_quantization
    )
    if not state_quantization.enabled:
        itemsize = jnp.dtype(precision.moment_dtype).itemsize
        total_params = sum(group.param_count for group in groups)
        return int(total_params * itemsize * 2)

    fallback_itemsize = jnp.dtype(state_quantization.fallback_dtype).itemsize
    total = 0
    for group in groups:
        if _quantize_group_state(group, state_quantization):
            total += 2 * estimate_quantized_moment_bytes(
                group.param_count,
                block_size=state_quantization.block_size,
                scale_dtype=state_quantization.scale_dtype,
            )
        else:
            total += int(group.param_count * fallback_itemsize * 2)
    return total


def _build_grouped_hybrid_transform(
    *,
    labels: Any,
    groups: tuple[CompiledGroup, ...],
    optimizer: OptimizerConfig,
    schedule: ScheduleConfig,
    gradient_policy: GradientPolicy,
    accumulation: AccumulationConfig,
    precision: PrecisionConfig,
    family: str,
    adam_b1: float,
    adam_b2: float,
    adam_eps: float,
    key: jax.Array,
    family_kwargs: dict[str, Any],
) -> optax.GradientTransformation:
    transforms = {}
    for index, group in enumerate(groups):
        if group.optimizer != optimizer.name:
            raise NotImplementedError(
                f"Unsupported optimizer {group.optimizer!r} for {optimizer.name!r} builder."
            )
        lr_schedule = build_schedule(
            schedule,
            peak_lr=group.effective_lr,
            total_steps=schedule.total_steps,
        )
        group_key = jax.random.fold_in(key, index)
        transforms[group.source_label] = _hybrid_group_transform(
            family,
            learning_rate=lr_schedule,
            weight_decay=group.weight_decay_value,
            b1=optimizer.b1,
            adam_b1=adam_b1,
            adam_b2=adam_b2,
            adam_eps=adam_eps,
            moment_dtype=precision.moment_dtype,
            axis_name=gradient_policy.axis_name,
            key=group_key,
            family_kwargs=family_kwargs,
        )

    chain_parts: list[optax.GradientTransformation] = []
    if gradient_policy.clip_global_norm is not None:
        chain_parts.append(_clip_by_policy(gradient_policy))
    chain_parts.append(optax.multi_transform(transforms, lambda _: labels))
    tx = optax.chain(*chain_parts)
    if gradient_policy.nonfinite == "skip":
        tx = optax.apply_if_finite(
            tx,
            max_consecutive_errors=gradient_policy.max_consecutive_nonfinite,
        )
    if accumulation.steps > 1:
        tx = optax.MultiSteps(
            tx,
            every_k_schedule=accumulation.steps,
            use_grad_mean=_multisteps_use_grad_mean(accumulation),
            accumulator_dtype=accumulation.accumulate_dtype,
        )
    return tx


def _hybrid_group_transform(
    family: str,
    *,
    learning_rate: Any,
    weight_decay: float,
    b1: float,
    adam_b1: float,
    adam_b2: float,
    adam_eps: float,
    moment_dtype: Any,
    axis_name: str | tuple[str, ...] | None,
    key: jax.Array,
    family_kwargs: dict[str, Any],
) -> optax.GradientTransformation:
    clean_kwargs = {
        key_: value
        for key_, value in family_kwargs.items()
        if value is not None
    }
    if family == "aurora":
        return aurora(
            learning_rate=learning_rate,
            b1=b1,
            weight_decay=weight_decay,
            mu_dtype=moment_dtype,
            axis_name=axis_name,
            adam_b1=adam_b1,
            adam_b2=adam_b2,
            adam_eps=adam_eps,
            key=key,
            **clean_kwargs,
        )
    if family == "prism":
        return prism(
            learning_rate=learning_rate,
            b1=b1,
            weight_decay=weight_decay,
            mu_dtype=moment_dtype,
            axis_name=axis_name,
            adam_b1=adam_b1,
            adam_b2=adam_b2,
            adam_eps=adam_eps,
            key=key,
            **clean_kwargs,
        )
    if family == "kron":
        return kron(
            learning_rate=learning_rate,
            b1=b1,
            weight_decay=weight_decay,
            mu_dtype=moment_dtype,
            axis_name=axis_name,
            key=key,
            **clean_kwargs,
        )
    if family == "muon":
        config = clean_kwargs.pop("config", {})
        del config
        return optax.contrib.muon(
            learning_rate=learning_rate,
            ns_steps=clean_kwargs.pop("ns_steps", 5),
            beta=b1,
            eps=clean_kwargs.pop("eps", adam_eps),
            weight_decay=weight_decay,
            mu_dtype=moment_dtype,
            nesterov=clean_kwargs.pop("nesterov", True),
            adaptive=clean_kwargs.pop("adaptive", False),
            preconditioning=clean_kwargs.pop("preconditioning", "frobenius"),
            adam_b1=adam_b1,
            adam_b2=adam_b2,
            adam_eps_root=0.0,
            adam_weight_decay=weight_decay,
            adam_learning_rate=learning_rate,
            consistent_rms=clean_kwargs.pop("consistent_rms", None),
        )
    if family == "soap":
        config = clean_kwargs.pop("config", {})
        del config
        clean_kwargs.pop("preconditioner_update_interval", None)
        return prism(
            learning_rate=learning_rate,
            b1=b1,
            weight_decay=weight_decay,
            mu_dtype=moment_dtype,
            axis_name=axis_name,
            adam_b1=adam_b1,
            adam_b2=adam_b2,
            adam_eps=adam_eps,
            key=key,
            mode="bidirectional",
            inv_steps=clean_kwargs.pop("inv_steps", 6),
            eps_gram=clean_kwargs.pop("eps_gram", 1e-6),
            **clean_kwargs,
        )
    raise ValueError(f"unknown hybrid optimizer family: {family!r}.")


def _build_grouped_schedule_free_transform(
    *,
    labels: Any,
    groups: tuple[CompiledGroup, ...],
    optimizer: OptimizerConfig,
    schedule: ScheduleConfig,
    gradient_policy: GradientPolicy,
    accumulation: AccumulationConfig,
    precision: PrecisionConfig,
    weighting_mode: str | WeightingMode,
    sf_b1: float,
    state_dtype: Any | None,
    key: jax.Array,
    schedule_free_plus: bool,
    sf_r: float,
    sf_c_warmup: int,
    sf_weight_lr_power: float,
    sf_use_lr_max: bool | None,
    sf_b1_anneal_steps: int,
    sf_b1_max: float | None,
    polyak: bool | None,
    use_adamc: bool | None,
    polyak_beta: float,
    polyak_f_star: float,
    polyak_axis_name: str | tuple[str, ...] | None,
) -> optax.GradientTransformation:
    polyak_enabled = _resolve_sf_bool(schedule_free_plus, polyak)
    use_adamc_enabled = _resolve_sf_bool(schedule_free_plus, use_adamc)
    use_lr_max_enabled = _resolve_sf_bool(schedule_free_plus, sf_use_lr_max)

    transforms = {}
    lr_schedules = {}
    decay_mask = _decay_mask_from_labels(labels, groups)
    for index, group in enumerate(groups):
        if group.optimizer != "schedule_free_adam":
            raise NotImplementedError(f"Unsupported optimizer: {group.optimizer!r}.")
        lr_schedule = build_schedule(
            schedule,
            peak_lr=group.effective_lr,
            total_steps=schedule.total_steps,
        )
        lr_schedules[group.source_label] = lr_schedule
        inner_weight_decay = 0.0 if use_adamc_enabled else group.weight_decay_value
        group_key = jax.random.fold_in(key, index)
        transforms[group.source_label] = adamw(
            learning_rate=lr_schedule,
            b1=optimizer.b1,
            b2=optimizer.b2,
            eps=optimizer.eps,
            eps_root=optimizer.eps_root,
            mu_dtype=precision.moment_dtype,
            weight_decay=inner_weight_decay,
            nesterov=optimizer.nesterov,
            use_magma=False,
            axis_name=gradient_policy.axis_name,
            key=group_key,
        )

    base_parts: list[optax.GradientTransformation] = []
    if gradient_policy.clip_global_norm is not None:
        base_parts.append(_clip_by_policy(gradient_policy))
    base_parts.append(optax.multi_transform(transforms, lambda _: labels))
    base_optimizer = optax.chain(*base_parts)

    def learning_rate_tree(count, params):
        del params
        return jax.tree.map(
            lambda label: None if label is None else lr_schedules[label](count),
            labels,
            is_leaf=lambda x: x is None,
        )

    tx = schedule_free(
        base_optimizer=base_optimizer,
        learning_rate=learning_rate_tree,
        b1=sf_b1,
        weighting_mode=weighting_mode,
        state_dtype=state_dtype,
        key=key,
        r=sf_r,
        c_warmup=sf_c_warmup,
        weight_lr_power=sf_weight_lr_power,
        use_lr_max=use_lr_max_enabled,
        b1_anneal_steps=sf_b1_anneal_steps,
        b1_max=sf_b1_max,
        polyak=polyak_enabled,
        polyak_beta=polyak_beta,
        polyak_f_star=polyak_f_star,
        polyak_axis_name=polyak_axis_name,
        lr_max_init=optimizer.eps,
        adamc_weight_decay=optimizer.weight_decay if use_adamc_enabled else 0.0,
        adamc_weight_decay_mask=decay_mask,
    )
    if gradient_policy.nonfinite == "skip":
        tx = optax.apply_if_finite(
            tx,
            max_consecutive_errors=gradient_policy.max_consecutive_nonfinite,
        )
    if accumulation.steps > 1:
        tx = optax.MultiSteps(
            tx,
            every_k_schedule=accumulation.steps,
            use_grad_mean=_multisteps_use_grad_mean(accumulation),
            accumulator_dtype=accumulation.accumulate_dtype,
        )
    return tx


def _build_grouped_galore_transform(
    *,
    labels: Any,
    groups: tuple[CompiledGroup, ...],
    optimizer: OptimizerConfig,
    schedule: ScheduleConfig,
    gradient_policy: GradientPolicy,
    accumulation: AccumulationConfig,
    precision: PrecisionConfig,
    galore: GaLoreConfig,
) -> optax.GradientTransformation:
    transforms = {}
    for group in groups:
        lr_schedule = build_schedule(
            schedule,
            peak_lr=group.effective_lr,
            total_steps=schedule.total_steps,
        )
        if group.optimizer == "galore_adamw":
            transforms[group.source_label] = galore_adamw(
                learning_rate=lr_schedule,
                rank=_galore_rank_for_group(galore, group.source_label),
                update_interval=galore.update_interval,
                scale=galore.scale,
                projection=galore.projection,
                basis_method=galore.basis_method,
                basis_dtype=galore.basis_dtype,
                state_on_basis_refresh=galore.state_on_basis_refresh,
                min_matrix_size=galore.min_matrix_size,
                b1=optimizer.b1,
                b2=optimizer.b2,
                eps=optimizer.eps,
                eps_root=optimizer.eps_root,
                mu_dtype=precision.moment_dtype,
                weight_decay=group.weight_decay_value,
            )
        elif group.optimizer == galore.fallback_optimizer == "adamw":
            transforms[group.source_label] = adamw(
                learning_rate=lr_schedule,
                b1=optimizer.b1,
                b2=optimizer.b2,
                eps=optimizer.eps,
                eps_root=optimizer.eps_root,
                mu_dtype=precision.moment_dtype,
                weight_decay=group.weight_decay_value,
                nesterov=optimizer.nesterov,
                use_magma=optimizer.use_magma,
                magma_p=optimizer.magma_p,
                magma_tau=optimizer.magma_tau,
                axis_name=gradient_policy.axis_name,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer for GaLore builder: {group.optimizer!r}.")

    chain_parts: list[optax.GradientTransformation] = []
    if gradient_policy.clip_global_norm is not None:
        chain_parts.append(_clip_by_policy(gradient_policy))
    chain_parts.append(optax.multi_transform(transforms, lambda _: labels))
    tx = optax.chain(*chain_parts)
    if gradient_policy.nonfinite == "skip":
        tx = optax.apply_if_finite(
            tx,
            max_consecutive_errors=gradient_policy.max_consecutive_nonfinite,
        )
    if accumulation.steps > 1:
        tx = optax.MultiSteps(
            tx,
            every_k_schedule=accumulation.steps,
            use_grad_mean=_multisteps_use_grad_mean(accumulation),
            accumulator_dtype=accumulation.accumulate_dtype,
        )
    return tx


def _build_factorized_adamw_transform(
    *,
    labels: Any,
    groups: tuple[CompiledGroup, ...],
    optimizer: OptimizerConfig,
    schedule: ScheduleConfig,
    gradient_policy: GradientPolicy,
    accumulation: AccumulationConfig,
    precision: PrecisionConfig,
) -> optax.GradientTransformation:
    lr_schedules = {
        group.source_label: build_schedule(
            schedule,
            peak_lr=group.effective_lr,
            total_steps=schedule.total_steps,
        )
        for group in groups
    }
    weight_decays = {
        group.source_label: group.weight_decay_value
        for group in groups
    }
    adam = scale_by_adam(
        b1=optimizer.b1,
        b2=optimizer.b2,
        eps=optimizer.eps,
        eps_root=optimizer.eps_root,
        mu_dtype=precision.moment_dtype,
        weight_decay=0.0,
        nesterov=optimizer.nesterov,
        use_magma=False,
        axis_name=gradient_policy.axis_name,
    )

    def init_fn(params):
        return _FactorizedAdamWState(
            count=jnp.zeros([], jnp.int32),
            inner_state=adam.init(params),
        )

    def update_fn(updates, state, params=None):
        adam_updates, inner_state = adam.update(
            updates,
            state.inner_state,
            params,
        )
        params_or_updates = adam_updates if params is None else params
        scaled = jax.tree.map(
            lambda update, param, label: _scale_factorized_adamw_leaf(
                update,
                param,
                label,
                count=state.count,
                lr_schedules=lr_schedules,
                weight_decays=weight_decays,
            ),
            adam_updates,
            params_or_updates,
            labels,
            is_leaf=lambda x: x is None,
        )
        return scaled, _FactorizedAdamWState(
            count=jax.lax.convert_element_type(state.count + 1, jnp.int32),
            inner_state=inner_state,
        )

    tx = optax.GradientTransformation(init_fn, update_fn)
    chain_parts: list[optax.GradientTransformation] = []
    if gradient_policy.clip_global_norm is not None:
        chain_parts.append(_clip_by_policy(gradient_policy))
    chain_parts.append(tx)
    tx = optax.chain(*chain_parts)
    if gradient_policy.nonfinite == "skip":
        tx = optax.apply_if_finite(
            tx,
            max_consecutive_errors=gradient_policy.max_consecutive_nonfinite,
        )
    if accumulation.steps > 1:
        tx = optax.MultiSteps(
            tx,
            every_k_schedule=accumulation.steps,
            use_grad_mean=_multisteps_use_grad_mean(accumulation),
            accumulator_dtype=accumulation.accumulate_dtype,
        )
    return tx


def _scale_factorized_adamw_leaf(
    update: Any,
    param: Any,
    label: Any,
    *,
    count: jax.Array,
    lr_schedules: Mapping[str, Any],
    weight_decays: Mapping[str, float],
) -> Any:
    if update is None:
        return None
    if label is None:
        return update
    label = str(label)
    scaled = update
    weight_decay = weight_decays[label]
    if weight_decay != 0.0 and param is not None:
        scaled = scaled + weight_decay * param.astype(jnp.float32)
    return (-lr_schedules[label](count) * scaled).astype(update.dtype)


def _build_grouped_transform(
    *,
    labels: Any,
    groups: tuple[CompiledGroup, ...],
    optimizer: OptimizerConfig,
    schedule: ScheduleConfig,
    gradient_policy: GradientPolicy,
    accumulation: AccumulationConfig,
    precision: PrecisionConfig,
    state_quantization: StateQuantizationConfig,
) -> optax.GradientTransformation:
    if _can_use_factorized_adamw(groups, optimizer, state_quantization):
        return _build_factorized_adamw_transform(
            labels=labels,
            groups=groups,
            optimizer=optimizer,
            schedule=schedule,
            gradient_policy=gradient_policy,
            accumulation=accumulation,
            precision=precision,
        )

    transforms = {}
    for index, group in enumerate(groups):
        if group.optimizer not in ("adamw", "adamw8"):
            raise NotImplementedError(f"Unsupported optimizer: {group.optimizer!r}.")
        lr_schedule = build_schedule(
            schedule,
            peak_lr=group.effective_lr,
            total_steps=schedule.total_steps,
        )
        if group.optimizer == "adamw8":
            if not state_quantization.enabled:
                raise ValueError("adamw8 groups require StateQuantizationConfig.enabled.")
            transforms[group.source_label] = adamw8(
                learning_rate=lr_schedule,
                b1=optimizer.b1,
                b2=optimizer.b2,
                eps=optimizer.eps,
                eps_root=optimizer.eps_root,
                weight_decay=group.weight_decay_value,
                block_size=state_quantization.block_size,
                block_layout=state_quantization.block_layout,
                min_size=state_quantization.min_size,
                scale_dtype=state_quantization.scale_dtype,
                fallback_dtype=state_quantization.fallback_dtype,
                stochastic_rounding=state_quantization.stochastic_rounding,
                quantize=_quantize_group_state(group, state_quantization),
                nesterov=optimizer.nesterov,
                use_magma=optimizer.use_magma,
                key=jax.random.fold_in(jax.random.PRNGKey(42), index),
            )
        else:
            transforms[group.source_label] = adamw(
                learning_rate=lr_schedule,
                b1=optimizer.b1,
                b2=optimizer.b2,
                eps=optimizer.eps,
                eps_root=optimizer.eps_root,
                mu_dtype=precision.moment_dtype,
                weight_decay=group.weight_decay_value,
                nesterov=optimizer.nesterov,
                use_magma=optimizer.use_magma,
                magma_p=optimizer.magma_p,
                magma_tau=optimizer.magma_tau,
                axis_name=gradient_policy.axis_name,
            )

    chain_parts: list[optax.GradientTransformation] = []
    if gradient_policy.clip_global_norm is not None:
        chain_parts.append(_clip_by_policy(gradient_policy))
    chain_parts.append(optax.multi_transform(transforms, lambda _: labels))
    tx = optax.chain(*chain_parts)
    if gradient_policy.nonfinite == "skip":
        tx = optax.apply_if_finite(
            tx,
            max_consecutive_errors=gradient_policy.max_consecutive_nonfinite,
        )
    elif gradient_policy.nonfinite == "raise":
        # Optax transformations cannot raise dynamically inside JIT. Validation
        # code accepts the policy so callers can enforce it outside this builder.
        pass
    if accumulation.steps > 1:
        tx = optax.MultiSteps(
            tx,
            every_k_schedule=accumulation.steps,
            use_grad_mean=_multisteps_use_grad_mean(accumulation),
            accumulator_dtype=accumulation.accumulate_dtype,
        )
    return tx


def _build_grouped_apollo_transform(
    *,
    labels: Any,
    groups: tuple[CompiledGroup, ...],
    optimizer: OptimizerConfig,
    schedule: ScheduleConfig,
    gradient_policy: GradientPolicy,
    accumulation: AccumulationConfig,
    precision: PrecisionConfig,
    apollo: APOLLOConfig,
) -> optax.GradientTransformation:
    transforms = {}
    for group in groups:
        lr_schedule = build_schedule(
            schedule,
            peak_lr=group.effective_lr,
            total_steps=schedule.total_steps,
        )
        if group.optimizer == "apollo_adamw":
            transforms[group.source_label] = apollo_adamw(
                learning_rate=lr_schedule,
                rank=apollo.rank,
                projection_seed=apollo.projection_seed,
                projection_refresh_interval=apollo.projection_refresh_interval,
                scaling=apollo.scaling,
                mini=apollo.mini,
                scale=_apollo_effective_scale(apollo),
                scale_front=apollo.scale_front,
                disable_norm_growth_limiter=apollo.disable_norm_growth_limiter,
                norm_growth_limiter=apollo.norm_growth_limiter,
                b1=optimizer.b1,
                b2=optimizer.b2,
                eps=apollo.eps,
                eps_root=optimizer.eps_root,
                mu_dtype=precision.moment_dtype,
                weight_decay=group.weight_decay_value,
            )
        elif group.optimizer == apollo.fallback_optimizer == "adamw":
            transforms[group.source_label] = adamw(
                learning_rate=lr_schedule,
                b1=optimizer.b1,
                b2=optimizer.b2,
                eps=optimizer.eps,
                eps_root=optimizer.eps_root,
                mu_dtype=precision.moment_dtype,
                weight_decay=group.weight_decay_value,
                nesterov=optimizer.nesterov,
                use_magma=optimizer.use_magma,
                magma_p=optimizer.magma_p,
                magma_tau=optimizer.magma_tau,
                axis_name=gradient_policy.axis_name,
            )
        else:
            raise NotImplementedError(
                f"Unsupported optimizer for APOLLO builder: {group.optimizer!r}."
            )

    chain_parts: list[optax.GradientTransformation] = []
    if gradient_policy.clip_global_norm is not None:
        chain_parts.append(_clip_by_policy(gradient_policy))
    chain_parts.append(optax.multi_transform(transforms, lambda _: labels))
    tx = optax.chain(*chain_parts)
    if gradient_policy.nonfinite == "skip":
        tx = optax.apply_if_finite(
            tx,
            max_consecutive_errors=gradient_policy.max_consecutive_nonfinite,
        )
    if accumulation.steps > 1:
        tx = optax.MultiSteps(
            tx,
            every_k_schedule=accumulation.steps,
            use_grad_mean=_multisteps_use_grad_mean(accumulation),
            accumulator_dtype=accumulation.accumulate_dtype,
        )
    return tx


def _schedule_free_eval_params(
    params: Any,
    state: optax.OptState | None,
    view: str,
) -> Any:
    if view == "optimizer":
        return params
    if view != "schedule_free":
        raise ValueError(f"unknown eval params view: {view!r}.")
    if state is None:
        raise ValueError("schedule-free eval_params requires optimizer state.")
    schedule_free_state = _unwrap_schedule_free_state(state)
    return schedule_free_eval_params(schedule_free_state, params)


def _unwrap_schedule_free_state(state: Any) -> Any:
    current = state
    for _ in range(8):
        if hasattr(current, "b1") and hasattr(current, "z"):
            return current
        if hasattr(current, "inner_opt_state"):
            current = current.inner_opt_state
            continue
        if hasattr(current, "inner_state"):
            current = current.inner_state
            continue
        break
    raise ValueError("could not find ScheduleFreeState inside optimizer state.")


def _decay_mask_from_labels(
    labels: Any,
    groups: tuple[CompiledGroup, ...],
) -> Any:
    decay_by_label = {
        group.source_label: group.weight_decay
        for group in groups
    }
    return jax.tree.map(
        lambda label: None if label is None else bool(decay_by_label[label]),
        labels,
        is_leaf=lambda x: x is None,
    )


def _resolve_sf_bool(enabled_by_plus: bool, value: bool | None) -> bool:
    return enabled_by_plus if value is None else bool(value)


def _make_report(
    fingerprint: str,
    logical_id_table_hash: str,
    model_state_structure_hash: str | None,
    groups: tuple[CompiledGroup, ...],
    schedule: ScheduleConfig,
    accumulation: AccumulationConfig,
    precision: PrecisionConfig,
    state_quantization: StateQuantizationConfig | None = None,
    *,
    warnings: tuple[str, ...],
) -> OptimizerReport:
    preview_lr = max((group.effective_lr for group in groups), default=0.0)
    schedule_preview = preview_schedule(
        schedule,
        peak_lr=preview_lr,
        total_steps=schedule.total_steps,
    )
    trainable_params = sum(group.param_count for group in groups)
    trainable_bytes = sum(group.byte_count for group in groups)
    return OptimizerReport(
        fingerprint=fingerprint,
        groups=groups,
        schedule_preview=schedule_preview,
        trainable_params=trainable_params,
        trainable_bytes=trainable_bytes,
        estimated_state_bytes=estimate_optimizer_state(
            groups,
            precision,
            state_quantization,
        ),
        total_steps=schedule.total_steps,
        logical_id_table_hash=logical_id_table_hash,
        model_state_structure_hash=model_state_structure_hash,
        accumulation_steps=accumulation.steps,
        schedule_step_counter=schedule.step_counter,
        state_policies=_state_policies(groups, precision, state_quantization),
        warnings=warnings,
    )


def _report_warnings(
    plan_warnings: tuple[str, ...],
    trainable: Any,
    precision: PrecisionConfig,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *plan_warnings,
                *_precision_warnings(trainable, precision),
            )
        )
    )


def _hybrid_profile_warnings(family: str) -> tuple[str, ...]:
    if family == "soap":
        return (
            "SOAP-style builder is experimental and uses Rollfast's "
            "bidirectional matrix-preconditioning path; it is not labeled "
            "paper-exact SOAP.",
        )
    if family == "muon":
        return (
            "Muon builder is experimental for pretrained fine-tuning until "
            "benchmarked on target workloads.",
        )
    return ()


def _hybrid_method_config(family: str, family_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(family_kwargs.get("config", {}))
    if family == "muon":
        return {
            "method": "muon",
            **config,
            "matrix_optimizer": "optax.contrib.muon",
            "fallback_optimizer": "adamw",
            "matrix_eligibility": "rank_2_parameters",
            "orthogonalization": "newton_schulz",
            "profile_fidelity": "experimental",
            "reference_validated": False,
        }
    if family == "soap":
        return {
            "method": "soap",
            **config,
            "matrix_optimizer": "prism_bidirectional_surrogate",
            "fallback_optimizer": "adamw",
            "matrix_eligibility": "rank_2_parameters",
            "preconditioner_update_interval": family_kwargs.get(
                "preconditioner_update_interval"
            ),
            "profile_fidelity": "experimental",
            "reference_validated": False,
            "reference_validation": "pending_soap_equation_level_fixture",
        }
    if family in {"aurora", "prism", "kron"}:
        return {
            "method": family,
            **{key: value for key, value in family_kwargs.items() if key != "config"},
            "profile_fidelity": "experimental",
        }
    return {"method": family}


def _precision_warnings(trainable: Any, precision: PrecisionConfig) -> tuple[str, ...]:
    dtypes = {
        jnp.dtype(leaf.dtype)
        for leaf in jax.tree.leaves(trainable, is_leaf=lambda x: x is None)
        if leaf is not None and hasattr(leaf, "dtype")
    }
    warnings: list[str] = []
    if jnp.dtype(jnp.float16) in dtypes and precision.loss_scale == "none":
        warnings.append("fp16 without loss scaling.")
    if (
        precision.master_params == "never"
        and dtypes & {jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)}
    ):
        warnings.append("low-precision stored parameters without master parameters.")
    return tuple(warnings)


def _state_policies(
    groups: tuple[CompiledGroup, ...],
    precision: PrecisionConfig,
    state_quantization: StateQuantizationConfig | None,
) -> dict[str, str]:
    if state_quantization is None or not state_quantization.enabled:
        dtype_name = jnp.dtype(precision.moment_dtype).name
        return {group.source_label: f"{dtype_name}_moments" for group in groups}
    fallback_name = jnp.dtype(state_quantization.fallback_dtype).name
    return {
        group.source_label: (
            "blockwise_int8_moments"
            if _quantize_group_state(group, state_quantization)
            else f"{fallback_name}_fallback_moments"
        )
        for group in groups
    }


def _quantize_group_state(
    group: CompiledGroup,
    state_quantization: StateQuantizationConfig,
) -> bool:
    if not state_quantization.enabled:
        return False
    if group.param_count < state_quantization.min_size:
        return False
    keep_tags = {tag.lower() for tag in state_quantization.keep_fp32_tags}
    group_terms = {tag.lower() for tag in group.tags}
    group_terms.update((group.source_label.lower(), group.role.lower()))
    return not any(
        keep_tag in term
        for keep_tag in keep_tags
        for term in group_terms
    )


def _can_use_factorized_adamw(
    groups: tuple[CompiledGroup, ...],
    optimizer: OptimizerConfig,
    state_quantization: StateQuantizationConfig,
) -> bool:
    return (
        groups
        and optimizer.name == "adamw"
        and not optimizer.use_magma
        and not state_quantization.enabled
        and all(group.optimizer == "adamw" for group in groups)
    )


def _galore_rank_for_group(galore: GaLoreConfig, label: str) -> int:
    if isinstance(galore.rank, Mapping):
        if label not in galore.rank:
            raise ValueError(f"GaLore rank mapping is missing label {label!r}.")
        return int(galore.rank[label])
    return int(galore.rank)


def _estimate_galore_state_bytes(
    trainable: Any,
    labels: Any,
    galore: GaLoreConfig,
    precision: PrecisionConfig,
) -> int:
    total = 0
    for leaf, label in zip(
        jax.tree.leaves(trainable, is_leaf=lambda x: x is None),
        jax.tree.leaves(labels, is_leaf=lambda x: x is None),
        strict=True,
    ):
        if leaf is None or label is None:
            continue
        total += projected_state_nbytes(
            tuple(leaf.shape),
            rank=_galore_rank_for_group(galore, str(label)),
            projection=galore.projection,
            basis_dtype=galore.basis_dtype,
            moment_dtype=precision.moment_dtype,
            min_matrix_size=galore.min_matrix_size,
        )
    return total


def _estimate_galore_refresh_communication_bytes(
    trainable: Any,
    labels: Any,
    galore: GaLoreConfig,
) -> int:
    total = 0
    for leaf, label in zip(
        jax.tree.leaves(trainable, is_leaf=lambda x: x is None),
        jax.tree.leaves(labels, is_leaf=lambda x: x is None),
        strict=True,
    ):
        if leaf is None or label is None:
            continue
        _galore_rank_for_group(galore, str(label))
        if len(leaf.shape) == 2 and int(leaf.size) >= galore.min_matrix_size:
            total += int(leaf.size * leaf.dtype.itemsize)
    return total


def _check_galore_sharding_policy(
    sharding: ShardingPolicy,
    *,
    refresh_communication_bytes: int,
) -> None:
    if not sharding.mesh_axes or refresh_communication_bytes == 0:
        return
    if sharding.allow_host_materialization:
        return
    raise ValueError(
        "GaLore basis refresh under mesh sharding may require a hidden "
        "host/device all-gather. Set ShardingPolicy.allow_host_materialization=True "
        "to opt in after reviewing the reported communication estimate."
    )


def _galore_sharding_warnings(
    sharding: ShardingPolicy,
    *,
    refresh_communication_bytes: int,
) -> tuple[str, ...]:
    if not sharding.mesh_axes or refresh_communication_bytes == 0:
        return ()
    if not sharding.allow_host_materialization:
        return ()
    return (
        "GaLore basis refresh may materialize full gradient matrices; "
        f"estimated refresh communication bytes={refresh_communication_bytes}.",
    )


def _galore_profile_warnings(galore: GaLoreConfig) -> tuple[str, ...]:
    if galore.state_on_basis_refresh != "transport":
        return ()
    return (
        "GaLore basis-aware projected-state transport is experimental and not "
        "reference-validated.",
    )


def _galore_method_config(
    galore: GaLoreConfig,
    sharding: ShardingPolicy,
    *,
    refresh_communication_bytes: int,
) -> dict[str, Any]:
    rank = dict(galore.rank) if isinstance(galore.rank, Mapping) else int(galore.rank)
    return {
        "method": "galore",
        "rank": rank,
        "update_interval": galore.update_interval,
        "scale": galore.scale,
        "projection": galore.projection,
        "basis_method": galore.basis_method,
        "basis_dtype": jnp.dtype(galore.basis_dtype).name,
        "state_on_basis_refresh": galore.state_on_basis_refresh,
        "state_transport_profile": (
            "experimental_basis_overlap"
            if galore.state_on_basis_refresh == "transport"
            else galore.state_on_basis_refresh
        ),
        "reference_validated": galore.state_on_basis_refresh != "transport",
        "min_matrix_size": galore.min_matrix_size,
        "basis_refresh_communication_bytes": refresh_communication_bytes,
        "basis_refresh_collective": (
            "host_or_device_all_gather"
            if sharding.mesh_axes and refresh_communication_bytes > 0
            else "local"
        ),
        "allow_host_materialization": sharding.allow_host_materialization,
    }


def _estimate_apollo_state_bytes(
    trainable: Any,
    apollo: APOLLOConfig,
    precision: PrecisionConfig,
) -> int:
    total = 0
    for leaf in jax.tree.leaves(trainable, is_leaf=lambda x: x is None):
        if leaf is None:
            continue
        total += apollo_state_nbytes(
            tuple(leaf.shape),
            rank=apollo.rank,
            mini=apollo.mini,
            moment_dtype=precision.moment_dtype,
        )
    return total


def _apollo_effective_scale(apollo: APOLLOConfig) -> float:
    if apollo.scale is not None:
        return float(apollo.scale)
    return 128.0 if apollo.mini else 1.0


def _apollo_method_config(apollo: APOLLOConfig) -> dict[str, Any]:
    refresh_policy = (
        "fixed"
        if apollo.projection_refresh_interval is None
        else f"every_{apollo.projection_refresh_interval}_steps"
    )
    return {
        "method": "apollo_mini" if apollo.mini else "apollo",
        **apollo.to_dict(),
        "effective_rank": 1 if apollo.mini else apollo.rank,
        "effective_scaling": "tensor" if apollo.mini else apollo.scaling,
        "effective_scale": _apollo_effective_scale(apollo),
        "projection_type": "std",
        "projection_orientation": "std_right_when_rows_gte_cols",
        "projection_matrix_variance": "normal/sqrt(rank)",
        "normalization_axes": "projected_l2_per_channel_or_tensor",
        "projection_refresh_policy": refresh_policy,
        "profile_fidelity": "experimental",
        "reference_validated": False,
        "reference_validation": "pending_author_implementation_compatibility_test",
    }


def _apollo_profile_warnings() -> tuple[str, ...]:
    return (
        "APOLLO/APOLLO-Mini implementation is experimental until validated "
        "against the authors' reference implementation.",
    )


def _multisteps_use_grad_mean(accumulation: AccumulationConfig) -> bool:
    return accumulation.normalization != "none"


__all__ = (
    "adamw8_from_plan",
    "adamw_from_plan",
    "apollo_adamw_from_plan",
    "compile_optimizer",
    "estimate_optimizer_state",
    "galore_adamw_from_plan",
    "hybrid_aurora_adam_from_plan",
    "hybrid_kron_adam_from_plan",
    "hybrid_prism_adam_from_plan",
    "muon_adam_from_plan",
    "optimizer_from_plan",
    "schedule_free_adam_from_plan",
    "soap_adam_from_plan",
)
