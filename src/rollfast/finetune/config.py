"""Configuration and report objects for plan-aware fine-tuning optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Mapping

import jax.numpy as jnp
import optax


SCHEMA_VERSION = 1

ScheduleKind = Literal[
    "constant",
    "warmup_cosine",
    "wsd",
    "linear",
    "polynomial",
]
StepCounter = Literal["optimizer", "micro"]
NonFinitePolicy = Literal["skip", "none", "raise"]
Reduction = Literal["mean", "sum"]
SAMNorm = Literal["global_l2"]
OptimizerName = Literal[
    "adamw",
    "adamw8",
    "schedule_free_adam",
    "aurora_adam",
    "prism_adam",
    "kron_adam",
]


@dataclass(frozen=True)
class ScheduleConfig:
    """Serializable learning-rate schedule configuration."""

    kind: ScheduleKind = "warmup_cosine"
    total_steps: int | None = None
    warmup_steps: int | None = None
    warmup_fraction: float = 0.05
    decay_steps: int | None = None
    decay_fraction: float = 0.10
    end_lr_ratio: float = 0.01
    power: float = 1.0
    step_counter: StepCounter = "optimizer"

    def __post_init__(self) -> None:
        if self.total_steps is not None and self.total_steps <= 0:
            raise ValueError("total_steps must be positive when provided.")
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative when provided.")
        if self.decay_steps is not None and self.decay_steps < 0:
            raise ValueError("decay_steps must be non-negative when provided.")
        _check_fraction("warmup_fraction", self.warmup_fraction)
        _check_fraction("decay_fraction", self.decay_fraction)
        if self.end_lr_ratio < 0.0:
            raise ValueError("end_lr_ratio must be non-negative.")
        if self.power <= 0.0:
            raise ValueError("power must be positive.")

    def resolved(self, total_steps: int | None = None) -> "ScheduleConfig":
        """Return a copy with ``total_steps`` filled in if supplied."""

        resolved_total = self.total_steps if total_steps is None else total_steps
        return replace(self, total_steps=resolved_total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "warmup_fraction": self.warmup_fraction,
            "decay_steps": self.decay_steps,
            "decay_fraction": self.decay_fraction,
            "end_lr_ratio": self.end_lr_ratio,
            "power": self.power,
            "step_counter": self.step_counter,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduleConfig":
        return cls(
            kind=data.get("kind", "warmup_cosine"),
            total_steps=data.get("total_steps"),
            warmup_steps=data.get("warmup_steps"),
            warmup_fraction=data.get("warmup_fraction", 0.05),
            decay_steps=data.get("decay_steps"),
            decay_fraction=data.get("decay_fraction", 0.10),
            end_lr_ratio=data.get("end_lr_ratio", 0.01),
            power=data.get("power", 1.0),
            step_counter=data.get("step_counter", "optimizer"),
        )


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer-family defaults before plan group multipliers are applied."""

    name: OptimizerName = "adamw"
    base_lr: float = 5e-4
    weight_decay: float = 0.05
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-6
    eps_root: float = 0.0
    nesterov: bool = False
    use_magma: bool = False
    magma_p: float = 0.5
    magma_tau: float = 2.0
    lora_b_lr_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.base_lr <= 0.0:
            raise ValueError("base_lr must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0.0 <= self.b1 < 1.0:
            raise ValueError("b1 must satisfy 0 <= b1 < 1.")
        if not 0.0 <= self.b2 < 1.0:
            raise ValueError("b2 must satisfy 0 <= b2 < 1.")
        if self.eps < 0.0:
            raise ValueError("eps must be non-negative.")
        if self.eps_root < 0.0:
            raise ValueError("eps_root must be non-negative.")
        if not 0.0 < self.magma_p <= 1.0:
            raise ValueError("magma_p must satisfy 0 < magma_p <= 1.")
        if self.magma_tau <= 0.0:
            raise ValueError("magma_tau must be positive.")
        if self.lora_b_lr_ratio is not None and self.lora_b_lr_ratio <= 0.0:
            raise ValueError("lora_b_lr_ratio must be positive when provided.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "base_lr": self.base_lr,
            "weight_decay": self.weight_decay,
            "b1": self.b1,
            "b2": self.b2,
            "eps": self.eps,
            "eps_root": self.eps_root,
            "nesterov": self.nesterov,
            "use_magma": self.use_magma,
            "magma_p": self.magma_p,
            "magma_tau": self.magma_tau,
            "lora_b_lr_ratio": self.lora_b_lr_ratio,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OptimizerConfig":
        return cls(
            name=data.get("name", "adamw"),
            base_lr=data.get("base_lr", 5e-4),
            weight_decay=data.get("weight_decay", 0.05),
            b1=data.get("b1", 0.9),
            b2=data.get("b2", 0.999),
            eps=data.get("eps", 1e-6),
            eps_root=data.get("eps_root", 0.0),
            nesterov=data.get("nesterov", False),
            use_magma=data.get("use_magma", False),
            magma_p=data.get("magma_p", 0.5),
            magma_tau=data.get("magma_tau", 2.0),
            lora_b_lr_ratio=data.get("lora_b_lr_ratio"),
        )


@dataclass(frozen=True)
class GradientPolicy:
    """Gradient preprocessing and finite-value behavior."""

    clip_global_norm: float | None = 1.0
    nonfinite: NonFinitePolicy = "skip"
    max_consecutive_nonfinite: int = 1
    axis_name: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.clip_global_norm is not None and self.clip_global_norm <= 0.0:
            raise ValueError("clip_global_norm must be positive when provided.")
        if self.max_consecutive_nonfinite <= 0:
            raise ValueError("max_consecutive_nonfinite must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "clip_global_norm": self.clip_global_norm,
            "nonfinite": self.nonfinite,
            "max_consecutive_nonfinite": self.max_consecutive_nonfinite,
            "axis_name": self.axis_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GradientPolicy":
        return cls(
            clip_global_norm=data.get("clip_global_norm", 1.0),
            nonfinite=data.get("nonfinite", "skip"),
            max_consecutive_nonfinite=data.get("max_consecutive_nonfinite", 1),
            axis_name=data.get("axis_name"),
        )


@dataclass(frozen=True)
class AccumulationConfig:
    """Microbatch accumulation settings."""

    steps: int = 1
    reduction: Reduction = "mean"
    accumulator_dtype: Any = jnp.float32
    step_counter: StepCounter = "optimizer"

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        _canonical_dtype(self.accumulator_dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "steps": self.steps,
            "reduction": self.reduction,
            "accumulator_dtype": _dtype_name(self.accumulator_dtype),
            "step_counter": self.step_counter,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AccumulationConfig":
        return cls(
            steps=data.get("steps", 1),
            reduction=data.get("reduction", "mean"),
            accumulator_dtype=_dtype_from_name(data.get("accumulator_dtype", "float32")),
            step_counter=data.get("step_counter", "optimizer"),
        )


@dataclass(frozen=True)
class PrecisionConfig:
    """Mixed-precision optimizer-state policy."""

    compute_dtype: Any = jnp.bfloat16
    moment_dtype: Any = jnp.float32
    param_dtype: Any | None = None
    loss_scale: float | None = None

    def __post_init__(self) -> None:
        _canonical_dtype(self.compute_dtype)
        _canonical_dtype(self.moment_dtype)
        if self.param_dtype is not None:
            _canonical_dtype(self.param_dtype)
        if self.loss_scale is not None and self.loss_scale <= 0.0:
            raise ValueError("loss_scale must be positive when provided.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "compute_dtype": _dtype_name(self.compute_dtype),
            "moment_dtype": _dtype_name(self.moment_dtype),
            "param_dtype": None if self.param_dtype is None else _dtype_name(self.param_dtype),
            "loss_scale": self.loss_scale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PrecisionConfig":
        param_dtype = data.get("param_dtype")
        return cls(
            compute_dtype=_dtype_from_name(data.get("compute_dtype", "bfloat16")),
            moment_dtype=_dtype_from_name(data.get("moment_dtype", "float32")),
            param_dtype=None if param_dtype is None else _dtype_from_name(param_dtype),
            loss_scale=data.get("loss_scale"),
        )


@dataclass(frozen=True)
class StateQuantizationConfig:
    """Blockwise low-precision state policy for AdamW moments."""

    enabled: bool = False
    bits: int = 8
    block_size: int = 2048
    min_size: int = 4096
    scale_dtype: Any = jnp.float32
    fallback_dtype: Any = jnp.float32
    stochastic_rounding: bool = True
    keep_fp32_tags: tuple[str, ...] = (
        "bias",
        "norm",
        "embed",
        "embedding",
        "prompt",
        "ia3",
        "scale_shift",
    )

    def __post_init__(self) -> None:
        if self.bits != 8:
            raise ValueError("only 8-bit optimizer-state quantization is supported.")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if self.min_size < 0:
            raise ValueError("min_size must be non-negative.")
        _canonical_dtype(self.scale_dtype)
        _canonical_dtype(self.fallback_dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": self.enabled,
            "bits": self.bits,
            "block_size": self.block_size,
            "min_size": self.min_size,
            "scale_dtype": _dtype_name(self.scale_dtype),
            "fallback_dtype": _dtype_name(self.fallback_dtype),
            "stochastic_rounding": self.stochastic_rounding,
            "keep_fp32_tags": self.keep_fp32_tags,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateQuantizationConfig":
        return cls(
            enabled=data.get("enabled", False),
            bits=data.get("bits", 8),
            block_size=data.get("block_size", 2048),
            min_size=data.get("min_size", 4096),
            scale_dtype=_dtype_from_name(data.get("scale_dtype", "float32")),
            fallback_dtype=_dtype_from_name(data.get("fallback_dtype", "float32")),
            stochastic_rounding=data.get("stochastic_rounding", True),
            keep_fp32_tags=tuple(
                data.get(
                    "keep_fp32_tags",
                    (
                        "bias",
                        "norm",
                        "embed",
                        "embedding",
                        "prompt",
                        "ia3",
                        "scale_shift",
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class GroupRule:
    """Rollfast-side override for groups emitted by a fine-tuning plan."""

    label: str | None = None
    label_prefix: str | None = None
    role: str | None = None
    tag: str | None = None
    min_depth: int | None = None
    max_depth: int | None = None
    lr_multiplier: float = 1.0
    weight_decay: bool | None = None
    optimizer: OptimizerName | None = None
    priority: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        if self.lr_multiplier <= 0.0:
            raise ValueError("lr_multiplier must be positive.")
        if self.min_depth is not None and self.max_depth is not None:
            if self.min_depth > self.max_depth:
                raise ValueError("min_depth cannot exceed max_depth.")


@dataclass(frozen=True)
class EMAConfig:
    enabled: bool = False
    decay: float = 0.9999
    update_every: int = 1
    start_step: int = 0
    debias: bool = False
    state_dtype: Any = jnp.float32
    include_tags: tuple[str, ...] = ()
    exclude_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("EMA decay must satisfy 0 <= decay < 1.")
        if self.update_every <= 0:
            raise ValueError("EMA update_every must be positive.")
        if self.start_step < 0:
            raise ValueError("EMA start_step must be non-negative.")
        _canonical_dtype(self.state_dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": self.enabled,
            "decay": self.decay,
            "update_every": self.update_every,
            "start_step": self.start_step,
            "debias": self.debias,
            "state_dtype": _dtype_name(self.state_dtype),
            "include_tags": self.include_tags,
            "exclude_tags": self.exclude_tags,
        }


@dataclass(frozen=True)
class SWAConfig:
    enabled: bool = False
    start_step: int | None = None
    start_fraction: float = 0.75
    frequency: int = 1
    averaging: Literal["uniform"] = "uniform"
    state_dtype: Any = jnp.float32

    def __post_init__(self) -> None:
        if self.start_step is not None and self.start_step < 0:
            raise ValueError("SWA start_step must be non-negative when provided.")
        _check_fraction("start_fraction", self.start_fraction)
        if self.frequency <= 0:
            raise ValueError("SWA frequency must be positive.")
        if self.averaging != "uniform":
            raise ValueError("SWA averaging currently supports only 'uniform'.")
        _canonical_dtype(self.state_dtype)

    def resolved_start_step(self, total_steps: int | None) -> int:
        if self.start_step is not None:
            return self.start_step
        if total_steps is None:
            return 0
        return int(total_steps * self.start_fraction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": self.enabled,
            "start_step": self.start_step,
            "start_fraction": self.start_fraction,
            "frequency": self.frequency,
            "averaging": self.averaging,
            "state_dtype": _dtype_name(self.state_dtype),
        }


@dataclass(frozen=True)
class SAMConfig:
    enabled: bool = False
    rho: float = 0.05
    adaptive: bool = False
    eta: float = 0.0
    norm: SAMNorm = "global_l2"
    eps: float = 1e-12
    perturb_bias: bool = True
    perturb_norm: bool = True
    axis_name: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.rho <= 0.0:
            raise ValueError("SAM rho must be positive.")
        if self.eta < 0.0:
            raise ValueError("SAM eta must be non-negative.")
        if self.norm != "global_l2":
            raise ValueError("SAM currently supports only norm='global_l2'.")
        if self.eps <= 0.0:
            raise ValueError("SAM eps must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": self.enabled,
            "rho": self.rho,
            "adaptive": self.adaptive,
            "eta": self.eta,
            "norm": self.norm,
            "eps": self.eps,
            "perturb_bias": self.perturb_bias,
            "perturb_norm": self.perturb_norm,
            "axis_name": self.axis_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SAMConfig":
        return cls(
            enabled=data.get("enabled", False),
            rho=data.get("rho", 0.05),
            adaptive=data.get("adaptive", False),
            eta=data.get("eta", 0.0),
            norm=data.get("norm", "global_l2"),
            eps=data.get("eps", 1e-12),
            perturb_bias=data.get("perturb_bias", True),
            perturb_norm=data.get("perturb_norm", True),
            axis_name=data.get("axis_name"),
        )


@dataclass(frozen=True)
class AdaLoRAControllerConfig:
    enabled: bool = False
    init_rank: int = 12
    target_rank: int | None = None
    min_rank: int = 1
    initial_warmup_fraction: float = 0.10
    final_tuning_fraction: float = 0.10
    allocation_interval: int | None = None
    beta1: float = 0.85
    beta2: float = 0.85
    orthogonal_reg_weight: float = 0.5
    score_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.init_rank < 1:
            raise ValueError("AdaLoRA init_rank must be positive.")
        if self.target_rank is not None and self.target_rank < 1:
            raise ValueError("AdaLoRA target_rank must be positive when provided.")
        if self.min_rank < 0:
            raise ValueError("AdaLoRA min_rank must be non-negative.")
        if self.target_rank is not None and self.target_rank < self.min_rank:
            raise ValueError("AdaLoRA target_rank must be >= min_rank.")
        if self.allocation_interval is not None and self.allocation_interval < 1:
            raise ValueError("AdaLoRA allocation_interval must be positive.")
        _check_fraction("initial_warmup_fraction", self.initial_warmup_fraction)
        _check_fraction("final_tuning_fraction", self.final_tuning_fraction)
        if self.initial_warmup_fraction + self.final_tuning_fraction > 1.0:
            raise ValueError(
                "AdaLoRA warmup and final tuning fractions must sum to at most 1."
            )
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("AdaLoRA beta1 must satisfy 0 <= beta1 < 1.")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdaLoRA beta2 must satisfy 0 <= beta2 < 1.")
        if self.orthogonal_reg_weight < 0.0:
            raise ValueError("AdaLoRA orthogonal_reg_weight must be non-negative.")
        if self.score_eps <= 0.0:
            raise ValueError("AdaLoRA score_eps must be positive.")

    @classmethod
    def compat(
        cls,
        *,
        init_rank: int = 12,
        target_rank: int = 8,
        tinit: int = 0,
        tfinal: int = 0,
        delta_t: int = 1,
        total_steps: int | None = None,
        beta1: float = 0.85,
        beta2: float = 0.85,
        orthogonal_reg_weight: float = 0.5,
    ) -> "AdaLoRAControllerConfig":
        if total_steps is None:
            initial_fraction = 0.0
            final_fraction = 0.0
        else:
            if total_steps <= 0:
                raise ValueError("total_steps must be positive when provided.")
            initial_fraction = tinit / total_steps
            final_fraction = tfinal / total_steps
        return cls(
            enabled=True,
            init_rank=init_rank,
            target_rank=target_rank,
            initial_warmup_fraction=initial_fraction,
            final_tuning_fraction=final_fraction,
            allocation_interval=delta_t,
            beta1=beta1,
            beta2=beta2,
            orthogonal_reg_weight=orthogonal_reg_weight,
        )

    def resolved_target_rank(self) -> int:
        return self.init_rank if self.target_rank is None else self.target_rank

    def resolved_interval(self, total_steps: int) -> int:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if self.allocation_interval is not None:
            return self.allocation_interval
        return max(1, total_steps // 100)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": self.enabled,
            "init_rank": self.init_rank,
            "target_rank": self.target_rank,
            "min_rank": self.min_rank,
            "initial_warmup_fraction": self.initial_warmup_fraction,
            "final_tuning_fraction": self.final_tuning_fraction,
            "allocation_interval": self.allocation_interval,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "orthogonal_reg_weight": self.orthogonal_reg_weight,
            "score_eps": self.score_eps,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdaLoRAControllerConfig":
        return cls(
            enabled=data.get("enabled", False),
            init_rank=data.get("init_rank", 12),
            target_rank=data.get("target_rank"),
            min_rank=data.get("min_rank", 1),
            initial_warmup_fraction=data.get("initial_warmup_fraction", 0.10),
            final_tuning_fraction=data.get("final_tuning_fraction", 0.10),
            allocation_interval=data.get("allocation_interval"),
            beta1=data.get("beta1", 0.85),
            beta2=data.get("beta2", 0.85),
            orthogonal_reg_weight=data.get("orthogonal_reg_weight", 0.5),
            score_eps=data.get("score_eps", 1e-8),
        )


@dataclass(frozen=True)
class PlanGroup:
    """Rollfast-owned normalized plan group."""

    label: str
    role: str
    depth: int | None
    lr_multiplier: float
    weight_decay: bool
    tags: frozenset[str]
    param_count: int = 0
    byte_count: int = 0
    leaf_count: int = 0


@dataclass(frozen=True)
class CompiledGroup:
    """One resolved optimizer group after Rollfast rules are applied."""

    source_label: str
    optimizer: OptimizerName
    base_lr: float
    plan_lr_multiplier: float
    rule_lr_multiplier: float
    effective_lr: float
    weight_decay: bool
    weight_decay_value: float
    role: str
    depth: int | None
    tags: frozenset[str]
    param_count: int
    byte_count: int
    leaf_count: int


@dataclass(frozen=True)
class SchedulePoint:
    step: int
    value: float


@dataclass(frozen=True)
class OptimizerReport:
    """Static report for a compiled plan-aware optimizer."""

    fingerprint: str
    groups: tuple[CompiledGroup, ...]
    schedule_preview: tuple[SchedulePoint, ...]
    trainable_params: int
    trainable_bytes: int
    estimated_state_bytes: int
    total_steps: int | None
    accumulation_steps: int = 1
    schedule_step_counter: StepCounter = "optimizer"
    state_policies: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def group_table(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "label": group.source_label,
                "optimizer": group.optimizer,
                "effective_lr": group.effective_lr,
                "weight_decay": group.weight_decay_value,
                "params": group.param_count,
                "bytes": group.byte_count,
                "state_policy": self.state_policies.get(group.source_label, "moments"),
                "role": group.role,
                "depth": group.depth,
            }
            for group in self.groups
        )

    def __str__(self) -> str:
        lines = [
            "Rollfast fine-tuning optimizer",
            f"fingerprint: {self.fingerprint[:16]}",
            f"trainable params: {self.trainable_params}",
            f"estimated state bytes: {self.estimated_state_bytes}",
            "groups:",
        ]
        for group in self.groups:
            lines.append(
                "  "
                f"{group.source_label}: opt={group.optimizer} "
                f"lr={group.effective_lr:.6g} wd={group.weight_decay_value:.6g} "
                f"params={group.param_count}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class OptimizerBundle:
    """Compiled Optax transformation plus static fine-tuning metadata."""

    tx: optax.GradientTransformation
    report: OptimizerReport
    optimizer_config: OptimizerConfig
    schedule_config: ScheduleConfig
    gradient_policy: GradientPolicy
    accumulation_config: AccumulationConfig
    precision_config: PrecisionConfig
    quantization_config: StateQuantizationConfig
    ema_config: EMAConfig = field(default_factory=EMAConfig)
    swa_config: SWAConfig = field(default_factory=SWAConfig)
    eval_fn: Callable[[Any, optax.OptState | None, str], Any] | None = None
    eval_params_kind: str = "identity"
    eval_views: tuple[str, ...] = ("optimizer",)
    default_eval_view: str = "optimizer"

    def init(self, params: Any) -> optax.OptState:
        return self.tx.init(params)

    def update(
        self,
        grads: Any,
        state: optax.OptState,
        params: Any | None = None,
        **extra_args: Any,
    ) -> tuple[Any, optax.OptState]:
        return self.tx.update(grads, state, params, **extra_args)

    def eval_params(
        self,
        params: Any,
        state: optax.OptState | None = None,
        *,
        view: str | None = None,
    ) -> Any:
        resolved_view = self.default_eval_view if view is None else view
        if self.eval_fn is not None:
            return self.eval_fn(params, state, resolved_view)
        if resolved_view != "optimizer":
            raise ValueError(f"unknown eval params view: {resolved_view!r}.")
        return params

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": self.report.fingerprint,
            "optimizer": self.optimizer_config.to_dict(),
            "schedule": self.schedule_config.to_dict(),
            "gradient_policy": self.gradient_policy.to_dict(),
            "accumulation": self.accumulation_config.to_dict(),
            "precision": self.precision_config.to_dict(),
            "state_quantization": self.quantization_config.to_dict(),
            "ema": self.ema_config.to_dict(),
            "swa": self.swa_config.to_dict(),
            "eval_params_kind": self.eval_params_kind,
            "eval_views": self.eval_views,
            "default_eval_view": self.default_eval_view,
            "groups": self.report.group_table(),
            "schedule_preview": [
                {"step": point.step, "value": point.value}
                for point in self.report.schedule_preview
            ],
        }


def _check_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} <= 1.")


def _canonical_dtype(dtype: Any) -> jnp.dtype:
    return jnp.dtype(dtype)


def _dtype_name(dtype: Any) -> str:
    return jnp.dtype(dtype).name


def _dtype_from_name(name: str) -> jnp.dtype:
    return jnp.dtype(name)


__all__ = (
    "SCHEMA_VERSION",
    "AccumulationConfig",
    "AdaLoRAControllerConfig",
    "CompiledGroup",
    "EMAConfig",
    "GradientPolicy",
    "GroupRule",
    "OptimizerBundle",
    "OptimizerConfig",
    "OptimizerReport",
    "PlanGroup",
    "PrecisionConfig",
    "SAMConfig",
    "SWAConfig",
    "ScheduleConfig",
    "SchedulePoint",
    "StateQuantizationConfig",
)
