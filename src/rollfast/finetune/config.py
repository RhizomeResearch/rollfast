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
    "custom",
]
StepCounter = Literal["optimizer", "micro"]
NonFinitePolicy = Literal["skip", "none", "raise"]
Normalization = Literal["examples", "tokens", "pairs", "pixels", "custom", "none"]
SAMNorm = Literal["global_l2"]
OptimizerName = Literal[
    "adamw",
    "adamw8",
    "schedule_free_adam",
    "aurora_adam",
    "prism_adam",
    "kron_adam",
]
ProfileFidelity = Literal[
    "safe_default",
    "paper_exact",
    "reference_implementation",
    "experimental",
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
class OptimizerProfile:
    """Declared exactness profile for an optimizer/controller method."""

    id: str
    method: str
    fidelity: ProfileFidelity
    reference_ids: tuple[str, ...]
    config: Mapping[str, Any]
    known_deviations: tuple[str, ...] = ()


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
    normalization: Normalization = "examples"
    remainder: Literal["error", "drop", "apply_with_true_normalizer"] = "error"
    accumulate_dtype: Any = jnp.float32
    reduce_after_accumulation: bool = True
    finite_policy: Literal["discard_window", "error"] = "discard_window"

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        _canonical_dtype(self.accumulate_dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "steps": self.steps,
            "normalization": self.normalization,
            "remainder": self.remainder,
            "accumulate_dtype": _dtype_name(self.accumulate_dtype),
            "reduce_after_accumulation": self.reduce_after_accumulation,
            "finite_policy": self.finite_policy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AccumulationConfig":
        return cls(
            steps=data.get("steps", 1),
            normalization=data.get("normalization", "examples"),
            remainder=data.get("remainder", "error"),
            accumulate_dtype=_dtype_from_name(data.get("accumulate_dtype", "float32")),
            reduce_after_accumulation=data.get("reduce_after_accumulation", True),
            finite_policy=data.get("finite_policy", "discard_window"),
        )


@dataclass(frozen=True)
class PrecisionConfig:
    """Mixed-precision optimizer-state policy."""

    expected_model_compute_dtype: Any | None = None
    gradient_dtype: Any = jnp.float32
    accumulation_dtype: Any = jnp.float32
    master_params: Literal["auto", "always", "never"] = "auto"
    master_param_dtype: Any = jnp.float32
    moment_dtype: Any = jnp.float32
    preconditioner_dtype: Any = jnp.float32
    update_compute_dtype: Any = jnp.float32
    cast_back: Literal["nearest", "stochastic"] = "nearest"
    loss_scale: Literal["none", "static", "dynamic"] = "none"
    static_loss_scale: float = 2**15
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    def __post_init__(self) -> None:
        if self.expected_model_compute_dtype is not None:
            _canonical_dtype(self.expected_model_compute_dtype)
        _canonical_dtype(self.gradient_dtype)
        _canonical_dtype(self.accumulation_dtype)
        _canonical_dtype(self.master_param_dtype)
        _canonical_dtype(self.moment_dtype)
        _canonical_dtype(self.preconditioner_dtype)
        _canonical_dtype(self.update_compute_dtype)
        if self.static_loss_scale <= 0.0:
            raise ValueError("static_loss_scale must be positive.")
        if self.growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1.")
        if not 0.0 < self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must satisfy 0 < value < 1.")
        if self.growth_interval <= 0:
            raise ValueError("growth_interval must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "expected_model_compute_dtype": None
            if self.expected_model_compute_dtype is None
            else _dtype_name(self.expected_model_compute_dtype),
            "gradient_dtype": _dtype_name(self.gradient_dtype),
            "accumulation_dtype": _dtype_name(self.accumulation_dtype),
            "master_params": self.master_params,
            "master_param_dtype": _dtype_name(self.master_param_dtype),
            "moment_dtype": _dtype_name(self.moment_dtype),
            "preconditioner_dtype": _dtype_name(self.preconditioner_dtype),
            "update_compute_dtype": _dtype_name(self.update_compute_dtype),
            "cast_back": self.cast_back,
            "loss_scale": self.loss_scale,
            "static_loss_scale": self.static_loss_scale,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PrecisionConfig":
        expected = data.get("expected_model_compute_dtype")
        return cls(
            expected_model_compute_dtype=None
            if expected is None
            else _dtype_from_name(expected),
            gradient_dtype=_dtype_from_name(data.get("gradient_dtype", "float32")),
            accumulation_dtype=_dtype_from_name(data.get("accumulation_dtype", "float32")),
            master_params=data.get("master_params", "auto"),
            master_param_dtype=_dtype_from_name(data.get("master_param_dtype", "float32")),
            moment_dtype=_dtype_from_name(data.get("moment_dtype", "float32")),
            preconditioner_dtype=_dtype_from_name(data.get("preconditioner_dtype", "float32")),
            update_compute_dtype=_dtype_from_name(data.get("update_compute_dtype", "float32")),
            cast_back=data.get("cast_back", "nearest"),
            loss_scale=data.get("loss_scale", "none"),
            static_loss_scale=data.get("static_loss_scale", 2**15),
            growth_factor=data.get("growth_factor", 2.0),
            backoff_factor=data.get("backoff_factor", 0.5),
            growth_interval=data.get("growth_interval", 2000),
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
    norm: SAMNorm = "global_l2"
    eps: float = 1e-12
    perturb_bias: bool = True
    perturb_norm: bool = True
    axis_name: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.rho <= 0.0:
            raise ValueError("SAM rho must be positive.")
        if self.norm != "global_l2":
            raise ValueError("SAM currently supports only norm='global_l2'.")
        if self.eps <= 0.0:
            raise ValueError("SAM eps must be positive.")

    @property
    def adaptive(self) -> Literal[False]:
        """Plain SAM is never adaptive; use ASAMConfig for adaptive SAM."""

        return False

    @property
    def eta(self) -> float:
        """Plain SAM has no ASAM eta term."""

        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": self.enabled,
            "rho": self.rho,
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
            norm=data.get("norm", "global_l2"),
            eps=data.get("eps", 1e-12),
            perturb_bias=data.get("perturb_bias", True),
            perturb_norm=data.get("perturb_norm", True),
            axis_name=data.get("axis_name"),
        )


@dataclass(frozen=True)
class ASAMConfig:
    enabled: bool = False
    rho: float = 0.5
    eta: float = 0.01
    norm: SAMNorm = "global_l2"
    eps: float = 1e-12
    perturb_bias: bool = False
    perturb_norm: bool = False
    axis_name: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.rho <= 0.0:
            raise ValueError("ASAM rho must be positive.")
        if self.eta < 0.0:
            raise ValueError("ASAM eta must be non-negative.")
        if self.norm != "global_l2":
            raise ValueError("ASAM currently supports only norm='global_l2'.")
        if self.eps <= 0.0:
            raise ValueError("ASAM eps must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": self.enabled,
            "rho": self.rho,
            "eta": self.eta,
            "norm": self.norm,
            "eps": self.eps,
            "perturb_bias": self.perturb_bias,
            "perturb_norm": self.perturb_norm,
            "axis_name": self.axis_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ASAMConfig":
        return cls(
            enabled=data.get("enabled", False),
            rho=data.get("rho", 0.5),
            eta=data.get("eta", 0.01),
            norm=data.get("norm", "global_l2"),
            eps=data.get("eps", 1e-12),
            perturb_bias=data.get("perturb_bias", False),
            perturb_norm=data.get("perturb_norm", False),
            axis_name=data.get("axis_name"),
        )


@dataclass(frozen=True)
class AdaLoRAControllerConfig:
    initial_budget: int = 12
    target_budget: int = 8
    t_init: int = 0
    t_final: int = 0
    allocation_interval: int = 1
    min_rank: int = 1
    beta_sensitivity: float = 0.85
    beta_uncertainty: float = 0.85
    orthogonal_reg_weight: float = 0.5
    score_eps: float = 1e-8
    final_support_frozen: bool = True
    tie_break: Literal["logical_id"] = "logical_id"

    def __post_init__(self) -> None:
        if self.initial_budget < 1:
            raise ValueError("AdaLoRA initial_budget must be positive.")
        if self.target_budget < 1:
            raise ValueError("AdaLoRA target_budget must be positive.")
        if self.t_init < 0 or self.t_final < 0:
            raise ValueError("AdaLoRA t_init and t_final must be non-negative.")
        if self.allocation_interval < 1:
            raise ValueError("AdaLoRA allocation_interval must be positive.")
        if self.min_rank < 0:
            raise ValueError("AdaLoRA min_rank must be non-negative.")
        if not 0.0 <= self.beta_sensitivity < 1.0:
            raise ValueError("AdaLoRA beta_sensitivity must satisfy 0 <= beta < 1.")
        if not 0.0 <= self.beta_uncertainty < 1.0:
            raise ValueError("AdaLoRA beta_uncertainty must satisfy 0 <= beta < 1.")
        if self.orthogonal_reg_weight < 0.0:
            raise ValueError("AdaLoRA orthogonal_reg_weight must be non-negative.")
        if self.score_eps <= 0.0:
            raise ValueError("AdaLoRA score_eps must be positive.")

    def resolved_interval(self, total_steps: int) -> int:
        del total_steps
        return self.allocation_interval

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "initial_budget": self.initial_budget,
            "target_budget": self.target_budget,
            "t_init": self.t_init,
            "t_final": self.t_final,
            "allocation_interval": self.allocation_interval,
            "min_rank": self.min_rank,
            "beta_sensitivity": self.beta_sensitivity,
            "beta_uncertainty": self.beta_uncertainty,
            "orthogonal_reg_weight": self.orthogonal_reg_weight,
            "score_eps": self.score_eps,
            "final_support_frozen": self.final_support_frozen,
            "tie_break": self.tie_break,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdaLoRAControllerConfig":
        return cls(
            initial_budget=data.get("initial_budget", 12),
            target_budget=data.get("target_budget", 8),
            t_init=data.get("t_init", 0),
            t_final=data.get("t_final", 0),
            allocation_interval=data.get("allocation_interval", 1),
            min_rank=data.get("min_rank", 1),
            beta_sensitivity=data.get("beta_sensitivity", 0.85),
            beta_uncertainty=data.get("beta_uncertainty", 0.85),
            orthogonal_reg_weight=data.get("orthogonal_reg_weight", 0.5),
            score_eps=data.get("score_eps", 1e-8),
            final_support_frozen=data.get("final_support_frozen", True),
            tie_break=data.get("tie_break", "logical_id"),
        )


@dataclass(frozen=True)
class LossBundle:
    """Summed loss contract for exact accumulation."""

    loss_sum: Any
    normalizer: Any
    metrics_sums: Mapping[str, Any]
    metric_normalizers: Mapping[str, Any]
    new_model_state: Any | None
    aux_sums: Mapping[str, Any] = field(default_factory=dict)
    aux_normalizers: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuxLossRule:
    """How Rollfast combines one Equimo-provided auxiliary loss."""

    name: str
    coefficient: float | ScheduleConfig
    normalization: Literal["raw", "examples", "tokens", "parameters"]
    counter: Literal["optimizer_step", "schedule_step"] = "optimizer_step"


@dataclass(frozen=True)
class NormalizedLeaf:
    """Canonical Rollfast view of one trainable plan leaf."""

    logical_id: str
    physical_path: tuple[str | int, ...]
    alias_group: str | None
    role: str
    depth: int | None
    tags: frozenset[str]
    shape: tuple[int, ...]
    dtype: str
    layout: str | None
    sharding_fingerprint: str | None
    source_label: str
    lr_multiplier: float
    weight_decay: bool


@dataclass(frozen=True)
class CompiledPolicyTrees:
    """Factorized per-leaf optimization policies."""

    optimizer_family: Any
    lr_multiplier: Any
    decay_coefficient: Any
    state_precision: Any
    clip_group: Any
    logical_ids: Any


@dataclass(frozen=True)
class AlgorithmSemantics:
    """Ownership declaration for optimizer LR and decay application."""

    lr_application: Literal["inside", "post_transform", "tree_aware"]
    weight_decay_owner: Literal["none", "algorithm", "external"]
    supports_leaf_lr_multiplier: bool
    supports_leaf_decay: bool


@dataclass(frozen=True)
class LoRAPlusConfig:
    """LoRA+ learning-rate ratio policy."""

    ratio_B_over_A: float = 16.0
    anchor: Literal["A", "B", "geometric_mean"] = "B"
    apply_same_schedule_shape: bool = True
    weight_decay_A: float = 0.0
    weight_decay_B: float = 0.0


@dataclass(frozen=True)
class ShardingPolicy:
    """Optimizer-state sharding and placement contract."""

    mesh_axes: tuple[str, ...] = ()
    data_axes: tuple[str, ...] = ("data",)
    parameter_axes: tuple[str, ...] = ()
    state_placement: Literal[
        "follow_param",
        "replicate_small_follow_large",
        "explicit",
    ] = "follow_param"
    small_state_threshold: int = 4096
    allow_host_materialization: bool = False


@dataclass(frozen=True)
class AccumulationState:
    grad_numerator: Any
    normalizer: Any
    metric_sums: Any
    metric_normalizers: Any
    microsteps_in_window: Any
    all_finite: Any
    pending_model_state: Any | None


@dataclass(frozen=True)
class StepCounters:
    microstep: Any
    attempted_update: Any
    successful_update: Any
    schedule_step: Any
    rank_step: Any
    average_step: Any
    loss_scale_growth_step: Any


@dataclass(frozen=True)
class RNGStreams:
    forward: Any
    sam: Any
    stochastic_rounding: Any
    quantization: Any
    controller: Any


@dataclass(frozen=True)
class FineTuneStepState:
    optimizer_state: Any
    model_state: Any | None
    master_params: Any | None
    accumulation: AccumulationState
    loss_scale: Any | None
    ema_state: Any | None
    swa_state: Any | None
    adalora_state: Any | None
    projection_state: Any | None
    schedule_free_state: Any | None
    counters: StepCounters
    rng: RNGStreams


@dataclass(frozen=True)
class GaLoreConfig:
    """GaLore optimizer projection configuration."""

    rank: int
    update_interval: int = 200
    scale: float = 0.25
    projection_type: Literal["std", "reverse_std", "right", "left", "full"] = "std"
    target: Literal["matrix_only"] = "matrix_only"
    fallback_optimizer: OptimizerName = "adamw"


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
    "AccumulationState",
    "AdaLoRAControllerConfig",
    "AlgorithmSemantics",
    "ASAMConfig",
    "AuxLossRule",
    "CompiledGroup",
    "CompiledPolicyTrees",
    "EMAConfig",
    "FineTuneStepState",
    "GaLoreConfig",
    "GradientPolicy",
    "GroupRule",
    "LoRAPlusConfig",
    "LossBundle",
    "NormalizedLeaf",
    "OptimizerBundle",
    "OptimizerConfig",
    "OptimizerProfile",
    "OptimizerReport",
    "PlanGroup",
    "PrecisionConfig",
    "RNGStreams",
    "SAMConfig",
    "SWAConfig",
    "ScheduleConfig",
    "SchedulePoint",
    "ShardingPolicy",
    "StepCounters",
    "StateQuantizationConfig",
)
