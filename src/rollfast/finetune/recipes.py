"""Fine-tuning optimizer recipe defaults."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from .config import (
    AccumulationConfig,
    EMAConfig,
    GradientPolicy,
    GroupRule,
    OptimizerConfig,
    PrecisionConfig,
    SWAConfig,
    ScheduleConfig,
)


@dataclass(frozen=True)
class FineTuneOptimizerRecipe:
    """Serializable collection of configs consumed by ``compile_optimizer``."""

    optimizer: OptimizerConfig
    schedule: ScheduleConfig
    gradient_policy: GradientPolicy = field(default_factory=GradientPolicy)
    accumulation: AccumulationConfig = field(default_factory=AccumulationConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    swa: SWAConfig = field(default_factory=SWAConfig)
    group_rules: tuple[GroupRule, ...] = ()


def full_ft_adamw(
    *,
    total_steps: int | None = None,
    base_lr: float = 5e-4,
    weight_decay: float = 0.05,
) -> FineTuneOptimizerRecipe:
    """Return AdamW defaults for full-model fine-tuning."""

    return FineTuneOptimizerRecipe(
        optimizer=OptimizerConfig(base_lr=base_lr, weight_decay=weight_decay),
        schedule=ScheduleConfig(kind="warmup_cosine", total_steps=total_steps),
    )


def partial_ft_adamw(
    *,
    total_steps: int | None = None,
    base_lr: float = 2e-4,
    weight_decay: float = 0.05,
) -> FineTuneOptimizerRecipe:
    """Return AdamW defaults for partial fine-tuning."""

    return FineTuneOptimizerRecipe(
        optimizer=OptimizerConfig(base_lr=base_lr, weight_decay=weight_decay),
        schedule=ScheduleConfig(kind="warmup_cosine", total_steps=total_steps),
    )


def linear_probe_adamw(
    *,
    total_steps: int | None = None,
    base_lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> FineTuneOptimizerRecipe:
    """Return AdamW defaults for training only a linear probe head."""

    return FineTuneOptimizerRecipe(
        optimizer=OptimizerConfig(base_lr=base_lr, weight_decay=weight_decay),
        schedule=ScheduleConfig(kind="warmup_cosine", total_steps=total_steps),
    )


def lora_adamw(
    *,
    total_steps: int | None = None,
    base_lr: float = 2e-4,
    weight_decay: float = 0.0,
    lora_b_lr_ratio: float | None = None,
) -> FineTuneOptimizerRecipe:
    """Return AdamW defaults for LoRA or LoRA+ fine-tuning."""

    return FineTuneOptimizerRecipe(
        optimizer=OptimizerConfig(
            base_lr=base_lr,
            weight_decay=weight_decay,
            lora_b_lr_ratio=lora_b_lr_ratio,
        ),
        schedule=ScheduleConfig(kind="warmup_cosine", total_steps=total_steps),
    )


def adapters_adamw(
    *,
    total_steps: int | None = None,
    base_lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> FineTuneOptimizerRecipe:
    """Return AdamW defaults for adapter-module fine-tuning."""

    return FineTuneOptimizerRecipe(
        optimizer=OptimizerConfig(base_lr=base_lr, weight_decay=weight_decay),
        schedule=ScheduleConfig(kind="warmup_cosine", total_steps=total_steps),
    )


def prompts_adamw(
    *,
    total_steps: int | None = None,
    base_lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> FineTuneOptimizerRecipe:
    """Return AdamW defaults for prompt-parameter fine-tuning."""

    return FineTuneOptimizerRecipe(
        optimizer=OptimizerConfig(base_lr=base_lr, weight_decay=weight_decay),
        schedule=ScheduleConfig(kind="warmup_cosine", total_steps=total_steps),
    )


def bitfit_adamw(
    *,
    total_steps: int | None = None,
    base_lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> FineTuneOptimizerRecipe:
    """Return AdamW defaults for bias-only fine-tuning."""

    return FineTuneOptimizerRecipe(
        optimizer=OptimizerConfig(base_lr=base_lr, weight_decay=weight_decay),
        schedule=ScheduleConfig(kind="warmup_cosine", total_steps=total_steps),
    )


__all__ = (
    "FineTuneOptimizerRecipe",
    "adapters_adamw",
    "bitfit_adamw",
    "full_ft_adamw",
    "linear_probe_adamw",
    "lora_adamw",
    "partial_ft_adamw",
    "prompts_adamw",
)
