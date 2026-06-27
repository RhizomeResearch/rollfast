"""Fine-tuning optimizer recipe defaults."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math

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


DEFAULT_NO_DECAY_TAGS = (
    "bias",
    "norm",
    "embedding.position",
    "embedding.class_token",
    "embedding.register_token",
    "embedding.distillation_token",
    "embedding.mask_token",
)


def no_decay_rules(
    tags: tuple[str, ...] = DEFAULT_NO_DECAY_TAGS,
    *,
    priority: int = 1,
) -> tuple[GroupRule, ...]:
    """Return common tag-based rules that force zero weight decay."""

    return tuple(
        GroupRule(
            tag=tag,
            weight_decay_value=0.0,
            priority=priority,
            name=f"no_decay:{tag}",
        )
        for tag in dict.fromkeys(str(tag) for tag in tags)
    )


def discriminative_adamw_rules(
    *,
    backbone_lr: float,
    head_lr: float,
    backbone_weight_decay: float = 0.05,
    head_weight_decay: float = 0.01,
    no_decay_tags: tuple[str, ...] = DEFAULT_NO_DECAY_TAGS,
    no_decay_priority: int = 1,
) -> tuple[float, float, tuple[GroupRule, ...]]:
    """Return base AdamW settings and rules for head/backbone fine-tuning."""

    _check_positive_finite("backbone_lr", backbone_lr)
    _check_positive_finite("head_lr", head_lr)
    _check_non_negative_finite("backbone_weight_decay", backbone_weight_decay)
    _check_non_negative_finite("head_weight_decay", head_weight_decay)

    rules = (
        GroupRule(
            role="head",
            lr_multiplier=float(head_lr) / float(backbone_lr),
            weight_decay_value=float(head_weight_decay),
            name="head_backbone:head",
        ),
        *no_decay_rules(no_decay_tags, priority=no_decay_priority),
    )
    return float(backbone_lr), float(backbone_weight_decay), rules


def head_backbone_adamw(
    *,
    backbone_lr: float,
    head_lr: float,
    total_steps: int | None = None,
    backbone_weight_decay: float = 0.05,
    head_weight_decay: float = 0.01,
    no_decay_tags: tuple[str, ...] = DEFAULT_NO_DECAY_TAGS,
    no_decay_priority: int = 1,
    eps: float = 1e-6,
) -> FineTuneOptimizerRecipe:
    """Return AdamW defaults for discriminative head/backbone fine-tuning."""

    base_lr, weight_decay, group_rules = discriminative_adamw_rules(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        backbone_weight_decay=backbone_weight_decay,
        head_weight_decay=head_weight_decay,
        no_decay_tags=no_decay_tags,
        no_decay_priority=no_decay_priority,
    )
    return FineTuneOptimizerRecipe(
        optimizer=OptimizerConfig(
            base_lr=base_lr,
            weight_decay=weight_decay,
            eps=eps,
        ),
        schedule=ScheduleConfig(kind="warmup_cosine", total_steps=total_steps),
        group_rules=group_rules,
    )


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


def _check_positive_finite(name: str, value: float) -> None:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")


def _check_non_negative_finite(name: str, value: float) -> None:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


__all__ = (
    "DEFAULT_NO_DECAY_TAGS",
    "FineTuneOptimizerRecipe",
    "adapters_adamw",
    "bitfit_adamw",
    "discriminative_adamw_rules",
    "full_ft_adamw",
    "head_backbone_adamw",
    "linear_probe_adamw",
    "lora_adamw",
    "no_decay_rules",
    "partial_ft_adamw",
    "prompts_adamw",
)
