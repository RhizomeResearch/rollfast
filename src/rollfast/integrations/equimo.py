"""Optional Equimo fine-tuning plan conveniences."""

from __future__ import annotations

from typing import Any

from rollfast.finetune import (
    OptimizerBundle,
    adamw_from_plan,
    compile_optimizer,
    make_plan_update_step,
)


def from_equimo_plan(plan: Any, **kwargs: Any) -> OptimizerBundle:
    """Compile an Equimo ``FineTunePlan`` using structural plan fields."""

    _validate_equimo_like_plan(plan)
    return compile_optimizer(plan, **kwargs)


def adamw_from_equimo_plan(plan: Any, **kwargs: Any) -> OptimizerBundle:
    """Build grouped AdamW from an Equimo ``FineTunePlan``."""

    _validate_equimo_like_plan(plan)
    return adamw_from_plan(plan, **kwargs)


def make_equimo_update_step(
    plan: Any, loss_fn: Any, optimizer: OptimizerBundle, **kwargs: Any
):
    """Return a single-update helper that calls ``plan.combine`` before loss."""

    _validate_equimo_like_plan(plan)
    _require_combine(plan)
    return make_plan_update_step(plan, loss_fn, optimizer, **kwargs)


def transition_lpft_stage(plan: Any, **kwargs: Any) -> OptimizerBundle:
    """Compile a new optimizer for an externally prepared LP-FT stage."""

    return from_equimo_plan(plan, **kwargs)


def _validate_equimo_like_plan(plan: Any) -> None:
    missing = [
        attr
        for attr in ("trainable", "frozen", "labels", "group_specs", "identities")
        if not hasattr(plan, attr)
    ]
    if missing:
        raise TypeError(
            "expected an Equimo-like FineTunePlan with attributes "
            "trainable, frozen, labels, group_specs, and identities; "
            f"missing {missing!r}."
        )


def _require_combine(plan: Any) -> None:
    if not hasattr(plan, "combine"):
        raise TypeError(
            "expected an Equimo-like FineTunePlan with a combine method for "
            "plan-aware update steps."
        )


__all__ = (
    "adamw_from_equimo_plan",
    "from_equimo_plan",
    "make_equimo_update_step",
    "transition_lpft_stage",
)
