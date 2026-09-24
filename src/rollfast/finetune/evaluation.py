"""Checkpoint-bound parameter views for evaluation and export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from ._protocols import CombinableFineTunePlanProtocol
from .averaging import AveragingState
from .config import OptimizerBundle
from .serialization import OptimizerStateCheckpoint, restore_state_checkpoint


@dataclass(frozen=True)
class EvaluationView:
    """A selected model and the checkpoint position that produced it.

    ``model`` contains the selected trainable leaves recombined with the plan's
    frozen leaves. ``model_state`` is the matching snapshot stored in the
    optimizer checkpoint, if the model has mutable state.
    """

    model: Any
    model_state: Any | None
    trainable: Any
    view: str
    model_checkpoint_id: str
    optimizer_checkpoint_id: str
    optimizer_fingerprint: str
    applied_updates: int
    average_updates: int | None


def bind_evaluation_view(
    bundle: OptimizerBundle,
    plan: CombinableFineTunePlanProtocol,
    trainable: Any,
    checkpoint: OptimizerStateCheckpoint,
    *,
    view: str,
    model_checkpoint_id: str,
    optimizer_checkpoint_id: str,
    applied_updates: int,
) -> EvaluationView:
    """Select one parameter view from a matching optimizer/model checkpoint.

    The caller supplies the training parameters and plan loaded from
    ``model_checkpoint_id`` and the applied-update position recorded with
    that model. The checkpoint's own counters and averaging step, when present,
    must agree with that position.
    """

    if not optimizer_checkpoint_id:
        raise ValueError("optimizer_checkpoint_id is required for evaluation.")
    if isinstance(applied_updates, bool) or not isinstance(applied_updates, int):
        raise TypeError("applied_updates must be a non-negative integer.")
    if applied_updates < 0:
        raise ValueError("applied_updates must be a non-negative integer.")
    if view not in bundle.eval_views:
        raise ValueError(f"unsupported evaluation view: {view!r}.")
    if (
        getattr(plan, "model_state", None) is not None
        and checkpoint.model_state is None
    ):
        raise ValueError("checkpoint is missing the model-state snapshot.")

    state = restore_state_checkpoint(
        bundle, checkpoint, model_checkpoint_id=model_checkpoint_id
    )
    recorded_updates = checkpoint.manifest.get("counters", {}).get("successful_update")
    if recorded_updates is not None and recorded_updates != applied_updates:
        raise ValueError(
            "applied_updates disagrees with the checkpoint's successful_update counter."
        )
    averaging_state = state if isinstance(state, AveragingState) else None
    if bundle.ema_config.enabled or bundle.swa_config.enabled:
        if averaging_state is None:
            raise ValueError("checkpoint is missing enabled averaging state.")
        if int(jax.device_get(averaging_state.step)) != applied_updates:
            raise ValueError("applied_updates disagrees with the averaging step.")

    selected = bundle.eval_params(trainable, state, view=view)
    average_updates = None
    if view == "ema" and averaging_state is not None:
        average_updates = int(jax.device_get(averaging_state.ema_count))
    elif view == "swa" and averaging_state is not None:
        average_updates = int(jax.device_get(averaging_state.swa_count))

    return EvaluationView(
        model=plan.combine(selected),
        model_state=checkpoint.model_state,
        trainable=selected,
        view=view,
        model_checkpoint_id=model_checkpoint_id,
        optimizer_checkpoint_id=optimizer_checkpoint_id,
        optimizer_fingerprint=bundle.report.fingerprint,
        applied_updates=applied_updates,
        average_updates=average_updates,
    )


__all__ = ("EvaluationView", "bind_evaluation_view")
