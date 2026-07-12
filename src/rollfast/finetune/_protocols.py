"""Structural protocols accepted by ``rollfast.finetune``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

PyTree = Any


@runtime_checkable
class GroupSpecProtocol(Protocol):
    """Minimum group metadata Rollfast consumes from a model library."""

    label: str
    role: str
    depth: int | None
    lr_multiplier: float
    weight_decay: bool
    tags: tuple[str, ...]


@runtime_checkable
class FineTunePlanProtocol(Protocol):
    """Minimum structural protocol for compiling a fine-tuning plan.

    Equimo's ``FineTunePlan`` satisfies this protocol, but Rollfast does not
    import Equimo to check it.
    """

    trainable: PyTree
    frozen: PyTree
    labels: PyTree
    group_specs: Mapping[str, GroupSpecProtocol]
    identities: PyTree


@runtime_checkable
class CombinableFineTunePlanProtocol(FineTunePlanProtocol, Protocol):
    """Fine-tuning plan accepted by plan-aware update helpers."""

    def combine(self, trainable: PyTree) -> Any: ...


__all__ = (
    "CombinableFineTunePlanProtocol",
    "FineTunePlanProtocol",
    "GroupSpecProtocol",
    "PyTree",
)
