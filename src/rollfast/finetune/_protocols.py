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
    """Structural fine-tuning plan protocol.

    Equimo's ``FineTunePlan`` satisfies this protocol, but Rollfast does not
    import Equimo to check it.
    """

    trainable: PyTree
    labels: PyTree
    group_specs: Mapping[str, GroupSpecProtocol]
    trainable_mask: PyTree
    report: Any

    def combine(self, trainable: PyTree | None = None) -> Any: ...


__all__ = ("FineTunePlanProtocol", "GroupSpecProtocol", "PyTree")
