"""Plan validation, normalization, and structural fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu

from ._protocols import FineTunePlanProtocol, GroupSpecProtocol
from .config import PlanGroup


@dataclass(frozen=True)
class NormalizedPlan:
    """Validated plan data owned by Rollfast."""

    trainable: Any
    labels: Any
    groups: dict[str, PlanGroup]
    fingerprint: str
    warnings: tuple[str, ...] = ()


def validate_plan(
    plan: FineTunePlanProtocol,
    *,
    allow_empty_groups: bool = False,
) -> NormalizedPlan:
    """Validate a structural fine-tuning plan and normalize its groups."""

    _require_attrs(plan, ("trainable", "labels", "group_specs"))
    _check_treedef_alignment(plan.trainable, plan.labels)

    raw_groups = dict(plan.group_specs)

    labels_seen: dict[str, dict[str, int]] = {}
    warnings: list[str] = []
    _validate_group_specs(raw_groups)

    leaves = jtu.tree_leaves(plan.trainable)
    label_leaves = jtu.tree_leaves(plan.labels)
    if len(leaves) != len(label_leaves):
        raise ValueError(
            "plan.trainable and plan.labels must have the same number of leaves."
        )

    for leaf, label in zip(leaves, label_leaves, strict=True):
        if leaf is None:
            if label is not None:
                raise ValueError("frozen/absent leaves must not carry labels.")
            continue

        if not _is_inexact_array(leaf):
            raise ValueError(
                "fine-tuning optimizer leaves must be inexact JAX/NumPy arrays; "
                f"got {type(leaf).__name__}."
            )
        if label is None:
            raise ValueError("every trainable array leaf must have a non-None label.")
        if label == "frozen":
            raise ValueError("frozen leaves must be absent, not labeled 'frozen'.")
        try:
            hash(label)
        except TypeError as error:
            raise ValueError("plan labels must be hashable.") from error
        if not isinstance(label, str):
            raise ValueError("plan labels must be strings.")
        if label not in raw_groups:
            raise ValueError(f"label {label!r} is missing from plan.group_specs.")

        arr = jnp.asarray(leaf)
        stats = labels_seen.setdefault(label, {"params": 0, "bytes": 0, "leaves": 0})
        stats["params"] += int(arr.size)
        stats["bytes"] += int(arr.size * arr.dtype.itemsize)
        stats["leaves"] += 1

    if not labels_seen:
        warnings.append("plan has no trainable array leaves.")

    unused = set(raw_groups) - set(labels_seen)
    if unused and not allow_empty_groups:
        raise ValueError(f"unused group_specs labels: {tuple(sorted(unused))!r}.")

    groups = {
        label: _normalize_group(label, spec, labels_seen.get(label))
        for label, spec in sorted(raw_groups.items())
        if allow_empty_groups or label in labels_seen
    }
    fingerprint = plan_fingerprint(plan.trainable, plan.labels, groups)
    return NormalizedPlan(
        trainable=plan.trainable,
        labels=plan.labels,
        groups=groups,
        fingerprint=fingerprint,
        warnings=tuple(warnings),
    )


def plan_fingerprint(
    trainable: Any,
    labels: Any,
    groups: dict[str, PlanGroup] | None = None,
) -> str:
    """Hash plan structure, leaf metadata, labels, and groups.

    Raw parameter values are intentionally excluded.
    """

    _check_treedef_alignment(trainable, labels)
    leaf_records = []
    label_leaves = jtu.tree_leaves(labels)
    for index, (path, leaf) in enumerate(jtu.tree_leaves_with_path(trainable)):
        label = label_leaves[index]
        if leaf is None:
            shape: tuple[int, ...] | None = None
            dtype = None
        else:
            arr = jnp.asarray(leaf)
            shape = tuple(int(dim) for dim in arr.shape)
            dtype = arr.dtype.name
        leaf_records.append(
            {
                "path": _path_to_string(path),
                "shape": shape,
                "dtype": dtype,
                "label": label,
            }
        )

    group_records = []
    if groups is not None:
        group_records = [
            {
                "label": group.label,
                "role": group.role,
                "depth": group.depth,
                "lr_multiplier": group.lr_multiplier,
                "weight_decay": group.weight_decay,
                "tags": sorted(group.tags),
                "param_count": group.param_count,
                "byte_count": group.byte_count,
                "leaf_count": group.leaf_count,
            }
            for _, group in sorted(groups.items())
        ]

    payload = {
        "treedef": str(jtu.tree_structure(trainable)),
        "leaves": leaf_records,
        "groups": group_records,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_attrs(obj: Any, attrs: tuple[str, ...]) -> None:
    missing = [attr for attr in attrs if not hasattr(obj, attr)]
    if missing:
        raise TypeError(f"fine-tuning plan is missing attributes: {missing!r}.")


def _check_treedef_alignment(trainable: Any, labels: Any) -> None:
    trainable_def = jtu.tree_structure(trainable)
    label_def = jtu.tree_structure(labels)
    if trainable_def != label_def:
        raise ValueError(
            "plan.trainable and plan.labels must have identical PyTree structure."
        )


def _validate_group_specs(groups: dict[str, GroupSpecProtocol]) -> None:
    for key, group in groups.items():
        _require_attrs(
            group,
            ("label", "role", "depth", "lr_multiplier", "weight_decay", "tags"),
        )
        if group.label != key:
            raise ValueError(
                f"group_specs key {key!r} does not match group.label {group.label!r}."
            )
        if not math.isfinite(float(group.lr_multiplier)):
            raise ValueError(f"group {key!r} has a non-finite lr_multiplier.")
        if float(group.lr_multiplier) <= 0.0:
            raise ValueError(f"group {key!r} lr_multiplier must be positive.")
        if key.endswith("_no_decay") and bool(group.weight_decay):
            raise ValueError(f"group {key!r} says no_decay but enables weight decay.")
        if (
            key.endswith("_decay")
            and not key.endswith("_no_decay")
            and not bool(group.weight_decay)
        ):
            raise ValueError(f"group {key!r} says decay but disables weight decay.")


def _normalize_group(
    label: str,
    group: GroupSpecProtocol,
    stats: dict[str, int] | None,
) -> PlanGroup:
    stats = {"params": 0, "bytes": 0, "leaves": 0} if stats is None else stats
    return PlanGroup(
        label=label,
        role=str(group.role),
        depth=None if group.depth is None else int(group.depth),
        lr_multiplier=float(group.lr_multiplier),
        weight_decay=bool(group.weight_decay),
        tags=frozenset(str(tag) for tag in group.tags),
        param_count=stats["params"],
        byte_count=stats["bytes"],
        leaf_count=stats["leaves"],
    )


def _is_inexact_array(value: Any) -> bool:
    if not hasattr(value, "dtype") or not hasattr(value, "shape"):
        return False
    return bool(jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact))


def _path_to_string(path: tuple[Any, ...]) -> str:
    return ".".join(_key_to_string(key) for key in path)


def _key_to_string(key: Any) -> str:
    if isinstance(key, jtu.DictKey):
        return str(key.key)
    if isinstance(key, jtu.SequenceKey):
        return str(key.idx)
    if isinstance(key, jtu.GetAttrKey):
        return str(key.name)
    if isinstance(key, jtu.FlattenedIndexKey):
        return str(key.key)
    return str(key)


__all__ = ("NormalizedPlan", "plan_fingerprint", "validate_plan")
