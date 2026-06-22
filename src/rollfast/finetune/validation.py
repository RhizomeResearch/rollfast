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
    frozen: Any
    labels: Any
    identities: Any
    groups: dict[str, PlanGroup]
    fingerprint: str
    logical_id_table_hash: str
    model_state_structure_hash: str | None
    warnings: tuple[str, ...] = ()


def validate_plan(
    plan: FineTunePlanProtocol,
    *,
    allow_empty_groups: bool = False,
) -> NormalizedPlan:
    """Validate a structural fine-tuning plan and normalize its groups."""

    _require_attrs(plan, ("trainable", "frozen", "labels", "group_specs", "identities"))
    _check_treedef_alignment(plan.trainable, plan.frozen, "plan.trainable", "plan.frozen")
    _check_treedef_alignment(plan.trainable, plan.labels, "plan.trainable", "plan.labels")
    _check_treedef_alignment(
        plan.trainable,
        plan.identities,
        "plan.trainable",
        "plan.identities",
    )

    raw_groups = dict(plan.group_specs)

    labels_seen: dict[str, dict[str, int]] = {}
    warnings: list[str] = []
    _validate_group_specs(raw_groups)

    leaves = _tree_leaves(plan.trainable)
    frozen_leaves = _tree_leaves(plan.frozen)
    label_leaves = _tree_leaves(plan.labels)
    identity_leaves = _tree_leaves(plan.identities)
    if len(leaves) != len(label_leaves) or len(leaves) != len(frozen_leaves):
        raise ValueError(
            "plan.trainable, plan.frozen, and plan.labels must have the same number of leaves."
        )

    for leaf, frozen_leaf, label, identity in zip(
        leaves,
        frozen_leaves,
        label_leaves,
        identity_leaves,
        strict=True,
    ):
        if leaf is None:
            if label is not None:
                raise ValueError("frozen/absent leaves must not carry labels.")
            if frozen_leaf is None:
                continue
            continue

        if frozen_leaf is not None:
            raise ValueError("trainable and frozen leaves must not both be non-None.")
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
        if identity is None:
            raise ValueError("every trainable array leaf must carry a logical identity.")

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
    fingerprint = plan_fingerprint(
        plan.trainable,
        plan.labels,
        groups,
        identities=plan.identities,
        lineage=getattr(plan, "lineage", None),
    )
    model_state = getattr(plan, "model_state", None)
    return NormalizedPlan(
        trainable=plan.trainable,
        frozen=plan.frozen,
        labels=plan.labels,
        identities=plan.identities,
        groups=groups,
        fingerprint=fingerprint,
        logical_id_table_hash=_logical_id_table_hash(plan.trainable, plan.identities),
        model_state_structure_hash=_model_state_structure_hash(model_state),
        warnings=tuple(warnings),
    )


def plan_fingerprint(
    trainable: Any,
    labels: Any,
    groups: dict[str, PlanGroup] | None = None,
    *,
    identities: Any | None = None,
    lineage: Any | None = None,
) -> str:
    """Hash plan structure, leaf metadata, labels, and groups.

    Raw parameter values are intentionally excluded.
    """

    _check_treedef_alignment(trainable, labels, "trainable", "labels")
    if identities is not None:
        _check_treedef_alignment(trainable, identities, "trainable", "identities")
    leaf_records = []
    label_leaves = _tree_leaves(labels)
    identity_leaves = (
        [None] * len(label_leaves)
        if identities is None
        else _tree_leaves(identities)
    )
    for index, (path, leaf) in enumerate(_tree_leaves_with_path(trainable)):
        label = label_leaves[index]
        identity = identity_leaves[index]
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
                "logical_id": _identity_field(identity, "logical_id"),
                "alias_group": _identity_field(identity, "alias_group"),
                "layout": _identity_metadata_field(identity, "layout"),
                "sharding_fingerprint": _identity_metadata_field(
                    identity,
                    "sharding_fingerprint",
                ),
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
        "treedef": str(jtu.tree_structure(trainable, is_leaf=_is_none_leaf)),
        "leaves": leaf_records,
        "groups": group_records,
        "lineage": _lineage_record(lineage),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_attrs(obj: Any, attrs: tuple[str, ...]) -> None:
    missing = [attr for attr in attrs if not hasattr(obj, attr)]
    if missing:
        raise TypeError(f"fine-tuning plan is missing attributes: {missing!r}.")


def _check_treedef_alignment(left: Any, right: Any, left_name: str, right_name: str) -> None:
    left_def = jtu.tree_structure(left, is_leaf=_is_none_leaf)
    right_def = jtu.tree_structure(right, is_leaf=_is_none_leaf)
    if left_def != right_def:
        raise ValueError(
            f"{left_name} and {right_name} must have identical PyTree structure."
        )


def _logical_id_table_hash(trainable: Any, identities: Any) -> str:
    records = []
    identity_leaves = _tree_leaves(identities)
    for index, (path, leaf) in enumerate(_tree_leaves_with_path(trainable)):
        if leaf is None:
            continue
        identity = identity_leaves[index]
        records.append(
            {
                "path": _path_to_string(path),
                "logical_id": _identity_field(identity, "logical_id"),
                "alias_group": _identity_field(identity, "alias_group"),
                "layout": _identity_metadata_field(identity, "layout"),
            }
        )
    return _hash_json(records)


def _model_state_structure_hash(model_state: Any | None) -> str | None:
    if model_state is None:
        return None
    records = []
    for path, leaf in jtu.tree_leaves_with_path(model_state):
        if leaf is None:
            shape = None
            dtype = None
        elif hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            arr = jnp.asarray(leaf)
            shape = tuple(int(dim) for dim in arr.shape)
            dtype = arr.dtype.name
        else:
            shape = ()
            dtype = type(leaf).__name__
        records.append(
            {
                "path": _path_to_string(path),
                "shape": shape,
                "dtype": dtype,
            }
        )
    return _hash_json(
        {
            "treedef": str(jtu.tree_structure(model_state)),
            "leaves": records,
        }
    )


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_none_leaf(value: Any) -> bool:
    return value is None


def _tree_leaves(tree: Any) -> list[Any]:
    return jtu.tree_leaves(tree, is_leaf=_is_none_leaf)


def _tree_leaves_with_path(tree: Any) -> list[tuple[Any, Any]]:
    return jtu.tree_leaves_with_path(tree, is_leaf=_is_none_leaf)


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


def _identity_field(identity: Any, name: str) -> Any:
    if identity is None:
        return None
    return getattr(identity, name, None)


def _identity_metadata_field(identity: Any, name: str) -> Any:
    value = _identity_field(identity, name)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        return tuple(
            item if isinstance(item, str | int | float | bool | None) else str(item)
            for item in value
        )
    return str(value)


def _lineage_record(lineage: Any | None) -> dict[str, Any]:
    if lineage is None:
        return {}
    return {
        "base_checkpoint_id": getattr(lineage, "base_checkpoint_id", None),
        "base_checkpoint_hash": getattr(lineage, "base_checkpoint_hash", None),
        "identity_stability": getattr(lineage, "identity_stability", None),
        "model_revision": getattr(lineage, "model_revision", None),
    }


__all__ = ("NormalizedPlan", "plan_fingerprint", "validate_plan")
