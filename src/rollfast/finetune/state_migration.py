"""Conservative optimizer-state migration for staged fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import jax

from rollfast.optim.adam8 import QuantizedBlocks, tree_state_nbytes

from ._protocols import FineTunePlanProtocol
from .builders import compile_optimizer
from .config import OptimizerBundle, SCHEMA_VERSION

StatePolicy = Literal[
    "reset_all",
    "preserve_shared",
    "preserve_by_path_and_shape",
    "preserve_exact_group",
]
CounterPolicy = Literal[
    "restart_schedule",
    "continue_global_step",
    "continue_optimizer_step_with_new_schedule",
]


@dataclass(frozen=True)
class OptimizerMigrationReport:
    """Static report describing a conservative optimizer-state migration."""

    schema_version: int
    state_policy: StatePolicy
    counter_policy: CounterPolicy
    old_fingerprint: str
    new_fingerprint: str
    preserved_state_leaves: tuple[str, ...]
    initialized_state_leaves: tuple[str, ...]
    dropped_state_leaves: tuple[str, ...]
    incompatible_state_leaves: tuple[str, ...]
    preserved_param_leaves: tuple[str, ...]
    initialized_param_leaves: tuple[str, ...]
    dropped_param_leaves: tuple[str, ...]
    changed_group_leaves: tuple[str, ...]
    old_state_bytes: int
    new_state_bytes: int
    schedule_counter_behavior: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_policy": self.state_policy,
            "counter_policy": self.counter_policy,
            "old_fingerprint": self.old_fingerprint,
            "new_fingerprint": self.new_fingerprint,
            "preserved_state_leaves": self.preserved_state_leaves,
            "initialized_state_leaves": self.initialized_state_leaves,
            "dropped_state_leaves": self.dropped_state_leaves,
            "incompatible_state_leaves": self.incompatible_state_leaves,
            "preserved_param_leaves": self.preserved_param_leaves,
            "initialized_param_leaves": self.initialized_param_leaves,
            "dropped_param_leaves": self.dropped_param_leaves,
            "changed_group_leaves": self.changed_group_leaves,
            "old_state_bytes": self.old_state_bytes,
            "new_state_bytes": self.new_state_bytes,
            "schedule_counter_behavior": self.schedule_counter_behavior,
        }


def reconfigure_optimizer(
    *,
    old_plan: FineTunePlanProtocol,
    old_bundle: OptimizerBundle,
    old_state: Any,
    new_plan: FineTunePlanProtocol,
    new_recipe: Any | None = None,
    new_bundle: OptimizerBundle | None = None,
    state_policy: StatePolicy = "preserve_shared",
    counter_policy: CounterPolicy = "restart_schedule",
    strict: bool = True,
    **compile_kwargs: Any,
) -> tuple[OptimizerBundle, Any, OptimizerMigrationReport]:
    """Compile a new bundle and migrate compatible optimizer state.

    The default policy preserves compatible shared moment leaves by parameter
    path and initializes newly trainable leaves. Schedule/count leaves are reset
    unless ``counter_policy='continue_global_step'`` is selected.
    """

    _validate_policy(state_policy, counter_policy)
    if new_bundle is None:
        new_bundle = compile_optimizer(new_plan, recipe=new_recipe, **compile_kwargs)
    new_state = new_bundle.init(new_plan.trainable)
    old_params = _param_records(old_plan)
    new_params = _param_records(new_plan)
    param_report = _param_report(old_params, new_params)
    if strict and param_report["incompatible"]:
        paths = ", ".join(param_report["incompatible"])
        raise ValueError(f"incompatible shared parameter leaves: {paths}")

    if state_policy == "reset_all":
        migrated_state = new_state
        preserved_state: tuple[str, ...] = ()
        initialized_state = _array_state_paths(new_state)
        incompatible_state: tuple[str, ...] = ()
    else:
        migrated_state, preserved_state, initialized_state, incompatible_state = (
            _migrate_state_tree(
                old_state,
                new_state,
                state_policy=state_policy,
                counter_policy=counter_policy,
            )
        )
        if strict and incompatible_state:
            paths = ", ".join(incompatible_state)
            raise ValueError(f"incompatible shared optimizer-state leaves: {paths}")

    old_state_paths = set(_array_state_paths(old_state))
    new_state_paths = set(_array_state_paths(new_state))
    dropped_state = tuple(sorted(old_state_paths - new_state_paths))
    report = OptimizerMigrationReport(
        schema_version=SCHEMA_VERSION,
        state_policy=state_policy,
        counter_policy=counter_policy,
        old_fingerprint=old_bundle.report.fingerprint,
        new_fingerprint=new_bundle.report.fingerprint,
        preserved_state_leaves=preserved_state,
        initialized_state_leaves=initialized_state,
        dropped_state_leaves=dropped_state,
        incompatible_state_leaves=incompatible_state,
        preserved_param_leaves=tuple(param_report["preserved"]),
        initialized_param_leaves=tuple(param_report["initialized"]),
        dropped_param_leaves=tuple(param_report["dropped"]),
        changed_group_leaves=tuple(param_report["changed_group"]),
        old_state_bytes=tree_state_nbytes(old_state),
        new_state_bytes=tree_state_nbytes(migrated_state),
        schedule_counter_behavior=_counter_behavior(counter_policy),
    )
    return new_bundle, migrated_state, report


def _migrate_state_tree(
    old_state: Any,
    new_state: Any,
    *,
    state_policy: StatePolicy,
    counter_policy: CounterPolicy,
) -> tuple[Any, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    old_exact = {
        _path_tokens(path): leaf
        for path, leaf in _path_leaves(old_state)
        if _state_leaf_has_storage(leaf)
    }
    old_moments = _moment_leaves_by_param(old_state)
    old_structured = _structured_leaves_by_param(old_state)
    new_path_leaves = _path_leaves(new_state)
    new_leaves = []
    preserved: list[str] = []
    initialized: list[str] = []
    incompatible: list[str] = []

    for path, leaf in new_path_leaves:
        tokens = _path_tokens(path)
        path_text = _format_tokens(tokens)
        replacement = None
        if _state_leaf_has_storage(leaf):
            candidate = _migration_candidate(
                tokens,
                old_exact,
                old_moments,
                old_structured,
                state_policy=state_policy,
                counter_policy=counter_policy,
            )
            if candidate is not None:
                if _compatible_leaf(candidate, leaf):
                    replacement = candidate
                    preserved.append(path_text)
                else:
                    incompatible.append(path_text)
            if replacement is None:
                initialized.append(path_text)
                replacement = leaf
        else:
            replacement = leaf
        new_leaves.append(replacement)

    treedef = jax.tree_util.tree_structure(new_state, is_leaf=_is_state_leaf)
    migrated = jax.tree_util.tree_unflatten(treedef, new_leaves)
    return (
        migrated,
        tuple(sorted(preserved)),
        tuple(sorted(initialized)),
        tuple(sorted(incompatible)),
    )


def _migration_candidate(
    tokens: tuple[str, ...],
    old_exact: Mapping[tuple[str, ...], Any],
    old_moments: Mapping[tuple[str, tuple[str, ...]], Any],
    old_structured: Mapping[tuple[str, tuple[str, ...]], Any],
    *,
    state_policy: StatePolicy,
    counter_policy: CounterPolicy,
) -> Any | None:
    if _is_counter_path(tokens):
        if counter_policy == "continue_global_step":
            return old_exact.get(tokens)
        return None
    if state_policy == "preserve_exact_group":
        return old_exact.get(tokens)
    moment_key = _moment_key(tokens)
    if moment_key is not None:
        return old_moments.get(moment_key)
    structured_key = _structured_key(tokens)
    if structured_key is not None:
        return old_structured.get(structured_key)
    if state_policy == "preserve_by_path_and_shape":
        return old_exact.get(tokens)
    return old_exact.get(tokens)


def _moment_leaves_by_param(state: Any) -> dict[tuple[str, tuple[str, ...]], Any]:
    moments = {}
    for path, leaf in _path_leaves(state):
        if not _state_leaf_has_storage(leaf):
            continue
        key = _moment_key(_path_tokens(path))
        if key is not None:
            moments[key] = leaf
    return moments


def _structured_leaves_by_param(state: Any) -> dict[tuple[str, tuple[str, ...]], Any]:
    structured = {}
    for path, leaf in _path_leaves(state):
        if not _state_leaf_has_storage(leaf):
            continue
        key = _structured_key(_path_tokens(path))
        if key is not None:
            structured[key] = leaf
    return structured


def _moment_key(tokens: tuple[str, ...]) -> tuple[str, tuple[str, ...]] | None:
    for index, token in enumerate(tokens):
        if token in ("attr:mu", "attr:nu"):
            return token, tokens[index + 1 :]
    return None


def _structured_key(tokens: tuple[str, ...]) -> tuple[str, tuple[str, ...]] | None:
    for index, token in enumerate(tokens):
        if token in ("attr:Qs_preconditioners", "attr:Ls_lipschitz"):
            return token, tokens[index + 1 :]
    return None


def _param_records(plan: FineTunePlanProtocol) -> dict[tuple[str, ...], dict[str, Any]]:
    trainable_paths = jax.tree_util.tree_flatten_with_path(
        plan.trainable,
        is_leaf=lambda x: x is None,
    )[0]
    labels_by_path = {
        _path_tokens(path): label
        for path, label in jax.tree_util.tree_flatten_with_path(
            plan.labels,
            is_leaf=lambda x: x is None,
        )[0]
    }
    identities_by_path = {
        _path_tokens(path): identity
        for path, identity in jax.tree_util.tree_flatten_with_path(
            plan.identities,
            is_leaf=lambda x: x is None,
        )[0]
    }
    records: dict[tuple[str, ...], dict[str, Any]] = {}
    for path, leaf in trainable_paths:
        if leaf is None:
            continue
        physical = _path_tokens(path)
        identity = identities_by_path.get(physical)
        logical_id = getattr(identity, "logical_id", None)
        key = ("logical", str(logical_id)) if logical_id else physical
        records[key] = {
            "shape": tuple(leaf.shape),
            "dtype": leaf.dtype,
            "label": labels_by_path.get(physical),
            "physical_path": physical,
        }
    return records


def _param_report(
    old: Mapping[tuple[str, ...], Mapping[str, Any]],
    new: Mapping[tuple[str, ...], Mapping[str, Any]],
) -> dict[str, list[str]]:
    old_paths = set(old)
    new_paths = set(new)
    shared = sorted(old_paths & new_paths)
    report = {
        "preserved": [],
        "initialized": [_format_tokens(path) for path in sorted(new_paths - old_paths)],
        "dropped": [_format_tokens(path) for path in sorted(old_paths - new_paths)],
        "changed_group": [],
        "incompatible": [],
    }
    for path in shared:
        old_record = old[path]
        new_record = new[path]
        if (
            old_record["shape"] == new_record["shape"]
            and old_record["dtype"] == new_record["dtype"]
        ):
            report["preserved"].append(_format_tokens(path))
        else:
            report["incompatible"].append(_format_tokens(path))
        if old_record["label"] != new_record["label"]:
            report["changed_group"].append(_format_tokens(path))
    return report


def _path_leaves(tree: Any) -> list[tuple[Any, Any]]:
    return jax.tree_util.tree_flatten_with_path(tree, is_leaf=_is_state_leaf)[0]


def _array_state_paths(tree: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            _format_tokens(_path_tokens(path))
            for path, leaf in _path_leaves(tree)
            if _state_leaf_has_storage(leaf)
        )
    )


def _path_tokens(path: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(_path_token(part) for part in path)


def _path_token(part: Any) -> str:
    if hasattr(part, "name"):
        return f"attr:{part.name}"
    if hasattr(part, "key"):
        return f"key:{part.key}"
    if hasattr(part, "idx"):
        return f"idx:{part.idx}"
    return repr(part)


def _format_tokens(tokens: tuple[str, ...]) -> str:
    return "/".join(tokens)


def _state_leaf_has_storage(leaf: Any) -> bool:
    return isinstance(leaf, QuantizedBlocks) or (
        hasattr(leaf, "shape") and hasattr(leaf, "dtype")
    )


def _is_state_leaf(leaf: Any) -> bool:
    return leaf is None or isinstance(leaf, QuantizedBlocks) or _is_masked_node(leaf)


def _is_masked_node(leaf: Any) -> bool:
    return leaf.__class__.__name__ == "MaskedNode"


def _compatible_leaf(old: Any, new: Any) -> bool:
    if isinstance(old, QuantizedBlocks) or isinstance(new, QuantizedBlocks):
        return (
            isinstance(old, QuantizedBlocks)
            and isinstance(new, QuantizedBlocks)
            and old.shape == new.shape
            and old.block_size == new.block_size
            and old.values.dtype == new.values.dtype
            and old.scales.dtype == new.scales.dtype
        )
    return (
        hasattr(old, "shape")
        and hasattr(new, "shape")
        and tuple(old.shape) == tuple(new.shape)
        and old.dtype == new.dtype
    )


def _is_counter_path(tokens: tuple[str, ...]) -> bool:
    return tokens and tokens[-1] in {
        "attr:count",
        "attr:notfinite_count",
        "attr:total_notfinite",
    }


def _counter_behavior(counter_policy: CounterPolicy) -> str:
    if counter_policy == "restart_schedule":
        return "initialized count leaves; stage schedule restarts"
    if counter_policy == "continue_global_step":
        return "preserved compatible count leaves"
    return "preserved moment state; schedule count leaves remain stage-local"


def _validate_policy(state_policy: str, counter_policy: str) -> None:
    if state_policy not in {
        "reset_all",
        "preserve_shared",
        "preserve_by_path_and_shape",
        "preserve_exact_group",
    }:
        raise ValueError(f"unknown state_policy: {state_policy!r}.")
    if counter_policy not in {
        "restart_schedule",
        "continue_global_step",
        "continue_optimizer_step_with_new_schedule",
    }:
        raise ValueError(f"unknown counter_policy: {counter_policy!r}.")


__all__ = (
    "CounterPolicy",
    "OptimizerMigrationReport",
    "StatePolicy",
    "reconfigure_optimizer",
)
