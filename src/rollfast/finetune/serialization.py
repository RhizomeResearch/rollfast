"""Backend-neutral optimizer-state manifests and trusted-local pickle helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any, Mapping

import jax

from .config import OptimizerBundle, SCHEMA_VERSION

_CHECKPOINT_FORMAT = "rollfast.finetune.optimizer_state"
_COMPATIBILITY_FIELDS = (
    "optimizer_config",
    "schedule_config",
    "gradient_policy",
    "accumulation_config",
    "precision_config",
    "sharding_config",
    "offload_policy",
    "quantization_metadata",
    "method_config",
    "ema",
    "swa",
    "eval_params_kind",
    "eval_views",
    "default_eval_view",
)
_COMPATIBILITY_ALIASES = {
    "optimizer_config": "optimizer",
    "schedule_config": "schedule",
    "accumulation_config": "accumulation",
    "precision_config": "precision",
    "sharding_config": "sharding",
    "offload_policy": "state_offload",
    "quantization_metadata": "state_quantization",
}


@dataclass(frozen=True)
class OptimizerStateCheckpoint:
    """Logical optimizer-state checkpoint.

    The state PyTree is intentionally left backend-neutral. Users can store this
    object with their checkpointing system, or use the pickle helpers with trusted
    local files for tests and scripts.
    """

    manifest: Mapping[str, Any]
    state: Any
    master_params: Any | None = None
    accumulation: Any | None = None
    model_state: Any | None = None
    loss_scale: Any | None = None
    rng: Any | None = None
    counters: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format: str = _CHECKPOINT_FORMAT
    schema_version: int = SCHEMA_VERSION


class OptimizerStateRestoreError(ValueError):
    """Raised when an optimizer-state checkpoint cannot be restored safely."""


def state_manifest(bundle: OptimizerBundle) -> dict[str, Any]:
    """Return the serializable optimizer manifest for ``bundle``."""

    return bundle.manifest()


def make_state_checkpoint(
    bundle: OptimizerBundle,
    state: Any,
    *,
    model_checkpoint_id: str,
    master_params: Any | None = None,
    accumulation: Any | None = None,
    model_state: Any | None = None,
    loss_scale: Any | None = None,
    rng: Any | None = None,
    counters: Any | None = None,
    base_model_value_hash: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OptimizerStateCheckpoint:
    """Package a bundle manifest and optimizer state PyTree."""

    if not model_checkpoint_id:
        raise ValueError("model_checkpoint_id is required for optimizer checkpoints.")
    if bundle.precision_config.master_params == "always" and master_params is None:
        raise ValueError(
            "master_params are required when precision.master_params='always'."
        )
    if bundle.precision_config.loss_scale != "none" and loss_scale is None:
        raise ValueError(
            "loss_scale state is required when precision.loss_scale is enabled."
        )
    return OptimizerStateCheckpoint(
        manifest={
            **state_manifest(bundle),
            "model_checkpoint_id": model_checkpoint_id,
            "base_model_value_hash": base_model_value_hash,
            "counters": _counters_to_dict(counters),
            "components": {
                "optimizer_state": True,
                "master_params": master_params is not None,
                "accumulation": accumulation is not None,
                "model_state": model_state is not None,
                "loss_scale": loss_scale is not None,
                "rng": rng is not None,
                "counters": counters is not None,
                "schedule_free": _contains_state_type(state, "ScheduleFreeState"),
            },
        },
        state=state,
        master_params=master_params,
        accumulation=accumulation,
        model_state=model_state,
        loss_scale=loss_scale,
        rng=rng,
        counters=counters,
        metadata={} if metadata is None else dict(metadata),
    )


def restore_state_checkpoint(
    bundle: OptimizerBundle,
    checkpoint: OptimizerStateCheckpoint,
    *,
    model_checkpoint_id: str,
    strict: bool = True,
) -> Any:
    """Validate and return the state PyTree from ``checkpoint``."""

    if not model_checkpoint_id:
        raise ValueError(
            "model_checkpoint_id is required for optimizer checkpoint restore."
        )
    _validate_checkpoint_schema(checkpoint)
    if strict:
        expected = bundle.report.fingerprint
        actual = checkpoint.manifest.get("fingerprint")
        if actual != expected:
            raise OptimizerStateRestoreError(
                "optimizer-state checkpoint fingerprint mismatch: "
                f"expected {expected!r}, got {actual!r}."
            )
        checkpoint_model_id = checkpoint.manifest.get("model_checkpoint_id")
        if checkpoint_model_id != model_checkpoint_id:
            raise OptimizerStateRestoreError(
                "optimizer-state checkpoint model checkpoint mismatch: "
                f"expected {model_checkpoint_id!r}, got {checkpoint_model_id!r}."
            )
        _validate_manifest_compatibility(bundle, checkpoint.manifest)
        checkpoint_logical_ids = checkpoint.manifest.get("logical_id_table_hash")
        expected_logical_ids = bundle.report.logical_id_table_hash
        if checkpoint_logical_ids != expected_logical_ids:
            raise OptimizerStateRestoreError(
                "optimizer-state checkpoint logical-ID table mismatch: "
                f"expected {expected_logical_ids!r}, got {checkpoint_logical_ids!r}."
            )
        checkpoint_model_state_hash = checkpoint.manifest.get(
            "model_state_structure_hash"
        )
        expected_model_state_hash = bundle.report.model_state_structure_hash
        if checkpoint_model_state_hash != expected_model_state_hash:
            raise OptimizerStateRestoreError(
                "optimizer-state checkpoint model-state structure mismatch: "
                f"expected {expected_model_state_hash!r}, got {checkpoint_model_state_hash!r}."
            )
        if _requires_schedule_free_state(bundle) and not _contains_state_type(
            checkpoint.state,
            "ScheduleFreeState",
        ):
            raise OptimizerStateRestoreError(
                "checkpoint is missing required schedule-free optimizer state."
            )
    return checkpoint.state


def save_state_checkpoint(
    path: str | Path,
    bundle: OptimizerBundle,
    state: Any,
    *,
    model_checkpoint_id: str,
    master_params: Any | None = None,
    accumulation: Any | None = None,
    model_state: Any | None = None,
    loss_scale: Any | None = None,
    rng: Any | None = None,
    counters: Any | None = None,
    base_model_value_hash: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OptimizerStateCheckpoint:
    """Save a trusted-local pickle checkpoint and return the logical checkpoint."""

    checkpoint = make_state_checkpoint(
        bundle,
        state,
        model_checkpoint_id=model_checkpoint_id,
        master_params=master_params,
        accumulation=accumulation,
        model_state=model_state,
        loss_scale=loss_scale,
        rng=rng,
        counters=counters,
        base_model_value_hash=base_model_value_hash,
        metadata=metadata,
    )
    with Path(path).open("wb") as handle:
        pickle.dump(checkpoint, handle)
    return checkpoint


def load_state_checkpoint(
    path: str | Path,
    bundle: OptimizerBundle | None = None,
    *,
    trusted: bool = False,
    model_checkpoint_id: str | None = None,
    strict: bool = True,
) -> OptimizerStateCheckpoint | Any:
    """Load a pickle checkpoint after explicit trusted-local acknowledgement.

    If ``bundle`` is supplied, validate and return only the state PyTree.
    Otherwise return the logical checkpoint object.
    """

    if not trusted:
        raise OptimizerStateRestoreError(
            "pickle checkpoints may execute code; only load trusted local files "
            "by passing trusted=True."
        )
    with Path(path).open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, OptimizerStateCheckpoint):
        raise OptimizerStateRestoreError("file does not contain a Rollfast checkpoint.")
    if bundle is None:
        _validate_checkpoint_schema(checkpoint)
        return checkpoint
    if model_checkpoint_id is None:
        raise ValueError("model_checkpoint_id is required when restoring a checkpoint.")
    return restore_state_checkpoint(
        bundle,
        checkpoint,
        model_checkpoint_id=model_checkpoint_id,
        strict=strict,
    )


def _validate_checkpoint_schema(checkpoint: OptimizerStateCheckpoint) -> None:
    if checkpoint.format != _CHECKPOINT_FORMAT:
        raise OptimizerStateRestoreError(
            f"unsupported optimizer-state format: {checkpoint.format!r}."
        )
    if checkpoint.schema_version != SCHEMA_VERSION:
        raise OptimizerStateRestoreError(
            "unsupported optimizer-state schema version: "
            f"{checkpoint.schema_version!r}."
        )
    if checkpoint.manifest.get("schema_version") != SCHEMA_VERSION:
        raise OptimizerStateRestoreError("checkpoint manifest schema version mismatch.")
    if "fingerprint" not in checkpoint.manifest:
        raise OptimizerStateRestoreError("checkpoint manifest is missing fingerprint.")
    if "model_checkpoint_id" not in checkpoint.manifest:
        raise OptimizerStateRestoreError(
            "checkpoint manifest is missing model_checkpoint_id."
        )
    components = checkpoint.manifest.get("components", {})
    if checkpoint.manifest.get("precision", {}).get("master_params") == "always":
        if not components.get("master_params", False):
            raise OptimizerStateRestoreError(
                "checkpoint is missing required master parameters."
            )
    if checkpoint.manifest.get("precision", {}).get("loss_scale", "none") != "none":
        if not components.get("loss_scale", False):
            raise OptimizerStateRestoreError(
                "checkpoint is missing required loss-scale state."
            )
    if "schedule_free" in checkpoint.manifest.get("eval_views", ()):
        if not components.get("schedule_free", False):
            raise OptimizerStateRestoreError(
                "checkpoint manifest is missing schedule-free state metadata."
            )


def _validate_manifest_compatibility(
    bundle: OptimizerBundle,
    saved_manifest: Mapping[str, Any],
) -> None:
    expected_view = _compatibility_view(bundle.manifest())
    saved_view = _compatibility_view(saved_manifest)
    for key in _COMPATIBILITY_FIELDS:
        expected = expected_view[key]
        saved = saved_view[key]
        if saved == expected:
            continue
        if key == "sharding_config":
            mismatch = "sharding mismatch (sharding_config)"
        elif key == "quantization_metadata":
            mismatch = "quantization metadata mismatch (quantization_metadata)"
        else:
            mismatch = f"compatibility mismatch for {key!r}"
        raise OptimizerStateRestoreError(
            f"optimizer-state checkpoint {mismatch}: "
            f"expected {expected!r}, got {saved!r}."
        )


def _compatibility_view(manifest: Mapping[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {}
    for key in _COMPATIBILITY_FIELDS:
        if key in manifest:
            view[key] = manifest[key]
            continue
        alias = _COMPATIBILITY_ALIASES.get(key)
        if alias is not None and alias in manifest:
            view[key] = manifest[alias]
            continue
        raise OptimizerStateRestoreError(
            f"checkpoint manifest is missing compatibility field {key!r}."
        )
    return view


def _requires_schedule_free_state(bundle: OptimizerBundle) -> bool:
    return "schedule_free" in bundle.eval_views


def _counters_to_dict(counters: Any | None) -> dict[str, int]:
    if counters is None:
        return {}
    fields = getattr(counters, "_fields", None)
    if fields is not None:
        return {
            field: int(jax.device_get(getattr(counters, field))) for field in fields
        }
    if hasattr(counters, "__dataclass_fields__"):
        return {
            field: int(jax.device_get(getattr(counters, field)))
            for field in counters.__dataclass_fields__
        }
    if isinstance(counters, Mapping):
        return {str(key): int(jax.device_get(value)) for key, value in counters.items()}
    raise TypeError("counters must be a mapping, dataclass, NamedTuple, or None.")


def _contains_state_type(tree: Any, type_name: str) -> bool:
    return any(
        type(leaf).__name__ == type_name
        for leaf in jax.tree_util.tree_leaves(
            tree,
            is_leaf=lambda leaf: type(leaf).__name__ == type_name,
        )
    )


__all__ = (
    "OptimizerStateCheckpoint",
    "OptimizerStateRestoreError",
    "load_state_checkpoint",
    "make_state_checkpoint",
    "restore_state_checkpoint",
    "save_state_checkpoint",
    "state_manifest",
)
