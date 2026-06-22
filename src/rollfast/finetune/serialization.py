"""Backend-neutral optimizer-state manifests and checkpoint helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any, Mapping

from .config import OptimizerBundle, SCHEMA_VERSION

_CHECKPOINT_FORMAT = "rollfast.finetune.optimizer_state"


@dataclass(frozen=True)
class OptimizerStateCheckpoint:
    """Logical optimizer-state checkpoint.

    The state PyTree is intentionally left backend-neutral. Users can store this
    object with their checkpointing system, or use the small pickle helpers for
    local tests and scripts.
    """

    manifest: Mapping[str, Any]
    state: Any
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
    metadata: Mapping[str, Any] | None = None,
) -> OptimizerStateCheckpoint:
    """Package a bundle manifest and optimizer state PyTree."""

    return OptimizerStateCheckpoint(
        manifest=state_manifest(bundle),
        state=state,
        metadata={} if metadata is None else dict(metadata),
    )


def restore_state_checkpoint(
    bundle: OptimizerBundle,
    checkpoint: OptimizerStateCheckpoint,
    *,
    strict: bool = True,
) -> Any:
    """Validate and return the state PyTree from ``checkpoint``."""

    _validate_checkpoint_schema(checkpoint)
    if strict:
        expected = bundle.report.fingerprint
        actual = checkpoint.manifest.get("fingerprint")
        if actual != expected:
            raise OptimizerStateRestoreError(
                "optimizer-state checkpoint fingerprint mismatch: "
                f"expected {expected!r}, got {actual!r}."
            )
    return checkpoint.state


def save_state_checkpoint(
    path: str | Path,
    bundle: OptimizerBundle,
    state: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> OptimizerStateCheckpoint:
    """Save a local pickle checkpoint and return the logical checkpoint."""

    checkpoint = make_state_checkpoint(bundle, state, metadata=metadata)
    with Path(path).open("wb") as handle:
        pickle.dump(checkpoint, handle)
    return checkpoint


def load_state_checkpoint(
    path: str | Path,
    bundle: OptimizerBundle | None = None,
    *,
    strict: bool = True,
) -> OptimizerStateCheckpoint | Any:
    """Load a local pickle checkpoint.

    If ``bundle`` is supplied, validate and return only the state PyTree.
    Otherwise return the logical checkpoint object.
    """

    with Path(path).open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, OptimizerStateCheckpoint):
        raise OptimizerStateRestoreError("file does not contain a Rollfast checkpoint.")
    if bundle is None:
        _validate_checkpoint_schema(checkpoint)
        return checkpoint
    return restore_state_checkpoint(bundle, checkpoint, strict=strict)


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


__all__ = (
    "OptimizerStateCheckpoint",
    "OptimizerStateRestoreError",
    "load_state_checkpoint",
    "make_state_checkpoint",
    "restore_state_checkpoint",
    "save_state_checkpoint",
    "state_manifest",
)
