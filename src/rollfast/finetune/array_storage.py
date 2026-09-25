"""Bounded, non-executable optimizer checkpoint storage on shared POSIX filesystems."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, NamedTuple, cast

import jax
import numpy as np
from jax.experimental import multihost_utils

from ._checkpoint_tree import (
    checkpoint_parts,
    describe_tree,
    match_tree_schema,
    restore_tree,
)
from .config import OptimizerBundle
from .serialization import (
    OptimizerStateCheckpoint,
    OptimizerStateRestoreError,
    _atomic_write,
    _sync_directory,
    _validate_checkpoint_schema,
    restore_state_checkpoint,
)

_FORMAT = "rollfast.finetune.array_state"
_VERSION = 1
_GENERATION = re.compile(r"[0-9a-f]{32}\Z")
_HEADER_LIMIT = 16_384


@dataclass(frozen=True)
class ArrayCheckpointLimits:
    """Maximum resources the array checkpoint reader may use."""

    max_manifest_bytes: int
    max_arrays: int
    max_shards: int
    max_array_bytes: int
    max_total_bytes: int

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


def save_array_state_checkpoint(
    path: str | Path,
    checkpoint: OptimizerStateCheckpoint,
    *,
    limits: ArrayCheckpointLimits,
) -> None:
    """Collectively write a safe checkpoint to a shared POSIX filesystem.

    All participating JAX processes must invoke this function in the same order.
    One checkpoint path has one writer at a time. A failed generation is never
    published, so a previous completed checkpoint remains readable. Each distinct
    shard of a sharded array is written once, by the process that holds its
    primary replica.
    """
    destination = Path(path)
    rank = jax.process_index()
    ranks = jax.process_count()
    identifier = uuid.uuid4().bytes if rank == 0 else bytes(16)
    identifier = bytes(
        np.asarray(
            multihost_utils.broadcast_one_to_all(
                np.frombuffer(identifier, dtype=np.uint8), is_source=rank == 0
            )
        )
    )
    generation = identifier.hex()
    generation_dir = _generation_root(destination) / generation
    error: Exception | None = None
    local_record: dict[str, Any] | None = None
    try:
        _validate_checkpoint_schema(checkpoint)
        schema, arrays = describe_tree(checkpoint_parts(checkpoint))
        array_nodes = _schema_array_nodes(schema)
        if "state_schema" not in checkpoint.manifest:
            raise ValueError("array storage requires a checkpoint state schema.")
        match_tree_schema(schema, checkpoint.manifest["state_schema"])
        if len(arrays) > limits.max_arrays:
            raise ValueError("checkpoint array count exceeds limit.")
        generation_dir.mkdir(parents=True, exist_ok=True)
        rank_dir = generation_dir / f"rank-{rank:05d}"
        rank_dir.mkdir(exist_ok=False)
        records = []
        total_bytes = 0
        shard_count = 0
        for index, array in enumerate(arrays):
            chunks = []
            shards = (
                # Replicas hold identical data; readers share the primary copy.
                [shard for shard in array.addressable_shards if shard.replica_id == 0]
                if _is_global_array(array, array_nodes[index])
                else (None,)
            )
            for number, shard in enumerate(shards):
                item = array if shard is None else shard.data
                logical_shape = list(item.shape)
                raw = _storage_bytes(item)
                size = int(raw.nbytes)
                if size > limits.max_array_bytes:
                    raise ValueError("checkpoint array exceeds per-array limit.")
                shard_count += 1
                if shard_count > limits.max_shards:
                    raise ValueError("checkpoint shard allocation exceeds limit.")
                filename = f"leaf-{index:06d}-shard-{number:06d}.npy"
                stored_size, sha256 = _write_shard(rank_dir / filename, raw)
                total_bytes += stored_size
                if total_bytes > limits.max_total_bytes:
                    raise ValueError("checkpoint shard allocation exceeds limit.")
                shard_index = (
                    _encode_index(shard.index, array.shape)
                    if shard is not None
                    else None
                )
                if shard_count == 1:
                    # Fail at save time on paths the reader would refuse.
                    _reject_symlinked_parents(rank_dir / filename)
                chunks.append(
                    {
                        "file": filename,
                        "shape": logical_shape,
                        "index": shard_index,
                        "size": stored_size,
                        "sha256": sha256,
                    }
                )
            records.append({"chunks": chunks})
        _sync_directory(rank_dir)
        local_record = {
            "schema": schema,
            "arrays": records,
            "manifest_sha256": _manifest_hash(checkpoint.manifest, limits),
            "metadata_sha256": _json_hash(checkpoint.metadata, limits),
        }
        _write_json(rank_dir / "record.json", local_record, limits.max_manifest_bytes)
        _sync_directory(rank_dir)
    except Exception as exc:  # noqa: BLE001 - every participant must reach the status exchange
        error = exc
    if not _all_succeeded(error is None):
        raise OptimizerStateRestoreError(
            f"checkpoint write failed on a participant: {error or 'another process failed'}."
        ) from error
    publish_error: Exception | None = None
    if rank == 0:
        try:
            assert local_record is not None
            records = [
                _read_json(
                    generation_dir / f"rank-{process:05d}" / "record.json",
                    limits.max_manifest_bytes,
                )
                for process in range(ranks)
            ]
            _validate_records(records, schema, limits)
            index = {
                "format": _FORMAT,
                "version": _VERSION,
                "process_count": ranks,
                "manifest": checkpoint.manifest,
                "metadata": checkpoint.metadata,
                "records": records,
            }
            index_sha256 = _write_json(
                generation_dir / "index.json", index, limits.max_manifest_bytes
            )
            _sync_directory(generation_dir)
            _sync_directory(_generation_root(destination))
            pointer = {
                "format": _FORMAT,
                "version": _VERSION,
                "generation": generation,
                "sha256": index_sha256,
            }
            _atomic_json(destination, pointer, limits.max_manifest_bytes)
        except Exception as exc:  # noqa: BLE001 - publish failures must reach every process
            publish_error = exc
    if not _all_succeeded(publish_error is None):
        raise OptimizerStateRestoreError(
            f"checkpoint publication failed: {publish_error or 'coordinator failed'}."
        ) from publish_error


def load_array_state_checkpoint(
    path: str | Path,
    bundle: OptimizerBundle,
    *,
    template: OptimizerStateCheckpoint,
    model_checkpoint_id: str,
    limits: ArrayCheckpointLimits,
    base_model_value_hash: str | None = None,
    applied_updates: int | None = None,
) -> OptimizerStateCheckpoint:
    """Load and strictly validate a checkpoint against a trusted local template.

    Distributed processes call this function together with the same path and
    checkpoint identity. A process may read shards written by another process.
    The restored arrays retain the template's sharding.
    """
    error: Exception | None = None
    prepared: _PreparedLoad | None = None
    try:
        prepared = _prepare_array_load(
            Path(path),
            bundle,
            template,
            model_checkpoint_id,
            limits,
            base_model_value_hash,
            applied_updates,
        )
    except Exception as exc:  # noqa: BLE001 - all processes must reach the status exchange
        error = exc
    if not _all_succeeded(error is None):
        raise OptimizerStateRestoreError(
            f"checkpoint read failed on a participant: {error or 'another process failed'}."
        ) from error
    assert prepared is not None
    if jax.process_count() > 1:
        generations = np.asarray(
            multihost_utils.process_allgather(
                np.frombuffer(bytes.fromhex(prepared.generation), dtype=np.uint8)
            )
        )
        if not np.all(generations == generations[0]):
            raise OptimizerStateRestoreError(
                "checkpoint generation changed across participating processes."
            )
    arrays = []
    for target, node, pieces in zip(
        prepared.targets, prepared.nodes, prepared.pieces, strict=True
    ):
        if _is_global_array(target, node):
            shards = [
                _place_piece(piece, shard.data, shard.device)
                for piece, shard in zip(pieces, target.addressable_shards, strict=True)
            ]
            arrays.append(
                jax.make_array_from_single_device_arrays(
                    target.shape, target.sharding, shards, dtype=target.dtype
                )
            )
        else:
            arrays.append(
                _place_piece(
                    pieces[0],
                    target,
                    target.device if isinstance(target, jax.Array) else None,
                )
            )
    parts = restore_tree(checkpoint_parts(template), prepared.schema, arrays)
    loaded = replace(
        template,
        manifest=prepared.index["manifest"],
        metadata=prepared.index["metadata"],
        **parts,
    )
    restore_state_checkpoint(
        bundle,
        loaded,
        model_checkpoint_id=model_checkpoint_id,
        template=template,
        base_model_value_hash=base_model_value_hash,
        applied_updates=applied_updates,
    )
    return loaded


class _PreparedLoad(NamedTuple):
    """Validated host data for one process, ready for device placement."""

    generation: str
    index: dict[str, Any]
    schema: dict[str, Any]
    targets: list[Any]
    nodes: list[dict[str, Any]]
    pieces: list[list[np.ndarray]]


def _prepare_array_load(
    destination: Path,
    bundle: OptimizerBundle,
    template: OptimizerStateCheckpoint,
    model_checkpoint_id: str,
    limits: ArrayCheckpointLimits,
    base_model_value_hash: str | None,
    applied_updates: int | None,
) -> _PreparedLoad:
    pointer = _read_json(destination, limits.max_manifest_bytes)
    if (
        pointer.get("format") != _FORMAT
        or pointer.get("version") != _VERSION
        or not isinstance(pointer.get("generation"), str)
        or not _GENERATION.fullmatch(pointer["generation"])
        or not isinstance(pointer.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", pointer["sha256"])
    ):
        raise OptimizerStateRestoreError("unsupported array checkpoint pointer.")
    generation_dir = _generation_root(destination) / pointer["generation"]
    index_file = generation_dir / "index.json"
    index = _read_json(
        index_file, limits.max_manifest_bytes, expected_sha256=pointer.get("sha256")
    )
    if index.get("format") != _FORMAT or index.get("version") != _VERSION:
        raise OptimizerStateRestoreError("unsupported array checkpoint schema version.")
    if index.get("process_count") != jax.process_count():
        raise OptimizerStateRestoreError("checkpoint process count mismatch.")
    records = index.get("records")
    expected_schema, expected_arrays = describe_tree(checkpoint_parts(template))
    expected_nodes = _schema_array_nodes(expected_schema)
    if not isinstance(records, list) or len(records) != jax.process_count():
        raise OptimizerStateRestoreError("checkpoint process records are incomplete.")
    try:
        _validate_records(records, expected_schema, limits)
        if records[0]["manifest_sha256"] != _manifest_hash(index["manifest"], limits):
            raise ValueError("checkpoint manifest differs from process records.")
        if records[0]["metadata_sha256"] != _json_hash(index["metadata"], limits):
            raise ValueError("checkpoint metadata differs from process records.")
        match_tree_schema(index["manifest"]["state_schema"], expected_schema)
    except (KeyError, TypeError, ValueError) as exc:
        raise OptimizerStateRestoreError(
            f"checkpoint state schema mismatch: {exc}"
        ) from exc
    restore_state_checkpoint(
        bundle,
        replace(template, manifest=index["manifest"], metadata=index["metadata"]),
        model_checkpoint_id=model_checkpoint_id,
        template=template,
        base_model_value_hash=base_model_value_hash,
        applied_updates=applied_updates,
    )
    rank = jax.process_index()
    prepared_pieces: list[list[np.ndarray]] = []
    for number, target in enumerate(expected_arrays):
        if _is_global_array(target, expected_nodes[number]):
            prepared_pieces.append(
                _load_global_pieces(generation_dir, records, number, target, limits)
            )
        else:
            chunks = records[rank]["arrays"][number]["chunks"]
            if len(chunks) != 1 or chunks[0].get("index") is not None:
                raise OptimizerStateRestoreError(
                    "checkpoint local array layout mismatch."
                )
            prepared_pieces.append(
                [_load_piece(generation_dir, rank, chunks[0], target, limits)]
            )
    return _PreparedLoad(
        generation=pointer["generation"],
        index=index,
        schema=records[rank]["schema"],
        targets=expected_arrays,
        nodes=expected_nodes,
        pieces=prepared_pieces,
    )


def _load_global_pieces(
    generation_dir: Path,
    records: list[dict[str, Any]],
    number: int,
    target: jax.Array,
    limits: ArrayCheckpointLimits,
) -> list[np.ndarray]:
    """Return one host piece per addressable shard of ``target``.

    Each distinct shard is stored once, possibly by another process, so chunks
    are matched by logical index; local replicas share one loaded piece.
    """
    # JSON text is a canonical, hashable key for untrusted index records.
    chunks: dict[str, tuple[int, dict[str, Any]]] = {}
    for owner, record in enumerate(records):
        for chunk in record["arrays"][number]["chunks"]:
            key = json.dumps(chunk.get("index"))
            if key in chunks:
                raise OptimizerStateRestoreError("checkpoint shard layout mismatch.")
            chunks[key] = (owner, chunk)
    loaded: dict[str, np.ndarray] = {}
    pieces = []
    for shard in target.addressable_shards:
        key = json.dumps(_encode_index(shard.index, target.shape))
        if key not in loaded:
            if key not in chunks:
                raise OptimizerStateRestoreError("checkpoint shard layout mismatch.")
            owner, chunk = chunks[key]
            loaded[key] = _load_piece(generation_dir, owner, chunk, shard.data, limits)
        pieces.append(loaded[key])
    return pieces


def _is_key_array(value: Any) -> bool:
    return isinstance(value, jax.Array) and jax.dtypes.issubdtype(
        value.dtype, jax.dtypes.prng_key
    )


def _is_global_array(value: Any, node: dict[str, Any]) -> bool:
    return isinstance(value, jax.Array) and node["scope"] == "global"


def _storage_bytes(array: Any) -> np.ndarray:
    if _is_key_array(array):
        array = jax.random.key_data(array)
    data = np.asarray(array)
    if data.dtype.hasobject:
        raise TypeError("object arrays cannot be stored in checkpoints.")
    return cast(np.ndarray, np.ascontiguousarray(data).view(np.uint8).reshape(-1))


def _load_piece(
    generation_dir: Path,
    rank: int,
    chunk: dict[str, Any],
    target: Any,
    limits: ArrayCheckpointLimits,
) -> Any:
    name = chunk.get("file")
    if not isinstance(name, str) or not re.fullmatch(
        r"leaf-\d{6}-shard-\d{6}\.npy", name
    ):
        raise OptimizerStateRestoreError("checkpoint shard filename is invalid.")
    path = generation_dir / f"rank-{rank:05d}" / name
    if chunk.get("shape") != list(target.shape):
        raise OptimizerStateRestoreError("checkpoint shard shape mismatch.")
    # Typed PRNG keys are stored as their raw key data.
    stored = jax.random.key_data(target) if _is_key_array(target) else target
    stored_dtype = np.dtype(stored.dtype)
    expected_bytes = stored.size * stored_dtype.itemsize
    if expected_bytes > limits.max_array_bytes:
        raise OptimizerStateRestoreError("checkpoint array exceeds per-array limit.")
    with _open_checkpoint_file(path) as handle:
        size = os.fstat(handle.fileno()).st_size
        if size != chunk.get("size") or size > limits.max_array_bytes + _HEADER_LIMIT:
            raise OptimizerStateRestoreError("checkpoint shard size mismatch.")
        # Read once so the checksummed bytes are exactly the decoded bytes.
        payload = handle.read(size + 1)
    if len(payload) != size:
        raise OptimizerStateRestoreError("checkpoint shard size mismatch.")
    if hashlib.sha256(payload).hexdigest() != chunk.get("sha256"):
        raise OptimizerStateRestoreError("checkpoint shard checksum mismatch.")
    stream = io.BytesIO(payload)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(
            stream, max_header_size=_HEADER_LIMIT
        )
    elif version in ((2, 0), (3, 0)):
        shape, _, dtype = np.lib.format.read_array_header_2_0(
            stream, max_header_size=_HEADER_LIMIT
        )
    else:
        raise OptimizerStateRestoreError("unsupported NPY version.")
    if dtype != np.dtype("uint8") or shape != (expected_bytes,):
        raise OptimizerStateRestoreError("checkpoint shard header mismatch.")
    if size != stream.tell() + expected_bytes:
        raise OptimizerStateRestoreError("checkpoint shard payload length mismatch.")
    data = np.frombuffer(
        payload, dtype=np.uint8, count=expected_bytes, offset=stream.tell()
    ).copy()
    return data.view(stored_dtype).reshape(stored.shape)


def _place_piece(data: np.ndarray, target: Any, device: Any | None) -> Any:
    if not isinstance(target, jax.Array):
        return data
    placed = jax.device_put(data, device)
    if _is_key_array(target):
        return jax.random.wrap_key_data(placed, impl=jax.random.key_impl(target))
    return placed


def _encode_index(
    index: tuple[slice | int, ...], shape: tuple[int, ...]
) -> list[list[int]]:
    output = []
    for item, dimension in zip(index, shape, strict=True):
        if isinstance(item, slice):
            if item.step not in (None, 1):
                raise ValueError("checkpoint shard slices require step 1.")
            output.append(
                [
                    int(item.start or 0),
                    int(dimension if item.stop is None else item.stop),
                ]
            )
        else:
            output.append([int(item), int(item) + 1])
    return output


def _generation_root(path: Path) -> Path:
    return path.parent / f".{path.name}.generations"


def _all_succeeded(success: bool) -> bool:
    if jax.process_count() == 1:
        return success
    states = multihost_utils.process_allgather(np.asarray(success, dtype=np.int8))
    return bool(np.all(np.asarray(states)))


def _validate_records(
    records: list[Any], schema: dict[str, Any], limits: ArrayCheckpointLimits
) -> None:
    expected_arrays = _schema_array_nodes(schema)
    if len(expected_arrays) > limits.max_arrays:
        raise ValueError("checkpoint array count exceeds limit.")
    total_bytes = 0
    total_shards = 0
    manifest_hash: str | None = None
    metadata_hash: str | None = None
    for record in records:
        if not isinstance(record, dict) or len(record.get("arrays", ())) != len(
            expected_arrays
        ):
            raise ValueError("checkpoint array record count mismatch.")
        match_tree_schema(record["schema"], schema)
        if manifest_hash is None:
            manifest_hash = record.get("manifest_sha256")
            metadata_hash = record.get("metadata_sha256")
        if (
            record.get("manifest_sha256") != manifest_hash
            or record.get("metadata_sha256") != metadata_hash
        ):
            raise ValueError("checkpoint process manifest mismatch.")
        for entry in record["arrays"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("chunks"), list):
                raise TypeError("checkpoint shard record is invalid.")
            for chunk in entry["chunks"]:
                if not isinstance(chunk, dict) or not isinstance(
                    chunk.get("size"), int
                ):
                    raise TypeError("checkpoint shard size is invalid.")
                if (
                    chunk["size"] < 0
                    or chunk["size"] > limits.max_array_bytes + _HEADER_LIMIT
                ):
                    raise ValueError("checkpoint shard exceeds per-array limit.")
                total_bytes += chunk["size"]
                total_shards += 1
                if (
                    total_bytes > limits.max_total_bytes
                    or total_shards > limits.max_shards
                ):
                    raise ValueError("checkpoint shard allocation exceeds limit.")


def _schema_array_nodes(schema: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        if node["kind"] == "array":
            nodes.append(node)
        for child in node.get("children", ()):
            visit(child)

    visit(schema)
    return nodes


def _json_bytes(value: Any, limit: int) -> bytes:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(encoded) > limit:
        raise ValueError("checkpoint manifest exceeds byte limit.")
    return encoded


def _json_hash(value: Any, limits: ArrayCheckpointLimits) -> str:
    return hashlib.sha256(_json_bytes(value, limits.max_manifest_bytes)).hexdigest()


def _manifest_hash(manifest: Any, limits: ArrayCheckpointLimits) -> str:
    normalized = dict(manifest)
    if "state_schema" in normalized:
        normalized["state_schema"] = _schema_without_scalar_values(
            normalized["state_schema"]
        )
    return _json_hash(normalized, limits)


def _schema_without_scalar_values(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            key: _schema_without_scalar_values(value)
            for key, value in schema.items()
            if key != "value"
        }
    if isinstance(schema, list):
        return [_schema_without_scalar_values(value) for value in schema]
    return schema


def _write_shard(path: Path, raw: np.ndarray) -> tuple[int, str]:
    """Durably create an NPY file for ``raw`` and return its size and SHA-256."""
    # Same bytes as ``np.save``; hashing while writing avoids reading the shard back.
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, np.lib.format.header_data_from_array_1_0(raw)
    )
    digest = hashlib.sha256()
    with path.open("xb") as handle:
        for block in (header.getvalue(), raw.data):
            handle.write(block)
            digest.update(block)
        handle.flush()
        os.fsync(handle.fileno())
        return handle.tell(), digest.hexdigest()


def _write_json(path: Path, value: Any, limit: int) -> str:
    """Durably create ``path`` with canonical JSON and return its SHA-256."""
    encoded = _json_bytes(value, limit)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Any, limit: int) -> None:
    with _atomic_write(path) as handle:
        handle.write(_json_bytes(value, limit))


def _read_json(
    path: Path, limit: int, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    with _open_checkpoint_file(path) as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > limit:
            raise OptimizerStateRestoreError("checkpoint manifest exceeds byte limit.")
        payload = handle.read(limit + 1)
    if len(payload) != size:
        raise OptimizerStateRestoreError("checkpoint manifest length mismatch.")
    if (
        expected_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise OptimizerStateRestoreError("checkpoint index checksum mismatch.")
    try:
        value = json.loads(payload, parse_constant=_reject_json_constant)
    except (UnicodeError, ValueError) as exc:
        raise OptimizerStateRestoreError(
            "checkpoint manifest is invalid JSON."
        ) from exc
    if not isinstance(value, dict):
        raise OptimizerStateRestoreError("checkpoint manifest must be an object.")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}.")


def _reject_symlinked_parents(path: Path) -> None:
    for parent in path.parents:
        if parent.exists() and parent.is_symlink():
            raise OptimizerStateRestoreError("checkpoint path traverses a symlink.")


@contextmanager
def _open_checkpoint_file(path: Path) -> Iterator[BinaryIO]:
    _reject_symlinked_parents(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OptimizerStateRestoreError("checkpoint entry is not a regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle
    finally:
        os.close(descriptor)


__all__ = (
    "ArrayCheckpointLimits",
    "load_array_state_checkpoint",
    "save_array_state_checkpoint",
)
