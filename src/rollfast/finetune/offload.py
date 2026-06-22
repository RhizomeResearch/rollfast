"""Explicit optimizer-state host/device offload helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.tree_util as jtu

from .config import OptimizerBundle, StateOffloadPolicy


def with_state_offload(
    bundle: OptimizerBundle,
    policy: StateOffloadPolicy | None = None,
) -> OptimizerBundle:
    """Return ``bundle`` with an explicit optimizer-state offload policy."""

    resolved = StateOffloadPolicy(enabled=True) if policy is None else policy
    return replace(bundle, offload_policy=resolved)


def offload_optimizer_state(
    state: Any,
    *,
    policy: StateOffloadPolicy | None = None,
) -> Any:
    """Move eligible optimizer-state array leaves according to ``policy``."""

    resolved = StateOffloadPolicy() if policy is None else policy
    if not resolved.enabled:
        return state
    if resolved.target == "device":
        return jtu.tree_map(
            lambda leaf: _device_leaf(leaf, resolved),
            state,
        )
    return jtu.tree_map(
        lambda leaf: _host_leaf(leaf, resolved),
        state,
    )


def restore_offloaded_optimizer_state(
    state: Any,
    *,
    reference: Any | None = None,
    policy: StateOffloadPolicy | None = None,
) -> Any:
    """Move an offloaded optimizer state back to device arrays."""

    resolved = StateOffloadPolicy(enabled=True, target="device") if policy is None else policy
    if not resolved.enabled:
        return state
    if reference is None:
        return jtu.tree_map(
            lambda leaf: _device_leaf(leaf, resolved),
            state,
        )
    return jtu.tree_map(
        lambda leaf, ref: _device_leaf(leaf, resolved, reference=ref),
        state,
        reference,
    )


def state_offload_manifest(
    policy: StateOffloadPolicy,
    state: Any | None = None,
) -> dict[str, Any]:
    """Return serializable offload policy metadata and optional state counts."""

    manifest = policy.to_dict()
    if state is None:
        return manifest
    eligible_leaves = 0
    eligible_bytes = 0
    for leaf in jtu.tree_leaves(state):
        nbytes = _leaf_nbytes(leaf)
        if nbytes >= policy.min_bytes and _is_array_like(leaf):
            eligible_leaves += 1
            eligible_bytes += nbytes
    manifest.update(
        {
            "eligible_leaves": eligible_leaves,
            "eligible_bytes": eligible_bytes,
        }
    )
    return manifest


def _host_leaf(leaf: Any, policy: StateOffloadPolicy) -> Any:
    if not _eligible_leaf(leaf, policy):
        return leaf
    return jax.device_get(leaf)


def _device_leaf(
    leaf: Any,
    policy: StateOffloadPolicy,
    *,
    reference: Any | None = None,
) -> Any:
    if not _eligible_leaf(leaf, policy):
        return leaf
    sharding = getattr(reference, "sharding", None)
    if policy.preserve_sharding and sharding is not None:
        return jax.device_put(leaf, sharding)
    return jax.device_put(leaf)


def _eligible_leaf(leaf: Any, policy: StateOffloadPolicy) -> bool:
    return _is_array_like(leaf) and _leaf_nbytes(leaf) >= policy.min_bytes


def _is_array_like(leaf: Any) -> bool:
    return hasattr(leaf, "shape") and hasattr(leaf, "dtype")


def _leaf_nbytes(leaf: Any) -> int:
    nbytes = getattr(leaf, "nbytes", None)
    return int(nbytes) if nbytes is not None else 0


__all__ = (
    "offload_optimizer_state",
    "restore_offloaded_optimizer_state",
    "state_offload_manifest",
    "with_state_offload",
)
