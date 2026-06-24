"""Shared helpers for lightweight fine-tuning benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import platform
import time
from typing import Any

import jax
import jax.numpy as jnp

import rollfast
import rollfast.finetune as rfft


@dataclass(frozen=True)
class TinyGroup:
    label: str
    role: str
    depth: int | None
    lr_multiplier: float
    weight_decay: bool
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TinyPlan:
    trainable: Any
    labels: Any
    group_specs: dict[str, TinyGroup]

    def combine(self, trainable=None):
        return self.trainable if trainable is None else trainable


def tiny_plan() -> TinyPlan:
    trainable = {
        "blocks": (
            {
                "w": jnp.ones((16, 16), dtype=jnp.float32),
                "b": jnp.ones((16,), dtype=jnp.float32),
            },
            {
                "w": jnp.ones((16, 16), dtype=jnp.float32) * 0.5,
                "b": jnp.ones((16,), dtype=jnp.float32),
            },
        ),
        "head": {"w": jnp.ones((16, 4), dtype=jnp.float32)},
    }
    labels = {
        "blocks": (
            {"w": "block_00_decay", "b": "block_00_no_decay"},
            {"w": "block_01_decay", "b": "block_01_no_decay"},
        ),
        "head": {"w": "head_decay"},
    }
    groups = {
        "block_00_decay": TinyGroup(
            "block_00_decay",
            role="backbone",
            depth=0,
            lr_multiplier=0.7,
            weight_decay=True,
            tags=("block",),
        ),
        "block_00_no_decay": TinyGroup(
            "block_00_no_decay",
            role="backbone",
            depth=0,
            lr_multiplier=0.7,
            weight_decay=False,
            tags=("block", "bias"),
        ),
        "block_01_decay": TinyGroup(
            "block_01_decay",
            role="backbone",
            depth=1,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("block",),
        ),
        "block_01_no_decay": TinyGroup(
            "block_01_no_decay",
            role="backbone",
            depth=1,
            lr_multiplier=1.0,
            weight_decay=False,
            tags=("block", "bias"),
        ),
        "head_decay": TinyGroup(
            "head_decay",
            role="head",
            depth=None,
            lr_multiplier=2.0,
            weight_decay=True,
            tags=("head",),
        ),
    }
    return TinyPlan(trainable=trainable, labels=labels, group_specs=groups)


def large_plan() -> TinyPlan:
    trainable = {
        "w0": jnp.ones((4096, 64), dtype=jnp.float32),
        "w1": jnp.ones((4096, 64), dtype=jnp.float32) * 0.5,
        "bias": jnp.ones((1024,), dtype=jnp.float32),
    }
    labels = {"w0": "matrix_decay", "w1": "matrix_decay", "bias": "bias_no_decay"}
    groups = {
        "matrix_decay": TinyGroup(
            "matrix_decay",
            role="backbone",
            depth=0,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("matrix",),
        ),
        "bias_no_decay": TinyGroup(
            "bias_no_decay",
            role="bias",
            depth=None,
            lr_multiplier=1.0,
            weight_decay=False,
            tags=("bias",),
        ),
    }
    return TinyPlan(trainable=trainable, labels=labels, group_specs=groups)


def metadata(*, warmup_steps: int, measured_steps: int) -> dict[str, Any]:
    devices = jax.devices()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "jax": jax.__version__,
        "jaxlib": getattr(jax.lib, "__version__", "unknown"),
        "rollfast": getattr(rollfast, "__version__", "unknown"),
        "devices": [str(device) for device in devices],
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "metrics_scope": "model-and-optimizer-only",
    }


def tree_l2_loss(params: Any, target: float = 0.0) -> jax.Array:
    leaves = [
        leaf
        for leaf in jax.tree.leaves(params, is_leaf=lambda x: x is None)
        if leaf is not None
    ]
    return sum(jnp.mean((leaf - target) ** 2) for leaf in leaves)


def ones_like_trainable(tree: Any) -> Any:
    return jax.tree.map(
        lambda leaf: None if leaf is None else jnp.ones_like(leaf),
        tree,
        is_leaf=lambda x: x is None,
    )


def benchmark_step(step_fn, state, params, *, warmup_steps: int, measured_steps: int):
    for _ in range(warmup_steps):
        params, state, _ = step_fn(params, state)
    jax.block_until_ready(jax.tree.leaves(params)[0])

    started = time.perf_counter()
    for _ in range(measured_steps):
        params, state, value = step_fn(params, state)
    jax.block_until_ready(jax.tree.leaves(params)[0])
    elapsed = time.perf_counter() - started
    return params, state, value, elapsed / measured_steps


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


__all__ = (
    "benchmark_step",
    "emit",
    "large_plan",
    "metadata",
    "ones_like_trainable",
    "rfft",
    "tiny_plan",
    "tree_l2_loss",
)
