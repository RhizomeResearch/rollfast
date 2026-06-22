from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


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
    trainable_mask: Any = None
    report: Any = None

    def combine(self, trainable=None):
        return self.trainable if trainable is None else trainable


def tiny_plan() -> TinyPlan:
    trainable = {
        "embed": None,
        "blocks": (
            {
                "w": jnp.ones((2, 2), dtype=jnp.float32),
                "b": jnp.ones((2,), dtype=jnp.float32),
            },
            {
                "w": jnp.ones((2, 2), dtype=jnp.float32) * 2.0,
                "b": jnp.ones((2,), dtype=jnp.float32),
            },
        ),
        "head": {"w": jnp.ones((2, 1), dtype=jnp.float32)},
    }
    labels = {
        "embed": None,
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
            lr_multiplier=0.5,
            weight_decay=True,
            tags=("block",),
        ),
        "block_00_no_decay": TinyGroup(
            "block_00_no_decay",
            role="backbone",
            depth=0,
            lr_multiplier=0.5,
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


def tiny_lora_plan() -> TinyPlan:
    trainable = {
        "lora_A": jnp.ones((2, 4), dtype=jnp.float32),
        "lora_B": jnp.ones((4, 2), dtype=jnp.float32),
    }
    labels = {"lora_A": "lora_A_decay", "lora_B": "lora_B_decay"}
    groups = {
        "lora_A_decay": TinyGroup(
            "lora_A_decay",
            role="peft",
            depth=None,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("lora", "lora.A"),
        ),
        "lora_B_decay": TinyGroup(
            "lora_B_decay",
            role="peft",
            depth=None,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("lora", "lora.B"),
        ),
    }
    return TinyPlan(trainable=trainable, labels=labels, group_specs=groups)
