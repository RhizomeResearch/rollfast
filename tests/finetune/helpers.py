from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.tree_util as jtu
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
class TinyIdentity:
    logical_id: str
    module_id: str
    leaf_role: str
    physical_path: tuple[str | int, ...]
    tags: frozenset[str]
    depth: int | None = None
    alias_group: str | None = None
    layout: Any | None = None
    segment: Any | None = None


@dataclass(frozen=True)
class TinyPlan:
    trainable: Any
    labels: Any
    group_specs: dict[str, TinyGroup]
    frozen: Any = None
    param_info: Any = None
    identities: Any = None
    model_state: Any = None
    state_policy: Any = None
    aux_losses: tuple[Any, ...] = ()
    lineage: Any = None
    trainable_mask: Any = None
    report: Any = None

    def __post_init__(self) -> None:
        if self.frozen is None:
            object.__setattr__(self, "frozen", _none_tree_like(self.trainable))
        if self.identities is None:
            object.__setattr__(self, "identities", _identity_tree(self.trainable))
        if self.param_info is None:
            object.__setattr__(self, "param_info", self.identities)

    def combine(self, trainable=None):
        return self.trainable if trainable is None else trainable


def _is_none_leaf(value: Any) -> bool:
    return value is None


def _none_tree_like(tree: Any) -> Any:
    return jtu.tree_map(lambda _: None, tree, is_leaf=_is_none_leaf)


def _identity_tree(tree: Any) -> Any:
    return jtu.tree_map_with_path(
        _identity_for_leaf,
        tree,
        is_leaf=_is_none_leaf,
    )


def _identity_for_leaf(path: tuple[Any, ...], leaf: Any) -> TinyIdentity | None:
    if leaf is None:
        return None
    tokens = tuple(_path_token(entry) for entry in path)
    logical_id = ".".join(str(token) for token in tokens)
    return TinyIdentity(
        logical_id=logical_id,
        module_id=".".join(str(token) for token in tokens[:-1]),
        leaf_role=str(tokens[-1]) if tokens else "leaf",
        physical_path=tokens,
        tags=frozenset(),
    )


def _path_token(entry: Any) -> str | int:
    if hasattr(entry, "key"):
        return entry.key
    if hasattr(entry, "idx"):
        return entry.idx
    if hasattr(entry, "name"):
        return entry.name
    return str(entry)


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
            tags=("lora", "lora.factor_A"),
        ),
        "lora_B_decay": TinyGroup(
            "lora_B_decay",
            role="peft",
            depth=None,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("lora", "lora.factor_B"),
        ),
    }
    return TinyPlan(trainable=trainable, labels=labels, group_specs=groups)
