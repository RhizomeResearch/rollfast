from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
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


def ones_like_trainable(tree):
    return jax.tree.map(
        lambda x: jnp.ones_like(x) if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
    )


def zeros_like_trainable(tree):
    return jax.tree.map(
        lambda x: jnp.zeros_like(x) if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
    )


def assert_tree_allclose(left, right):
    for lhs, rhs in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_allclose(lhs, rhs)


def assert_rng_equal(left, right):
    for name in (
        "forward",
        "sam",
        "stochastic_rounding",
        "quantization",
        "controller",
    ):
        np.testing.assert_allclose(getattr(left, name), getattr(right, name))


def large_plan() -> TinyPlan:
    trainable = {
        "w": jnp.linspace(-1.0, 1.0, 8192, dtype=jnp.float32).reshape(128, 64),
        "embed": jnp.ones((4096,), dtype=jnp.float32) * 0.5,
        "bias": jnp.ones((4096,), dtype=jnp.float32),
    }
    labels = {
        "w": "large_decay",
        "embed": "embed_decay",
        "bias": "bias_no_decay",
    }
    groups = {
        "large_decay": TinyGroup(
            "large_decay",
            role="backbone",
            depth=0,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("block",),
        ),
        "embed_decay": TinyGroup(
            "embed_decay",
            role="embedding.patch",
            depth=None,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=(),
        ),
        "bias_no_decay": TinyGroup(
            "bias_no_decay",
            role="head",
            depth=None,
            lr_multiplier=1.0,
            weight_decay=False,
            tags=("bias",),
        ),
    }
    return TinyPlan(trainable=trainable, labels=labels, group_specs=groups)


def leaf_estimation_plan(*, mixed: bool) -> TinyPlan:
    trainable = {
        "large": jnp.ones((4097 if mixed else 2048,), dtype=jnp.float32),
        "small_a": jnp.ones((2048,), dtype=jnp.float32),
        "small_b": jnp.ones((33 if mixed else 2048,), dtype=jnp.float32),
    }
    labels = {name: "shared" for name in trainable}
    groups = {
        "shared": TinyGroup(
            "shared",
            role="backbone",
            depth=0,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("block",),
        )
    }
    if mixed:
        trainable["sensitive"] = jnp.ones((5000,), dtype=jnp.float32)
        labels["sensitive"] = "sensitive"
        groups["sensitive"] = TinyGroup(
            "sensitive",
            role="backbone",
            depth=0,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("bias",),
        )
    return TinyPlan(trainable=trainable, labels=labels, group_specs=groups)
