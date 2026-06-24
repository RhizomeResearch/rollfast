"""Factorized optimizer compiler tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import rollfast.finetune as rfft

from .helpers import tiny_plan


def _ones_like_trainable(tree):
    return jax.tree.map(
        lambda x: jnp.ones_like(x) if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
    )


def test_factorized_adamw_has_no_partition_state():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)

    assert not _contains_state_type(state, "PartitionState")
    assert _contains_state_type(state, "_FactorizedAdamWState")


def test_factorized_adamw_matches_manual_first_group_updates():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        base_lr=1e-3,
        weight_decay=0.1,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    grads = _ones_like_trainable(plan.trainable)
    updates, _ = bundle.update(grads, state, plan.trainable)
    groups = {group.source_label: group for group in bundle.report.groups}
    adam_unit = 1.0 / (1.0 + bundle.optimizer_config.eps)

    assert jnp.allclose(
        updates["blocks"][0]["w"],
        -groups["block_00_decay"].effective_lr
        * (adam_unit + 0.1 * plan.trainable["blocks"][0]["w"]),
    )
    assert jnp.allclose(
        updates["blocks"][0]["b"],
        -groups["block_00_no_decay"].effective_lr * adam_unit,
    )
    assert groups["head_decay"].effective_lr == pytest.approx(2e-3)


def _contains_state_type(state, name: str) -> bool:
    if type(state).__name__ == name:
        return True
    if isinstance(state, tuple):
        return any(_contains_state_type(item, name) for item in state)
    if hasattr(state, "_fields"):
        return any(
            _contains_state_type(getattr(state, field), name) for field in state._fields
        )
    if isinstance(state, dict):
        return any(_contains_state_type(item, name) for item in state.values())
    return False
