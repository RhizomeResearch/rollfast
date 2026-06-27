import jax.numpy as jnp
import pytest

import rollfast.finetune as rfft

from .helpers import TinyGroup, TinyPlan


def _head_backbone_plan() -> TinyPlan:
    trainable = {
        "block": {
            "w": jnp.ones((2, 2), dtype=jnp.float32),
            "b": jnp.ones((2,), dtype=jnp.float32),
        },
        "head": {
            "w": jnp.ones((2, 1), dtype=jnp.float32),
            "b": jnp.ones((1,), dtype=jnp.float32),
        },
    }
    labels = {
        "block": {"w": "block_decay", "b": "block_no_decay"},
        "head": {"w": "head_decay", "b": "head_no_decay"},
    }
    groups = {
        "block_decay": TinyGroup(
            "block_decay",
            role="block",
            depth=0,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("block",),
        ),
        "block_no_decay": TinyGroup(
            "block_no_decay",
            role="block",
            depth=0,
            lr_multiplier=1.0,
            weight_decay=False,
            tags=("block", "bias"),
        ),
        "head_decay": TinyGroup(
            "head_decay",
            role="head",
            depth=None,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("head",),
        ),
        "head_no_decay": TinyGroup(
            "head_no_decay",
            role="head",
            depth=None,
            lr_multiplier=1.0,
            weight_decay=False,
            tags=("head", "bias"),
        ),
    }
    return TinyPlan(trainable=trainable, labels=labels, group_specs=groups)


def test_head_backbone_adamw_recipe_compiles_expected_policy():
    bundle = rfft.compile_optimizer(
        _head_backbone_plan(),
        recipe=rfft.head_backbone_adamw(
            total_steps=10,
            backbone_lr=1e-5,
            head_lr=1e-3,
            backbone_weight_decay=0.05,
            head_weight_decay=0.01,
            no_decay_tags=("bias",),
            eps=1e-8,
        ),
        gradient_policy=rfft.GradientPolicy(clip_global_norm=None),
    )
    groups = {group.source_label: group for group in bundle.report.groups}

    assert groups["block_decay"].effective_lr == pytest.approx(1e-5)
    assert groups["block_decay"].weight_decay_value == pytest.approx(0.05)
    assert groups["block_no_decay"].weight_decay_value == 0.0
    assert groups["head_decay"].effective_lr == pytest.approx(1e-3)
    assert groups["head_decay"].weight_decay_value == pytest.approx(0.01)
    assert groups["head_no_decay"].effective_lr == pytest.approx(1e-3)
    assert groups["head_no_decay"].weight_decay_value == 0.0
    assert groups["head_no_decay"].matched_rule_names == (
        "head_backbone:head",
        "no_decay:bias",
    )
    assert bundle.optimizer_config.eps == pytest.approx(1e-8)


def test_discriminative_adamw_rules_return_direct_builder_inputs():
    base_lr, weight_decay, rules = rfft.discriminative_adamw_rules(
        backbone_lr=2e-5,
        head_lr=2e-3,
        backbone_weight_decay=0.02,
        head_weight_decay=0.01,
        no_decay_tags=("bias",),
    )
    bundle = rfft.adamw_from_plan(
        _head_backbone_plan(),
        total_steps=10,
        base_lr=base_lr,
        weight_decay=weight_decay,
        group_rules=rules,
        schedule="constant",
        clip_global_norm=None,
    )
    groups = {group.source_label: group for group in bundle.report.groups}

    assert groups["block_decay"].effective_lr == pytest.approx(2e-5)
    assert groups["head_decay"].effective_lr == pytest.approx(2e-3)
    assert groups["head_decay"].weight_decay_value == pytest.approx(0.01)
    assert groups["head_no_decay"].weight_decay_value == 0.0
