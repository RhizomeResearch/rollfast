import jax
import jax.numpy as jnp
import optax
import pytest
import equinox as eqx
from typing import Any

import rollfast.finetune as rfft

from .helpers import TinyGroup, TinyPlan, tiny_lora_plan, tiny_plan


class CallableTree(eqx.Module):
    w: Any

    def __call__(self, x):
        return x


def test_adamw_from_plan_reports_effective_group_lrs():
    bundle = rfft.adamw_from_plan(
        tiny_plan(),
        total_steps=20,
        base_lr=1e-3,
        weight_decay=0.05,
        schedule="constant",
        clip_global_norm=None,
    )
    groups = {group.source_label: group for group in bundle.report.groups}

    assert groups["block_00_decay"].effective_lr == pytest.approx(5e-4)
    assert groups["block_01_decay"].effective_lr == pytest.approx(1e-3)
    assert groups["head_decay"].effective_lr == pytest.approx(2e-3)
    assert bundle.report.trainable_params == 14
    assert bundle.report.estimated_state_bytes == 14 * 4 * 2


def test_adamw_from_plan_updates_no_decay_leaf_only_from_gradient():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=20,
        base_lr=1e-3,
        weight_decay=0.1,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    zeros = jax.tree.map(
        lambda x: jnp.zeros_like(x) if x is not None else None,
        plan.trainable,
        is_leaf=lambda x: x is None,
    )

    updates, _ = bundle.update(zeros, state, plan.trainable)

    assert jnp.allclose(updates["blocks"][0]["b"], 0.0)
    assert jnp.all(updates["blocks"][0]["w"] < 0.0)


def test_adamw_from_plan_has_no_array_state_for_frozen_absent_leaf():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)

    state_shapes = [
        tuple(leaf.shape)
        for leaf in jax.tree.leaves(state)
        if hasattr(leaf, "shape") and leaf.shape
    ]
    assert (3,) not in state_shapes


def test_lora_plus_ratio_is_visible_in_report():
    bundle = rfft.adamw_from_plan(
        tiny_lora_plan(),
        total_steps=20,
        base_lr=2e-4,
        weight_decay=0.0,
        schedule="constant",
        lora_b_lr_ratio=16.0,
    )
    groups = {group.source_label: group for group in bundle.report.groups}

    assert groups["lora_A_decay"].effective_lr == pytest.approx(2e-4)
    assert groups["lora_B_decay"].effective_lr == pytest.approx(3.2e-3)


def test_report_exposes_verbose_rule_breakdown_and_unmatched_rule_warning():
    bundle = rfft.adamw_from_plan(
        tiny_plan(),
        total_steps=20,
        base_lr=1e-3,
        weight_decay=0.05,
        schedule="constant",
        clip_global_norm=None,
        group_rules=(
            rfft.GroupRule(
                label="head_decay",
                lr_multiplier=2.0,
                name="head_rule",
            ),
            rfft.GroupRule(label="missing_decay", name="missing_rule"),
        ),
    )
    groups = {group.source_label: group for group in bundle.report.groups}
    default_rows = {row["label"]: row for row in bundle.report.group_table()}
    verbose_rows = {
        row["label"]: row for row in bundle.report.group_table(verbose=True)
    }

    assert groups["head_decay"].matched_rule_names == ("head_rule",)
    assert "matched_rules" not in default_rows["head_decay"]
    assert verbose_rows["head_decay"]["base_lr"] == pytest.approx(1e-3)
    assert verbose_rows["head_decay"]["plan_lr_multiplier"] == pytest.approx(2.0)
    assert verbose_rows["head_decay"]["rule_lr_multiplier"] == pytest.approx(2.0)
    assert verbose_rows["head_decay"]["matched_rules"] == ("head_rule",)
    assert bundle.report.policy_table() == bundle.report.group_table(verbose=True)
    assert "group rule 'missing_rule' matched no groups." in bundle.report.warnings


def test_lora_plus_ratio_controls_update_magnitude():
    plan = tiny_lora_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=20,
        base_lr=2e-4,
        weight_decay=0.0,
        schedule="constant",
        clip_global_norm=None,
        lora_b_lr_ratio=16.0,
        b1=0.0,
        b2=0.0,
        eps=0.0,
    )
    state = bundle.init(plan.trainable)
    grads = jax.tree.map(
        lambda x: jnp.ones_like(x) if x is not None else None,
        plan.trainable,
        is_leaf=lambda x: x is None,
    )

    updates, _ = bundle.update(grads, state, plan.trainable)

    a_update = jnp.mean(jnp.abs(updates["lora_A"]))
    b_update = jnp.mean(jnp.abs(updates["lora_B"]))
    assert b_update / a_update == pytest.approx(16.0)


def test_moment_dtype_controls_adam_state_precision():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
        moment_dtype=jnp.bfloat16,
    )
    state = bundle.init(plan.trainable)
    state_dtypes = {
        leaf.dtype
        for leaf in jax.tree.leaves(state)
        if hasattr(leaf, "dtype") and leaf.shape
    }

    assert jnp.dtype(jnp.bfloat16) in state_dtypes


def test_precision_warnings_report_fp16_without_loss_scale_or_master_params():
    plan = TinyPlan(
        trainable={"w": jnp.ones((2,), dtype=jnp.float16)},
        labels={"w": "w_decay"},
        group_specs={
            "w_decay": TinyGroup(
                "w_decay",
                role="head",
                depth=None,
                lr_multiplier=1.0,
                weight_decay=True,
            )
        },
    )
    bundle = rfft.compile_optimizer(
        plan,
        optimizer=rfft.OptimizerConfig(weight_decay=0.0),
        schedule=rfft.ScheduleConfig(kind="constant", total_steps=5),
        gradient_policy=rfft.GradientPolicy(clip_global_norm=None),
        precision=rfft.PrecisionConfig(
            master_params="never",
            loss_scale="none",
        ),
    )

    assert "fp16 without loss scaling." in bundle.report.warnings
    assert (
        "low-precision stored parameters without master parameters."
        in bundle.report.warnings
    )


def test_make_update_step_updates_trainable_tree():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)

    def loss_fn(params):
        return sum(
            jnp.sum(leaf**2) for leaf in jax.tree.leaves(params) if leaf is not None
        )

    step = rfft.make_update_step(loss_fn, bundle)
    updated, state, loss = step(plan.trainable, state)

    assert loss > 0.0
    assert state is not None
    assert not jnp.allclose(updated["head"]["w"], plan.trainable["head"]["w"])


def test_accumulation_wraps_optimizer_and_waits_for_full_microsteps():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
        accumulation_steps=2,
    )
    state = bundle.init(plan.trainable)
    grads = jax.tree.map(
        lambda x: jnp.ones_like(x) if x is not None else None,
        plan.trainable,
        is_leaf=lambda x: x is None,
    )

    updates_1, state = bundle.update(grads, state, plan.trainable)
    updates_2, _ = bundle.update(grads, state, plan.trainable)

    assert all(
        jnp.allclose(leaf, 0.0)
        for leaf in jax.tree.leaves(updates_1)
        if leaf is not None
    )
    assert any(
        not jnp.allclose(leaf, 0.0)
        for leaf in jax.tree.leaves(updates_2)
        if leaf is not None
    )


def test_bundle_manifest_is_serializable_shape_metadata():
    bundle = rfft.adamw_from_plan(
        tiny_plan(),
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
    )
    manifest = bundle.manifest()

    assert manifest["fingerprint"] == bundle.report.fingerprint
    assert manifest["optimizer"]["name"] == "adamw"
    assert manifest["groups"]


def test_bundle_updates_can_be_applied_with_optax():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    grads = jax.tree.map(
        lambda x: jnp.ones_like(x) if x is not None else None,
        plan.trainable,
        is_leaf=lambda x: x is None,
    )

    updates, state = bundle.update(grads, state, plan.trainable)
    updated = optax.apply_updates(plan.trainable, updates)

    assert state is not None
    assert updated["embed"] is None
    assert not jnp.allclose(updated["blocks"][1]["w"], plan.trainable["blocks"][1]["w"])


def test_callable_label_tree_is_treated_as_static_labels():
    plan = TinyPlan(
        trainable=CallableTree(jnp.ones((2,), dtype=jnp.float32)),
        labels=CallableTree("w_decay"),
        group_specs={
            "w_decay": TinyGroup(
                "w_decay",
                role="head",
                depth=None,
                lr_multiplier=1.0,
                weight_decay=True,
            )
        },
    )
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )

    state = bundle.init(plan.trainable)

    assert state is not None


def test_empty_all_frozen_plan_compiles_to_empty_report():
    plan = TinyPlan(trainable={"w": None}, labels={"w": None}, group_specs={})
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)

    assert bundle.report.trainable_params == 0
    assert bundle.report.groups == ()
    assert bundle.report.warnings == ("plan has no trainable array leaves.",)
    assert all(getattr(leaf, "shape", ()) == () for leaf in jax.tree.leaves(state))
