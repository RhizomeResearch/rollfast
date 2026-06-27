import jax
import jax.numpy as jnp
import optax
import pytest

import rollfast.finetune as rfft

from .helpers import tiny_lora_plan, tiny_plan


def _ones_like_trainable(tree):
    return jax.tree.map(
        lambda x: jnp.ones_like(x) if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
    )


def _zeros_like_trainable(tree):
    return jax.tree.map(
        lambda x: jnp.zeros_like(x) if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
    )


def test_schedule_free_adam_from_plan_reports_grouped_lrs():
    bundle = rfft.schedule_free_adam_from_plan(
        tiny_plan(),
        total_steps=20,
        base_lr=1e-3,
        schedule="wsd",
        weight_decay=0.05,
        clip_global_norm=None,
    )
    groups = {group.source_label: group for group in bundle.report.groups}

    assert bundle.eval_params_kind == "schedule_free"
    assert groups["block_00_decay"].optimizer == "schedule_free_adam"
    assert groups["block_00_decay"].effective_lr == pytest.approx(5e-4)
    assert groups["head_decay"].effective_lr == pytest.approx(2e-3)
    assert groups["block_00_no_decay"].weight_decay_value == 0.0
    method_config = bundle.manifest()["method_config"]
    assert method_config["method"] == "schedule_free_adam"
    assert method_config["profile_fidelity"] == "safe_default"
    assert method_config["paper_profile"] is False
    assert method_config["known_deviations"] == ("uses_external_wsd_schedule",)
    assert method_config["config"]["external_schedule"] == "wsd"


def test_schedule_free_adamc_rejects_multiple_weight_decay_values():
    with pytest.raises(ValueError, match="Schedule-Free AdamC"):
        rfft.schedule_free_adam_from_plan(
            tiny_plan(),
            total_steps=20,
            schedule="wsd",
            schedule_free_plus=True,
            group_rules=(
                rfft.GroupRule(role="backbone", weight_decay_value=0.05),
                rfft.GroupRule(role="head", weight_decay_value=0.01),
            ),
        )


def test_schedule_free_adam_updates_and_returns_eval_params():
    plan = tiny_plan()
    bundle = rfft.schedule_free_adam_from_plan(
        plan,
        total_steps=20,
        schedule="wsd",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(
        _ones_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )
    params = optax.apply_updates(plan.trainable, updates)
    updates, state = bundle.update(
        _ones_like_trainable(params),
        state,
        params,
    )
    params = optax.apply_updates(params, updates)
    eval_params = bundle.eval_params(params, state)

    assert eval_params["embed"] is None
    assert eval_params["head"]["w"].shape == plan.trainable["head"]["w"].shape
    assert not jnp.allclose(eval_params["head"]["w"], params["head"]["w"])


def test_schedule_free_eval_params_unwraps_accumulation_and_finite_guard_state():
    plan = tiny_plan()
    bundle = rfft.schedule_free_adam_from_plan(
        plan,
        total_steps=20,
        schedule="wsd",
        clip_global_norm=None,
        accumulation_steps=2,
    )
    state = bundle.init(plan.trainable)
    grads = _ones_like_trainable(plan.trainable)
    updates, state = bundle.update(grads, state, plan.trainable)
    params = optax.apply_updates(plan.trainable, updates)
    updates, state = bundle.update(grads, state, params)
    params = optax.apply_updates(params, updates)

    eval_params = bundle.eval_params(params, state)

    assert eval_params["blocks"][1]["w"].shape == params["blocks"][1]["w"].shape


def test_schedule_free_no_decay_leaf_unchanged_under_zero_gradients():
    plan = tiny_plan()
    bundle = rfft.schedule_free_adam_from_plan(
        plan,
        total_steps=20,
        schedule="wsd",
        weight_decay=0.1,
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    updates, _ = bundle.update(
        _zeros_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )

    assert jnp.allclose(updates["blocks"][0]["b"], 0.0)
    assert jnp.all(updates["blocks"][0]["w"] < 0.0)


def test_schedule_free_lora_plus_ratio_is_visible_in_report():
    bundle = rfft.schedule_free_adam_from_plan(
        tiny_lora_plan(),
        total_steps=20,
        base_lr=2e-4,
        schedule="wsd",
        weight_decay=0.0,
        lora_b_lr_ratio=16.0,
    )
    groups = {group.source_label: group for group in bundle.report.groups}

    assert groups["lora_A_decay"].effective_lr == pytest.approx(2e-4)
    assert groups["lora_B_decay"].effective_lr == pytest.approx(3.2e-3)


def test_schedule_free_requires_total_steps():
    with pytest.raises(TypeError, match="required keyword-only argument"):
        rfft.schedule_free_adam_from_plan(tiny_plan())
