"""APOLLO optimizer builder tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

import rollfast.finetune as rfft
from rollfast.optim.apollo import APOLLOLeafState, apollo_adamw

from .helpers import TinyGroup, TinyPlan, tiny_plan


def _ones_like_trainable(tree):
    return jax.tree.map(
        lambda x: jnp.ones_like(x) if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
    )


def _apollo_leaf_states(state):
    return [
        leaf
        for leaf in jax.tree.leaves(state, is_leaf=lambda x: isinstance(x, APOLLOLeafState))
        if isinstance(leaf, APOLLOLeafState)
    ]


def _matrix_plan() -> TinyPlan:
    trainable = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    labels = {"w": "matrix_decay"}
    groups = {
        "matrix_decay": TinyGroup(
            "matrix_decay",
            role="backbone",
            depth=0,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("matrix",),
        )
    }
    return TinyPlan(trainable=trainable, labels=labels, group_specs=groups)


def test_apollo_from_plan_updates_and_reports_projected_state():
    plan = _matrix_plan()
    bundle = rfft.apollo_adamw_from_plan(
        plan,
        apollo=rfft.APOLLOConfig(rank=1, projection_seed=7),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(
        _ones_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )
    params = optax.apply_updates(plan.trainable, updates)
    groups = {group.source_label: group for group in bundle.report.groups}
    projected_states = [leaf for leaf in _apollo_leaf_states(state) if leaf.projected]

    assert groups["matrix_decay"].optimizer == "apollo_adamw"
    assert bundle.report.estimated_state_bytes < 4 * 4 * 4 * 2
    assert bundle.report.state_policies["matrix_decay"] == "apollo_channel_scaling"
    assert not jnp.allclose(params["w"], plan.trainable["w"])
    assert projected_states
    assert {leaf.orientation for leaf in projected_states} == {"right"}
    assert all(leaf.mu.shape[0] == 1 for leaf in projected_states)


def test_apollo_manifest_records_projection_contract():
    plan = tiny_plan()
    config = rfft.APOLLOConfig(
        rank=3,
        projection_seed=123,
        projection_refresh_interval=5,
        scaling="tensor",
        scale=4.0,
        scale_front=True,
        disable_norm_growth_limiter=True,
        norm_growth_limiter=1.02,
    )
    bundle = rfft.apollo_adamw_from_plan(
        plan,
        apollo=config,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    method_config = bundle.manifest()["method_config"]

    assert method_config["method"] == "apollo"
    assert method_config["rank"] == 3
    assert method_config["projection_seed"] == 123
    assert method_config["projection_refresh_interval"] == 5
    assert method_config["projection_refresh_policy"] == "every_5_steps"
    assert method_config["effective_scaling"] == "tensor"
    assert method_config["scale"] == 4.0
    assert method_config["effective_scale"] == 4.0
    assert method_config["scale_front"] is True
    assert method_config["disable_norm_growth_limiter"] is True
    assert method_config["eps"] == 1e-6
    assert method_config["projection_type"] == "std"
    assert method_config["projection_orientation"] == "std_right_when_rows_gte_cols"
    assert method_config["projection_matrix_variance"] == "normal/sqrt(rank)"
    assert (
        method_config["adam_update_profile"]
        == "author_step_size_uncorrected_second_moment"
    )
    assert method_config["scaling_denominator_eps"] == 1e-8
    assert method_config["normalization_axes"] == "projected_l2_per_channel_or_tensor"
    assert method_config["profile_fidelity"] == "experimental"
    assert method_config["reference_validated"] is False
    assert method_config["reference_validation"] == "pending_author_implementation_compatibility_test"
    assert any("reference implementation" in warning for warning in bundle.report.warnings)


def test_apollo_default_eps_matches_author_profile():
    assert rfft.APOLLOConfig().eps == 1e-6
    assert rfft.APOLLOConfig.from_dict({}).eps == 1e-6


def test_apollo_mini_uses_rank_one_tensor_scaling():
    plan = tiny_plan()
    bundle = rfft.apollo_adamw_from_plan(
        plan,
        apollo=rfft.APOLLOConfig(rank=8, mini=True),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    projected_states = [leaf for leaf in _apollo_leaf_states(state) if leaf.projected]
    manifest = bundle.manifest()

    assert manifest["method_config"]["method"] == "apollo_mini"
    assert manifest["method_config"]["effective_rank"] == 1
    assert manifest["method_config"]["effective_scaling"] == "tensor"
    assert manifest["method_config"]["scale"] is None
    assert manifest["method_config"]["effective_scale"] == 128.0
    assert set(bundle.report.state_policies.values()) == {"apollo_mini_tensor_scaling"}
    assert projected_states
    assert all(leaf.projection.shape[0] == 1 for leaf in projected_states)


def test_apollo_projection_refresh_changes_projection_matrix():
    plan = _matrix_plan()
    bundle = rfft.apollo_adamw_from_plan(
        plan,
        apollo=rfft.APOLLOConfig(
            rank=1,
            projection_seed=11,
            projection_refresh_interval=1,
        ),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    grads = _ones_like_trainable(plan.trainable)
    state = bundle.init(plan.trainable)
    _, state = bundle.update(grads, state, plan.trainable)
    first_projection = _apollo_leaf_states(state)[0].projection
    _, state = bundle.update(grads, state, plan.trainable)
    second_projection = _apollo_leaf_states(state)[0].projection

    assert not jnp.allclose(first_projection, second_projection)


def test_apollo_mini_builder_default_scale_matches_reference_profile():
    plan = _matrix_plan()
    reference_scaled = rfft.apollo_adamw_from_plan(
        plan,
        apollo=rfft.APOLLOConfig(rank=8, mini=True),
        total_steps=10,
        base_lr=1.0,
        schedule="constant",
        clip_global_norm=None,
    )
    unit_scaled = rfft.apollo_adamw_from_plan(
        plan,
        apollo=rfft.APOLLOConfig(rank=8, mini=True, scale=1.0),
        total_steps=10,
        base_lr=1.0,
        schedule="constant",
        clip_global_norm=None,
    )
    grads = _ones_like_trainable(plan.trainable)

    reference_updates, _ = reference_scaled.update(
        grads,
        reference_scaled.init(plan.trainable),
        plan.trainable,
    )
    unit_updates, _ = unit_scaled.update(
        grads,
        unit_scaled.init(plan.trainable),
        plan.trainable,
    )

    ratio = jnp.linalg.norm(reference_updates["w"]) / jnp.linalg.norm(unit_updates["w"])
    assert jnp.allclose(ratio, jnp.sqrt(jnp.array(128.0)), rtol=1e-5)


def test_apollo_direct_scale_controls_first_update_norm():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = _ones_like_trainable(params)
    scaled = apollo_adamw(
        learning_rate=1.0,
        rank=1,
        projection_seed=17,
        scale=9.0,
    )
    unit = apollo_adamw(
        learning_rate=1.0,
        rank=1,
        projection_seed=17,
        scale=1.0,
    )

    scaled_updates, _ = scaled.update(grads, scaled.init(params), params)
    unit_updates, _ = unit.update(grads, unit.init(params), params)

    ratio = jnp.linalg.norm(scaled_updates["w"]) / jnp.linalg.norm(unit_updates["w"])
    assert jnp.allclose(ratio, 3.0, rtol=1e-5)


def test_apollo_first_update_matches_author_step_equation():
    params = {
        "w": jnp.array(
            [[1.0, -2.0], [0.5, 3.0], [-1.5, 2.5]],
            dtype=jnp.float32,
        )
    }
    grads = {
        "w": jnp.array(
            [[0.25, -0.5], [1.5, -0.75], [-1.0, 0.125]],
            dtype=jnp.float32,
        )
    }
    learning_rate = 0.25
    weight_decay = 0.2
    b1 = 0.9
    b2 = 0.999
    eps = 1e-6
    scale = 4.0
    tx = apollo_adamw(
        learning_rate=learning_rate,
        rank=2,
        projection_seed=17,
        scale=scale,
        disable_norm_growth_limiter=True,
        b1=b1,
        b2=b2,
        eps=eps,
        weight_decay=weight_decay,
    )

    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    leaf = _apollo_leaf_states(state)[0]
    projected_grad = (grads["w"] @ leaf.projection.T).T
    exp_avg = (1.0 - b1) * projected_grad
    exp_avg_sq = (1.0 - b2) * jnp.square(projected_grad)
    projected_update = exp_avg / (jnp.sqrt(exp_avg_sq) + eps)
    factors = jnp.linalg.norm(projected_update, axis=0) / (
        jnp.linalg.norm(projected_grad, axis=0) + 1e-8
    )
    scaled_grad = grads["w"] * factors[:, None] * jnp.sqrt(scale)
    step_size = learning_rate * jnp.sqrt(1.0 - b2) / (1.0 - b1)
    expected = (
        -step_size * scaled_grad
        - learning_rate * weight_decay * params["w"]
    )

    assert jnp.allclose(updates["w"], expected, rtol=1e-5, atol=1e-6)
