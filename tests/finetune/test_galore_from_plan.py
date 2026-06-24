"""GaLore optimizer builder tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

import rollfast.finetune as rfft
from rollfast.optim.galore import GaLoreLeafState

from .helpers import tiny_plan


def _ones_like_trainable(tree):
    return jax.tree.map(
        lambda x: jnp.ones_like(x) if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
    )


def _galore_leaf_states(state):
    return [
        leaf
        for leaf in jax.tree.leaves(
            state, is_leaf=lambda x: isinstance(x, GaLoreLeafState)
        )
        if isinstance(leaf, GaLoreLeafState)
    ]


def test_galore_from_plan_updates_and_reports_projected_state():
    plan = tiny_plan()
    bundle = rfft.galore_adamw_from_plan(
        plan,
        galore=rfft.GaLoreConfig(rank=1, min_matrix_size=1, update_interval=1),
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
    projected_states = [leaf for leaf in _galore_leaf_states(state) if leaf.projected]

    assert groups["block_00_decay"].optimizer == "galore_adamw"
    assert bundle.report.estimated_state_bytes < 14 * 4 * 2
    assert bundle.report.state_policies["block_00_decay"] == "galore_projected_moments"
    assert not jnp.allclose(params["blocks"][0]["w"], plan.trainable["blocks"][0]["w"])
    assert projected_states
    assert {leaf.orientation for leaf in projected_states} <= {"left", "right"}
    assert all(min(leaf.mu.shape) == 1 for leaf in projected_states)


def test_galore_auto_projection_manifest_marks_rollfast_profile():
    plan = tiny_plan()
    bundle = rfft.galore_adamw_from_plan(
        plan,
        galore=rfft.GaLoreConfig(rank=1, min_matrix_size=1, projection="auto"),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    method_config = bundle.manifest()["method_config"]

    assert method_config["projection_profile"] == "rollfast_memory_min_auto"
    assert method_config["reference_projection"] == "galore_std"
    assert method_config["profile_fidelity"] == "safe_default"
    assert method_config["reference_validated"] is False
    assert method_config["known_deviations"] == (
        "projection_auto_uses_memory_min_orientation_not_galore_std",
    )
    assert any("projection='auto'" in warning for warning in bundle.report.warnings)


def test_galore_explicit_projection_manifest_can_be_reference_validated():
    plan = tiny_plan()
    bundle = rfft.galore_adamw_from_plan(
        plan,
        galore=rfft.GaLoreConfig(rank=1, min_matrix_size=1, projection="right"),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    method_config = bundle.manifest()["method_config"]

    assert method_config["projection_profile"] == "galore_explicit_right"
    assert method_config["known_deviations"] == ()
    assert method_config["profile_fidelity"] == "reference_implementation"
    assert method_config["reference_validated"] is True


def test_galore_min_matrix_size_falls_back_to_full_moments():
    plan = tiny_plan()
    bundle = rfft.galore_adamw_from_plan(
        plan,
        galore=rfft.GaLoreConfig(rank=1, min_matrix_size=5),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    leaf_states = _galore_leaf_states(state)

    assert bundle.report.estimated_state_bytes == 14 * 4 * 2
    assert leaf_states
    assert all(not leaf.projected for leaf in leaf_states)


def test_galore_basis_refresh_canonicalizes_svd_signs():
    plan = tiny_plan()
    bundle = rfft.galore_adamw_from_plan(
        plan,
        galore=rfft.GaLoreConfig(rank=1, min_matrix_size=1, update_interval=1),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    grads = _ones_like_trainable(plan.trainable)
    _, state = bundle.update(grads, state, plan.trainable)

    for leaf in _galore_leaf_states(state):
        if not leaf.projected:
            continue
        basis = leaf.basis_left if leaf.orientation == "left" else leaf.basis_right
        pivot = jnp.argmax(jnp.abs(basis), axis=0)
        signs = jnp.take_along_axis(basis, pivot[None, :], axis=0)[0]
        assert jnp.all(signs >= 0.0)


def test_galore_transport_refresh_is_experimental_and_preserves_shapes():
    plan = tiny_plan()
    bundle = rfft.galore_adamw_from_plan(
        plan,
        galore=rfft.GaLoreConfig(
            rank=1,
            min_matrix_size=1,
            update_interval=1,
            state_on_basis_refresh="transport",
        ),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    grads = _ones_like_trainable(plan.trainable)
    state = bundle.init(plan.trainable)
    _, state = bundle.update(grads, state, plan.trainable)
    first_shapes = tuple(
        leaf.mu.shape for leaf in _galore_leaf_states(state) if leaf.projected
    )
    second_grads = jax.tree.map(
        lambda x: (
            (jnp.arange(x.size, dtype=x.dtype).reshape(x.shape) + 1.0)
            if x is not None
            else None
        ),
        plan.trainable,
        is_leaf=lambda x: x is None,
    )
    _, state = bundle.update(second_grads, state, plan.trainable)
    projected_states = [leaf for leaf in _galore_leaf_states(state) if leaf.projected]
    manifest = bundle.manifest()

    assert tuple(leaf.mu.shape for leaf in projected_states) == first_shapes
    assert all(jnp.all(leaf.nu >= 0.0) for leaf in projected_states)
    assert manifest["method_config"]["state_on_basis_refresh"] == "transport"
    assert (
        manifest["method_config"]["state_transport_profile"]
        == "experimental_basis_overlap"
    )
    assert manifest["method_config"]["reference_validated"] is False
    assert (
        "basis_transport_experimental_not_reference_validated"
        in manifest["method_config"]["known_deviations"]
    )
    assert any("experimental" in warning for warning in bundle.report.warnings)


def test_galore_rank_mapping_must_cover_labels():
    plan = tiny_plan()
    with pytest.raises(ValueError, match="missing label"):
        rfft.galore_adamw_from_plan(
            plan,
            galore=rfft.GaLoreConfig(
                rank={
                    "block_00_decay": 1,
                },
                min_matrix_size=1,
            ),
            total_steps=10,
            schedule="constant",
            clip_global_norm=None,
        )


def test_galore_sharded_refresh_rejects_hidden_all_gather():
    plan = tiny_plan()
    sharding = rfft.ShardingPolicy(
        mesh_axes=("data", "model"),
        data_axes=("data",),
        parameter_axes=("model",),
        allow_host_materialization=False,
    )

    with pytest.raises(ValueError, match="hidden .*all-gather"):
        rfft.galore_adamw_from_plan(
            plan,
            galore=rfft.GaLoreConfig(rank=1, min_matrix_size=1, update_interval=1),
            total_steps=10,
            schedule="constant",
            clip_global_norm=None,
            sharding=sharding,
        )


def test_galore_sharded_refresh_allows_no_projected_matrices():
    plan = tiny_plan()
    sharding = rfft.ShardingPolicy(
        mesh_axes=("data", "model"),
        data_axes=("data",),
        parameter_axes=("model",),
        allow_host_materialization=False,
    )

    bundle = rfft.galore_adamw_from_plan(
        plan,
        galore=rfft.GaLoreConfig(rank=1, min_matrix_size=10_000),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        sharding=sharding,
    )

    assert bundle.manifest()["method_config"]["basis_refresh_communication_bytes"] == 0


def test_galore_sharded_refresh_opt_in_reports_communication_cost():
    plan = tiny_plan()
    sharding = rfft.ShardingPolicy(
        mesh_axes=("data", "model"),
        data_axes=("data",),
        parameter_axes=("model",),
        allow_host_materialization=True,
    )

    bundle = rfft.galore_adamw_from_plan(
        plan,
        galore=rfft.GaLoreConfig(rank=1, min_matrix_size=1, update_interval=1),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        sharding=sharding,
    )
    method_config = bundle.manifest()["method_config"]

    assert method_config["basis_refresh_collective"] == "host_or_device_all_gather"
    assert method_config["basis_refresh_communication_bytes"] > 0
    assert method_config["allow_host_materialization"] is True
    assert any(
        "estimated refresh communication bytes" in warning
        for warning in bundle.report.warnings
    )
