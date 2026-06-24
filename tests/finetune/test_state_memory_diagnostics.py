import jax.numpy as jnp

import rollfast.finetune as rfft

from .test_adamw8_from_plan import large_plan
from .helpers import tiny_plan


def test_state_memory_summary_measures_adamw_state_categories():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    summary = rfft.optimizer_state_memory_summary(bundle, state)

    assert summary.optimizer == "adamw"
    assert summary.total_bytes >= bundle.report.estimated_state_bytes
    assert summary.by_category["first_moment"] == 14 * 4
    assert summary.by_category["second_moment"] == 14 * 4
    assert summary.preconditioner_bytes == 0
    assert sum(summary.by_placement.values()) == summary.total_bytes
    assert (
        summary.replicated_bytes
        + summary.globally_sharded_bytes
        + summary.unsharded_bytes
        == summary.total_bytes
    )
    assert all(leaf.placement for leaf in summary.leaves)
    assert (
        summary.to_dict()["estimated_state_bytes"]
        == bundle.report.estimated_state_bytes
    )


def test_state_memory_summary_accounts_for_blockwise_int8_state():
    plan = large_plan()
    quantization = rfft.StateQuantizationConfig(
        enabled=True,
        block_size=512,
        min_size=1024,
        stochastic_rounding=False,
    )
    fp32_bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    q_bundle = rfft.adamw8_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        state_quantization=quantization,
    )
    fp32_summary = rfft.optimizer_state_memory_summary(
        fp32_bundle,
        fp32_bundle.init(plan.trainable),
    )
    q_summary = rfft.optimizer_state_memory_summary(
        q_bundle,
        q_bundle.init(plan.trainable),
    )

    assert q_summary.total_bytes < fp32_summary.total_bytes
    assert any(leaf.storage == "blockwise_int8" for leaf in q_summary.leaves)
    assert (
        q_summary.by_category["first_moment"] < fp32_summary.by_category["first_moment"]
    )


def test_state_memory_summary_reports_kron_preconditioner_factors():
    plan = tiny_plan()
    bundle = rfft.hybrid_kron_adam_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        preconditioner_update_probability=1.0,
    )
    state = bundle.init(plan.trainable)
    summary = rfft.optimizer_state_memory_summary(bundle, state)
    factor_shapes = {factor.shape for factor in summary.preconditioner_factors}
    storages = {factor.storage for factor in summary.preconditioner_factors}

    assert summary.optimizer == "kron_adam"
    assert summary.preconditioner_bytes > 0
    assert summary.by_category["preconditioner"] == summary.preconditioner_bytes
    assert (2, 2) in factor_shapes
    assert "matrix_factor" in storages
    assert "diagonal_factor" in storages


def test_state_memory_estimate_predicts_kron_preconditioner_factors_before_init():
    plan = tiny_plan()
    bundle = rfft.hybrid_kron_adam_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        preconditioner_update_probability=1.0,
    )

    estimate = rfft.estimate_optimizer_state_memory(plan, bundle)
    measured = rfft.optimizer_state_memory_summary(bundle, bundle.init(plan.trainable))

    assert estimate.optimizer == "kron_adam"
    assert estimate.preconditioner_bytes == measured.preconditioner_bytes
    assert (
        estimate.by_category["preconditioner_aux"]
        == measured.by_category["preconditioner_aux"]
    )
    assert estimate.by_category["first_moment"] == measured.by_category["first_moment"]
    assert "finite guards" in estimate.warnings[0]
    assert estimate.total_bytes < measured.total_bytes
    assert estimate.to_dict()["preconditioner_bytes"] == measured.preconditioner_bytes


def test_state_memory_estimate_honors_kron_preconditioner_dtype():
    plan = tiny_plan()
    bundle = rfft.hybrid_kron_adam_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        preconditioner_dtype=jnp.bfloat16,
        preconditioner_update_probability=1.0,
    )

    estimate = rfft.estimate_optimizer_state_memory(
        plan,
        bundle,
        preconditioner_dtype=jnp.bfloat16,
    )
    measured = rfft.optimizer_state_memory_summary(bundle, bundle.init(plan.trainable))

    assert estimate.preconditioner_bytes == measured.preconditioner_bytes
    assert {factor.dtype for factor in estimate.preconditioner_factors} == {"bfloat16"}


def test_state_memory_estimate_reports_replicated_and_sharded_policy_bytes():
    plan = tiny_plan()
    replicated_bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        sharding=rfft.ShardingPolicy(
            mesh_axes=("data",),
            state_placement="replicate_small_follow_large",
            small_state_threshold=10_000,
        ),
    )
    sharded_bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        sharding=rfft.ShardingPolicy(
            mesh_axes=("data",),
            state_placement="replicate_small_follow_large",
            small_state_threshold=0,
        ),
    )

    replicated = rfft.estimate_optimizer_state_memory(plan, replicated_bundle)
    sharded = rfft.estimate_optimizer_state_memory(plan, sharded_bundle)

    assert replicated.replicated_bytes == replicated.total_bytes
    assert replicated.globally_sharded_bytes == 0
    assert sharded.globally_sharded_bytes == sharded.total_bytes
    assert sharded.replicated_bytes == 0
    assert replicated.to_dict()["by_placement"]["replicated"] == replicated.total_bytes


def test_state_memory_summary_reports_no_preconditioners_for_aurora_prism():
    plan = tiny_plan()
    for builder in (
        rfft.hybrid_aurora_adam_from_plan,
        rfft.hybrid_prism_adam_from_plan,
    ):
        bundle = builder(
            plan,
            total_steps=10,
            schedule="constant",
            clip_global_norm=None,
        )
        summary = rfft.optimizer_state_memory_summary(
            bundle, bundle.init(plan.trainable)
        )

        assert summary.preconditioner_bytes == 0
        assert "first_moment" in summary.by_category


def test_state_memory_summary_serializes_quantized_leaf_metadata():
    plan = large_plan()
    bundle = rfft.adamw8_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        state_quantization=rfft.StateQuantizationConfig(
            enabled=True,
            block_size=512,
            min_size=1024,
            stochastic_rounding=False,
        ),
    )
    summary = rfft.optimizer_state_memory_summary(bundle, bundle.init(plan.trainable))
    first_quantized = next(
        leaf for leaf in summary.leaves if leaf.storage == "blockwise_int8"
    )

    assert "scale" in first_quantized.dtype
    assert isinstance(first_quantized.bytes, int)
    assert first_quantized.shape == (128, 64)
    assert jnp.dtype("int8").name in first_quantized.dtype
