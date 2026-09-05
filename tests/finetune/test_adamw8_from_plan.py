import jax
import jax.numpy as jnp
import optax

import rollfast.finetune as rfft
from rollfast.optim.adam8 import (
    DYNAMIC_SIGNED_CODEBOOK_ID,
    DYNAMIC_UNSIGNED_CODEBOOK_ID,
    QuantizedBlocks,
    tree_state_nbytes,
)

from .helpers import ones_like_trainable, large_plan, leaf_estimation_plan


def _quantized_leaves(tree):
    return [
        leaf
        for leaf in jax.tree.leaves(
            tree,
            is_leaf=lambda x: isinstance(x, QuantizedBlocks),
        )
        if isinstance(leaf, QuantizedBlocks)
    ]


def test_adamw8_from_plan_quantizes_eligible_group_state_and_reports_bytes():
    plan = large_plan()
    quantization = rfft.StateQuantizationConfig(
        enabled=True,
        block_size=512,
        min_size=1024,
        stochastic_rounding=False,
    )
    bundle = rfft.adamw8_from_plan(
        plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
        weight_decay=0.0,
        state_quantization=quantization,
    )
    fp32_bundle = rfft.adamw_from_plan(
        plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
        weight_decay=0.0,
    )
    state = bundle.init(plan.trainable)
    fp32_state = fp32_bundle.init(plan.trainable)
    groups = {group.source_label: group for group in bundle.report.groups}

    assert bundle.optimizer_config.name == "adamw8"
    assert bundle.manifest()["state_quantization"]["enabled"] is True
    assert bundle.manifest()["state_quantization"]["block_layout"] == "shard_local"
    assert (
        bundle.manifest()["state_quantization"]["first_moment_codebook_id"]
        == DYNAMIC_SIGNED_CODEBOOK_ID
    )
    assert (
        bundle.manifest()["state_quantization"]["second_moment_codebook_id"]
        == DYNAMIC_UNSIGNED_CODEBOOK_ID
    )
    assert (
        rfft.StateQuantizationConfig.from_dict(bundle.manifest()["state_quantization"])
        == quantization
    )
    group_rows = {row["label"]: row for row in bundle.report.group_table()}
    assert groups["large_decay"].optimizer == "adamw8"
    assert groups["bias_no_decay"].optimizer == "adamw8"
    assert group_rows["large_decay"]["state_policy"] == "blockwise_int8_moments"
    assert group_rows["embed_decay"]["state_policy"] == "float32_fallback_moments"
    assert group_rows["bias_no_decay"]["state_policy"] == "float32_fallback_moments"
    quantized_leaves = _quantized_leaves(state)
    assert quantized_leaves
    assert {leaf.codebook_id for leaf in quantized_leaves} == {
        DYNAMIC_SIGNED_CODEBOOK_ID,
        DYNAMIC_UNSIGNED_CODEBOOK_ID,
    }
    assert (
        bundle.report.estimated_state_bytes < fp32_bundle.report.estimated_state_bytes
    )
    assert tree_state_nbytes(state) < tree_state_nbytes(fp32_state)


def test_adamw8_keeps_sensitive_tagged_group_in_fp32_state():
    plan = large_plan()
    bundle = rfft.adamw8_from_plan(
        plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
        state_quantization=rfft.StateQuantizationConfig(
            enabled=True,
            block_size=512,
            min_size=1,
            stochastic_rounding=False,
        ),
    )
    state = bundle.init(plan.trainable)
    quantized_shapes = {leaf.shape for leaf in _quantized_leaves(state)}
    fp32_shapes = {
        tuple(leaf.shape)
        for leaf in jax.tree.leaves(state)
        if hasattr(leaf, "dtype") and leaf.dtype == jnp.float32 and leaf.shape
    }

    assert (128, 64) in quantized_shapes
    assert (4096,) in fp32_shapes


def test_adamw8_report_estimates_subthreshold_leaves_individually():
    plan = leaf_estimation_plan(mixed=False)
    bundle = rfft.adamw8_from_plan(
        plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
        state_quantization=rfft.StateQuantizationConfig(
            enabled=True,
            block_size=512,
            min_size=4096,
            stochastic_rounding=False,
        ),
    )
    state = bundle.init(plan.trainable)
    summary = rfft.optimizer_state_memory_summary(bundle, state)
    expected_moment_bytes = 3 * 2048 * 4 * 2

    assert not _quantized_leaves(state)
    assert bundle.report.estimated_state_bytes == expected_moment_bytes
    assert (
        summary.by_category["first_moment"] + summary.by_category["second_moment"]
        == expected_moment_bytes
    )


def test_adamw8_from_plan_update_stays_close_to_fp32_after_quantized_storage():
    plan = large_plan()
    quantization = rfft.StateQuantizationConfig(
        enabled=True,
        block_size=512,
        min_size=1024,
        stochastic_rounding=False,
    )
    fp32_bundle = rfft.adamw_from_plan(
        plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
        weight_decay=0.01,
    )
    q_bundle = rfft.adamw8_from_plan(
        plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
        weight_decay=0.01,
        state_quantization=quantization,
    )
    grads = ones_like_trainable(plan.trainable)
    fp32_state = fp32_bundle.init(plan.trainable)
    q_state = q_bundle.init(plan.trainable)
    fp32_updates, fp32_state = fp32_bundle.update(grads, fp32_state, plan.trainable)
    q_updates, q_state = q_bundle.update(grads, q_state, plan.trainable)
    fp32_params = optax.apply_updates(plan.trainable, fp32_updates)
    q_params = optax.apply_updates(plan.trainable, q_updates)

    fp32_updates, _ = fp32_bundle.update(grads, fp32_state, fp32_params)
    q_updates, _ = q_bundle.update(grads, q_state, q_params)

    assert jnp.allclose(q_updates["w"], fp32_updates["w"], rtol=0.03, atol=3e-4)
    assert jnp.allclose(
        q_updates["bias"],
        fp32_updates["bias"],
        rtol=0.0,
        atol=1e-7,
    )
