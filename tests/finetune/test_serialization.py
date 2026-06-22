import jax
import jax.numpy as jnp
import pytest

import rollfast.finetune as rfft

from .helpers import tiny_lora_plan, tiny_plan


def _ones_like_trainable(tree):
    return jax.tree.map(
        lambda x: jnp.ones_like(x) if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
    )


def _assert_tree_allclose(left, right):
    leaves = zip(
        jax.tree.leaves(left, is_leaf=lambda x: x is None),
        jax.tree.leaves(right, is_leaf=lambda x: x is None),
        strict=True,
    )
    for lhs, rhs in leaves:
        if lhs is None:
            assert rhs is None
        elif hasattr(lhs, "dtype"):
            assert jnp.allclose(lhs, rhs)
        else:
            assert lhs == rhs


def test_state_checkpoint_round_trip_preserves_next_update():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        ema=rfft.EMAConfig(enabled=True, decay=0.5),
    )
    state = bundle.init(plan.trainable)
    grads = _ones_like_trainable(plan.trainable)
    updates, state = bundle.update(grads, state, plan.trainable)
    params = jax.tree.map(
        lambda p, u: p + u if p is not None else None,
        plan.trainable,
        updates,
        is_leaf=lambda x: x is None,
    )

    checkpoint = rfft.make_state_checkpoint(
        bundle,
        state,
        model_checkpoint_id="model-step-1",
        metadata={"epoch": 1},
    )
    restored = rfft.restore_state_checkpoint(
        bundle,
        checkpoint,
        model_checkpoint_id="model-step-1",
    )
    updates_a, _ = bundle.update(grads, state, params)
    updates_b, _ = bundle.update(grads, restored, params)

    assert checkpoint.metadata["epoch"] == 1
    _assert_tree_allclose(updates_a, updates_b)


def test_state_checkpoint_rejects_mismatched_plan_fingerprint():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(plan, total_steps=10, schedule="constant")
    checkpoint = rfft.make_state_checkpoint(
        bundle,
        bundle.init(plan.trainable),
        model_checkpoint_id="model-step-1",
    )
    other = rfft.adamw_from_plan(
        tiny_lora_plan(),
        total_steps=10,
        schedule="constant",
    )

    with pytest.raises(rfft.OptimizerStateRestoreError, match="fingerprint mismatch"):
        rfft.restore_state_checkpoint(
            other,
            checkpoint,
            model_checkpoint_id="model-step-1",
        )


def test_state_checkpoint_rejects_mismatched_model_checkpoint_id():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(plan, total_steps=10, schedule="constant")
    checkpoint = rfft.make_state_checkpoint(
        bundle,
        bundle.init(plan.trainable),
        model_checkpoint_id="model-step-1",
    )

    with pytest.raises(rfft.OptimizerStateRestoreError, match="model checkpoint mismatch"):
        rfft.restore_state_checkpoint(
            bundle,
            checkpoint,
            model_checkpoint_id="model-step-2",
        )


def test_state_checkpoint_rejects_mismatched_sharding_policy():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        sharding=rfft.ShardingPolicy(mesh_axes=("data",)),
    )
    checkpoint = rfft.make_state_checkpoint(
        bundle,
        bundle.init(plan.trainable),
        model_checkpoint_id="model-step-1",
    )
    other = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        sharding=rfft.ShardingPolicy(mesh_axes=("data", "model")),
    )

    with pytest.raises(rfft.OptimizerStateRestoreError, match="sharding mismatch"):
        rfft.restore_state_checkpoint(
            other,
            checkpoint,
            model_checkpoint_id="model-step-1",
        )


def test_state_checkpoint_rejects_mismatched_logical_id_table_hash():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(plan, total_steps=10, schedule="constant")
    checkpoint = rfft.make_state_checkpoint(
        bundle,
        bundle.init(plan.trainable),
        model_checkpoint_id="model-step-1",
    )
    incompatible = rfft.OptimizerStateCheckpoint(
        manifest={
            **checkpoint.manifest,
            "logical_id_table_hash": "other-logical-table",
        },
        state=checkpoint.state,
        metadata=checkpoint.metadata,
    )

    with pytest.raises(rfft.OptimizerStateRestoreError, match="logical-ID table mismatch"):
        rfft.restore_state_checkpoint(
            bundle,
            incompatible,
            model_checkpoint_id="model-step-1",
        )


def test_state_checkpoint_rejects_mismatched_quantization_metadata():
    plan = tiny_plan()
    bundle = rfft.adamw8_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        state_quantization=rfft.StateQuantizationConfig(
            enabled=True,
            block_size=16,
            min_size=1,
            stochastic_rounding=False,
        ),
    )
    checkpoint = rfft.make_state_checkpoint(
        bundle,
        bundle.init(plan.trainable),
        model_checkpoint_id="model-step-1",
    )
    incompatible = rfft.OptimizerStateCheckpoint(
        manifest={
            **checkpoint.manifest,
            "state_quantization": {
                **checkpoint.manifest["state_quantization"],
                "block_layout": "logical_global",
            },
        },
        state=checkpoint.state,
        metadata=checkpoint.metadata,
    )

    with pytest.raises(rfft.OptimizerStateRestoreError, match="quantization metadata mismatch"):
        rfft.restore_state_checkpoint(
            bundle,
            incompatible,
            model_checkpoint_id="model-step-1",
        )


def test_state_checkpoint_includes_required_master_and_loss_scale_components():
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(
        master_params="always",
        loss_scale="dynamic",
    )
    bundle = rfft.compile_optimizer(
        plan,
        optimizer=rfft.OptimizerConfig(weight_decay=0.0),
        schedule=rfft.ScheduleConfig(kind="constant", total_steps=10),
        gradient_policy=rfft.GradientPolicy(clip_global_norm=None),
        precision=precision,
    )
    master = rfft.make_master_params(plan.trainable, precision)
    loss_scale = rfft.init_loss_scale_state(precision)
    state = bundle.init(master)

    checkpoint = rfft.make_state_checkpoint(
        bundle,
        state,
        model_checkpoint_id="model-step-1",
        master_params=master,
        loss_scale=loss_scale,
    )
    restored = rfft.restore_state_checkpoint(
        bundle,
        checkpoint,
        model_checkpoint_id="model-step-1",
    )

    assert checkpoint.master_params is master
    assert checkpoint.loss_scale == loss_scale
    assert checkpoint.manifest["components"]["master_params"] is True
    assert checkpoint.manifest["components"]["loss_scale"] is True
    assert restored is state


def test_state_checkpoint_rejects_missing_required_master_or_loss_scale():
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(
        master_params="always",
        loss_scale="dynamic",
    )
    bundle = rfft.compile_optimizer(
        plan,
        optimizer=rfft.OptimizerConfig(weight_decay=0.0),
        schedule=rfft.ScheduleConfig(kind="constant", total_steps=10),
        gradient_policy=rfft.GradientPolicy(clip_global_norm=None),
        precision=precision,
    )
    master = rfft.make_master_params(plan.trainable, precision)
    state = bundle.init(master)

    with pytest.raises(ValueError, match="master_params"):
        rfft.make_state_checkpoint(
            bundle,
            state,
            model_checkpoint_id="model-step-1",
            loss_scale=rfft.init_loss_scale_state(precision),
        )
    with pytest.raises(ValueError, match="loss_scale"):
        rfft.make_state_checkpoint(
            bundle,
            state,
            model_checkpoint_id="model-step-1",
            master_params=master,
        )


def test_state_checkpoint_save_and_load(tmp_path):
    plan = tiny_plan()
    bundle = rfft.schedule_free_adam_from_plan(
        plan,
        total_steps=10,
        schedule="wsd",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    path = tmp_path / "optimizer.rfopt"

    checkpoint = rfft.save_state_checkpoint(
        path,
        bundle,
        state,
        model_checkpoint_id="model-step-1",
    )
    loaded = rfft.load_state_checkpoint(path)
    restored = rfft.load_state_checkpoint(
        path,
        bundle,
        model_checkpoint_id="model-step-1",
    )

    assert loaded.manifest["fingerprint"] == checkpoint.manifest["fingerprint"]
    assert loaded.manifest["model_checkpoint_id"] == "model-step-1"
    assert loaded.manifest["components"]["schedule_free"] is True
    assert restored is not None


def test_schedule_free_checkpoint_rejects_missing_schedule_free_state():
    plan = tiny_plan()
    bundle = rfft.schedule_free_adam_from_plan(
        plan,
        total_steps=10,
        schedule="wsd",
        clip_global_norm=None,
    )
    checkpoint = rfft.make_state_checkpoint(
        bundle,
        bundle.init(plan.trainable),
        model_checkpoint_id="model-step-1",
    )
    missing_state = rfft.OptimizerStateCheckpoint(
        manifest=checkpoint.manifest,
        state=(),
        metadata=checkpoint.metadata,
    )
    missing_metadata = rfft.OptimizerStateCheckpoint(
        manifest={
            **checkpoint.manifest,
            "components": {
                **checkpoint.manifest["components"],
                "schedule_free": False,
            },
        },
        state=checkpoint.state,
        metadata=checkpoint.metadata,
    )

    with pytest.raises(rfft.OptimizerStateRestoreError, match="schedule-free optimizer state"):
        rfft.restore_state_checkpoint(
            bundle,
            missing_state,
            model_checkpoint_id="model-step-1",
        )
    with pytest.raises(rfft.OptimizerStateRestoreError, match="schedule-free state metadata"):
        rfft.restore_state_checkpoint(
            bundle,
            missing_metadata,
            model_checkpoint_id="model-step-1",
        )


def test_state_manifest_contains_eval_views_and_averaging_configs():
    plan = tiny_plan()
    sharding = rfft.ShardingPolicy(
        mesh_axes=("data", "model"),
        parameter_axes=("model",),
        state_placement="replicate_small_follow_large",
    )
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        sharding=sharding,
        ema=rfft.EMAConfig(enabled=True, decay=0.5),
        swa=rfft.SWAConfig(enabled=True, start_step=1),
    )
    manifest = rfft.state_manifest(bundle)

    assert manifest["fingerprint"] == bundle.report.fingerprint
    assert manifest["plan_fingerprint"] == bundle.report.fingerprint
    assert manifest["rollfast_revision"]
    assert manifest["model_checkpoint_id"] is None
    assert manifest["base_model_value_hash"] is None
    assert manifest["optimizer_profiles"] == ("adamw",)
    assert manifest["optimizer_config"] == manifest["optimizer"]
    assert manifest["schedule_config"] == manifest["schedule"]
    assert manifest["accumulation_config"] == manifest["accumulation"]
    assert manifest["precision_config"] == manifest["precision"]
    assert manifest["sharding_config"] == manifest["sharding"]
    assert manifest["quantization_metadata"] == manifest["state_quantization"]
    assert manifest["group_table"] == manifest["groups"]
    assert manifest["logical_id_table_hash"] == bundle.report.logical_id_table_hash
    assert manifest["model_state_structure_hash"] is None
    assert manifest["counters"] == {}
    assert manifest["serializable"] is True
    assert manifest["ema"]["enabled"] is True
    assert manifest["swa"]["enabled"] is True
    assert manifest["sharding"] == sharding.to_dict()
    assert set(manifest["eval_views"]) == {"optimizer", "ema", "swa"}


def test_state_checkpoint_manifest_records_runtime_counters():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(plan, total_steps=10, schedule="constant")
    counters = rfft.StepCounters(
        microstep=jnp.asarray(3, dtype=jnp.int32),
        attempted_update=jnp.asarray(3, dtype=jnp.int32),
        successful_update=jnp.asarray(2, dtype=jnp.int32),
        schedule_step=jnp.asarray(2, dtype=jnp.int32),
        rank_step=jnp.asarray(2, dtype=jnp.int32),
        average_step=jnp.asarray(2, dtype=jnp.int32),
        loss_scale_growth_step=jnp.asarray(0, dtype=jnp.int32),
    )

    checkpoint = rfft.make_state_checkpoint(
        bundle,
        bundle.init(plan.trainable),
        model_checkpoint_id="model-step-3",
        counters=counters,
    )

    assert checkpoint.manifest["counters"]["microstep"] == 3
    assert checkpoint.manifest["counters"]["successful_update"] == 2
    assert checkpoint.manifest["components"]["counters"] is True
