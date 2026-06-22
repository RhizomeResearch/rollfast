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
        metadata={"epoch": 1},
    )
    restored = rfft.restore_state_checkpoint(bundle, checkpoint)
    updates_a, _ = bundle.update(grads, state, params)
    updates_b, _ = bundle.update(grads, restored, params)

    assert checkpoint.metadata["epoch"] == 1
    _assert_tree_allclose(updates_a, updates_b)


def test_state_checkpoint_rejects_mismatched_plan_fingerprint():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(plan, total_steps=10, schedule="constant")
    checkpoint = rfft.make_state_checkpoint(bundle, bundle.init(plan.trainable))
    other = rfft.adamw_from_plan(
        tiny_lora_plan(),
        total_steps=10,
        schedule="constant",
    )

    with pytest.raises(rfft.OptimizerStateRestoreError, match="fingerprint mismatch"):
        rfft.restore_state_checkpoint(other, checkpoint)


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

    checkpoint = rfft.save_state_checkpoint(path, bundle, state)
    loaded = rfft.load_state_checkpoint(path)
    restored = rfft.load_state_checkpoint(path, bundle)

    assert loaded.manifest["fingerprint"] == checkpoint.manifest["fingerprint"]
    assert restored is not None


def test_state_manifest_contains_eval_views_and_averaging_configs():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        ema=rfft.EMAConfig(enabled=True, decay=0.5),
        swa=rfft.SWAConfig(enabled=True, start_step=1),
    )
    manifest = rfft.state_manifest(bundle)

    assert manifest["fingerprint"] == bundle.report.fingerprint
    assert manifest["ema"]["enabled"] is True
    assert manifest["swa"]["enabled"] is True
    assert set(manifest["eval_views"]) == {"optimizer", "ema", "swa"}
