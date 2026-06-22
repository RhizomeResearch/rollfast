import jax
import jax.numpy as jnp
import optax
import pytest

import rollfast.finetune as rfft

from .helpers import tiny_plan


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


@pytest.mark.parametrize(
    ("builder", "optimizer_name"),
    (
        (rfft.hybrid_aurora_adam_from_plan, "aurora_adam"),
        (rfft.hybrid_prism_adam_from_plan, "prism_adam"),
        (rfft.hybrid_kron_adam_from_plan, "kron_adam"),
    ),
)
def test_hybrid_from_plan_updates_and_reports_groups(builder, optimizer_name):
    plan = tiny_plan()
    kwargs = {"total_steps": 10, "schedule": "constant", "clip_global_norm": None}
    if optimizer_name == "kron_adam":
        kwargs["preconditioner_update_probability"] = 1.0
    bundle = builder(plan, **kwargs)
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(
        _ones_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )
    params = optax.apply_updates(plan.trainable, updates)
    groups = {group.source_label: group for group in bundle.report.groups}

    assert state is not None
    assert params["embed"] is None
    assert not jnp.allclose(params["blocks"][0]["w"], plan.trainable["blocks"][0]["w"])
    assert groups["block_00_decay"].optimizer == optimizer_name
    assert groups["block_00_decay"].effective_lr == pytest.approx(2.5e-4)
    assert groups["block_00_no_decay"].weight_decay_value == 0.0


def test_hybrid_no_decay_leaf_unchanged_under_zero_gradients():
    plan = tiny_plan()
    bundle = rfft.hybrid_aurora_adam_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        weight_decay=0.1,
        clip_global_norm=None,
        polar_ns_iters=2,
    )
    state = bundle.init(plan.trainable)
    updates, _ = bundle.update(
        _zeros_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )

    assert jnp.allclose(updates["blocks"][0]["b"], 0.0)
    assert jnp.all(updates["blocks"][0]["w"] < 0.0)


def test_hybrid_supports_ema_eval_view_and_manifest():
    plan = tiny_plan()
    bundle = rfft.hybrid_prism_adam_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        ns_iters=2,
        ema=rfft.EMAConfig(enabled=True, decay=0.5),
    )
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(
        _ones_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )
    params = optax.apply_updates(plan.trainable, updates)

    assert bundle.eval_params(params, state, view="ema")["head"]["w"].shape == (
        2,
        1,
    )
    assert bundle.manifest()["groups"][0]["optimizer"] == "prism_adam"
