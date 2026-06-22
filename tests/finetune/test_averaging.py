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


def test_adamw_eval_params_identity_without_averaging():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )

    assert bundle.eval_params(plan.trainable) is plan.trainable
    with pytest.raises(ValueError, match="unknown eval params view"):
        bundle.eval_params(plan.trainable, view="ema")


def test_ema_view_updates_after_optimizer_step():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        ema=rfft.EMAConfig(enabled=True, decay=0.5),
    )
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(
        _ones_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )
    params = optax.apply_updates(plan.trainable, updates)
    ema_params = bundle.eval_params(params, state, view="ema")

    assert bundle.default_eval_view == "ema"
    assert "ema" in bundle.eval_views
    assert not jnp.allclose(ema_params["head"]["w"], params["head"]["w"])
    assert not jnp.allclose(ema_params["head"]["w"], plan.trainable["head"]["w"])
    assert bundle.manifest()["ema"]["enabled"] is True


def test_swa_view_starts_at_resolved_fraction():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=4,
        schedule="constant",
        clip_global_norm=None,
        swa=rfft.SWAConfig(enabled=True, start_fraction=0.5),
    )
    state = bundle.init(plan.trainable)
    params = plan.trainable
    grads = _ones_like_trainable(params)

    updates, state = bundle.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    assert bundle.eval_params(params, state, view="swa")["head"]["w"].shape == (
        2,
        1,
    )
    assert jnp.allclose(
        bundle.eval_params(params, state, view="swa")["head"]["w"],
        params["head"]["w"],
    )

    updates, state = bundle.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    swa_params = bundle.eval_params(params, state, view="swa")

    assert "swa" in bundle.eval_views
    assert not jnp.allclose(swa_params["head"]["w"], plan.trainable["head"]["w"])


def test_averaging_does_not_advance_on_withheld_accumulation_microstep():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        accumulation_steps=2,
        ema=rfft.EMAConfig(enabled=True, decay=0.5),
    )
    state = bundle.init(plan.trainable)
    grads = _ones_like_trainable(plan.trainable)

    updates, state = bundle.update(grads, state, plan.trainable)
    params = optax.apply_updates(plan.trainable, updates)
    assert state.ema_count == 0
    assert jnp.allclose(
        bundle.eval_params(params, state, view="ema")["head"]["w"],
        plan.trainable["head"]["w"],
    )

    updates, state = bundle.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    assert state.ema_count == 1
    assert not jnp.allclose(
        bundle.eval_params(params, state, view="ema")["head"]["w"],
        plan.trainable["head"]["w"],
    )


def test_schedule_free_keeps_named_eval_views_with_ema():
    plan = tiny_plan()
    bundle = rfft.schedule_free_adam_from_plan(
        plan,
        total_steps=10,
        schedule="wsd",
        clip_global_norm=None,
        ema=rfft.EMAConfig(enabled=True, decay=0.5),
    )
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(
        _ones_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )
    params = optax.apply_updates(plan.trainable, updates)

    assert bundle.default_eval_view == "ema"
    assert {"optimizer", "schedule_free", "ema"} <= set(bundle.eval_views)
    assert bundle.eval_params(params, state, view="schedule_free")["head"][
        "w"
    ].shape == params["head"]["w"].shape
    assert bundle.eval_params(params, state, view="ema")["head"]["w"].shape == params[
        "head"
    ]["w"].shape


def test_ema_tag_filters_average_included_groups_only():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        ema=rfft.EMAConfig(enabled=True, decay=0.5, include_tags=("head",)),
    )
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(
        _ones_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )
    params = optax.apply_updates(plan.trainable, updates)
    ema_params = bundle.eval_params(params, state, view="ema")

    assert not jnp.allclose(ema_params["head"]["w"], params["head"]["w"])
    assert jnp.allclose(
        ema_params["blocks"][0]["w"],
        params["blocks"][0]["w"],
    )
