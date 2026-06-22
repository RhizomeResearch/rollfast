"""Stateful SAM/ASAM step tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rollfast.finetune as rfft

from .helpers import tiny_plan


def _bundle(plan, *, accumulation_steps: int = 1):
    return rfft.adamw_from_plan(
        plan,
        total_steps=10,
        base_lr=1e-3,
        weight_decay=0.0,
        b1=0.0,
        b2=0.0,
        schedule="constant",
        clip_global_norm=None,
        accumulation_steps=accumulation_steps,
    )


def _loss_with_state(model, model_state, key):
    sample = jax.random.normal(key, ())
    loss = jnp.sum(model["head"]["w"] ** 2)
    return (loss, {"sample": sample}), {"updates": model_state["updates"] + 1}


def _loss_with_aggregate_state(model, model_state, key):
    sample = jax.random.normal(key, ())
    del model_state
    loss = jnp.sum(model["head"]["w"] ** 2)
    return (loss, {"sample": sample}), {"updates": jnp.asarray(1, dtype=jnp.int32)}


def _aggregate_updates(old_state, proposed_state):
    return {"updates": old_state["updates"] + proposed_state["updates"]}


def _assert_tree_allclose(left, right):
    for lhs, rhs in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if hasattr(lhs, "dtype"):
            np.testing.assert_allclose(lhs, rhs)


def _assert_rng_equal(left, right):
    for name in (
        "forward",
        "sam",
        "stochastic_rounding",
        "quantization",
        "controller",
    ):
        np.testing.assert_allclose(getattr(left, name), getattr(right, name))


def test_stateful_sam_replays_same_key_and_commits_first_pass_state_only():
    plan = tiny_plan()
    bundle = _bundle(plan)
    opt_state = bundle.init(plan.trainable)
    model_state = {"updates": jnp.asarray(0, dtype=jnp.int32)}
    rng = rfft.init_rng_streams(42)
    step = rfft.make_stateful_sam_step(
        plan=plan,
        base_optimizer=bundle,
        loss_fn=_loss_with_state,
        has_aux=True,
    )

    params, opt_state, model_state, rng_after, info = step(
        plan.trainable,
        opt_state,
        model_state,
        rng,
    )

    assert bool(info.all_finite)
    assert int(info.failed_pass) == 0
    assert model_state["updates"] == 1
    assert info.proposed_model_state["updates"] == 1
    assert info.perturbed_model_state["updates"] == 1
    np.testing.assert_allclose(info.aux["sample"], info.perturbed_aux["sample"])
    assert not jnp.array_equal(rng_after.sam, rng.sam)
    np.testing.assert_allclose(rng_after.forward, rng.forward)
    assert not jnp.allclose(params["head"]["w"], plan.trainable["head"]["w"])
    assert opt_state is not None


def test_stateful_sam_skips_transaction_when_second_pass_is_nonfinite():
    plan = tiny_plan()
    bundle = _bundle(plan)
    opt_state = bundle.init(plan.trainable)
    old_opt_state = opt_state
    model_state = {"updates": jnp.asarray(0, dtype=jnp.int32)}
    rng = rfft.init_rng_streams(42)
    base_head_sum = jnp.sum(plan.trainable["head"]["w"])

    def loss_fn(model, model_state, key):
        del key
        loss = jnp.sum(model["head"]["w"] ** 2)
        perturbed = jnp.abs(jnp.sum(model["head"]["w"]) - base_head_sum) > 1e-6
        loss = jnp.where(perturbed, jnp.asarray(jnp.inf), loss)
        return loss, {"updates": model_state["updates"] + 1}

    step = rfft.make_stateful_sam_step(
        plan=plan,
        base_optimizer=bundle,
        loss_fn=loss_fn,
    )
    params, opt_state, model_state, rng_after, info = step(
        plan.trainable,
        opt_state,
        model_state,
        rng,
    )

    assert not bool(info.all_finite)
    assert int(info.failed_pass) == 2
    assert model_state["updates"] == 0
    _assert_tree_allclose(params, plan.trainable)
    _assert_tree_allclose(opt_state, old_opt_state)
    _assert_rng_equal(rng_after, rng)


def test_stateful_sam_aggregate_policy_commits_first_pass_via_hook():
    plan = tiny_plan()
    bundle = _bundle(plan)
    opt_state = bundle.init(plan.trainable)
    model_state = {"updates": jnp.asarray(5, dtype=jnp.int32)}
    rng = rfft.init_rng_streams(42)
    step = rfft.make_stateful_sam_step(
        plan=plan,
        base_optimizer=bundle,
        loss_fn=_loss_with_aggregate_state,
        has_aux=True,
        state_policy="optimizer_step_aggregate",
        model_state_aggregator=_aggregate_updates,
    )

    _, _, model_state, _, info = step(
        plan.trainable,
        opt_state,
        model_state,
        rng,
    )

    assert bool(info.all_finite)
    assert model_state["updates"] == 6
    assert info.proposed_model_state["updates"] == 1
    assert info.perturbed_model_state["updates"] == 1


def test_stateful_sam_rejects_accumulation_and_aggregate_state_policy():
    plan = tiny_plan()

    with pytest.raises(ValueError, match="accumulation_steps=1"):
        rfft.make_stateful_sam_step(
            plan=plan,
            base_optimizer=_bundle(plan, accumulation_steps=2),
            loss_fn=_loss_with_state,
        )
    with pytest.raises(NotImplementedError, match="aggregation hook"):
        rfft.make_stateful_sam_step(
            plan=plan,
            base_optimizer=_bundle(plan),
            loss_fn=_loss_with_state,
            state_policy="optimizer_step_aggregate",
        )
