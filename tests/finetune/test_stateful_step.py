"""Model-state and RNG-aware update-step tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rollfast.finetune as rfft

from .helpers import tiny_plan


def _bundle(plan, precision):
    return rfft.compile_optimizer(
        plan,
        optimizer=rfft.OptimizerConfig(
            base_lr=1e-3,
            weight_decay=0.0,
            b1=0.0,
            b2=0.0,
        ),
        schedule=rfft.ScheduleConfig(kind="constant", total_steps=10),
        gradient_policy=rfft.GradientPolicy(clip_global_norm=None),
        precision=precision,
    )


def _loss_with_state(model, model_state, key):
    del key
    return (
        jnp.sum(model["head"]["w"].astype(jnp.float32) ** 2),
        {"updates": model_state["updates"] + 1},
    )


def _nonfinite_loss_with_state(model, model_state, key):
    del key
    return (
        jnp.sum(model["head"]["w"].astype(jnp.float32)) * jnp.asarray(jnp.inf),
        {"updates": model_state["updates"] + 1},
    )


def _loss_with_aggregate_state(model, model_state, key):
    del model_state, key
    return (
        jnp.sum(model["head"]["w"].astype(jnp.float32) ** 2),
        {"updates": jnp.asarray(1, dtype=jnp.int32)},
    )


def _nonfinite_loss_with_aggregate_state(model, model_state, key):
    del model_state, key
    return (
        jnp.sum(model["head"]["w"].astype(jnp.float32)) * jnp.asarray(jnp.inf),
        {"updates": jnp.asarray(1, dtype=jnp.int32)},
    )


def _aggregate_updates(old_state, proposed_state):
    return {"updates": old_state["updates"] + proposed_state["updates"]}


def _assert_tree_allclose(left, right):
    for lhs, rhs in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
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


def test_stateful_step_commits_model_state_and_advances_rng_on_success():
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(master_params="always")
    bundle = _bundle(plan, precision)
    master = rfft.make_master_params(plan.trainable, precision)
    state = bundle.init(master)
    model_state = {"updates": jnp.asarray(0, dtype=jnp.int32)}
    rng = rfft.init_rng_streams(0)
    step = rfft.make_stateful_loss_scaled_master_update_step(
        plan,
        _loss_with_state,
        bundle,
        state_policy="microbatch_sequential",
    )

    visible, master, state, model_state, scale_state, rng_after, info = step(
        plan.trainable,
        master,
        state,
        model_state,
        None,
        rng,
    )

    assert bool(info.all_finite)
    assert scale_state is None
    assert model_state["updates"] == 1
    assert info.proposed_model_state["updates"] == 1
    assert not jnp.array_equal(rng_after.forward, rng.forward)
    assert master["head"]["w"][0, 0] != plan.trainable["head"]["w"][0, 0]
    assert visible["head"]["w"][0, 0] != plan.trainable["head"]["w"][0, 0]


def test_stateful_step_skips_state_and_rng_on_nonfinite_update():
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(
        master_params="always",
        loss_scale="dynamic",
        static_loss_scale=8.0,
    )
    bundle = _bundle(plan, precision)
    master = rfft.make_master_params(plan.trainable, precision)
    state = bundle.init(master)
    model_state = {"updates": jnp.asarray(0, dtype=jnp.int32)}
    scale_state = rfft.init_loss_scale_state(precision)
    rng = rfft.init_rng_streams(0)
    step = rfft.make_stateful_loss_scaled_master_update_step(
        plan,
        _nonfinite_loss_with_state,
        bundle,
        state_policy="microbatch_sequential",
    )

    visible, new_master, new_state, model_state, scale_state, rng_after, info = step(
        plan.trainable,
        master,
        state,
        model_state,
        scale_state,
        rng,
    )

    assert not bool(info.all_finite)
    assert model_state["updates"] == 0
    assert info.proposed_model_state["updates"] == 1
    np.testing.assert_allclose(scale_state.loss_scale, 4.0)
    _assert_tree_allclose(visible, plan.trainable)
    _assert_tree_allclose(new_master, master)
    _assert_tree_allclose(new_state, state)
    _assert_rng_equal(rng_after, rng)


def test_stateful_step_rejects_aggregate_policy_without_hook():
    plan = tiny_plan()
    bundle = _bundle(plan, rfft.PrecisionConfig())

    with pytest.raises(NotImplementedError, match="aggregation hook"):
        rfft.make_stateful_loss_scaled_master_update_step(
            plan,
            _loss_with_state,
            bundle,
            state_policy="optimizer_step_aggregate",
        )


def test_stateful_step_aggregate_policy_commits_via_hook():
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(master_params="always")
    bundle = _bundle(plan, precision)
    master = rfft.make_master_params(plan.trainable, precision)
    state = bundle.init(master)
    model_state = {"updates": jnp.asarray(5, dtype=jnp.int32)}
    rng = rfft.init_rng_streams(0)
    step = rfft.make_stateful_loss_scaled_master_update_step(
        plan,
        _loss_with_aggregate_state,
        bundle,
        state_policy="optimizer_step_aggregate",
        model_state_aggregator=_aggregate_updates,
    )

    *_, model_state, _, _, info = step(
        plan.trainable,
        master,
        state,
        model_state,
        None,
        rng,
    )

    assert bool(info.all_finite)
    assert info.proposed_model_state["updates"] == 1
    assert model_state["updates"] == 6


def test_stateful_step_aggregate_policy_withholds_nonfinite_hook_result():
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(master_params="always")
    bundle = _bundle(plan, precision)
    master = rfft.make_master_params(plan.trainable, precision)
    state = bundle.init(master)
    model_state = {"updates": jnp.asarray(5, dtype=jnp.int32)}
    rng = rfft.init_rng_streams(0)
    step = rfft.make_stateful_loss_scaled_master_update_step(
        plan,
        _nonfinite_loss_with_aggregate_state,
        bundle,
        state_policy="optimizer_step_aggregate",
        model_state_aggregator=_aggregate_updates,
    )

    *_, model_state, _, _, info = step(
        plan.trainable,
        master,
        state,
        model_state,
        None,
        rng,
    )

    assert not bool(info.all_finite)
    assert info.proposed_model_state["updates"] == 1
    assert model_state["updates"] == 5


def test_finetune_step_state_initializes_master_loss_scale_accumulators_and_rng():
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(
        master_params="always",
        loss_scale="static",
        static_loss_scale=16.0,
    )
    bundle = _bundle(plan, precision)

    state = rfft.init_finetune_step_state(
        bundle,
        plan.trainable,
        model_state={"updates": jnp.asarray(0, dtype=jnp.int32)},
        rng_seed=123,
    )

    assert state.master_params is not None
    assert state.loss_scale.loss_scale == 16.0
    assert state.accumulation.normalizer == 0.0
    assert state.accumulation.microsteps_in_window == 0
    assert state.counters.microstep == 0
    assert state.rng.forward.shape == (2,)
    assert state.optimizer_state is not None
    for leaf in jax.tree.leaves(
        state.accumulation.grad_numerator,
        is_leaf=lambda x: x is None,
    ):
        if leaf is not None:
            assert leaf.dtype == bundle.accumulation_config.accumulate_dtype
            assert jnp.all(leaf == 0)


def test_finetune_update_step_advances_success_counters_and_commits_state():
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(master_params="always")
    bundle = _bundle(plan, precision)
    step_state = rfft.init_finetune_step_state(
        bundle,
        plan.trainable,
        model_state={"updates": jnp.asarray(0, dtype=jnp.int32)},
    )
    step = rfft.make_finetune_update_step(
        plan,
        _loss_with_state,
        bundle,
        state_policy="microbatch_sequential",
    )

    visible, step_state, info = step(plan.trainable, step_state)

    assert bool(info.all_finite)
    assert step_state.model_state["updates"] == 1
    assert step_state.counters.microstep == 1
    assert step_state.counters.attempted_update == 1
    assert step_state.counters.successful_update == 1
    assert step_state.counters.schedule_step == 1
    assert step_state.counters.rank_step == 1
    assert step_state.counters.average_step == 1
    assert step_state.counters.loss_scale_growth_step == 0
    assert step_state.accumulation.microsteps_in_window == 0
    assert visible["head"]["w"][0, 0] != plan.trainable["head"]["w"][0, 0]


def test_finetune_update_step_skips_success_counters_on_nonfinite_update():
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(
        master_params="always",
        loss_scale="dynamic",
        static_loss_scale=8.0,
    )
    bundle = _bundle(plan, precision)
    step_state = rfft.init_finetune_step_state(
        bundle,
        plan.trainable,
        model_state={"updates": jnp.asarray(0, dtype=jnp.int32)},
    )
    old_master = step_state.master_params
    old_optimizer_state = step_state.optimizer_state
    old_rng = step_state.rng
    step = rfft.make_finetune_update_step(
        plan,
        _nonfinite_loss_with_state,
        bundle,
        state_policy="microbatch_sequential",
    )

    visible, step_state, info = step(plan.trainable, step_state)

    assert not bool(info.all_finite)
    assert step_state.model_state["updates"] == 0
    assert step_state.counters.microstep == 1
    assert step_state.counters.attempted_update == 1
    assert step_state.counters.successful_update == 0
    assert step_state.counters.schedule_step == 0
    assert step_state.counters.rank_step == 0
    assert step_state.counters.average_step == 0
    assert step_state.counters.loss_scale_growth_step == 0
    np.testing.assert_allclose(step_state.loss_scale.loss_scale, 4.0)
    _assert_tree_allclose(visible, plan.trainable)
    _assert_tree_allclose(step_state.master_params, old_master)
    _assert_tree_allclose(step_state.optimizer_state, old_optimizer_state)
    _assert_rng_equal(step_state.rng, old_rng)


def test_finetune_update_step_rejects_internal_accumulation_until_stateful_path_exists():
    plan = tiny_plan()
    bundle = rfft.compile_optimizer(
        plan,
        optimizer=rfft.OptimizerConfig(weight_decay=0.0),
        schedule=rfft.ScheduleConfig(kind="constant", total_steps=10),
        gradient_policy=rfft.GradientPolicy(clip_global_norm=None),
        accumulation=rfft.AccumulationConfig(steps=2),
    )

    with pytest.raises(NotImplementedError, match="accumulation steps=1"):
        rfft.make_finetune_update_step(plan, _loss_with_state, bundle)
