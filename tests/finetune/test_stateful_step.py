"""Model-state and RNG-aware update-step tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rollfast.finetune as rfft

from .helpers import assert_rng_equal, assert_tree_allclose, tiny_plan


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
    assert_tree_allclose(visible, plan.trainable)
    assert_tree_allclose(new_master, master)
    assert_tree_allclose(new_state, state)
    assert_rng_equal(rng_after, rng)


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
    assert state.accumulation.loss_sum == 0.0
    assert state.accumulation.normalizer == 0.0
    assert state.accumulation.aux_sums == {}
    assert state.accumulation.aux_normalizers == {}
    assert state.accumulation.microsteps_in_window == 0
    assert not bool(state.accumulation.pending_model_state_valid)
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
    assert_tree_allclose(visible, plan.trainable)
    assert_tree_allclose(step_state.master_params, old_master)
    assert_tree_allclose(step_state.optimizer_state, old_optimizer_state)
    assert_rng_equal(step_state.rng, old_rng)


def test_gradient_cast_overflow_rolls_back_then_accepts_next_step():
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(
        master_params="always",
        gradient_dtype=jnp.float16,
        loss_scale="dynamic",
        static_loss_scale=8.0,
    )
    bundle = rfft.compile_optimizer(
        plan,
        optimizer=rfft.OptimizerConfig(base_lr=1e-3, weight_decay=0.0),
        schedule=rfft.ScheduleConfig(kind="constant", total_steps=10),
        gradient_policy=rfft.GradientPolicy(clip_global_norm=None),
        precision=precision,
        ema=rfft.EMAConfig(enabled=True, decay=0.5),
    )
    initial = rfft.init_finetune_step_state(
        bundle,
        plan.trainable,
        model_state={"updates": jnp.asarray(0)},
        rng_seed=42,
    )

    def loss(model, model_state, key, factor):
        del key
        return (
            jnp.sum(model["head"]["w"]) * factor,
            {"updates": model_state["updates"] + 1},
        )

    step = rfft.make_finetune_update_step(
        plan, loss, bundle, state_policy="microbatch_sequential"
    )
    rejected_params, rejected, info = step(plan.trainable, initial, jnp.asarray(1e10))
    assert jnp.isfinite(info.value)
    assert not bool(info.all_finite)
    assert_tree_allclose(rejected_params, plan.trainable)
    assert_tree_allclose(rejected.optimizer_state, initial.optimizer_state)
    assert rejected.model_state["updates"] == 0
    assert_rng_equal(rejected.rng, initial.rng)
    assert rejected.counters.attempted_update == 1
    assert rejected.counters.successful_update == 0
    assert rejected.counters.schedule_step == 0
    assert rejected.optimizer_state.ema_count == 0
    assert rejected.loss_scale.loss_scale == 4.0

    accepted_params, accepted, info = step(rejected_params, rejected, jnp.asarray(1.0))
    assert bool(info.all_finite)
    assert not jnp.array_equal(
        accepted_params["head"]["w"], plan.trainable["head"]["w"]
    )
    assert accepted.model_state["updates"] == 1
    assert accepted.counters.attempted_update == 2
    assert accepted.counters.successful_update == 1
    assert accepted.counters.schedule_step == 1
    assert accepted.optimizer_state.ema_count == 1
    assert not jnp.array_equal(accepted.rng.forward, initial.rng.forward)


def test_jitted_stateful_step_rejects_gradient_cast_overflow():
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(gradient_dtype=jnp.float16)
    bundle = _bundle(plan, precision)
    step = jax.jit(
        rfft.make_stateful_loss_scaled_master_update_step(
            plan,
            lambda model, state, key, factor: (
                jnp.sum(model["head"]["w"]) * factor,
                {"updates": state["updates"] + 1},
            ),
            bundle,
            state_policy="microbatch_sequential",
        )
    )
    initial_model_state = {"updates": jnp.asarray(0)}
    initial_opt_state = bundle.init(plan.trainable)
    result = step(
        plan.trainable,
        None,
        initial_opt_state,
        initial_model_state,
        None,
        None,
        jnp.asarray(1e10),
    )
    visible, _, opt_state, model_state, _, _, info = result
    assert not bool(info.all_finite)
    assert_tree_allclose(visible, plan.trainable)
    assert_tree_allclose(opt_state, initial_opt_state)
    assert model_state["updates"] == 0


def test_equinox_state_is_preserved_for_forward_rollback():
    eqx = pytest.importorskip("equinox")
    batch_norm, initial_state = eqx.nn.make_with_state(eqx.nn.BatchNorm)(
        1, axis_name="batch", mode="batch"
    )
    plan = tiny_plan()
    bundle = _bundle(plan, rfft.PrecisionConfig())

    def loss(model, model_state, key, factor):
        del key
        values, next_state = jax.vmap(
            lambda value: batch_norm(value, model_state),
            axis_name="batch",
            out_axes=(0, None),
        )(jnp.asarray([[1.0], [3.0]]))
        return jnp.sum(model["head"]["w"]) * jnp.mean(values**2) * factor, next_state

    step = rfft.make_stateful_loss_scaled_master_update_step(
        plan, loss, bundle, state_policy="microbatch_sequential"
    )
    initial_opt_state = bundle.init(plan.trainable)
    rejected = step(
        plan.trainable,
        None,
        initial_opt_state,
        initial_state,
        None,
        None,
        jnp.asarray(jnp.inf),
    )
    assert not bool(rejected[-1].all_finite)
    assert_tree_allclose(rejected[3], initial_state)
    assert len(jax.tree.leaves(initial_state)) == 3

    accepted = step(*rejected[:6], jnp.asarray(1.0))
    assert bool(accepted[-1].all_finite)
    assert len(jax.tree.leaves(accepted[3])) == 3
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(initial_state), jax.tree.leaves(accepted[3]), strict=True
        )
    )


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
