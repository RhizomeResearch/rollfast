"""Attempt-indexed schedule clock and skipped-attempt tests."""

from __future__ import annotations

import pickle
from dataclasses import replace

import jax
import jax.numpy as jnp
import optax
import pytest

import rollfast.finetune as rfft
from rollfast.finetune.builders import _FactorizedAdamWState

from .helpers import tiny_plan

_CLOCK_STATES = (optax.ScaleByScheduleState, _FactorizedAdamWState)
FAMILIES = ("adamw", "muon_adam")


def _schedule(step_counter):
    # Every slot of this WSD schedule has a distinct learning rate except the
    # plateau, and the last slot reaches the floor.
    return rfft.ScheduleConfig(
        kind="wsd",
        total_steps=8,
        warmup_steps=1,
        decay_steps=4,
        end_lr_ratio=0.01,
        step_counter=step_counter,
    )


def _build(family, step_counter, **kwargs):
    plan = tiny_plan()
    common = dict(
        total_steps=8,
        base_lr=1e-2,
        schedule=_schedule(step_counter),
        weight_decay=0.1,
        clip_global_norm=None,
        **kwargs,
    )
    if family == "adamw":
        return plan, rfft.adamw_from_plan(plan, **common)
    return plan, rfft.muon_adam_from_plan(plan, **common)


def _loss(model, model_state, key, scale):
    del key
    return scale * sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(model)), model_state


def _is_clock(node):
    return isinstance(node, _CLOCK_STATES)


def _clocks(state):
    return {
        int(node.count)
        for node in jax.tree.leaves(state, is_leaf=_is_clock)
        if _is_clock(node)
    }


def _with_clocks(state, value):
    return jax.tree.map(
        lambda node: (
            node._replace(count=jnp.full_like(node.count, value))
            if _is_clock(node)
            else node
        ),
        state,
        is_leaf=_is_clock,
    )


def _assert_trees_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for lhs, rhs in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jax.dtypes.issubdtype(lhs.dtype, jax.dtypes.prng_key):
            lhs, rhs = jax.random.key_data(lhs), jax.random.key_data(rhs)
        assert jnp.array_equal(lhs, rhs)


@pytest.mark.parametrize("family", FAMILIES)
def test_applied_updates_use_the_learning_rate_of_their_attempt_slot(family):
    plan, bundle = _build(family, "attempt")
    _, reference = _build(family, "optimizer")
    step = jax.jit(rfft.make_finetune_update_step(plan, _loss, bundle))
    skip = jax.jit(rfft.make_skip_attempt_step(bundle))
    reference_step = jax.jit(rfft.make_finetune_update_step(plan, _loss, reference))
    params = reference_params = plan.trainable
    state = rfft.init_finetune_step_state(bundle, params)
    reference_state = rfft.init_finetune_step_state(reference, reference_params)
    attempts = (
        "apply",
        "apply",
        "skip",
        "nonfinite",
        "apply",
        "skip",
        "apply",
        "apply",
    )

    for slot, attempt in enumerate(attempts):
        before_params, before_state = params, state
        if attempt == "skip":
            state = skip(state)
        else:
            scale = jnp.asarray(jnp.nan if attempt == "nonfinite" else 1.0)
            params, state, info = step(params, state, scale)
            assert bool(info.update_applied) == (attempt == "apply")

        if attempt == "apply":
            # The reference counts applied updates; forcing its clock to the
            # attempt slot gives the update that slot's learning rate.
            reference_state = replace(
                reference_state,
                optimizer_state=_with_clocks(reference_state.optimizer_state, slot),
            )
            reference_params, reference_state, _ = reference_step(
                reference_params, reference_state, jnp.asarray(1.0)
            )
            _assert_trees_equal(params, reference_params)
            _assert_trees_equal(
                _with_clocks(state.optimizer_state, 0),
                _with_clocks(reference_state.optimizer_state, 0),
            )
        else:
            # Moments, Adam's count, and weight decay leave parameters alone.
            _assert_trees_equal(params, before_params)
            _assert_trees_equal(
                _with_clocks(state.optimizer_state, 0),
                _with_clocks(before_state.optimizer_state, 0),
            )
        assert _clocks(state.optimizer_state) == {slot + 1}
        assert state.counters.attempted_update == slot + 1
        assert state.counters.schedule_step == slot + 1
        assert state.counters.microstep == slot + 1

    assert state.counters.successful_update == attempts.count("apply")


@pytest.mark.parametrize("family", FAMILIES)
def test_attempt_clock_without_skips_matches_optimizer_clock(family):
    plan, attempt_bundle = _build(family, "attempt")
    _, optimizer_bundle = _build(family, "optimizer")
    results = []
    for bundle in (attempt_bundle, optimizer_bundle):
        step = jax.jit(rfft.make_finetune_update_step(plan, _loss, bundle))
        params = plan.trainable
        state = rfft.init_finetune_step_state(bundle, params)
        for _ in range(3):
            params, state, _ = step(params, state, jnp.asarray(1.0))
        results.append((params, state))

    _assert_trees_equal(results[0], results[1])


def test_skipped_attempt_leaves_the_optimizer_clock_alone():
    plan, bundle = _build("adamw", "optimizer")
    state = rfft.init_finetune_step_state(bundle, plan.trainable)

    skipped = jax.jit(rfft.make_skip_attempt_step(bundle))(state)

    _assert_trees_equal(skipped.optimizer_state, state.optimizer_state)
    assert skipped.counters.microstep == 1
    assert skipped.counters.attempted_update == 1
    assert skipped.counters.successful_update == 0
    assert skipped.counters.schedule_step == 0


@pytest.mark.parametrize(("step_counter", "clock"), (("attempt", 1), ("optimizer", 0)))
def test_rejected_bundle_update_consumes_its_slot_under_attempt_clock(
    step_counter, clock
):
    plan, bundle = _build("adamw", step_counter)
    state = bundle.init(plan.trainable)
    nonfinite = jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.nan), plan.trainable)

    updates, next_state, status = jax.jit(bundle.update_with_status)(
        nonfinite, state, plan.trainable
    )

    assert not bool(status.update_applied)
    assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(updates))
    assert _clocks(next_state) == {clock}
    assert next_state.total_nonfinite == 1
    _assert_trees_equal(
        _with_clocks(next_state.inner_state, 0), _with_clocks(state.inner_state, 0)
    )


def test_resume_after_skipped_attempt_matches_uninterrupted_run():
    plan, bundle = _build("adamw", "attempt")
    step = jax.jit(rfft.make_finetune_update_step(plan, _loss, bundle))
    skip = jax.jit(rfft.make_skip_attempt_step(bundle))
    params, state, _ = step(
        plan.trainable,
        rfft.init_finetune_step_state(bundle, plan.trainable),
        jnp.asarray(1.0),
    )
    state = skip(state)

    checkpoint = pickle.loads(
        pickle.dumps(
            rfft.make_step_state_checkpoint(bundle, state, model_checkpoint_id="m")
        )
    )
    rfft.restore_state_checkpoint(bundle, checkpoint, model_checkpoint_id="m")
    resumed = rfft.step_state_from_checkpoint(checkpoint)

    def finish(params, state):
        params, state, _ = step(params, state, jnp.asarray(jnp.nan))
        params, state, _ = step(params, state, jnp.asarray(1.0))
        state = skip(state)
        params, state, _ = step(params, state, jnp.asarray(1.0))
        return params, state

    _assert_trees_equal(finish(params, resumed), finish(params, state))


def test_accumulation_window_is_one_attempt():
    plan, bundle = _build("adamw", "attempt", accumulation_steps=2)
    step = jax.jit(rfft.make_finetune_update_step(plan, _loss, bundle))
    params = plan.trainable
    state = rfft.init_finetune_step_state(bundle, params)

    clocks = []
    for scale in (1.0, 1.0, 1.0, jnp.nan, 1.0, 1.0):
        params, state, _ = step(params, state, jnp.asarray(scale))
        clocks.append(_clocks(state.optimizer_state))

    # Mid-window microbatches leave the clock; the discarded window consumes one.
    assert clocks == [{0}, {1}, {1}, {2}, {2}, {3}]
    assert state.counters.attempted_update == 3
    assert state.counters.schedule_step == 3
    assert state.counters.successful_update == 2
    with pytest.raises(NotImplementedError, match="accumulation"):
        rfft.make_skip_attempt_step(bundle)


def test_empty_loss_bundle_window_consumes_its_slot():
    plan, bundle = _build("adamw", "attempt")

    def bundled_loss(params, weight):
        return rfft.LossBundle(
            loss_sum=weight * sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(params)),
            normalizer=weight,
            metrics_sums={},
            metric_normalizers={},
            new_model_state=None,
        )

    accumulation = rfft.AccumulationConfig(steps=2)
    step = jax.jit(
        rfft.make_accumulating_loss_bundle_update_step(
            bundled_loss, bundle, accumulation=accumulation
        )
    )
    params = plan.trainable
    opt_state = bundle.init(params)
    window = rfft.init_accumulation_state(params, accumulation)

    applied = []
    for weight in (1.0, 1.0, 0.0, 0.0, 1.0, 1.0):
        before = params
        params, opt_state, window, info = step(
            params, opt_state, window, jnp.asarray(weight)
        )
        applied.append(bool(info.update_applied))
        if weight == 0.0:
            _assert_trees_equal(params, before)

    assert applied == [False, True, False, False, False, True]
    assert _clocks(opt_state) == {3}


@pytest.mark.parametrize(
    "build",
    (
        lambda plan, schedule: rfft.apollo_adamw_from_plan(
            plan, total_steps=8, schedule=schedule
        ),
        lambda plan, schedule: rfft.galore_adamw_from_plan(
            plan, galore=rfft.GaLoreConfig(rank=1), total_steps=8, schedule=schedule
        ),
        lambda plan, schedule: rfft.schedule_free_adam_from_plan(
            plan, total_steps=8, schedule=schedule
        ),
    ),
    ids=("apollo_adamw", "galore_adamw", "schedule_free_adam"),
)
def test_builders_with_shared_learning_rate_counts_reject_attempt_clock(build):
    plan = tiny_plan()
    build(plan, _schedule("optimizer"))

    with pytest.raises(NotImplementedError, match="step_counter='optimizer'"):
        build(plan, _schedule("attempt"))
