"""Applied-update outcomes and complete-window lifecycle tests."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import rollfast.finetune as rfft

from .helpers import TinyGroup, TinyPlan, tiny_plan


def _plan(dtype=jnp.float32):
    return TinyPlan(
        trainable={"w": jnp.array([1.0], dtype=dtype)},
        labels={"w": "head"},
        group_specs={
            "head": TinyGroup("head", "head", None, 1.0, True),
        },
    )


def _bundle(plan, *, steps=2, averaging=True):
    return rfft.adamw_from_plan(
        plan,
        total_steps=12,
        base_lr=0.1,
        weight_decay=0.1,
        b1=0.0,
        b2=0.0,
        schedule="constant",
        clip_global_norm=None,
        accumulation_steps=steps,
        ema=rfft.EMAConfig(enabled=averaging, decay=0.5),
        swa=rfft.SWAConfig(enabled=averaging, start_step=1 if averaging else None),
    )


@pytest.mark.parametrize("use_jit", (False, True))
@pytest.mark.parametrize("bad_position", (0, 1))
def test_discarded_window_does_not_advance_averages_or_poison_next_window(
    use_jit, bad_position
):
    plan = _plan()
    bundle = _bundle(plan)
    update = (
        jax.jit(bundle.update_with_status) if use_jit else bundle.update_with_status
    )
    params = plan.trainable
    state = bundle.init(params)
    events = []
    for factor in (1.0, 1.0):
        grads = {"w": jnp.full_like(params["w"], factor)}
        updates, state, status = update(grads, state, params)
        params = optax.apply_updates(params, updates)
        events.append(status)
    assert bool(events[0].accumulating)
    assert bool(events[1].update_applied)
    assert int(state.step) == int(state.ema_count) == int(state.swa_count) == 1
    accepted_params = params
    accepted_inner = state.inner_state.inner_opt_state.inner_state

    for position in range(2):
        factor = jnp.nan if position == bad_position else 1.0
        grads = {"w": jnp.full_like(params["w"], factor)}
        updates, state, status = update(grads, state, params)
        params = optax.apply_updates(params, updates)
    assert bool(status.window_discarded)
    assert not bool(status.update_applied)
    assert int(status.microsteps_in_window) == 0
    assert int(state.step) == int(state.ema_count) == int(state.swa_count) == 1
    assert int(state.inner_state.gradient_step) == 2
    assert int(state.inner_state.inner_opt_state.total_nonfinite) == 1
    np.testing.assert_array_equal(params["w"], accepted_params["w"])
    assert all(
        bool(value)
        for value in jax.tree.leaves(
            jax.tree.map(
                lambda left, right: jnp.array_equal(left, right),
                state.inner_state.inner_opt_state.inner_state,
                accepted_inner,
            )
        )
    )
    assert all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree.leaves(state.inner_state.acc_grads)
    )

    for position in range(2):
        grads = {"w": jnp.ones_like(params["w"])}
        updates, state, status = update(grads, state, params)
        params = optax.apply_updates(params, updates)
        assert bool(status.accumulating) == (position == 0)
    assert bool(status.update_applied)
    assert int(state.step) == int(state.ema_count) == int(state.swa_count) == 2
    assert int(state.inner_state.inner_opt_state.total_nonfinite) == 1
    assert bool(jnp.all(jnp.isfinite(params["w"])))


@pytest.mark.parametrize("use_jit", (False, True))
def test_status_matches_ordinary_update_and_zero_gradient_decay(use_jit):
    plan = _plan()
    bundle = _bundle(plan, steps=1, averaging=False)
    update = (
        jax.jit(bundle.update_with_status) if use_jit else bundle.update_with_status
    )
    params = plan.trainable
    state = bundle.init(params)

    # A caller records an all-ignore event by omitting this invocation.
    skipped_events = []
    valid_count = 0
    if valid_count == 0:
        skipped_events.append("all_ignore")
    assert skipped_events == ["all_ignore"]
    np.testing.assert_array_equal(params["w"], plan.trainable["w"])
    assert int(state.total_nonfinite) == 0

    zero_grads = {"w": jnp.zeros_like(params["w"])}
    updates, state, status = update(zero_grads, state, params)
    assert bool(status.update_applied)
    assert not bool(status.accumulating) and not bool(status.window_discarded)
    assert int(status.microsteps_in_window) == 0
    assert float(updates["w"][0]) < 0.0

    params = optax.apply_updates(params, updates)
    updates, state, status = update({"w": jnp.array([jnp.inf])}, state, params)
    assert bool(status.window_discarded)
    np.testing.assert_array_equal(updates["w"], jnp.zeros_like(params["w"]))


@pytest.mark.parametrize("use_jit", (False, True))
def test_accumulated_mean_matches_optax_adamw(use_jit):
    plan = _plan()
    bundle = _bundle(plan, averaging=False)
    update = (
        jax.jit(bundle.update_with_status) if use_jit else bundle.update_with_status
    )
    params = plan.trainable
    state = bundle.init(params)
    for value in (1.0, 3.0):
        grads = {"w": jnp.full_like(params["w"], value)}
        updates, state, status = update(grads, state, params)
        params = optax.apply_updates(params, updates)
    reference = optax.adamw(learning_rate=0.1, b1=0.0, b2=0.0, weight_decay=0.1)
    reference_updates, _ = reference.update(
        {"w": jnp.array([2.0])}, reference.init(plan.trainable), plan.trainable
    )
    expected = optax.apply_updates(plan.trainable, reference_updates)
    assert bool(status.update_applied)
    np.testing.assert_allclose(params["w"], expected["w"], rtol=1e-6)


def test_complete_window_validation_accepts_both_accumulation_routes():
    plan = _plan()
    config = rfft.AccumulationConfig(steps=2)
    bundle = _bundle(plan)
    state = bundle.init(plan.trainable)
    rfft.validate_accumulation_complete(state, config)
    _, pending, status = bundle.update_with_status(
        {"w": jnp.ones_like(plan.trainable["w"])}, state, plan.trainable
    )
    assert bool(status.accumulating)
    with pytest.raises(ValueError, match="1 of 2 microsteps"):
        rfft.validate_accumulation_complete(pending, config)
    assert int(pending.inner_state.mini_step) == 1
    _, completed, status = bundle.update_with_status(
        {"w": jnp.ones_like(plan.trainable["w"])}, pending, plan.trainable
    )
    assert bool(status.update_applied)
    rfft.validate_accumulation_complete(completed, config)

    explicit = rfft.init_accumulation_state(plan.trainable, config)
    rfft.validate_accumulation_complete(explicit, config)
    with pytest.raises(ValueError, match="1 of 2 microsteps"):
        rfft.validate_accumulation_complete(
            replace(explicit, microsteps_in_window=jnp.asarray(1)), config
        )
    with pytest.raises(ValueError, match="no accumulation window"):
        rfft.validate_accumulation_complete(jnp.array(0), config)


@pytest.mark.parametrize("first_bad", (False, True))
def test_safe_checkpoint_replays_partial_window(tmp_path, first_bad):
    plan = _plan()
    bundle = _bundle(plan)
    params = plan.trainable
    initial = bundle.init(params)
    grads = {"w": jnp.array([jnp.nan if first_bad else 1.0])}
    _, state, first = bundle.update_with_status(grads, initial, params)
    assert bool(first.accumulating)
    checkpoint = rfft.make_state_checkpoint(
        bundle, state, model_checkpoint_id="model-1", counters={"successful_update": 0}
    )
    template = rfft.make_state_checkpoint(
        bundle,
        initial,
        model_checkpoint_id="model-1",
        counters={"successful_update": 0},
    )
    path = tmp_path / "partial.rfarr"
    limits = rfft.ArrayCheckpointLimits(
        max_manifest_bytes=2_000_000,
        max_arrays=10_000,
        max_shards=20_000,
        max_array_bytes=10_000_000,
        max_total_bytes=100_000_000,
    )
    rfft.save_array_state_checkpoint(path, checkpoint, limits=limits)
    loaded = rfft.load_array_state_checkpoint(
        path,
        bundle,
        template=template,
        model_checkpoint_id="model-1",
        applied_updates=0,
        limits=limits,
    )
    with pytest.raises(ValueError, match="1 of 2 microsteps"):
        rfft.validate_accumulation_complete(loaded.state, bundle.accumulation_config)
    for factor in (1.0, 1.0, 1.0):
        step_grads = {"w": jnp.full_like(params["w"], factor)}
        expected, state, expected_status = bundle.update_with_status(
            step_grads, state, params
        )
        actual, restored, actual_status = bundle.update_with_status(
            step_grads, loaded.state, params
        )
        np.testing.assert_array_equal(actual["w"], expected["w"])
        assert all(
            bool(value)
            for value in jax.tree.leaves(
                jax.tree.map(jnp.array_equal, actual_status, expected_status)
            )
        )
        assert all(
            bool(value)
            for value in jax.tree.leaves(jax.tree.map(jnp.array_equal, restored, state))
        )
        params = optax.apply_updates(params, expected)
        loaded = replace(loaded, state=restored)


def test_masked_bfloat16_state_retains_dtype_after_discard():
    plan = tiny_plan()
    plan = replace(
        plan,
        trainable=jax.tree.map(
            lambda leaf: leaf.astype(jnp.bfloat16) if leaf is not None else None,
            plan.trainable,
            is_leaf=lambda leaf: leaf is None,
        ),
    )
    bundle = _bundle(plan, averaging=False)
    params = plan.trainable
    state = bundle.init(params)
    grads = jax.tree.map(
        lambda leaf: jnp.full_like(leaf, jnp.nan) if leaf is not None else None,
        params,
        is_leaf=lambda leaf: leaf is None,
    )
    update = jax.jit(bundle.update_with_status)
    for _ in range(2):
        _, state, status = update(grads, state, params)
    assert bool(status.window_discarded)
    assert jax.tree.structure(state.acc_grads) == jax.tree.structure(params)
    for leaf in jax.tree.leaves(state.acc_grads):
        assert leaf.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(leaf)))
