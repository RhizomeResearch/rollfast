"""Exact LossBundle accumulation tests."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import optax
import pytest

import rollfast.finetune as rfft

from .helpers import TinyGroup, TinyPlan


def _single_weight_plan() -> TinyPlan:
    return TinyPlan(
        trainable={"w": jnp.asarray(1.0, dtype=jnp.float32)},
        labels={"w": "w_no_decay"},
        group_specs={
            "w_no_decay": TinyGroup(
                "w_no_decay",
                role="head",
                depth=None,
                lr_multiplier=1.0,
                weight_decay=False,
            )
        },
    )


def test_loss_bundle_microbatch_accumulation_uses_true_normalizer():
    plan = _single_weight_plan()
    x = jnp.asarray([0.0, 2.0, 4.0], dtype=jnp.float32)
    normalizers = jnp.asarray([1.0, 3.0, 6.0], dtype=jnp.float32)
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        base_lr=1e-2,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
    )

    def scalar_loss(params, values, weights):
        residual = params["w"] - values
        return jnp.sum(weights * residual**2) / jnp.sum(weights)

    def bundled_loss(params, values, weights):
        residual = params["w"] - values
        loss_sum = jnp.sum(weights * residual**2)
        normalizer = jnp.sum(weights)
        return rfft.LossBundle(
            loss_sum=loss_sum,
            normalizer=normalizer,
            metrics_sums={},
            metric_normalizers={},
            new_model_state=None,
        )

    scalar_step = rfft.make_update_step(scalar_loss, bundle)
    bundle_step = rfft.make_loss_bundle_update_step(
        bundled_loss,
        bundle,
        microbatch_axis=0,
    )
    scalar_state = bundle.init(plan.trainable)
    bundle_state = bundle.init(plan.trainable)

    scalar_params, _, scalar_value = scalar_step(
        plan.trainable,
        scalar_state,
        x,
        normalizers,
    )
    bundle_params, _, bundle_value = bundle_step(
        plan.trainable,
        bundle_state,
        x,
        normalizers,
    )

    assert jnp.allclose(bundle_params["w"], scalar_params["w"])
    assert jnp.allclose(bundle_value.loss_sum, scalar_value)
    assert jnp.asarray(bundle_value.normalizer) == 1


@pytest.mark.parametrize("jit", [False, True])
def test_stateful_loss_bundle_accumulation_matches_full_weighted_update(jit):
    plan = _single_weight_plan()
    x = jnp.asarray([0.0, 2.0, 4.0], dtype=jnp.float32)
    normalizers = jnp.asarray([1.0, 3.0, 6.0], dtype=jnp.float32)
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        base_lr=1e-2,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
        b1=0.0,
        b2=0.0,
    )
    accumulation = rfft.AccumulationConfig(steps=3)

    def scalar_loss(params, values, weights):
        residual = params["w"] - values
        return jnp.sum(weights * residual**2) / jnp.sum(weights)

    def bundled_loss(params, value, weight):
        residual = params["w"] - value
        return rfft.LossBundle(
            loss_sum=weight * residual**2,
            normalizer=weight,
            metrics_sums={},
            metric_normalizers={},
            new_model_state=None,
        )

    scalar_step = rfft.make_update_step(scalar_loss, bundle)
    accumulating_step = rfft.make_accumulating_loss_bundle_update_step(
        bundled_loss,
        bundle,
        accumulation=accumulation,
    )
    if jit:
        accumulating_step = jax.jit(accumulating_step)
    scalar_state = bundle.init(plan.trainable)
    accum_state = bundle.init(plan.trainable)
    accumulation_state = rfft.init_accumulation_state(plan.trainable, accumulation)

    scalar_params, _, _ = scalar_step(
        plan.trainable,
        scalar_state,
        x,
        normalizers,
    )
    params = plan.trainable
    applied = []
    for value, weight in zip(x, normalizers, strict=True):
        params, accum_state, accumulation_state, info = accumulating_step(
            params,
            accum_state,
            accumulation_state,
            value,
            weight,
        )
        applied.append(bool(info.update_applied))

    assert applied == [False, False, True]
    assert accumulation_state.microsteps_in_window == 0
    assert accumulation_state.normalizer == 0.0
    assert jnp.allclose(params["w"], scalar_params["w"])


def test_accumulation_state_initializes_complete_loss_bundle_storage():
    plan = _single_weight_plan()
    accumulation = rfft.AccumulationConfig(steps=2, accumulate_dtype=jnp.float32)

    state = rfft.init_accumulation_state(plan.trainable, accumulation)

    assert state.loss_sum == 0.0
    assert state.loss_sum.dtype == accumulation.accumulate_dtype
    assert state.normalizer.dtype == accumulation.accumulate_dtype
    assert state.aux_sums == {}
    assert state.aux_normalizers == {}
    assert not bool(state.pending_model_state_valid)


def test_stateful_loss_bundle_accumulation_returns_complete_weighted_bundle():
    plan = _single_weight_plan()
    optimizer = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        base_lr=1e-2,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
    )
    accumulation = rfft.AccumulationConfig(steps=2)

    def bundled_loss(params, value, weight, aux_weight):
        residual = params["w"] - value
        return rfft.LossBundle(
            loss_sum=weight * residual**2,
            normalizer=weight,
            metrics_sums={"absolute_error": weight * jnp.abs(residual)},
            metric_normalizers={"absolute_error": weight},
            new_model_state=None,
            aux_sums={"penalty": aux_weight * params["w"] ** 2},
            aux_normalizers={"penalty": aux_weight},
        )

    step = rfft.make_accumulating_loss_bundle_update_step(
        bundled_loss,
        optimizer,
        accumulation=accumulation,
    )
    params = plan.trainable
    optimizer_state = optimizer.init(params)
    state = rfft.init_accumulation_state(params, accumulation)
    inputs = (
        (jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(2.0)),
        (jnp.asarray(3.0), jnp.asarray(3.0), jnp.asarray(4.0)),
    )

    for value, weight, aux_weight in inputs:
        params, optimizer_state, state, info = step(
            params,
            optimizer_state,
            state,
            value,
            weight,
            aux_weight,
        )

    assert bool(info.update_applied)
    assert jnp.allclose(info.loss_bundle.loss_sum, 3.25)
    assert jnp.allclose(info.loss_bundle.metrics_sums["absolute_error"], 1.75)
    assert jnp.allclose(info.loss_bundle.aux_sums["penalty"], 1.0)
    assert info.loss_bundle.normalizer == 1.0
    assert info.loss_bundle.metric_normalizers["absolute_error"] == 1.0
    assert info.loss_bundle.aux_normalizers["penalty"] == 1.0
    assert state.loss_sum == 0.0
    assert state.normalizer == 0.0
    assert state.metric_sums["absolute_error"] == 0.0
    assert state.aux_sums["penalty"] == 0.0


def test_stateful_loss_bundle_accumulation_aggregates_model_state_payloads():
    plan = _single_weight_plan()
    x = jnp.asarray([0.0, 2.0, 4.0, 6.0], dtype=jnp.float32)
    normalizers = jnp.asarray([1.0, 3.0, 6.0, 2.0], dtype=jnp.float32)
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        base_lr=1e-2,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
        b1=0.0,
        b2=0.0,
    )
    accumulation = rfft.AccumulationConfig(steps=2)

    def aggregate(left, right):
        return {
            "count": left["count"] + right["count"],
            "normalizer": left["normalizer"] + right["normalizer"],
        }

    def bundled_loss(params, value, weight):
        residual = params["w"] - value
        return rfft.LossBundle(
            loss_sum=weight * residual**2,
            normalizer=weight,
            metrics_sums={},
            metric_normalizers={},
            new_model_state={
                "count": jnp.asarray(1, dtype=jnp.int32),
                "normalizer": weight,
            },
        )

    accumulating_step = rfft.make_accumulating_loss_bundle_update_step(
        bundled_loss,
        bundle,
        accumulation=accumulation,
        model_state_aggregator=aggregate,
    )
    params = plan.trainable
    state = bundle.init(params)
    accumulation_state = rfft.init_accumulation_state(params, accumulation)
    completed_counts = []
    for value, weight in zip(x, normalizers, strict=True):
        params, state, accumulation_state, info = accumulating_step(
            params,
            state,
            accumulation_state,
            value,
            weight,
        )
        if info.update_applied:
            completed_counts.append(int(info.loss_bundle.new_model_state["count"]))

    assert bool(info.update_applied)
    assert completed_counts == [2, 2]
    assert info.loss_bundle.new_model_state["count"] == 2
    assert jnp.allclose(
        info.loss_bundle.new_model_state["normalizer"],
        jnp.sum(normalizers[-2:]),
    )
    assert not bool(accumulation_state.pending_model_state_valid)


def test_stateful_loss_bundle_accumulation_discards_nonfinite_window():
    plan = _single_weight_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        base_lr=1e-2,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
        b1=0.0,
        b2=0.0,
    )
    accumulation = rfft.AccumulationConfig(steps=2)

    def aggregate(left, right):
        return {"count": left["count"] + right["count"]}

    def bundled_loss(params, multiplier):
        return rfft.LossBundle(
            loss_sum=params["w"] * multiplier,
            normalizer=jnp.asarray(1.0, dtype=jnp.float32),
            metrics_sums={"value": multiplier},
            metric_normalizers={"value": jnp.asarray(1.0, dtype=jnp.float32)},
            new_model_state={"count": jnp.asarray(1, dtype=jnp.int32)},
            aux_sums={"penalty": multiplier},
            aux_normalizers={"penalty": jnp.asarray(1.0, dtype=jnp.float32)},
        )

    state = bundle.init(plan.trainable)
    old_state = state
    accumulation_state = rfft.init_accumulation_state(plan.trainable, accumulation)
    step = rfft.make_accumulating_loss_bundle_update_step(
        bundled_loss,
        bundle,
        accumulation=accumulation,
        model_state_aggregator=aggregate,
    )

    params, state, accumulation_state, info = step(
        plan.trainable,
        state,
        accumulation_state,
        jnp.asarray(jnp.inf),
    )
    params, state, accumulation_state, info = step(
        params,
        state,
        accumulation_state,
        jnp.asarray(1.0, dtype=jnp.float32),
    )

    assert not bool(info.update_applied)
    assert not bool(info.all_finite)
    assert accumulation_state.microsteps_in_window == 0
    assert accumulation_state.loss_sum == 0.0
    assert accumulation_state.normalizer == 0.0
    assert accumulation_state.metric_sums["value"] == 0.0
    assert accumulation_state.metric_normalizers["value"] == 0.0
    assert accumulation_state.aux_sums["penalty"] == 0.0
    assert accumulation_state.aux_normalizers["penalty"] == 0.0
    assert not bool(accumulation_state.pending_model_state_valid)
    assert jnp.allclose(params["w"], plan.trainable["w"])
    for left, right in zip(
        jax.tree.leaves(state), jax.tree.leaves(old_state), strict=True
    ):
        if hasattr(left, "dtype"):
            assert jnp.allclose(left, right)

    for _ in range(2):
        params, state, accumulation_state, info = step(
            params,
            state,
            accumulation_state,
            jnp.asarray(1.0, dtype=jnp.float32),
        )

    assert bool(info.update_applied)
    assert bool(info.all_finite)
    assert info.loss_bundle.loss_sum == 1.0
    assert info.loss_bundle.metrics_sums["value"] == 1.0
    assert info.loss_bundle.aux_sums["penalty"] == 1.0
    assert info.loss_bundle.new_model_state["count"] == 2
    assert not bool(accumulation_state.pending_model_state_valid)


def test_jitted_accumulation_replaces_invalid_model_state_payload():
    plan = _single_weight_plan()
    optimizer = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        base_lr=1e-2,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
        b1=0.0,
        b2=0.0,
    )
    accumulation = rfft.AccumulationConfig(steps=2)

    def aggregate(left, right):
        return {"count": left["count"] + right["count"]}

    def bundled_loss(params, multiplier):
        return rfft.LossBundle(
            loss_sum=params["w"] * multiplier,
            normalizer=jnp.asarray(1.0, dtype=jnp.float32),
            metrics_sums={},
            metric_normalizers={},
            new_model_state={"count": jnp.asarray(1, dtype=jnp.int32)},
        )

    step = jax.jit(
        rfft.make_accumulating_loss_bundle_update_step(
            bundled_loss,
            optimizer,
            accumulation=accumulation,
            model_state_aggregator=aggregate,
        )
    )
    params = plan.trainable
    optimizer_state = optimizer.init(params)
    state = rfft.init_accumulation_state(
        params,
        accumulation,
        pending_model_state={"count": jnp.asarray(0, dtype=jnp.int32)},
    )

    completed_counts = []
    for _ in range(4):
        params, optimizer_state, state, info = step(
            params,
            optimizer_state,
            state,
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        if info.update_applied:
            completed_counts.append(int(info.loss_bundle.new_model_state["count"]))

    assert completed_counts == [2, 2]
    assert not bool(state.pending_model_state_valid)


def test_jitted_accumulation_executes_optimizer_only_at_finite_boundaries():
    plan = _single_weight_plan()
    optimizer = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        base_lr=1e-2,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
    )
    callback_count = 0

    def count_update(_):
        nonlocal callback_count
        callback_count += 1

    def counting_update(updates, state, params=None):
        del params
        jax.debug.callback(count_update, jnp.asarray(0), ordered=True)
        return updates, state

    counting_tx = optax.GradientTransformation(
        lambda params: (),
        counting_update,
    )
    optimizer = replace(optimizer, tx=optax.chain(counting_tx, optimizer.tx))
    accumulation = rfft.AccumulationConfig(steps=3)

    def bundled_loss(params, multiplier):
        return rfft.LossBundle(
            loss_sum=params["w"] * multiplier,
            normalizer=jnp.asarray(1.0, dtype=jnp.float32),
            metrics_sums={},
            metric_normalizers={},
            new_model_state=None,
        )

    step = jax.jit(
        rfft.make_accumulating_loss_bundle_update_step(
            bundled_loss,
            optimizer,
            accumulation=accumulation,
        )
    )

    def run_window(multipliers):
        params = plan.trainable
        optimizer_state = optimizer.init(params)
        state = rfft.init_accumulation_state(params, accumulation)
        observed_counts = []
        for multiplier in multipliers:
            params, optimizer_state, state, info = step(
                params,
                optimizer_state,
                state,
                jnp.asarray(multiplier, dtype=jnp.float32),
            )
            jax.block_until_ready((params, optimizer_state, state, info))
            jax.effects_barrier()
            observed_counts.append(callback_count)
        return observed_counts

    assert run_window((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)) == [0, 0, 1, 1, 1, 2]
    callback_count = 0
    assert run_window((jnp.inf, 1.0, 1.0)) == [0, 0, 0]


def test_stateful_loss_bundle_accumulation_rejects_pre_wrapped_optimizer():
    plan = _single_weight_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
        accumulation_steps=2,
    )

    def bundled_loss(params):
        return rfft.LossBundle(
            loss_sum=params["w"] ** 2,
            normalizer=jnp.asarray(1.0, dtype=jnp.float32),
            metrics_sums={},
            metric_normalizers={},
            new_model_state=None,
        )

    with pytest.raises(ValueError, match="accumulation steps=1"):
        rfft.make_accumulating_loss_bundle_update_step(
            bundled_loss,
            bundle,
            accumulation=rfft.AccumulationConfig(steps=2),
        )
