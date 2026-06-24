"""Exact LossBundle accumulation tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
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


def test_stateful_loss_bundle_accumulation_matches_full_weighted_update():
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


def test_stateful_loss_bundle_accumulation_aggregates_model_state_payloads():
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
    for value, weight in zip(x, normalizers, strict=True):
        params, state, accumulation_state, info = accumulating_step(
            params,
            state,
            accumulation_state,
            value,
            weight,
        )

    assert bool(info.update_applied)
    assert info.loss_bundle.new_model_state["count"] == 3
    assert jnp.allclose(
        info.loss_bundle.new_model_state["normalizer"],
        jnp.sum(normalizers),
    )


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

    def bundled_loss(params, multiplier):
        return rfft.LossBundle(
            loss_sum=params["w"] * multiplier,
            normalizer=jnp.asarray(1.0, dtype=jnp.float32),
            metrics_sums={},
            metric_normalizers={},
            new_model_state=None,
        )

    state = bundle.init(plan.trainable)
    old_state = state
    accumulation_state = rfft.init_accumulation_state(plan.trainable, accumulation)
    step = rfft.make_accumulating_loss_bundle_update_step(
        bundled_loss,
        bundle,
        accumulation=accumulation,
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
    assert jnp.allclose(params["w"], plan.trainable["w"])
    for left, right in zip(
        jax.tree.leaves(state), jax.tree.leaves(old_state), strict=True
    ):
        if hasattr(left, "dtype"):
            assert jnp.allclose(left, right)


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
