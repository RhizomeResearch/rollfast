"""Master-parameter precision step tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import rollfast.finetune as rfft

from .helpers import TinyGroup, TinyPlan


def _plan(dtype) -> TinyPlan:
    return TinyPlan(
        trainable={"w": jnp.asarray([1.0], dtype=dtype)},
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


def _bundle(plan: TinyPlan, precision: rfft.PrecisionConfig) -> rfft.OptimizerBundle:
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


def _loss_fn(params):
    return jnp.sum(params["w"].astype(jnp.float32) ** 2)


def _assert_tree_allclose(left, right):
    for lhs, rhs in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_allclose(lhs, rhs)


def test_master_params_auto_resolves_for_low_precision_only():
    fp32 = _plan(jnp.float32)
    bf16 = _plan(jnp.bfloat16)
    precision = rfft.PrecisionConfig(master_params="auto")

    assert rfft.make_master_params(fp32.trainable, precision) is None
    assert rfft.make_master_params(bf16.trainable, precision)["w"].dtype == jnp.float32


def test_master_update_step_keeps_fp32_master_and_casts_visible_params():
    plan = _plan(jnp.bfloat16)
    precision = rfft.PrecisionConfig(master_params="always")
    bundle = rfft.compile_optimizer(
        plan,
        optimizer=rfft.OptimizerConfig(base_lr=1e-3, weight_decay=0.0),
        schedule=rfft.ScheduleConfig(kind="constant", total_steps=10),
        gradient_policy=rfft.GradientPolicy(clip_global_norm=None),
        precision=precision,
    )
    master = rfft.make_master_params(plan.trainable, precision)
    state = bundle.init(master)

    def loss_fn(params):
        return jnp.sum(params["w"].astype(jnp.float32) ** 2)

    step = rfft.make_master_update_step(loss_fn, bundle)
    visible, master, state, loss = step(plan.trainable, master, state)

    assert loss > 0.0
    assert state is not None
    assert visible["w"].dtype == jnp.bfloat16
    assert master["w"].dtype == jnp.float32
    assert master["w"][0] < 1.0


def test_static_loss_scaled_master_step_matches_unscaled_step():
    plan = _plan(jnp.float16)
    base_precision = rfft.PrecisionConfig(master_params="always")
    scaled_precision = rfft.PrecisionConfig(
        master_params="always",
        loss_scale="static",
        static_loss_scale=8.0,
    )
    base_bundle = _bundle(plan, base_precision)
    scaled_bundle = _bundle(plan, scaled_precision)
    base_master = rfft.make_master_params(plan.trainable, base_precision)
    scaled_master = rfft.make_master_params(plan.trainable, scaled_precision)
    base_state = base_bundle.init(base_master)
    scaled_state = scaled_bundle.init(scaled_master)

    base_step = rfft.make_master_update_step(_loss_fn, base_bundle)
    scaled_step = rfft.make_loss_scaled_master_update_step(_loss_fn, scaled_bundle)

    base_visible, base_master, _, base_loss = base_step(
        plan.trainable,
        base_master,
        base_state,
    )
    (
        scaled_visible,
        scaled_master,
        _,
        scale_state,
        scaled_loss,
        all_finite,
    ) = scaled_step(
        plan.trainable,
        scaled_master,
        scaled_state,
        rfft.init_loss_scale_state(scaled_precision),
    )

    assert bool(all_finite)
    assert scale_state.loss_scale == 8.0
    np.testing.assert_allclose(scaled_loss, base_loss)
    _assert_tree_allclose(scaled_visible, base_visible)
    _assert_tree_allclose(scaled_master, base_master)


def test_dynamic_loss_scale_skips_nonfinite_update_and_backs_off():
    plan = _plan(jnp.float16)
    precision = rfft.PrecisionConfig(
        master_params="always",
        loss_scale="dynamic",
        static_loss_scale=8.0,
        backoff_factor=0.5,
    )
    bundle = _bundle(plan, precision)
    master = rfft.make_master_params(plan.trainable, precision)
    state = bundle.init(master)
    scale_state = rfft.init_loss_scale_state(precision)

    def nonfinite_loss(params):
        return jnp.sum(params["w"].astype(jnp.float32)) * jnp.asarray(jnp.inf)

    step = rfft.make_loss_scaled_master_update_step(nonfinite_loss, bundle)
    visible, new_master, new_state, scale_state, loss, all_finite = step(
        plan.trainable,
        master,
        state,
        scale_state,
    )

    assert not bool(all_finite)
    assert jnp.isinf(loss)
    np.testing.assert_allclose(scale_state.loss_scale, 4.0)
    _assert_tree_allclose(visible, plan.trainable)
    _assert_tree_allclose(new_master, master)
    _assert_tree_allclose(new_state, state)


def test_dynamic_loss_scale_grows_after_successful_updates():
    plan = _plan(jnp.float16)
    precision = rfft.PrecisionConfig(
        master_params="always",
        loss_scale="dynamic",
        static_loss_scale=2.0,
        growth_factor=2.0,
        growth_interval=2,
    )
    bundle = _bundle(plan, precision)
    master = rfft.make_master_params(plan.trainable, precision)
    state = bundle.init(master)
    scale_state = rfft.init_loss_scale_state(precision)
    step = rfft.make_loss_scaled_master_update_step(_loss_fn, bundle)

    visible, master, state, scale_state, _, all_finite = step(
        plan.trainable,
        master,
        state,
        scale_state,
    )
    assert bool(all_finite)
    np.testing.assert_allclose(scale_state.loss_scale, 2.0)
    assert scale_state.growth_tracker == 1

    _, _, _, scale_state, _, all_finite = step(
        visible,
        master,
        state,
        scale_state,
    )
    assert bool(all_finite)
    np.testing.assert_allclose(scale_state.loss_scale, 4.0)
    assert scale_state.growth_tracker == 0
