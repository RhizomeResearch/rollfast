"""Behavior and work avoidance in fine-tuning performance paths."""

import importlib
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import rollfast.finetune as rfft
from rollfast.finetune import step as steps
from rollfast.optim.adam8 import quantize_blocks

from .helpers import TinyGroup, TinyPlan, tiny_plan


def _assert_close(actual, expected, *, rtol=1e-6):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert jnp.asarray(a).dtype == jnp.asarray(b).dtype
        np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-6)


def _bundle():
    return rfft.adamw_from_plan(
        tiny_plan(), total_steps=10, schedule="constant", clip_global_norm=None
    )


def test_adalora_allocates_only_at_cadence_and_final_boundary(monkeypatch):
    module = importlib.import_module("rollfast.finetune.adalora")
    controller = rfft.make_adalora_controller(
        {"a": 4, "b": 4},
        total_steps=10,
        config=rfft.AdaLoRAControllerConfig(
            initial_budget=8,
            target_budget=4,
            t_init=2,
            t_final=2,
            allocation_interval=3,
        ),
    )
    state = controller.init()
    original = module.allocate_rank_mask
    calls = []

    def observed(*args, **kwargs):
        jax.debug.callback(lambda: calls.append(True))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "allocate_rank_mask", observed)
    update = jax.jit(controller.update)
    importance = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    for step in range(1, 11):
        previous = state
        state = jax.block_until_ready(update(state, importance))
        if step not in (2, 5, 8):
            np.testing.assert_array_equal(
                state.current_support, previous.current_support
            )
        assert len(calls) == sum(boundary <= step for boundary in (2, 5, 8))
        assert int(state.last_allocation_step) == step
    skipped = jax.block_until_ready(update(state, importance, applied=False))
    _assert_close(skipped, state)
    assert len(calls) == 3


@pytest.mark.parametrize("kind", ["ema", "swa"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_averaging_skips_casts_and_preserves_cadence(monkeypatch, kind, dtype):
    module = importlib.import_module("rollfast.finetune.averaging")
    calls = []
    original = module._cast_tree

    def observed(*args):
        jax.debug.callback(lambda: calls.append(True))
        return original(*args)

    monkeypatch.setattr(module, "_cast_tree", observed)
    config = (
        rfft.EMAConfig(
            enabled=True, state_dtype=dtype, start_step=2, update_every=3, decay=0.5
        )
        if kind == "ema"
        else rfft.SWAConfig(enabled=True, state_dtype=dtype, frequency=3)
    )

    def update(old, count, params, step, applied):
        if kind == "ema":
            return module._update_ema(
                old,
                count,
                params,
                config,
                step,
                applied,
                mask={"w": True, "none": None},
            )
        return module._update_swa(
            old, count, params, config, step, applied, swa_start_step=2
        )

    update = jax.jit(update)
    old = {"w": jnp.ones(4, dtype=dtype), "none": None}
    params = {"w": jnp.full(4, 3.0, dtype=jnp.float32), "none": None}
    for step, applied in ((1, True), (2, False), (3, True)):
        result = jax.block_until_ready(update(old, jnp.array(0), params, step, applied))
        _assert_close(result, (old, jnp.array(0)))
    assert not calls
    result, count = jax.block_until_ready(update(old, jnp.array(0), params, 2, True))
    expected = jnp.full(4, 2.0 if kind == "ema" else 3.0, dtype=dtype)
    np.testing.assert_array_equal(result["w"], expected)
    assert int(count) == 1 and len(calls) == 1


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_sam_scan_retains_sum_order_broadcasting_and_aux(dtype, reduction):
    def loss(params, sample, offset, *, scale):
        value = (
            jnp.sum((params["w"].astype(jnp.float32) * sample + offset) ** 2) * scale
        )
        return value, {"sample_sum": jnp.sum(sample), "none": None}

    value_and_grad = jax.value_and_grad(loss, has_aux=True)
    params = {"w": jnp.asarray(0.5, dtype=dtype), "none": None}
    batch = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    offset = jnp.asarray([1.0, 2.0])

    def evaluate(params, batch):
        return steps._evaluate_sam_value_and_grad(
            value_and_grad,
            params,
            (batch, offset),
            {"scale": jnp.asarray(2.0)},
            microbatch_axis=1,
            microbatch_count=batch.shape[1],
            microbatch_reduction=reduction,
        )

    expected = value_and_grad(params, batch[:, 0], offset, scale=2.0)
    for index in range(1, batch.shape[1]):
        item = value_and_grad(params, batch[:, index], offset, scale=2.0)
        expected = jax.tree.map(lambda a, b: a + b, expected, item)
    if reduction == "mean":
        expected = jax.tree.map(lambda x: x * 0.25, expected)
    for fn in (evaluate, jax.jit(evaluate)):
        _assert_close(
            fn(params, batch), expected, rtol=0.01 if dtype == jnp.bfloat16 else 1e-6
        )
    differentiated = jax.jit(jax.grad(lambda p: evaluate(p, batch)[0][0]))(params)
    _assert_close(
        differentiated, expected[1], rtol=0.01 if dtype == jnp.bfloat16 else 1e-6
    )
    small = jax.make_jaxpr(evaluate)(params, batch).jaxpr
    large = jax.make_jaxpr(evaluate)(params, jnp.tile(batch, (1, 8))).jaxpr
    assert len(small.eqns) == len(large.eqns)
    assert any(eqn.primitive.name == "scan" for eqn in large.eqns)


def test_loss_bundle_scan_preserves_all_totals_and_last_model_state():
    params = {"w": jnp.asarray(0.5)}
    values = jnp.arange(1, 5, dtype=jnp.float32)

    def loss(params, value):
        return rfft.LossBundle(
            loss_sum=(params["w"] - value) ** 2,
            normalizer=value,
            metrics_sums={"metric": value**2},
            metric_normalizers={"metric": value},
            new_model_state={"last": value},
            aux_sums={"aux": value * 2},
            aux_normalizers={"aux": jnp.asarray(1.0)},
        )

    def evaluate(params, values):
        return steps._evaluate_loss_bundle_value_and_grad(
            jax.value_and_grad(steps._loss_bundle_sum, has_aux=True),
            loss,
            params,
            (values,),
            {},
            microbatch_axis=0,
            microbatch_count=None,
        )

    for fn in (evaluate, jax.jit(evaluate)):
        bundle, grads = fn(params, values)
        np.testing.assert_allclose(
            bundle.loss_sum, jnp.sum((params["w"] - values) ** 2)
        )
        np.testing.assert_array_equal(bundle.normalizer, jnp.sum(values))
        np.testing.assert_array_equal(bundle.metrics_sums["metric"], jnp.sum(values**2))
        np.testing.assert_array_equal(
            bundle.metric_normalizers["metric"], jnp.sum(values)
        )
        np.testing.assert_array_equal(bundle.aux_sums["aux"], jnp.sum(values) * 2)
        assert float(bundle.aux_normalizers["aux"]) == 4.0
        assert float(bundle.new_model_state["last"]) == 4.0
        np.testing.assert_allclose(grads["w"], jnp.sum(2 * (params["w"] - values)))


@pytest.mark.parametrize("state_dtype", [jnp.float32, jnp.bfloat16])
def test_rejected_scaled_steps_bypass_optimizer_and_preserve_promoted_state(
    state_dtype,
):
    def update(grads, state, params=None):
        return (
            jax.tree.map(lambda g: -0.1 * jnp.sin(g), grads),
            state.astype(jnp.float32) + 1,
        )

    bundle = replace(
        _bundle(),
        tx=optax.GradientTransformation(
            lambda _: jnp.array(3.0, dtype=state_dtype), update
        ),
    )
    visible = {"w": jnp.array(1.0, dtype=jnp.bfloat16)}
    master = {"w": jnp.array(1.0)}
    state = bundle.init(master)

    def apply(g, loss):
        return steps._apply_scaled_master_grads(
            bundle, visible, master, state, g, jnp.array(2.0), loss
        )

    graph = jax.make_jaxpr(apply)({"w": jnp.array(2.0)}, jnp.array(1.0)).jaxpr
    conditional = next(eqn for eqn in graph.eqns if eqn.primitive.name == "cond")
    rejected, accepted = conditional.params["branches"]
    assert any(eqn.primitive.name == "sin" for eqn in accepted.jaxpr.eqns)
    assert not any(eqn.primitive.name == "sin" for eqn in rejected.jaxpr.eqns)
    apply = jax.jit(apply)
    for value, loss in ((float("nan"), 1.0), (float("inf"), 1.0), (2.0, float("nan"))):
        result = jax.block_until_ready(apply({"w": jnp.array(value)}, jnp.array(loss)))
        _assert_close(result[:2], (visible, master))
        assert result[2].dtype == jnp.float32 and float(result[2]) == 3.0
        assert not bool(result[3])
    result = jax.block_until_ready(apply({"w": jnp.array(2.0)}, jnp.array(1.0)))
    assert float(result[2]) == 4.0 and bool(result[3])


def test_rejected_scaled_steps_preserve_custom_optimizer_effects():
    calls = []

    def update(grads, state, params=None):
        jax.debug.callback(lambda: calls.append(True))
        return grads, state + 1

    bundle = replace(
        _bundle(), tx=optax.GradientTransformation(lambda _: jnp.array(0), update)
    )
    apply = jax.jit(
        lambda grad: steps._apply_scaled_master_grads(
            bundle,
            jnp.array(1.0),
            jnp.array(1.0),
            jnp.array(0),
            grad,
            jnp.array(1.0),
            jnp.array(1.0),
        )
    )
    visible, master, state, finite = jax.block_until_ready(apply(jnp.array(jnp.nan)))
    assert len(calls) == 1
    assert float(visible) == float(master) == 1.0
    assert int(state) == 0 and not bool(finite)


def test_offload_batches_only_eligible_leaves(monkeypatch):
    state = {
        "large": jnp.arange(16),
        "small": jnp.array(1),
        "none": None,
        "quantized": quantize_blocks(jnp.arange(64, dtype=jnp.float32), block_size=8),
    }
    original = jax.device_get
    calls = []

    def observed(value):
        calls.append(len(jax.tree.leaves(value)))
        return original(value)

    monkeypatch.setattr(jax, "device_get", observed)
    actual = rfft.offload_optimizer_state(
        state, policy=rfft.StateOffloadPolicy(enabled=True, min_bytes=16)
    )
    assert calls == [3]
    assert actual["small"] is state["small"]
    assert actual["none"] is None
    assert isinstance(actual["large"], np.ndarray)
    assert isinstance(actual["quantized"].values, np.ndarray)
    _assert_close(actual, state)


def test_factorized_adamw_evaluates_each_group_schedule_once(monkeypatch):
    module = importlib.import_module("rollfast.finetune.builders")
    original = module.build_schedule
    calls = []

    def observed(*args, **kwargs):
        schedule = original(*args, **kwargs)

        def evaluate(count):
            calls.append(True)
            return schedule(count)

        return evaluate

    monkeypatch.setattr(module, "build_schedule", observed)
    plan = TinyPlan(
        trainable={"a": jnp.ones(3), "b": jnp.ones(2)},
        labels={"a": "head", "b": "head"},
        group_specs={"head": TinyGroup("head", "head", None, 1.0, False)},
    )
    bundle = rfft.compile_optimizer(
        plan,
        schedule=rfft.ScheduleConfig(kind="warmup_cosine", total_steps=10),
        gradient_policy=rfft.GradientPolicy(clip_global_norm=None, nonfinite="none"),
    )
    state = bundle.init(plan.trainable)
    calls.clear()
    bundle.update(plan.trainable, state, plan.trainable)
    assert calls == [True]
    calls.clear()
    jax.make_jaxpr(bundle.update)(plan.trainable, state, plan.trainable)
    assert calls == [True]
