"""Precision modes of the composite fine-tuning steps and their reports."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rollfast.finetune as rfft

from .helpers import TinyGroup, TinyPlan, assert_rng_equal, assert_tree_allclose


def _plan(dtype, values=(1.0, -2.0, 0.5, 3.0)) -> TinyPlan:
    return TinyPlan(
        trainable={"w": jnp.asarray(values, dtype=dtype)},
        labels={"w": "head"},
        group_specs={"head": TinyGroup("head", "head", None, 1.0, False)},
    )


def _bundle(plan, precision, *, base_lr=1e-2, **kwargs) -> rfft.OptimizerBundle:
    return rfft.compile_optimizer(
        plan,
        optimizer=rfft.OptimizerConfig(base_lr=base_lr, weight_decay=0.0),
        schedule=rfft.ScheduleConfig(kind="constant", total_steps=100),
        gradient_policy=rfft.GradientPolicy(clip_global_norm=None),
        precision=precision,
        **kwargs,
    )


def _quadratic_loss(model, model_state, key, target):
    del key
    return jnp.sum((model["w"].astype(jnp.float32) - target) ** 2), model_state


def _scaled_linear_loss(model, model_state, key, factor):
    del key
    return jnp.sum(model["w"].astype(jnp.float32)) * factor, model_state


_MODES = {
    "full_precision": (jnp.float32, rfft.PrecisionConfig()),
    "master_mixed_precision": (jnp.bfloat16, rfft.bf16_mixed_precision()),
    "fp16_dynamic_loss_scale": (
        jnp.float16,
        rfft.fp16_mixed_precision(static_loss_scale=2.0**8),
    ),
    "master_free_stochastic": (jnp.bfloat16, rfft.bf16_master_free_precision()),
}


@pytest.mark.parametrize("mode", _MODES)
def test_finetune_update_step_trains_under_jit_and_scan_in_each_mode(mode):
    dtype, precision = _MODES[mode]
    plan = _plan(dtype)
    bundle = _bundle(plan, precision)
    state = rfft.init_finetune_step_state(bundle, plan.trainable, rng_seed=0)
    step = rfft.make_finetune_update_step(plan, _quadratic_loss, bundle)

    @jax.jit
    def train(params, state):
        def body(carry, _):
            params, state, info = step(*carry, jnp.zeros(4))
            return (params, state), info.value

        return jax.lax.scan(body, (params, state), length=20)

    (params, state), losses = train(plan.trainable, state)

    assert params["w"].dtype == dtype
    assert losses[-1] < losses[0]
    assert int(state.counters.successful_update) == 20
    keeps_masters = bundle.report.precision.master_params
    assert (state.master_params is not None) == keeps_masters
    if keeps_masters:
        assert state.master_params["w"].dtype == jnp.float32
        np.testing.assert_array_equal(
            params["w"], state.master_params["w"].astype(dtype)
        )


def test_master_free_stochastic_rounding_keeps_updates_below_bf16_spacing():
    plan = _plan(jnp.bfloat16, values=np.ones(4096))
    # Adam's constant-gradient step is the learning rate: a quarter of the
    # BF16 spacing below one, which round-to-nearest discards.
    learning_rate = 2.0**-10
    steps = 8
    means = {}
    for rounding in ("nearest", "stochastic"):
        precision = rfft.PrecisionConfig(
            master_params="never", parameter_rounding=rounding
        )
        bundle = _bundle(plan, precision, base_lr=learning_rate)
        state = rfft.init_finetune_step_state(bundle, plan.trainable, rng_seed=3)
        step = jax.jit(
            rfft.make_finetune_update_step(plan, _scaled_linear_loss, bundle)
        )
        params = plan.trainable
        for _ in range(steps):
            params, state, info = step(params, state, jnp.asarray(1.0))
            assert bool(info.update_applied)
        assert state.master_params is None
        assert params["w"].dtype == jnp.bfloat16
        means[rounding] = float(jnp.mean(params["w"].astype(jnp.float32)))

    assert means["nearest"] == 1.0
    np.testing.assert_allclose(
        1.0 - means["stochastic"], steps * learning_rate, rtol=0.05
    )


def test_stochastic_rounding_stream_advances_only_with_applied_updates():
    plan = _plan(jnp.bfloat16)
    bundle = _bundle(plan, rfft.bf16_master_free_precision())
    initial = rfft.init_finetune_step_state(bundle, plan.trainable, rng_seed=0)
    step = jax.jit(rfft.make_finetune_update_step(plan, _scaled_linear_loss, bundle))

    params, rejected, info = step(plan.trainable, initial, jnp.asarray(jnp.inf))
    assert not bool(info.update_applied)
    assert_tree_allclose(params, plan.trainable)
    assert_rng_equal(rejected.rng, initial.rng)

    params, accepted, info = step(params, rejected, jnp.asarray(1.0))
    assert bool(info.update_applied)
    assert not jnp.array_equal(
        accepted.rng.stochastic_rounding, initial.rng.stochastic_rounding
    )
    replayed, _, _ = step(plan.trainable, initial, jnp.asarray(1.0))
    np.testing.assert_array_equal(replayed["w"], params["w"])


def test_stochastic_rounding_requires_rng_and_an_rng_aware_helper():
    plan = _plan(jnp.bfloat16)
    bundle = _bundle(plan, rfft.bf16_master_free_precision())
    stateful = rfft.make_stateful_loss_scaled_master_update_step(
        plan, _scaled_linear_loss, bundle
    )

    with pytest.raises(ValueError, match="RNGStreams.stochastic_rounding"):
        stateful(
            plan.trainable,
            None,
            bundle.init(plan.trainable),
            None,
            None,
            None,
            jnp.asarray(1.0),
        )
    for make in (
        lambda: rfft.make_master_update_step(lambda p: jnp.sum(p["w"]), bundle),
        lambda: rfft.make_loss_scaled_master_update_step(
            lambda p: jnp.sum(p["w"]), bundle
        ),
        lambda: rfft.make_update_step(lambda p: jnp.sum(p["w"]), bundle),
    ):
        with pytest.raises(ValueError, match="parameter_rounding='stochastic'"):
            make()


def test_direct_update_helpers_warn_about_ignored_precision_policy():
    def loss(params):
        return jnp.sum(params["w"].astype(jnp.float32))

    masters = _bundle(_plan(jnp.bfloat16), rfft.bf16_mixed_precision())
    scaled = _bundle(_plan(jnp.float16), rfft.fp16_mixed_precision())

    with pytest.warns(UserWarning, match="ignores the bundle's master parameters;"):
        rfft.make_update_step(loss, masters)
    with pytest.warns(UserWarning, match="master parameters and loss scaling"):
        rfft.make_update_step(loss, scaled)


def test_candidate_overflow_is_rejected_without_lowering_the_loss_scale():
    # A finite gradient whose update leaves the float16 range must be rejected
    # without being mistaken for a gradient overflow.
    plan = _plan(jnp.float16, values=(65000.0,))
    precision = rfft.fp16_mixed_precision(static_loss_scale=8.0)
    bundle = _bundle(plan, precision, base_lr=1000.0)
    initial = rfft.init_finetune_step_state(bundle, plan.trainable)
    step = rfft.make_finetune_update_step(plan, _scaled_linear_loss, bundle)

    params, state, info = step(plan.trainable, initial, jnp.asarray(-1.0))

    assert bool(info.grads_finite)
    assert not bool(info.all_finite) and not bool(info.update_applied)
    assert_tree_allclose(params, plan.trainable)
    assert_tree_allclose(state.master_params, initial.master_params)
    assert float(state.loss_scale.loss_scale) == 8.0
    assert int(state.counters.attempted_update) == 1
    assert int(state.counters.successful_update) == 0


def test_dynamic_loss_scale_stays_within_its_bounds():
    plan = _plan(jnp.float16)
    precision = rfft.PrecisionConfig(
        loss_scale="dynamic",
        static_loss_scale=4.0,
        min_loss_scale=2.0,
        max_loss_scale=8.0,
        growth_interval=1,
    )
    bundle = _bundle(plan, precision)
    step = jax.jit(
        rfft.make_loss_scaled_master_update_step(
            lambda params, factor: jnp.sum(params["w"].astype(jnp.float32)) * factor,
            bundle,
        )
    )
    visible = plan.trainable
    master = rfft.make_master_params(visible, precision)
    opt_state = bundle.init(master)
    scale = rfft.init_loss_scale_state(precision)

    observed = []
    for factor in (1.0, 1.0, jnp.inf, jnp.inf, jnp.inf):
        visible, master, opt_state, scale, _, _ = step(
            visible, master, opt_state, scale, jnp.asarray(factor)
        )
        observed.append(float(scale.loss_scale))

    assert observed == [8.0, 8.0, 4.0, 2.0, 2.0]


def test_finetune_sam_step_matches_stateful_sam_without_masters():
    plan = _plan(jnp.float32)
    bundle = _bundle(plan, rfft.PrecisionConfig())
    rng = rfft.init_rng_streams(7)
    state = rfft.init_finetune_step_state(bundle, plan.trainable, rng=rng)
    target = jnp.full(4, 3.0)

    composite = rfft.make_finetune_sam_step(
        plan=plan, base_optimizer=bundle, loss_fn=_quadratic_loss
    )
    direct = rfft.make_stateful_sam_step(
        plan=plan, base_optimizer=bundle, loss_fn=_quadratic_loss
    )
    params, state, info = composite(plan.trainable, state, target)
    expected, opt_state, _, rng, _ = direct(
        plan.trainable, bundle.init(plan.trainable), None, rng, target
    )

    assert bool(info.update_applied)
    assert float(info.perturbed_loss) > float(info.loss)
    assert_tree_allclose(params, expected)
    assert_tree_allclose(state.optimizer_state, opt_state)
    assert_rng_equal(state.rng, rng)
    assert int(state.counters.successful_update) == 1


def test_finetune_sam_step_updates_masters_and_backs_off_on_overflow():
    plan = _plan(jnp.float16)
    precision = rfft.fp16_mixed_precision(static_loss_scale=8.0)
    bundle = _bundle(plan, precision)
    initial = rfft.init_finetune_step_state(bundle, plan.trainable)
    step = jax.jit(
        rfft.make_finetune_sam_step(
            plan=plan, base_optimizer=bundle, loss_fn=_quadratic_loss
        )
    )

    params, state, info = step(plan.trainable, initial, jnp.full(4, 3.0))
    assert bool(info.update_applied)
    assert params["w"].dtype == jnp.float16
    assert state.master_params["w"].dtype == jnp.float32
    np.testing.assert_array_equal(
        params["w"], state.master_params["w"].astype(jnp.float16)
    )
    assert not jnp.array_equal(state.master_params["w"], initial.master_params["w"])

    rejected_params, rejected, info = step(params, state, jnp.full(4, jnp.inf))
    assert not bool(info.all_finite)
    assert int(info.failed_pass) == 1
    assert_tree_allclose(rejected_params, params)
    assert_tree_allclose(rejected.master_params, state.master_params)
    assert float(rejected.loss_scale.loss_scale) == 4.0
    assert int(rejected.counters.successful_update) == 1


def test_finetune_steps_require_masters_in_step_state():
    plan = _plan(jnp.bfloat16)
    bundle = _bundle(plan, rfft.bf16_mixed_precision())
    state = replace(
        rfft.init_finetune_step_state(bundle, plan.trainable), master_params=None
    )
    step = rfft.make_finetune_update_step(plan, _quadratic_loss, bundle)

    with pytest.raises(ValueError, match="master_params is None"):
        step(plan.trainable, state, jnp.zeros(4))


def test_step_state_rejects_masters_with_less_precision_than_visible_params():
    plan = _plan(jnp.float32)
    bundle = _bundle(plan, rfft.PrecisionConfig(master_params="always"))

    with pytest.raises(ValueError, match="expected at least float32"):
        rfft.init_finetune_step_state(
            bundle,
            plan.trainable,
            master_params={"w": plan.trainable["w"].astype(jnp.bfloat16)},
        )
    with pytest.raises(ValueError, match="shape"):
        rfft.init_finetune_step_state(
            bundle, plan.trainable, master_params={"w": jnp.ones(3)}
        )


def test_complex_low_precision_trainables_keep_complex_masters():
    trainable = {
        "w": jnp.asarray([1.0, 2.0], dtype=jnp.bfloat16),
        "z": jnp.asarray([1 + 1j, 2 - 1j], dtype=jnp.complex64),
    }
    plan = TinyPlan(
        trainable=trainable,
        labels={"w": "head", "z": "head"},
        group_specs={"head": TinyGroup("head", "head", None, 1.0, False)},
    )
    bundle = _bundle(plan, rfft.bf16_mixed_precision())
    state = rfft.init_finetune_step_state(bundle, trainable)

    def loss(model, model_state, key):
        del key
        value = jnp.sum(model["w"].astype(jnp.float32) ** 2)
        return value + jnp.sum(jnp.abs(model["z"]) ** 2), model_state

    params, state, info = rfft.make_finetune_update_step(plan, loss, bundle)(
        trainable, state
    )

    assert bool(info.update_applied)
    assert state.master_params["z"].dtype == jnp.complex64
    assert params["z"].dtype == jnp.complex64
    assert bool(jnp.all(jnp.imag(params["z"]) != jnp.imag(trainable["z"])))


@pytest.mark.parametrize(
    ("dtype", "precision", "mode", "masters"),
    (
        (jnp.float32, rfft.PrecisionConfig(), "full_precision", ()),
        (
            jnp.bfloat16,
            rfft.bf16_mixed_precision(),
            "master_mixed_precision",
            ("float32",),
        ),
        (
            jnp.float16,
            rfft.fp16_mixed_precision(),
            "master_mixed_precision",
            ("float32",),
        ),
        (jnp.bfloat16, rfft.bf16_master_free_precision(), "master_free_stochastic", ()),
        (
            jnp.bfloat16,
            rfft.PrecisionConfig(master_params="never"),
            "master_free_nearest",
            (),
        ),
    ),
)
def test_precision_report_describes_the_resolved_mode(dtype, precision, mode, masters):
    plan = _plan(dtype)
    report = _bundle(plan, precision).report

    assert report.precision.mode == mode
    assert report.precision.master_param_dtypes == masters
    assert report.precision.trainable_dtypes == (jnp.dtype(dtype).name,)
    assert report.precision.visible_param_bytes == 4 * jnp.dtype(dtype).itemsize
    assert report.precision.master_param_bytes == (16 if masters else 0)
    assert mode in str(report)
    manifest = _bundle(plan, precision).manifest()
    assert manifest["resolved_precision"] == report.precision.to_dict()


def test_precision_report_describes_moment_rounding():
    plan = _plan(jnp.bfloat16)
    bf16_moments = rfft.PrecisionConfig(moment_dtype=jnp.bfloat16)

    adamw = rfft.adamw_from_plan(plan, total_steps=4, precision=bf16_moments)
    assert adamw.report.precision.moment_storage == "bfloat16"
    assert adamw.report.precision.moment_rounding == "stochastic"
    galore = rfft.galore_adamw_from_plan(
        _plan(jnp.bfloat16, values=np.ones((4, 4))),
        total_steps=4,
        galore=rfft.GaLoreConfig(rank=1),
        precision=bf16_moments,
    )
    assert galore.report.precision.moment_rounding == "nearest"
    quantized = rfft.adamw8_from_plan(plan, total_steps=4)
    assert quantized.report.precision.moment_storage.startswith("blockwise_int8")


def test_builders_accept_precision_and_reject_conflicting_shorthands():
    plan = _plan(jnp.bfloat16)
    bundle = rfft.adamw_from_plan(
        plan, total_steps=4, precision=rfft.bf16_master_free_precision()
    )
    assert bundle.precision_config == rfft.bf16_master_free_precision()

    with pytest.raises(ValueError, match="moment_dtype conflicts"):
        rfft.adamw_from_plan(
            plan,
            total_steps=4,
            moment_dtype=jnp.float32,
            precision=rfft.PrecisionConfig(moment_dtype=jnp.bfloat16),
        )
    with pytest.raises(ValueError, match="preconditioner_dtype conflicts"):
        rfft.hybrid_kron_adam_from_plan(
            plan,
            total_steps=4,
            preconditioner_dtype=jnp.bfloat16,
            precision=rfft.PrecisionConfig(),
        )
    with pytest.raises(ValueError, match="kron_adam only"):
        rfft.adamw_from_plan(
            plan,
            total_steps=4,
            precision=rfft.PrecisionConfig(preconditioner_dtype=jnp.bfloat16),
        )
    kron = rfft.hybrid_kron_adam_from_plan(
        plan, total_steps=4, preconditioner_dtype=jnp.bfloat16
    )
    assert kron.precision_config.preconditioner_dtype == jnp.bfloat16


@pytest.mark.parametrize(
    ("dtype", "precision", "kwargs", "match"),
    (
        (
            jnp.float16,
            rfft.PrecisionConfig(
                master_params="never", parameter_rounding="stochastic"
            ),
            {},
            "bfloat16 parameters only",
        ),
        (
            jnp.float32,
            rfft.PrecisionConfig(
                master_params="never", parameter_rounding="stochastic"
            ),
            {},
            "requires bfloat16",
        ),
        (
            jnp.bfloat16,
            rfft.PrecisionConfig(accumulation_dtype=jnp.bfloat16),
            {},
            "must match",
        ),
        (
            jnp.bfloat16,
            rfft.PrecisionConfig(moment_dtype=jnp.bfloat16),
            {"state_quantization": rfft.StateQuantizationConfig(enabled=True)},
            "fallback_dtype",
        ),
    ),
)
def test_compile_rejects_precision_the_plan_cannot_honor(
    dtype, precision, kwargs, match
):
    with pytest.raises(ValueError, match=match):
        _bundle(_plan(dtype), precision, **kwargs)


def test_stochastic_rounding_noise_differs_across_partitions_not_replicas():
    plan = _plan(jnp.bfloat16, values=np.ones(256))
    bundle = rfft.compile_optimizer(
        plan,
        optimizer=rfft.OptimizerConfig(base_lr=2.0**-10, weight_decay=0.0),
        schedule=rfft.ScheduleConfig(kind="constant", total_steps=10),
        gradient_policy=rfft.GradientPolicy(
            clip_global_norm=None,
            axis_name=("data", "model"),
            partition_axis_names=("model",),
            replicated_axis_names=("data",),
        ),
        precision=rfft.bf16_master_free_precision(),
    )
    state = rfft.init_finetune_step_state(bundle, plan.trainable, rng_seed=0)
    step = rfft.make_finetune_update_step(plan, _scaled_linear_loss, bundle)

    def shard_step(params, state):
        return step(params, state, jnp.asarray(1.0))[0]["w"]

    replicated = (None, None)
    params = jax.jit(
        jax.vmap(
            jax.vmap(shard_step, in_axes=replicated, axis_name="model", axis_size=2),
            in_axes=replicated,
            axis_name="data",
            axis_size=2,
        )
    )(plan.trainable, state)

    np.testing.assert_array_equal(params[0], params[1])
    assert not jnp.array_equal(params[0, 0], params[0, 1])
