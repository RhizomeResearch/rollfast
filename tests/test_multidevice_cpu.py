"""Collective and sharding checks for forced multi-device CPU CI."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import rollfast.finetune as rfft
from rollfast import adamw, apply_updates
from rollfast.finetune.transforms import clip_by_global_norm
from rollfast.optim.sam import global_l2_norm, sam_perturbation
from tests.finetune.helpers import TinyGroup, TinyPlan

pytestmark = pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason="requires XLA_FLAGS=--xla_force_host_platform_device_count=4",
)


def test_collective_norm_clipping_and_sam_use_all_devices():
    device_count = jax.local_device_count()
    values = jnp.arange(1, device_count * 2 + 1, dtype=jnp.float32).reshape(
        device_count, 2
    )
    expected_norm = np.linalg.norm(np.asarray(values))

    @partial(jax.pmap, axis_name="devices")
    def distributed_norm(local_values):
        return global_l2_norm({"w": local_values}, axis_name="devices")

    @partial(jax.pmap, axis_name="devices")
    def distributed_clip(local_values):
        tx = clip_by_global_norm(1.0, axis_name="devices")
        clipped, _ = tx.update({"w": local_values}, tx.init(None))
        return clipped["w"]

    @partial(jax.pmap, axis_name="devices")
    def distributed_sam(local_values):
        perturbation, perturbation_norm = sam_perturbation(
            {"w": local_values},
            rho=0.5,
            axis_name="devices",
        )
        return perturbation["w"], perturbation_norm

    norms = distributed_norm(values)
    clipped = distributed_clip(values)
    perturbation, perturbation_norms = distributed_sam(values)

    np.testing.assert_allclose(norms, expected_norm, rtol=1e-6)
    np.testing.assert_allclose(clipped, values / expected_norm, rtol=1e-6)
    np.testing.assert_allclose(
        perturbation,
        values * (0.5 / expected_norm),
        rtol=1e-6,
    )
    np.testing.assert_allclose(perturbation_norms, 0.5, rtol=1e-6)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_adamw_state_and_updates_preserve_multi_device_sharding(dtype):
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    sharding = NamedSharding(mesh, P("data", None))
    shape = (jax.local_device_count() * 2, 2)
    params = {"w": jax.device_put(jnp.ones(shape, dtype=dtype), sharding)}
    grads = {"w": jax.device_put(jnp.full(shape, 0.25, dtype=jnp.float32), sharding)}

    tx = adamw(learning_rate=1e-2)
    state = tx.init(params)
    moment_leaves = [
        leaf
        for leaf in jax.tree.leaves(state)
        if hasattr(leaf, "shape") and leaf.shape == params["w"].shape
    ]

    assert len(moment_leaves) >= 2
    assert all(
        leaf.sharding.is_equivalent_to(sharding, leaf.ndim) for leaf in moment_leaves
    )

    updates, state = tx.update(grads, state, params)
    updated = apply_updates(params, updates, stochastic=False)

    assert updates["w"].sharding.is_equivalent_to(sharding, updates["w"].ndim)
    assert updated["w"].sharding.is_equivalent_to(sharding, updated["w"].ndim)
    assert all(
        leaf.sharding.is_equivalent_to(sharding, leaf.ndim)
        for leaf in jax.tree.leaves(state)
        if hasattr(leaf, "shape") and leaf.shape == params["w"].shape
    )


def test_discarded_accumulation_preserves_sharding_and_recovers():
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    params = {
        "w": jax.device_put(
            jnp.ones((jax.local_device_count() * 2,), dtype=jnp.float32),
            sharding,
        )
    }
    plan = TinyPlan(
        trainable=params,
        labels={"w": "head"},
        group_specs={"head": TinyGroup("head", "head", None, 1.0, False)},
    )
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=8,
        schedule="constant",
        clip_global_norm=None,
        accumulation_steps=2,
    )
    state = bundle.init(params)
    update = jax.jit(bundle.update_with_status)
    bad = {"w": jax.device_put(jnp.full(params["w"].shape, jnp.nan), sharding)}
    for _ in range(2):
        _, state, status = update(bad, state, params)
    assert bool(status.window_discarded)
    assert state.acc_grads["w"].sharding.is_equivalent_to(sharding, 1)
    assert bool(jnp.all(jnp.isfinite(state.acc_grads["w"])))

    good = {"w": jax.device_put(jnp.ones(params["w"].shape), sharding)}
    for _ in range(2):
        updates, state, status = update(good, state, params)
    assert bool(status.update_applied)
    assert updates["w"].sharding.is_equivalent_to(sharding, 1)


@pytest.mark.parametrize("spec", [P("devices", None), P(None, "devices"), P()])
@pytest.mark.parametrize("block_size", [8, 32])
def test_quantized_zero_initialization_preserves_derived_sharding(spec, block_size):
    from rollfast.optim.adam8 import _init_moment_leaf, quantize_blocks
    from rollfast.utils import zeros_like_preserving_sharding

    mesh = Mesh(np.asarray(jax.devices()), ("devices",))
    sharding = NamedSharding(mesh, spec)
    params = jax.device_put(jnp.ones((8, 8), dtype=jnp.bfloat16), sharding)

    def reference(param):
        return quantize_blocks(
            zeros_like_preserving_sharding(param, jnp.float32),
            block_size=block_size,
            quantizer="dynamic_signed",
        )

    def initialize(param):
        return _init_moment_leaf(
            param,
            block_size=block_size,
            min_size=0,
            scale_dtype=jnp.float32,
            fallback_dtype=jnp.float32,
            block_layout="shard_local",
            quantize=True,
            quantizer="dynamic_signed",
        )

    for init, ref in (
        (initialize, reference),
        (jax.jit(initialize), jax.jit(reference)),
    ):
        actual, expected = init(params), ref(params)
        assert jax.tree.structure(actual) == jax.tree.structure(expected)
        for a, b in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_array_equal(a, b)
            assert a.dtype == b.dtype
            assert a.sharding.is_equivalent_to(b.sharding, b.ndim)


@pytest.mark.parametrize("declared_axis", [False, True])
def test_rejected_master_steps_keep_collective_participation(declared_axis):
    import rollfast.finetune as rfft
    from rollfast.finetune.step import _commit_update, _unscale_gradients
    from tests.finetune.helpers import tiny_plan

    def update(grads, state, params=None):
        total = jax.lax.psum(grads["w"], "devices")
        return {"w": -0.1 * total}, state + 1

    bundle = rfft.adamw_from_plan(tiny_plan(), total_steps=10)
    bundle = replace(
        bundle,
        tx=optax.GradientTransformation(lambda _: jnp.array(0), update),
        gradient_policy=rfft.GradientPolicy(
            axis_name="devices" if declared_axis else None, nonfinite="none"
        ),
    )

    @partial(jax.pmap, axis_name="devices")
    def apply(local_grad):
        grads, finite = _unscale_gradients(
            bundle, {"w": local_grad}, jnp.array(1.0), jnp.array(1.0)
        )
        commit = _commit_update(
            bundle,
            {"w": jnp.ones(2, dtype=jnp.bfloat16)},
            {"w": jnp.ones(2)},
            jnp.array(0),
            grads,
            finite,
            sr_key=None,
        )
        return commit.master, commit.opt_state, commit.accepted

    count = jax.local_device_count()
    grads = jnp.ones((count, 2)).at[0].set(jnp.nan)
    master, state, accepted = apply(grads)
    if declared_axis:
        # Partition participants agree, so one overflow rejects every replica.
        np.testing.assert_array_equal(master["w"], np.ones(grads.shape))
        np.testing.assert_array_equal(state, [0] * count)
        np.testing.assert_array_equal(accepted, [False] * count)
        return
    expected = np.full(grads.shape, 1.0 - 0.1 * (count - 1))
    expected[0] = 1.0
    np.testing.assert_allclose(master["w"], expected, rtol=1e-6)
    np.testing.assert_array_equal(state, [0] + [1] * (count - 1))
    np.testing.assert_array_equal(accepted, [False] + [True] * (count - 1))


def test_eager_bundle_init_places_parameter_state_like_parameters():
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    sharding = NamedSharding(mesh, P("data", None))
    params = {
        "w": jax.device_put(jnp.ones((jax.local_device_count() * 2, 2)), sharding)
    }
    plan = TinyPlan(
        trainable=params,
        labels={"w": "head"},
        group_specs={"head": TinyGroup("head", "head", None, 1.0, True)},
    )
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=4,
        schedule="constant",
        accumulation_steps=2,
        ema=rfft.EMAConfig(enabled=True),
    )

    # Traced parameters carry no sharding inside jax.jit, so init runs eagerly.
    state = bundle.init(params)

    parameter_state = [
        leaf for leaf in jax.tree.leaves(state) if leaf.shape == params["w"].shape
    ]
    assert len(parameter_state) == 4  # Adam moments, accumulator, and EMA.
    assert all(
        leaf.sharding.is_equivalent_to(sharding, leaf.ndim) for leaf in parameter_state
    )


@pytest.mark.parametrize(
    "precision",
    [rfft.bf16_mixed_precision(), rfft.bf16_master_free_precision()],
    ids=["masters", "master_free_stochastic"],
)
def test_low_precision_finetune_step_preserves_parameter_sharding(precision):
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    sharding = NamedSharding(mesh, P("data", None))
    shape = (jax.local_device_count() * 2, 4)
    params = {"w": jax.device_put(jnp.ones(shape, dtype=jnp.bfloat16), sharding)}
    plan = TinyPlan(
        trainable=params,
        labels={"w": "head"},
        group_specs={"head": TinyGroup("head", "head", None, 1.0, False)},
    )
    bundle = rfft.compile_optimizer(
        plan,
        optimizer=rfft.OptimizerConfig(base_lr=1e-2, weight_decay=0.0),
        schedule=rfft.ScheduleConfig(kind="constant", total_steps=4),
        precision=precision,
    )
    state = rfft.init_finetune_step_state(bundle, params, rng_seed=0)

    def loss(model, model_state, key):
        del key
        return jnp.sum(model["w"].astype(jnp.float32) ** 2), model_state

    step = jax.jit(rfft.make_finetune_update_step(plan, loss, bundle))
    updated, state, info = step(params, state)

    assert bool(info.update_applied)
    assert updated["w"].dtype == jnp.bfloat16
    assert updated["w"].sharding.is_equivalent_to(sharding, 2)
    if state.master_params is not None:
        assert state.master_params["w"].sharding.is_equivalent_to(sharding, 2)
