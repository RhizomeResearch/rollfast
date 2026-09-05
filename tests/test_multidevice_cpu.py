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

from rollfast import adamw
from rollfast.finetune.transforms import clip_by_global_norm
from rollfast.optim.sam import global_l2_norm, sam_perturbation

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


def test_adamw_state_and_updates_preserve_multi_device_sharding():
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    sharding = NamedSharding(mesh, P("data", None))
    shape = (jax.local_device_count() * 2, 2)
    params = {"w": jax.device_put(jnp.ones(shape, dtype=jnp.float32), sharding)}
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
    updated = optax.apply_updates(params, updates)

    assert updates["w"].sharding.is_equivalent_to(sharding, updates["w"].ndim)
    assert updated["w"].sharding.is_equivalent_to(sharding, updated["w"].ndim)
    assert all(
        leaf.sharding.is_equivalent_to(sharding, leaf.ndim)
        for leaf in jax.tree.leaves(state)
        if hasattr(leaf, "shape") and leaf.shape == params["w"].shape
    )


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
    from rollfast.finetune.step import _apply_scaled_master_grads
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
        return _apply_scaled_master_grads(
            bundle,
            {"w": jnp.ones(2, dtype=jnp.bfloat16)},
            {"w": jnp.ones(2)},
            jnp.array(0),
            {"w": local_grad},
            jnp.array(1.0),
            jnp.array(1.0),
        )

    grads = jnp.ones((jax.local_device_count(), 2)).at[0].set(jnp.nan)
    _, master, state, finite = apply(grads)
    expected = np.full(grads.shape, 1.0 - 0.1 * (jax.local_device_count() - 1))
    expected[0] = 1.0
    np.testing.assert_allclose(master["w"], expected, rtol=1e-6)
    np.testing.assert_array_equal(state, [0] + [1] * (jax.local_device_count() - 1))
    np.testing.assert_array_equal(
        finite, [False] + [True] * (jax.local_device_count() - 1)
    )
