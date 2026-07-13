"""Collective and sharding checks for forced multi-device CPU CI."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
import pytest

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
