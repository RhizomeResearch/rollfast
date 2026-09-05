"""Mesh-aware global norm tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import rollfast.finetune as rfft
from rollfast.finetune.transforms import clip_by_global_norm
from rollfast.optim.sam import global_l2_norm, sam_perturbation
from rollfast.utils import resolve_partition_norm_axis_name

from .helpers import tiny_plan


def test_partition_norm_axis_resolution_filters_replicated_axes():
    assert (
        resolve_partition_norm_axis_name(
            axis_name=("data", "model"),
            replicated_axis_names=("data",),
        )
        == "model"
    )
    assert (
        resolve_partition_norm_axis_name(
            axis_name=("data",),
            replicated_axis_names=("data",),
        )
        is None
    )
    assert resolve_partition_norm_axis_name(
        axis_name=("data", "model"),
        partition_axis_names=("model", "tensor"),
        replicated_axis_names=("data",),
    ) == ("model", "tensor")


def test_global_l2_norm_reduces_only_partition_axes():
    def norm_on_shard(grad):
        return global_l2_norm(
            {"w": grad},
            axis_name=("data", "model"),
            replicated_axis_names=("data",),
        )

    # Two replicas of four model shards, each containing two gradient values.
    grads = jnp.ones((2, 4, 2), dtype=jnp.float32)
    norms = jax.jit(
        jax.vmap(jax.vmap(norm_on_shard, axis_name="model"), axis_name="data")
    )(grads)
    np.testing.assert_allclose(norms, jnp.sqrt(8.0))


def test_global_norm_clip_uses_explicit_partition_axes():
    tx = clip_by_global_norm(
        1.0,
        axis_name=("data", "model"),
        partition_axis_names=("model",),
        replicated_axis_names=("data",),
    )

    def clip_shard(grad):
        updates, _ = tx.update({"w": grad}, tx.init(None))
        return updates["w"]

    grads = jnp.ones((2, 4, 2), dtype=jnp.float32)
    updates = jax.jit(
        jax.vmap(jax.vmap(clip_shard, axis_name="model"), axis_name="data")
    )(grads)
    np.testing.assert_allclose(updates, grads / jnp.sqrt(8.0), rtol=1e-5)


def test_global_norm_clip_uses_complex_magnitude_and_preserves_phase():
    tx = clip_by_global_norm(1.0)
    updates, _ = tx.update(
        {"w": jnp.asarray([1.0 + 2.0j], dtype=jnp.complex64)},
        tx.init(None),
    )

    np.testing.assert_allclose(jnp.abs(updates["w"]), 1.0, rtol=1e-5)
    assert updates["w"].dtype == jnp.complex64
    assert jnp.imag(updates["w"][0]) != 0.0


def test_sam_perturbation_uses_partition_axes():
    def perturb_shard(grad):
        perturbation, norm = sam_perturbation(
            {"w": grad},
            rho=0.5,
            axis_name=("data", "model"),
            partition_axis_names=("model",),
            replicated_axis_names=("data",),
        )
        return perturbation["w"], norm

    grads = jnp.ones((2, 4, 2), dtype=jnp.float32)
    perturbation, norms = jax.jit(
        jax.vmap(jax.vmap(perturb_shard, axis_name="model"), axis_name="data")
    )(grads)
    np.testing.assert_allclose(perturbation, 0.5 * grads / jnp.sqrt(8.0), rtol=1e-5)
    np.testing.assert_allclose(norms, 0.5, rtol=1e-5)


def test_builder_derives_norm_axes_from_sharding_policy():
    plan = tiny_plan()
    sharding = rfft.ShardingPolicy(
        mesh_axes=("data", "model"),
        data_axes=("data",),
        parameter_axes=("model",),
    )

    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=4,
        schedule="constant",
        axis_name=("data", "model"),
        sharding=sharding,
    )

    assert bundle.gradient_policy.partition_axis_names == ("model",)
    assert bundle.gradient_policy.replicated_axis_names == ("data",)
