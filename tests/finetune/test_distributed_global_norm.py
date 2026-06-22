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
    assert (
        resolve_partition_norm_axis_name(
            axis_name=("data", "model"),
            partition_axis_names=("model", "tensor"),
            replicated_axis_names=("data",),
        )
        == ("model", "tensor")
    )


def test_global_l2_norm_reduces_only_partition_axes(monkeypatch):
    calls = []

    def fake_psum(x, axis_name):
        calls.append(axis_name)
        return x * 4.0

    monkeypatch.setattr(jax.lax, "psum", fake_psum)

    norm = global_l2_norm(
        {"w": jnp.ones((2,), dtype=jnp.float32)},
        axis_name=("data", "model"),
        replicated_axis_names=("data",),
    )

    assert calls == ["model"]
    np.testing.assert_allclose(norm, jnp.sqrt(8.0))


def test_global_norm_clip_uses_explicit_partition_axes(monkeypatch):
    calls = []

    def fake_psum(x, axis_name):
        calls.append(axis_name)
        return x * 4.0

    monkeypatch.setattr(jax.lax, "psum", fake_psum)
    tx = clip_by_global_norm(
        1.0,
        axis_name=("data", "model"),
        partition_axis_names=("model",),
        replicated_axis_names=("data",),
    )

    updates, _ = tx.update(
        {"w": jnp.ones((2,), dtype=jnp.float32)},
        tx.init(None),
    )

    assert calls == ["model"]
    np.testing.assert_allclose(updates["w"], jnp.ones((2,)) / jnp.sqrt(8.0), rtol=1e-5)


def test_sam_perturbation_uses_partition_axes(monkeypatch):
    calls = []

    def fake_psum(x, axis_name):
        calls.append(axis_name)
        return x * 4.0

    monkeypatch.setattr(jax.lax, "psum", fake_psum)

    _, perturbation_norm = sam_perturbation(
        {"w": jnp.ones((2,), dtype=jnp.float32)},
        rho=0.5,
        axis_name=("data", "model"),
        partition_axis_names=("model",),
        replicated_axis_names=("data",),
    )

    assert calls == ["model", "model"]
    np.testing.assert_allclose(perturbation_norm, 0.5, rtol=1e-5)


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
