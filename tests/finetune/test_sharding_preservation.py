"""Sharding-preservation tests for fine-tuning state leaves."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import optax

import rollfast.finetune as rfft
from rollfast.optim.adam8 import QuantizedBlocks

from .helpers import TinyGroup, TinyPlan


def _named_sharding() -> NamedSharding:
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",))
    return NamedSharding(mesh, P())


def _sharded_matrix_plan(dtype=jnp.float32) -> tuple[TinyPlan, NamedSharding]:
    sharding = _named_sharding()
    trainable = {
        "w": jax.device_put(jnp.ones((4, 4), dtype=dtype), sharding),
    }
    labels = {"w": "matrix_decay"}
    groups = {
        "matrix_decay": TinyGroup(
            "matrix_decay",
            role="backbone",
            depth=0,
            lr_multiplier=1.0,
            weight_decay=True,
            tags=("matrix",),
        )
    }
    return TinyPlan(trainable=trainable, labels=labels, group_specs=groups), sharding


def _array_leaves(tree):
    return [
        leaf
        for leaf in jax.tree.leaves(tree)
        if hasattr(leaf, "sharding") and hasattr(leaf, "shape")
    ]


def test_master_params_and_accumulators_preserve_named_sharding():
    plan, sharding = _sharded_matrix_plan(jnp.bfloat16)
    precision = rfft.PrecisionConfig(master_params="always")
    accumulation = rfft.AccumulationConfig(steps=2)

    master = rfft.make_master_params(plan.trainable, precision)
    accumulation_state = rfft.init_accumulation_state(plan.trainable, accumulation)

    assert master["w"].dtype == jnp.float32
    assert master["w"].sharding == sharding
    assert accumulation_state.grad_numerator["w"].dtype == jnp.float32
    assert accumulation_state.grad_numerator["w"].sharding == sharding


def test_adamw_moments_and_updates_preserve_named_sharding():
    plan, sharding = _sharded_matrix_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=4,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    moment_leaves = [
        leaf for leaf in _array_leaves(state) if leaf.shape == plan.trainable["w"].shape
    ]

    assert len(moment_leaves) >= 2
    assert all(leaf.sharding == sharding for leaf in moment_leaves)

    grads = {"w": jax.device_put(jnp.ones((4, 4), dtype=jnp.float32), sharding)}
    updates, state = bundle.update(grads, state, plan.trainable)
    updated = optax.apply_updates(plan.trainable, updates)

    assert updates["w"].sharding == sharding
    assert updated["w"].sharding == sharding


def test_averaging_state_preserves_named_sharding():
    plan, sharding = _sharded_matrix_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=4,
        schedule="constant",
        clip_global_norm=None,
        ema=rfft.EMAConfig(enabled=True, debias=True),
        swa=rfft.SWAConfig(enabled=True, start_step=0),
    )
    state = bundle.init(plan.trainable)

    assert state.ema_params["w"].sharding == sharding
    assert state.swa_params["w"].sharding == sharding


def test_adamw8_quantized_blocks_preserve_named_sharding():
    plan, sharding = _sharded_matrix_plan()
    bundle = rfft.adamw8_from_plan(
        plan,
        total_steps=4,
        schedule="constant",
        clip_global_norm=None,
        state_quantization=rfft.StateQuantizationConfig(
            enabled=True,
            block_size=4,
            min_size=1,
            stochastic_rounding=False,
        ),
    )
    state = bundle.init(plan.trainable)
    blocks = [
        leaf
        for leaf in jax.tree.leaves(state, is_leaf=lambda x: isinstance(x, QuantizedBlocks))
        if isinstance(leaf, QuantizedBlocks)
    ]

    assert blocks
    assert all(block.values.sharding == sharding for block in blocks)
    assert all(block.scales.sharding == sharding for block in blocks)
