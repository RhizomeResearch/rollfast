from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import rollfast.finetune as rfft

from .helpers import tiny_plan


def test_state_offload_policy_round_trip_and_bundle_manifest():
    policy = rfft.StateOffloadPolicy(
        enabled=True,
        target="host",
        min_bytes=16,
        preserve_sharding=False,
    )
    assert rfft.StateOffloadPolicy.from_dict(policy.to_dict()) == policy

    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    bundle = rfft.with_state_offload(bundle, policy)

    assert bundle.manifest()["state_offload"]["enabled"] is True
    assert bundle.manifest()["state_offload"]["min_bytes"] == 16


def test_optimizer_state_offload_and_restore_thresholds():
    policy = rfft.StateOffloadPolicy(enabled=True, target="host", min_bytes=8)
    state = {
        "large": jnp.ones((4,), dtype=jnp.float32),
        "small": jnp.ones((), dtype=jnp.float32),
    }

    offloaded = rfft.offload_optimizer_state(state, policy=policy)

    assert isinstance(offloaded["large"], np.ndarray)
    assert isinstance(offloaded["small"], jax.Array)
    manifest = rfft.state_offload_manifest(policy, state)
    assert manifest["eligible_leaves"] == 1
    assert manifest["eligible_bytes"] == 16

    restored = rfft.restore_offloaded_optimizer_state(offloaded, policy=policy)

    assert isinstance(restored["large"], jax.Array)
    assert jnp.allclose(restored["large"], state["large"])
