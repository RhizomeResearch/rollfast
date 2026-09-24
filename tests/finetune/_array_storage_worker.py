"""Two-process CPU fixture for array checkpoint storage tests."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import rollfast.finetune as rfft

from .helpers import TinyGroup, TinyPlan


def main(address: str, rank: int, path: Path) -> None:
    jax.distributed.initialize(
        coordinator_address=address,
        num_processes=2,
        process_id=rank,
        initialization_timeout=30,
        heartbeat_timeout_seconds=15,
        shutdown_timeout_seconds=5,
    )
    try:
        sharding = NamedSharding(Mesh(np.asarray(jax.devices()), ("data",)), P("data"))
        values = jax.make_array_from_process_local_data(
            sharding, np.asarray([rank + 1], dtype=np.float32)
        )
        plan = TinyPlan(
            trainable={"w": values},
            labels={"w": "head"},
            group_specs={
                "head": TinyGroup(
                    "head", "head", None, lr_multiplier=1.0, weight_decay=True
                )
            },
        )
        bundle = rfft.adamw_from_plan(
            plan, total_steps=4, schedule="constant", clip_global_norm=None
        )
        state = bundle.init(plan.trainable)
        rng = rfft.init_rng_streams(jax.random.key(rank + 100))
        saved = rfft.make_state_checkpoint(
            bundle,
            state,
            model_checkpoint_id="distributed-model-0",
            counters={"successful_update": 0},
            rng=rng,
        )
        template = rfft.make_state_checkpoint(
            bundle,
            bundle.init(plan.trainable),
            model_checkpoint_id="distributed-model-0",
            counters={"successful_update": 0},
            rng=rfft.init_rng_streams(jax.random.key(0)),
        )
        limits = rfft.ArrayCheckpointLimits(
            2_000_000, 10_000, 20_000, 10_000_000, 100_000_000
        )
        rfft.save_array_state_checkpoint(path, saved, limits=limits)
        pointer_before_failure = path.read_bytes()
        invalid = (
            replace(saved, manifest={**saved.manifest, "state_schema": {}})
            if rank == 1
            else saved
        )
        try:
            rfft.save_array_state_checkpoint(path, invalid, limits=limits)
        except rfft.OptimizerStateRestoreError as error:
            assert "participant" in str(error)
        else:
            raise AssertionError("a failed participant must abort publication")
        assert path.read_bytes() == pointer_before_failure
        loaded = rfft.load_array_state_checkpoint(
            path,
            bundle,
            template=template,
            model_checkpoint_id="distributed-model-0",
            applied_updates=0,
            limits=limits,
        )
        assert jnp.array_equal(loaded.rng.forward, rng.forward)
        for actual, expected in zip(
            jax.tree.leaves(loaded.state), jax.tree.leaves(state), strict=True
        ):
            for actual_shard, expected_shard in zip(
                actual.addressable_shards, expected.addressable_shards, strict=True
            ):
                np.testing.assert_array_equal(actual_shard.data, expected_shard.data)
                assert actual_shard.index == expected_shard.index
        print(f"rank {rank} restored", flush=True)
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]))
