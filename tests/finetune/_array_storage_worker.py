"""Two-process CPU fixture for array checkpoint storage tests."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import rollfast.finetune as rfft
from rollfast.finetune import array_storage
from rollfast.utils import zeros_like_preserving_sharding

from .helpers import TinyGroup, TinyPlan


def main(address: str, rank: int, path: Path) -> None:
    jax.distributed.initialize(
        coordinator_address=address,
        num_processes=2,
        process_id=rank,
        initialization_timeout=30,
    )
    try:
        mesh = Mesh(np.asarray(jax.devices()), ("data",))
        values = jax.make_array_from_process_local_data(
            NamedSharding(mesh, P("data")), np.asarray([rank + 1], dtype=np.float32)
        )
        # Rank 1 holds no primary replica, so it restores rank 0's shard files.
        replicated = jax.make_array_from_process_local_data(
            NamedSharding(mesh, P()), np.asarray([3.0, 4.0], dtype=np.float32)
        )
        plan = TinyPlan(
            trainable={"w": values, "r": replicated},
            labels={"w": "head", "r": "head"},
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
        # Nonzero moments make misplaced cross-process shards observable.
        _, state = bundle.update(plan.trainable, state, plan.trainable)
        rng = rfft.init_rng_streams(jax.random.key(rank + 100))
        saved = rfft.make_state_checkpoint(
            bundle,
            state,
            model_checkpoint_id="distributed-model-0",
            counters={"successful_update": 1},
            rng=rng,
        )
        template = rfft.make_state_checkpoint(
            bundle,
            jax.tree.map(zeros_like_preserving_sharding, state),
            model_checkpoint_id="distributed-model-0",
            counters={"successful_update": 0},
            rng=rfft.init_rng_streams(jax.random.key(0)),
        )
        limits = rfft.ArrayCheckpointLimits(
            2_000_000, 10_000, 20_000, 10_000_000, 100_000_000
        )
        rfft.save_array_state_checkpoint(path, saved, limits=limits)
        pointer_before_failure = path.read_bytes()
        generation = json.loads(pointer_before_failure)["generation"]
        index = json.loads(
            (
                array_storage._generation_root(path) / generation / "index.json"
            ).read_text()
        )
        for number in range(len(index["records"][0]["arrays"])):
            shard_indices = [
                json.dumps(chunk["index"])
                for record in index["records"]
                for chunk in record["arrays"][number]["chunks"]
                if chunk["index"] is not None
            ]
            assert len(shard_indices) == len(set(shard_indices)), shard_indices
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
            applied_updates=1,
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
