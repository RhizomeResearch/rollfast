"""Two-process CPU fixture for array checkpoint storage tests."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import rollfast.finetune as rfft
from rollfast.finetune import array_storage
from rollfast.utils import zeros_like_preserving_sharding

from .helpers import ARRAY_LIMITS, TinyGroup, TinyPlan


def _assert_state_on(actual_tree, expected_tree, template_tree) -> None:
    for actual, expected, target in zip(
        jax.tree.leaves(actual_tree),
        jax.tree.leaves(expected_tree),
        jax.tree.leaves(template_tree),
        strict=True,
    ):
        assert actual.sharding.is_equivalent_to(target.sharding, actual.ndim)
        # Every process reaches this collective for the same leaves in order.
        value = np.asarray(
            expected
            if expected.is_fully_addressable
            else multihost_utils.process_allgather(expected, tiled=True)
        )
        for shard in actual.addressable_shards:
            np.testing.assert_array_equal(shard.data, value[shard.index])


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
        rfft.save_array_state_checkpoint(path, saved, limits=ARRAY_LIMITS)
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
            rfft.save_array_state_checkpoint(path, invalid, limits=ARRAY_LIMITS)
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
            limits=ARRAY_LIMITS,
        )
        assert jnp.array_equal(loaded.rng.forward, rng.forward)
        _assert_state_on(loaded.state, state, template.state)

        # Eager init keeps counters on one device per process; the stored
        # counters are replicated across processes, so every process slices them.
        init_state = bundle.init(plan.trainable)
        restored = rfft.load_array_state_checkpoint(
            path,
            bundle,
            template=replace(template, state=init_state),
            model_checkpoint_id="distributed-model-0",
            applied_updates=1,
            limits=ARRAY_LIMITS,
        )
        _assert_state_on(restored.state, state, init_state)

        # Per-process values may fill a shared array only when every process
        # stored the same bytes: init counters agree, the accumulation does not.
        per_process_path = path.with_name("per-process.rfarr")
        rfft.save_array_state_checkpoint(
            per_process_path,
            rfft.make_state_checkpoint(
                bundle,
                init_state,
                model_checkpoint_id="distributed-model-0",
                accumulation={"value": jnp.asarray([float(rank)])},
                counters={"successful_update": 0},
                rng=rng,
            ),
            limits=ARRAY_LIMITS,
        )
        shared_template = replace(template, accumulation={"value": jnp.zeros((1,))})
        restored = rfft.load_array_state_checkpoint(
            per_process_path,
            bundle,
            template=shared_template,
            model_checkpoint_id="distributed-model-0",
            limits=ARRAY_LIMITS,
        )
        _assert_state_on(restored.state, init_state, template.state)
        np.testing.assert_array_equal(restored.accumulation["value"], [rank])
        shared_value = jax.make_array_from_process_local_data(
            NamedSharding(mesh, P()), np.zeros((1,), dtype=np.float32)
        )
        try:
            rfft.load_array_state_checkpoint(
                per_process_path,
                bundle,
                template=replace(shared_template, accumulation={"value": shared_value}),
                model_checkpoint_id="distributed-model-0",
                limits=ARRAY_LIMITS,
            )
        except rfft.OptimizerStateRestoreError as error:
            assert "per-process values differ" in str(error), error
        else:
            raise AssertionError("differing per-process values must be rejected")
        print(f"rank {rank} restored", flush=True)
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]))
