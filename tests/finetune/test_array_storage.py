"""Strict optimizer checkpoint restoration and non-executable storage tests."""

import hashlib
import io
import json
import os
import socket
import subprocess
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import rollfast.finetune as rfft
from rollfast.utils import zeros_like_preserving_sharding

from .helpers import TinyGroup, TinyPlan, ones_like_trainable, tiny_plan

LIMITS = rfft.ArrayCheckpointLimits(
    max_manifest_bytes=2_000_000,
    max_arrays=10_000,
    max_shards=20_000,
    max_array_bytes=10_000_000,
    max_total_bytes=100_000_000,
)


def _bundle(*, averaging=False, dtype=jnp.float32):
    plan = tiny_plan()
    if dtype != jnp.float32:
        plan = replace(
            plan,
            trainable=jax.tree.map(
                lambda x: x.astype(dtype) if x is not None else None,
                plan.trainable,
                is_leaf=lambda x: x is None,
            ),
        )
    if dtype == jnp.float32:
        bundle = rfft.adamw_from_plan(
            plan,
            total_steps=5,
            schedule="constant",
            ema=rfft.EMAConfig(enabled=averaging),
            swa=rfft.SWAConfig(enabled=averaging, start_step=0 if averaging else None),
        )
    else:
        bundle = rfft.compile_optimizer(
            plan,
            optimizer=rfft.OptimizerConfig(),
            schedule=rfft.ScheduleConfig(kind="constant", total_steps=5),
            gradient_policy=rfft.GradientPolicy(nonfinite="none"),
            ema=rfft.EMAConfig(enabled=averaging),
            swa=rfft.SWAConfig(enabled=averaging, start_step=0 if averaging else None),
        )
    return plan, bundle


def _checkpoints(plan, bundle):
    state = bundle.init(plan.trainable)
    _, state = bundle.update(ones_like_trainable(plan.trainable), state, plan.trainable)
    saved = rfft.make_state_checkpoint(
        bundle,
        state,
        model_checkpoint_id="training-model-1",
        base_model_value_hash="model-digest",
        counters={"successful_update": 1},
        rng=rfft.init_rng_streams(jax.random.key(123)),
    )
    template = rfft.make_state_checkpoint(
        bundle,
        bundle.init(plan.trainable),
        model_checkpoint_id="training-model-1",
        base_model_value_hash="model-digest",
        counters={"successful_update": 0},
        rng=rfft.init_rng_streams(jax.random.key(0)),
    )
    return saved, template


@pytest.mark.parametrize("averaging", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_safe_adamw_round_trip_preserves_next_update(tmp_path, averaging, dtype):
    plan, bundle = _bundle(averaging=averaging, dtype=dtype)
    saved, template = _checkpoints(plan, bundle)
    path = tmp_path / "optimizer.rfarr"

    rfft.save_array_state_checkpoint(path, saved, limits=LIMITS)
    loaded = rfft.load_array_state_checkpoint(
        path,
        bundle,
        template=template,
        model_checkpoint_id="training-model-1",
        base_model_value_hash="model-digest",
        applied_updates=1,
        limits=LIMITS,
    )

    assert loaded.manifest["counters"]["successful_update"] == 1
    assert loaded.counters["successful_update"] == 1
    assert jnp.array_equal(loaded.rng.forward, saved.rng.forward)
    grads = ones_like_trainable(plan.trainable)
    expected, _ = jax.jit(bundle.update)(grads, saved.state, plan.trainable)
    actual, _ = jax.jit(bundle.update)(grads, loaded.state, plan.trainable)
    for left, right in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_allclose(
            left, right, rtol=2e-2 if dtype == jnp.bfloat16 else 1e-6
        )


def test_strict_restore_rejects_changed_group_and_state_shape():
    plan, bundle = _bundle()
    checkpoint, template = _checkpoints(plan, bundle)
    changed = rfft.adamw_from_plan(
        plan,
        total_steps=5,
        schedule="constant",
        group_rules=(rfft.GroupRule(name="head-rate", role="head", lr_multiplier=3.0),),
    )
    assert changed.report.fingerprint == bundle.report.fingerprint
    with pytest.raises(rfft.OptimizerStateRestoreError, match="effective group table"):
        rfft.restore_state_checkpoint(
            changed, checkpoint, model_checkpoint_id="training-model-1"
        )
    with pytest.raises(rfft.OptimizerStateRestoreError, match="state schema"):
        rfft.restore_state_checkpoint(
            bundle,
            replace(checkpoint, state=()),
            model_checkpoint_id="training-model-1",
        )
    with pytest.raises(rfft.OptimizerStateRestoreError, match="template mismatch"):
        rfft.restore_state_checkpoint(
            bundle,
            replace(
                checkpoint,
                manifest={
                    key: value
                    for key, value in checkpoint.manifest.items()
                    if key != "state_schema"
                },
                state=(),
            ),
            model_checkpoint_id="training-model-1",
            template=template,
        )


def test_safe_reader_rejects_wrong_identity_and_limits(tmp_path):
    plan, bundle = _bundle()
    saved, template = _checkpoints(plan, bundle)
    path = tmp_path / "optimizer.rfarr"
    rfft.save_array_state_checkpoint(path, saved, limits=LIMITS)

    with pytest.raises(
        rfft.OptimizerStateRestoreError, match="model checkpoint mismatch"
    ):
        rfft.load_array_state_checkpoint(
            path, bundle, template=template, model_checkpoint_id="wrong", limits=LIMITS
        )

    with pytest.raises(
        rfft.OptimizerStateRestoreError, match="base model hash mismatch"
    ):
        rfft.load_array_state_checkpoint(
            path,
            bundle,
            template=template,
            model_checkpoint_id="training-model-1",
            base_model_value_hash="wrong",
            limits=LIMITS,
        )
    with pytest.raises(
        rfft.OptimizerStateRestoreError, match="allocation exceeds limit"
    ):
        rfft.load_array_state_checkpoint(
            path,
            bundle,
            template=template,
            model_checkpoint_id="training-model-1",
            limits=replace(LIMITS, max_total_bytes=1),
        )


def test_array_shards_are_plain_npy_byte_files(tmp_path):
    from rollfast.finetune import array_storage
    from rollfast.finetune._checkpoint_tree import checkpoint_parts, describe_tree

    plan, bundle = _bundle()
    saved, _ = _checkpoints(plan, bundle)
    path = tmp_path / "optimizer.rfarr"
    rfft.save_array_state_checkpoint(path, saved, limits=LIMITS)
    pointer = json.loads(path.read_text())
    generation = array_storage._generation_root(path) / pointer["generation"]
    index = json.loads((generation / "index.json").read_bytes())
    index_digest = hashlib.sha256((generation / "index.json").read_bytes())
    assert index_digest.hexdigest() == pointer["sha256"]
    _, leaves = describe_tree(checkpoint_parts(saved))

    entries = index["records"][0]["arrays"]
    assert len(entries) == len(leaves)
    for leaf, entry in zip(leaves, entries, strict=True):
        (chunk,) = entry["chunks"]
        stored = (generation / "rank-00000" / chunk["file"]).read_bytes()
        expected = np.asarray(
            jax.random.key_data(leaf)
            if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            else leaf
        ).tobytes()
        reference = io.BytesIO()
        np.save(reference, np.frombuffer(expected, dtype=np.uint8), allow_pickle=False)
        assert stored == reference.getvalue()
        assert chunk["size"] == len(stored)
        assert chunk["sha256"] == hashlib.sha256(stored).hexdigest()


def test_array_checkpoint_paths_reject_symlinked_directories(tmp_path):
    plan, bundle = _bundle()
    saved, template = _checkpoints(plan, bundle)
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(
        rfft.OptimizerStateRestoreError, match="write failed.*traverses a symlink"
    ):
        rfft.save_array_state_checkpoint(
            linked / "optimizer.rfarr", saved, limits=LIMITS
        )
    assert not (real / "optimizer.rfarr").exists()

    rfft.save_array_state_checkpoint(real / "optimizer.rfarr", saved, limits=LIMITS)
    with pytest.raises(rfft.OptimizerStateRestoreError, match="traverses a symlink"):
        rfft.load_array_state_checkpoint(
            linked / "optimizer.rfarr",
            bundle,
            template=template,
            model_checkpoint_id="training-model-1",
            limits=LIMITS,
        )


def test_safe_round_trip_preserves_accumulation_and_precision_parts(tmp_path):
    plan = tiny_plan()
    precision = rfft.PrecisionConfig(master_params="always", loss_scale="dynamic")
    bundle = rfft.compile_optimizer(
        plan,
        optimizer=rfft.OptimizerConfig(weight_decay=0.0),
        schedule=rfft.ScheduleConfig(kind="constant", total_steps=4),
        gradient_policy=rfft.GradientPolicy(clip_global_norm=None),
        precision=precision,
    )
    master = rfft.make_master_params(plan.trainable, precision)
    loss_scale = rfft.init_loss_scale_state(precision)
    accumulation = rfft.init_accumulation_state(
        plan.trainable, rfft.AccumulationConfig(steps=2)
    )
    checkpoint = rfft.make_state_checkpoint(
        bundle,
        bundle.init(master),
        model_checkpoint_id="training-model-0",
        master_params=master,
        loss_scale=loss_scale,
        accumulation=replace(accumulation, microsteps_in_window=jnp.asarray(1)),
        counters={"successful_update": 0},
    )
    template = rfft.make_state_checkpoint(
        bundle,
        bundle.init(master),
        model_checkpoint_id="training-model-0",
        master_params=master,
        loss_scale=loss_scale,
        accumulation=accumulation,
        counters={"successful_update": 0},
    )
    path = tmp_path / "precision.rfarr"
    rfft.save_array_state_checkpoint(path, checkpoint, limits=LIMITS)
    loaded = rfft.load_array_state_checkpoint(
        path,
        bundle,
        template=template,
        model_checkpoint_id="training-model-0",
        applied_updates=0,
        limits=LIMITS,
    )

    assert int(loaded.accumulation.microsteps_in_window) == 1
    assert loaded.loss_scale == loss_scale
    for actual, expected in zip(
        jax.tree.leaves(loaded.master_params), jax.tree.leaves(master), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_strict_restore_rejects_missing_declared_component():
    plan, bundle = _bundle()
    checkpoint, _ = _checkpoints(plan, bundle)

    with pytest.raises(rfft.OptimizerStateRestoreError, match="rng component presence"):
        rfft.restore_state_checkpoint(
            bundle,
            replace(checkpoint, rng=None),
            model_checkpoint_id="training-model-1",
        )


@pytest.mark.skipif(jax.local_device_count() < 4, reason="requires four devices")
@pytest.mark.parametrize(
    ("mesh_shape", "spec", "distinct_shards"),
    [((4,), P(), 1), ((2, 2), P("data", None), 2)],
    ids=["replicated", "partially-replicated"],
)
def test_replicated_shards_are_written_once_and_restored_to_every_replica(
    tmp_path, mesh_shape, spec, distinct_shards
):
    from rollfast.finetune import array_storage
    from rollfast.finetune._checkpoint_tree import checkpoint_parts, describe_tree

    devices = np.asarray(jax.devices()[:4]).reshape(mesh_shape)
    mesh = Mesh(devices, ("data", "model")[: len(mesh_shape)])
    weight = jnp.arange(32, dtype=jnp.float32).reshape(8, 4)
    plan = TinyPlan(
        trainable={"w": jax.device_put(weight, NamedSharding(mesh, spec))},
        labels={"w": "head"},
        group_specs={"head": TinyGroup("head", "head", None, 1.0, True)},
    )
    bundle = rfft.adamw_from_plan(
        plan, total_steps=4, schedule="constant", clip_global_norm=None
    )
    state = bundle.init(plan.trainable)
    _, state = bundle.update(plan.trainable, state, plan.trainable)
    saved = rfft.make_state_checkpoint(
        bundle,
        state,
        model_checkpoint_id="replicated-model-0",
        counters={"successful_update": 1},
    )
    # Updated counters are replicated across the mesh, so zero the saved layout.
    template = rfft.make_state_checkpoint(
        bundle,
        jax.tree.map(zeros_like_preserving_sharding, state),
        model_checkpoint_id="replicated-model-0",
        counters={"successful_update": 0},
    )
    path = tmp_path / "replicated.rfarr"
    rfft.save_array_state_checkpoint(path, saved, limits=LIMITS)

    pointer = json.loads(path.read_text())
    generation = array_storage._generation_root(path) / pointer["generation"]
    index_file = generation / "index.json"
    index = json.loads(index_file.read_text())
    (record,) = index["records"]
    _, leaves = describe_tree(checkpoint_parts(saved))
    moment_chunks = [
        entry["chunks"]
        for leaf, entry in zip(leaves, record["arrays"], strict=True)
        if leaf.shape == weight.shape
    ]
    assert len(moment_chunks) == 2
    for chunks in moment_chunks:
        assert len({json.dumps(chunk["index"]) for chunk in chunks}) == len(chunks)
        assert len(chunks) == distinct_shards
    assert len(list(generation.glob("rank-00000/*.npy"))) == sum(
        len(entry["chunks"]) for entry in record["arrays"]
    )

    restored = rfft.load_array_state_checkpoint(
        path,
        bundle,
        template=template,
        model_checkpoint_id="replicated-model-0",
        applied_updates=1,
        limits=LIMITS,
    )
    for actual, expected in zip(
        jax.tree.leaves(restored.state), jax.tree.leaves(state), strict=True
    ):
        assert actual.sharding == expected.sharding
        for actual_shard, expected_shard in zip(
            actual.addressable_shards, expected.addressable_shards, strict=True
        ):
            assert actual_shard.index == expected_shard.index
            np.testing.assert_array_equal(actual_shard.data, expected_shard.data)

    # A repeated shard index, as in per-replica layouts, is ambiguous.
    moment_chunks[0].append(dict(moment_chunks[0][0]))
    index_file.write_text(json.dumps(index))
    pointer["sha256"] = hashlib.sha256(index_file.read_bytes()).hexdigest()
    path.write_text(json.dumps(pointer))
    with pytest.raises(rfft.OptimizerStateRestoreError, match="shard layout mismatch"):
        rfft.load_array_state_checkpoint(
            path,
            bundle,
            template=template,
            model_checkpoint_id="replicated-model-0",
            applied_updates=1,
            limits=LIMITS,
        )


def _shape_only(tree):
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=leaf.sharding
        ),
        tree,
    )


def _assert_restored_onto(restored, expected, template):
    for actual, value, target in zip(
        jax.tree.leaves(restored),
        jax.tree.leaves(expected),
        jax.tree.leaves(template),
        strict=True,
    ):
        assert actual.sharding.is_equivalent_to(target.sharding, actual.ndim)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(value))


def test_shape_only_templates_restore_without_template_arrays(tmp_path):
    plan, bundle = _bundle()
    saved, template = _checkpoints(plan, bundle)
    path = tmp_path / "optimizer.rfarr"
    rfft.save_array_state_checkpoint(path, saved, limits=LIMITS)

    abstract = replace(template, state=_shape_only(template.state))
    loaded = rfft.load_array_state_checkpoint(
        path,
        bundle,
        template=abstract,
        model_checkpoint_id="training-model-1",
        applied_updates=1,
        limits=LIMITS,
    )
    _assert_restored_onto(loaded.state, saved.state, template.state)

    with pytest.raises(rfft.OptimizerStateRestoreError, match="must be concrete"):
        rfft.load_array_state_checkpoint(
            path,
            bundle,
            template=replace(
                abstract,
                rng=replace(template.rng, forward=_shape_only(template.rng.forward)),
            ),
            model_checkpoint_id="training-model-1",
            limits=LIMITS,
        )
    with pytest.raises(rfft.OptimizerStateRestoreError, match="shape-only arrays"):
        rfft.save_array_state_checkpoint(
            tmp_path / "abstract.rfarr", abstract, limits=LIMITS
        )


def _placement_checkpoint(bundle, state, updates):
    return rfft.make_state_checkpoint(
        bundle,
        state,
        model_checkpoint_id="placement-model-0",
        counters={"successful_update": updates},
    )


def _load_placement(path, bundle, template_state):
    return rfft.load_array_state_checkpoint(
        path,
        bundle,
        template=_placement_checkpoint(bundle, template_state, 0),
        model_checkpoint_id="placement-model-0",
        limits=LIMITS,
    )


def _trained_placement_state(trainable):
    plan = TinyPlan(
        trainable=trainable,
        labels={name: "head" for name in trainable},
        group_specs={"head": TinyGroup("head", "head", None, 1.0, True)},
    )
    bundle = rfft.adamw_from_plan(
        plan, total_steps=4, schedule="constant", clip_global_norm=None
    )
    state = bundle.init(plan.trainable)
    # Updates replicate scalar counters that init leaves on one device.
    _, state = bundle.update(plan.trainable, state, plan.trainable)
    return plan, bundle, state


@pytest.mark.skipif(jax.local_device_count() < 4, reason="requires four devices")
def test_trained_state_restores_onto_eager_init_and_shape_only_templates(tmp_path):
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("data",))
    plan, bundle, state = _trained_placement_state(
        {
            "w": jax.device_put(
                jnp.arange(32, dtype=jnp.float32).reshape(8, 4),
                NamedSharding(mesh, P("data", None)),
            ),
            "r": jax.device_put(
                jnp.arange(4, dtype=jnp.float32), NamedSharding(mesh, P())
            ),
        }
    )
    saved = _placement_checkpoint(bundle, state, 1)
    path = tmp_path / "trained.rfarr"
    rfft.save_array_state_checkpoint(path, saved, limits=LIMITS)
    init_state = bundle.init(plan.trainable)

    for template_state in (init_state, _shape_only(init_state)):
        restored = _load_placement(path, bundle, template_state)
        _assert_restored_onto(restored.state, state, template_state)
    rfft.restore_state_checkpoint(
        bundle,
        saved,
        model_checkpoint_id="placement-model-0",
        template=_placement_checkpoint(bundle, init_state, 0),
    )

    # One device is filled by stitching the four stored row shards together.
    single_device = jax.tree.map(
        lambda leaf: jax.device_put(leaf, jax.devices()[0]), init_state
    )
    restored = _load_placement(path, bundle, single_device)
    _assert_restored_onto(restored.state, state, single_device)


@pytest.mark.skipif(jax.local_device_count() < 4, reason="requires four devices")
def test_replicated_and_single_device_placements_are_interchangeable(tmp_path):
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("data",))
    _, bundle, replicated = _trained_placement_state(
        {"r": jax.device_put(jnp.arange(4.0), NamedSharding(mesh, P()))}
    )
    single_device = jax.tree.map(
        lambda leaf: jax.device_put(leaf, jax.devices()[1]), replicated
    )

    for saved, target in ((replicated, single_device), (single_device, replicated)):
        path = tmp_path / f"{len(list(tmp_path.iterdir()))}.rfarr"
        rfft.save_array_state_checkpoint(
            path, _placement_checkpoint(bundle, saved, 1), limits=LIMITS
        )
        template = jax.tree.map(zeros_like_preserving_sharding, target)
        restored = _load_placement(path, bundle, template)
        _assert_restored_onto(restored.state, saved, template)


@pytest.mark.skipif(jax.local_device_count() < 4, reason="requires four devices")
def test_sharded_state_restores_onto_a_different_partition(tmp_path):
    from rollfast.finetune import array_storage

    devices = np.asarray(jax.devices()[:4])
    rows = Mesh(devices, ("data",))
    grid = Mesh(devices.reshape(2, 2), ("data", "model"))
    weight = jnp.arange(64, dtype=jnp.float32).reshape(8, 8)
    _, bundle, state = _trained_placement_state(
        {"w": jax.device_put(weight, NamedSharding(rows, P("data", None)))}
    )
    path = tmp_path / "sharded.rfarr"
    rfft.save_array_state_checkpoint(
        path, _placement_checkpoint(bundle, state, 1), limits=LIMITS
    )

    # Each target piece overlaps several stored row shards or a part of one.
    for sharding in (
        NamedSharding(rows, P(None, "data")),
        NamedSharding(grid, P("data", "model")),
        jax.sharding.SingleDeviceSharding(jax.devices()[3]),
    ):
        template = jax.tree.map(
            lambda leaf, sharding=sharding: jax.ShapeDtypeStruct(
                leaf.shape,
                leaf.dtype,
                sharding=sharding if leaf.shape == weight.shape else leaf.sharding,
            ),
            state,
        )
        restored = _load_placement(path, bundle, template)
        _assert_restored_onto(restored.state, state, template)

    # Stored shards must tile each array exactly once.
    pointer = json.loads(path.read_text())
    index_file = array_storage._generation_root(path) / pointer["generation"]
    index_file = index_file / "index.json"
    original = json.loads(index_file.read_text())
    moment = next(
        number
        for number, entry in enumerate(original["records"][0]["arrays"])
        if len(entry["chunks"]) == 4
    )
    for tamper in (
        lambda chunks: chunks.pop(),
        lambda chunks: chunks[1].update(index=[[1, 3], [0, 8]]),
    ):
        index = json.loads(json.dumps(original))
        tamper(index["records"][0]["arrays"][moment]["chunks"])
        index_file.write_text(json.dumps(index))
        pointer["sha256"] = hashlib.sha256(index_file.read_bytes()).hexdigest()
        path.write_text(json.dumps(pointer))
        with pytest.raises(
            rfft.OptimizerStateRestoreError, match="shard layout mismatch"
        ):
            _load_placement(path, bundle, state)


def test_corruption_and_failed_publication_preserve_last_complete_checkpoint(
    tmp_path, monkeypatch
):
    from rollfast.finetune import array_storage

    plan, bundle = _bundle()
    saved, template = _checkpoints(plan, bundle)
    path = tmp_path / "optimizer.rfarr"
    rfft.save_array_state_checkpoint(path, saved, limits=LIMITS)
    original_pointer = path.read_bytes()

    def fail_publish(*args, **kwargs):
        raise OSError("publication failed")

    with monkeypatch.context() as patch:
        patch.setattr(array_storage, "_atomic_json", fail_publish)
        with pytest.raises(rfft.OptimizerStateRestoreError, match="publication failed"):
            rfft.save_array_state_checkpoint(path, saved, limits=LIMITS)
    assert path.read_bytes() == original_pointer
    loaded = rfft.load_array_state_checkpoint(
        path,
        bundle,
        template=template,
        model_checkpoint_id="training-model-1",
        limits=LIMITS,
    )
    assert loaded.manifest["counters"]["successful_update"] == 1

    pointer = array_storage._read_json(path, LIMITS.max_manifest_bytes)
    generation = array_storage._generation_root(path) / pointer["generation"]
    shard = next(generation.glob("rank-00000/*.npy"))
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(
        rfft.OptimizerStateRestoreError, match="size mismatch|checksum mismatch"
    ):
        rfft.load_array_state_checkpoint(
            path,
            bundle,
            template=template,
            model_checkpoint_id="training-model-1",
            limits=LIMITS,
        )


def test_safe_reader_rejects_unsafe_shard_name(tmp_path):
    from rollfast.finetune import array_storage

    plan, bundle = _bundle()
    saved, template = _checkpoints(plan, bundle)
    path = tmp_path / "optimizer.rfarr"
    rfft.save_array_state_checkpoint(path, saved, limits=LIMITS)
    pointer = json.loads(path.read_text())
    index_file = (
        array_storage._generation_root(path) / pointer["generation"] / "index.json"
    )
    index = json.loads(index_file.read_text())
    index["records"][0]["arrays"][0]["chunks"][0]["file"] = "../../outside.npy"
    index_file.write_text(json.dumps(index))
    pointer["sha256"] = hashlib.sha256(index_file.read_bytes()).hexdigest()
    path.write_text(json.dumps(pointer))

    with pytest.raises(rfft.OptimizerStateRestoreError, match="filename is invalid"):
        rfft.load_array_state_checkpoint(
            path,
            bundle,
            template=template,
            model_checkpoint_id="training-model-1",
            limits=LIMITS,
        )


def test_distributed_shards_and_local_rng_round_trip(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.lower().endswith("_proxy"):
            del environment[name]
    environment.update(
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false",
        OMP_NUM_THREADS="1",
    )
    workers = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.finetune._array_storage_worker",
                address,
                str(rank),
                str(tmp_path / "shared.rfarr"),
            ],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(2)
    ]
    try:
        outputs = [worker.communicate(timeout=45) for worker in workers]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
    for worker, (stdout, stderr) in zip(workers, outputs, strict=True):
        assert worker.returncode == 0, stdout + stderr
        assert "restored" in stdout
