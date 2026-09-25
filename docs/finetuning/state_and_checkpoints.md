# Optimizer State And Checkpoints

For reader selection and migration from an application-owned checkpoint format, see
[fine-tuning compatibility](compatibility.md).

Rollfast optimizer checkpoints are backend-neutral: the logical object contains a serializable manifest and the
optimizer state PyTree. Use your preferred array checkpoint backend, or the pickle helpers only for trusted local files
in scripts and tests.

```python
checkpoint = rfft.make_state_checkpoint(
    optim,
    opt_state,
    model_checkpoint_id="model-step-100",
    metadata={"step": step},
)
restored_state = rfft.restore_state_checkpoint(
    optim,
    checkpoint,
    model_checkpoint_id="model-step-100",
)
```

Strict restore compares the plan fingerprint in the checkpoint manifest against the current optimizer bundle. A mismatch
fails before the state is used, but the fingerprint and model checkpoint ID do not authenticate a checkpoint:

```python
restored_state = rfft.restore_state_checkpoint(
    optim,
    checkpoint,
    model_checkpoint_id="model-step-100",
    strict=True,
)
```

The local helpers use pickle, which may execute code while loading. Pass `trusted=True` only as an explicit
acknowledgement that the file is trusted and local:

```python
rfft.save_state_checkpoint(
    "optimizer.rfopt",
    optim,
    opt_state,
    model_checkpoint_id="model-step-100",
)
opt_state = rfft.load_state_checkpoint(
    "optimizer.rfopt",
    optim,
    trusted=True,
    model_checkpoint_id="model-step-100",
)
```

For a non-executable checkpoint on a shared POSIX filesystem, use the array helpers. Supply explicit limits and rebuild
a trusted template from the same compiled plan and expected component structures. All participating JAX processes call
save and load in the same order. The template supplies Python container types, static fields and target sharding; the
file supplies only metadata and numeric array bytes. See
[`examples/finetuning/array_checkpoint.py`](../../examples/finetuning/array_checkpoint.py) for a runnable example.

```python
limits = rfft.ArrayCheckpointLimits(
    max_manifest_bytes=2_000_000,
    max_arrays=10_000,
    max_shards=20_000,
    max_array_bytes=100_000_000,
    max_total_bytes=1_000_000_000,
)
checkpoint = rfft.make_state_checkpoint(
    optim,
    opt_state,
    model_checkpoint_id=training_model_id,
    base_model_value_hash=training_model_digest,
    counters={"successful_update": applied_updates},
)
rfft.save_array_state_checkpoint("optimizer.rfarr", checkpoint, limits=limits)

# After restoring the matching training model, compile its plan again and
# construct a template with matching optional component structures.
template = rfft.make_state_checkpoint(
    optim,
    optim.init(trainable),
    model_checkpoint_id=training_model_id,
    base_model_value_hash=training_model_digest,
    counters={"successful_update": 0},
)
checkpoint = rfft.load_array_state_checkpoint(
    "optimizer.rfarr",
    optim,
    template=template,
    model_checkpoint_id=training_model_id,
    base_model_value_hash=training_model_digest,
    applied_updates=applied_updates,
    limits=limits,
)
opt_state = checkpoint.state
```

The reader validates the plan fingerprint, effective group table, optimizer configuration, expected model identity,
component tree, leaf shape and dtype, sharding layout, file lengths and SHA-256 checksums before returning state. These
checks detect incompatibility and accidental damage; they do not authenticate the checkpoint's origin. Apply source
admission and access controls separately. Pass the training model's identity and successful-update count from an
independently restored model artifact. An evaluated EMA or SWA model is a distinct artifact.

The array format is versioned separately from the logical optimizer checkpoint schema. It writes uncompressed NumPy
arrays with pickle disabled and supports bf16, complex arrays and typed PRNG keys without numeric conversion. The reader
requires finite caller-selected limits. Checkpoint metadata must be JSON serializable. It restores onto the same process
count and the same partition of each sharded array; changing the partition requires an explicit migration outside this
reader. Each distinct shard is written once, by the process that holds its primary replica, and restored to every device
that holds a copy. A shared POSIX filesystem is required for multiple processes because a process may read shards
written by another. Run only one writer per path at a time.

Writers create immutable generations and atomically replace a small index after every participant has written and synced
its shards. A failed write leaves the previous completed generation readable. Incomplete generations are never loaded;
administrators can remove them after excluding concurrent writers and readers. Old generations remain available until
separately retired. File checksums do not replace a signed or otherwise authenticated artifact inventory.

Existing logical checkpoints without `state_schema` remain loadable through the older trusted path. Pass `template=` to
`restore_state_checkpoint` when qualifying such a checkpoint's tree. The array reader requires a checkpoint produced
with the new structural schema and does not load pickle files. The pickle helper now also writes through a temporary
file before replacing its destination, but its read path still executes pickle and requires `trusted=True`.

Equimo model/delta serialization remains separate. Save the Equimo model or fine-tuning bundle with Equimo, and save the
Rollfast optimizer state alongside it with the same training step metadata.

For evaluation, retain the matching model-state snapshot and successful-update count in the checkpoint. Pass the loaded
checkpoint to `bind_evaluation_view` with an explicit view and both artifact IDs; the returned model, model state and
view metadata can then travel together to validation or export. See [Fine-Tuning Averaging](averaging.md) for an
example.
