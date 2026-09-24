# Optimizer State And Checkpoints

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

When checkpoints cross a trust boundary, use a backend-native, non-executable array checkpoint format instead of these
pickle helpers.

Equimo model/delta serialization remains separate. Save the Equimo model or fine-tuning bundle with Equimo, and save the
Rollfast optimizer state alongside it with the same training step metadata.

For evaluation, retain the matching model-state snapshot and successful-update count in the checkpoint. Pass the loaded
checkpoint to `bind_evaluation_view` with an explicit view and both artifact IDs; the returned model, model state and
view metadata can then travel together to validation or export. See [Fine-Tuning Averaging](averaging.md) for an
example.
