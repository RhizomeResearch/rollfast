# Optimizer State And Checkpoints

Rollfast optimizer checkpoints are backend-neutral: the logical object contains
a serializable manifest and the optimizer state PyTree. Use your preferred
array checkpoint backend, or the local pickle helpers for scripts and tests.

```python
checkpoint = rfft.make_state_checkpoint(
    optim,
    opt_state,
    metadata={"step": step},
)
restored_state = rfft.restore_state_checkpoint(optim, checkpoint)
```

Strict restore compares the plan fingerprint in the checkpoint manifest against
the current optimizer bundle. A mismatch fails before the state is used:

```python
restored_state = rfft.restore_state_checkpoint(optim, checkpoint, strict=True)
```

Local helper:

```python
rfft.save_state_checkpoint("optimizer.rfopt", optim, opt_state)
opt_state = rfft.load_state_checkpoint("optimizer.rfopt", optim)
```

Equimo model/delta serialization remains separate. Save the Equimo model or
fine-tuning bundle with Equimo, and save the Rollfast optimizer state alongside
it with the same training step metadata.
