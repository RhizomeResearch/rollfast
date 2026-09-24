# Fine-Tuning Averaging

Rollfast keeps EMA and SWA on the optimizer side. Equimo still owns the model tree and frozen/trainable partition;
Rollfast maintains extra parameter views inside the optimizer state.

```python
import rollfast.finetune as rfft

optim = rfft.adamw_from_plan(
    plan,
    total_steps=20_000,
    ema=rfft.EMAConfig(enabled=True, decay=0.9999),
    swa=rfft.SWAConfig(enabled=True, start_fraction=0.75),
)
```

Use named views when validating or checkpointing:

```python
from rollfast import apply_updates

trainable = apply_updates(trainable, updates, stochastic=False)
eval_ema = optim.eval_params(trainable, opt_state, view="ema")
eval_swa = optim.eval_params(trainable, opt_state, view="swa")
eval_model = plan.combine(eval_ema)
```

`view="optimizer"` returns the ordinary training parameters. Schedule-Free bundles additionally expose
`view="schedule_free"`.

Select a view explicitly when creating an evaluated model from a checkpoint. Use `"optimizer"` for ordinary training
parameters, even when EMA or SWA is enabled:

```python
checkpoint = rfft.make_state_checkpoint(
    optim,
    opt_state,
    model_checkpoint_id=model_checkpoint_id,
    model_state=model_state,
    counters={"successful_update": applied_updates},
)
evaluated = rfft.bind_evaluation_view(
    optim,
    plan,
    trainable,
    checkpoint,
    view="optimizer",
    model_checkpoint_id=model_checkpoint_id,
    optimizer_checkpoint_id=optimizer_checkpoint_id,
    applied_updates=applied_updates,
)
```

`evaluated.model` contains the selected trainable leaves combined with the plan's frozen leaves. Use it with
`evaluated.model_state` for validation and export. The result also records the requested view, both checkpoint IDs,
optimizer fingerprint, applied-update position and, for EMA/SWA, the number of recorded averages. Keep that result with
the evaluated artifact so a different view has a distinct identity. The caller must load `plan` and `trainable` from
`model_checkpoint_id`; Rollfast validates the optimizer checkpoint against that ID and checks stored counters when
available. A plan with mutable model state requires a model-state snapshot in the checkpoint.

EMA and SWA views raise `ValueError` until their first average is recorded. A view unavailable in the compiled bundle
also raises `ValueError`. Select an averaged view outside `jax.jit`; optimizer updates remain JIT compatible.

EMA/SWA state advances only when an optimizer step is actually applied. With gradient accumulation, withheld microsteps
do not update the averages.

These wrappers reconstruct the updated training parameters using deterministic rounding. Apply returned updates with
`stochastic=False` so that the training parameters and the parameter values used for averaging agree.

EMA can be restricted to plan groups by tag:

```python
optim = rfft.adamw_from_plan(
    plan,
    total_steps=20_000,
    ema=rfft.EMAConfig(
        enabled=True,
        decay=0.9999,
        include_tags=("head",),
        exclude_tags=("bias",),
    ),
)
```

Included leaves use the EMA average in `view="ema"`. Excluded leaves return the current optimizer parameters, so
evaluation trees keep the same structure as the training tree.
