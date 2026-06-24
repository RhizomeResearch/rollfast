# Fine-Tuning Averaging

Rollfast keeps EMA and SWA on the optimizer side. Equimo still owns the model
tree and frozen/trainable partition; Rollfast maintains extra parameter views
inside the optimizer state.

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
trainable = optax.apply_updates(trainable, updates)
eval_ema = optim.eval_params(trainable, opt_state, view="ema")
eval_swa = optim.eval_params(trainable, opt_state, view="swa")
eval_model = plan.combine(eval_ema)
```

`view="optimizer"` returns the ordinary training parameters. Schedule-Free
bundles additionally expose `view="schedule_free"`.

EMA/SWA state advances only when an optimizer step is actually applied. With
gradient accumulation, withheld microsteps do not update the averages.

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

Included leaves use the EMA average in `view="ema"`. Excluded leaves return the
current optimizer parameters, so evaluation trees keep the same structure as the
training tree.
