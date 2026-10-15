# Low-Precision Fine-Tuning

`PrecisionConfig` decides where parameter updates accumulate when trainable parameters are stored in BF16 or FP16. Every
compiled bundle resolves the policy against the plan and reports the result in `bundle.report.precision`, which is also
recorded in the manifest as `resolved_precision`:

```python
optim = rfft.adamw_from_plan(
    plan, total_steps=10_000, precision=rfft.bf16_mixed_precision()
)
print(optim.report.precision)
# master_mixed_precision: params=bfloat16 masters=float32 cast_back=nearest moments=float32 loss_scale=none
```

## Modes

| `report.precision.mode`  | Trainable storage    | Where updates accumulate                                                    | Extra bytes per parameter | Preset                                             |
| ------------------------ | -------------------- | --------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------- |
| `full_precision`         | float32 or wider     | The parameters                                                              | 0                         | `PrecisionConfig()`                                |
| `master_mixed_precision` | BF16/FP16 model copy | Float32 masters, cast back to the model dtype with round-to-nearest         | 4                         | `bf16_mixed_precision()`, `fp16_mixed_precision()` |
| `master_free_stochastic` | BF16                 | The BF16 parameters, through stochastic rounding of each float32 update sum | 0                         | `bf16_master_free_precision()`                     |
| `master_free_nearest`    | BF16/FP16            | The parameters with round-to-nearest; updates below half a spacing vanish   | 0                         | `PrecisionConfig(master_params="never")`           |

With float32 masters, the cast-back does not need stochastic rounding: the masters keep every residual update, and the
model copy is only a rounded view of them, so `cast_back` supports only `"nearest"`. Stochastic rounding replaces the
masters when their four bytes per parameter matter. It rounds each parameter up or down with probabilities that make the
stored value correct in expectation, so updates smaller than the BF16 spacing still move the parameters on average. It
applies to BF16 parameters only; FP16 parameters need masters and loss scaling. `master_free_nearest` is reported with a
warning because small updates are lost; it suits short runs whose updates stay well above the BF16 spacing.

`master_params="auto"` creates masters when any trainable leaf is BF16 or FP16. A master leaf never has less precision
than its model leaf, and complex leaves keep complex masters.

## Steps that honor the policy

Use the composite steps, which carry masters, loss scale, RNG streams, and counters in one `FineTuneStepState` PyTree
and therefore run under `jax.jit` and `jax.lax.scan`:

```python
optim = rfft.adamw_from_plan(
    plan, total_steps=10_000, precision=rfft.bf16_mixed_precision()
)
state = rfft.init_finetune_step_state(
    optim,
    bf16_trainable,
    master_params=fp32_trainable,  # The original float32 weights, before the cast.
    rng_seed=0,
)
step = jax.jit(rfft.make_finetune_update_step(plan, loss_fn, optim))
bf16_trainable, state, info = step(bf16_trainable, state, batch)
```

`loss_fn(model, model_state, key, *args)` returns `(loss, new_model_state)`. Without `master_params`, the state builds
masters by upcasting `bf16_trainable`, which keeps the rounding already applied to the weights. The masters must have
the model parameters' structure and shapes, and at least their precision.

| Helper                                                                           | Masters | Loss scaling | Stochastic parameter rounding | Notes                                                 |
| -------------------------------------------------------------------------------- | ------- | ------------ | ----------------------------- | ----------------------------------------------------- |
| `make_finetune_update_step`                                                      | Yes     | Yes          | Yes                           | Supports gradient accumulation                        |
| `make_finetune_sam_step`                                                         | Yes     | Yes          | Yes                           | Requires `accumulation_steps=1`                       |
| `make_stateful_loss_scaled_master_update_step`                                   | Yes     | Yes          | Yes                           | Separate state arguments                              |
| `make_loss_scaled_master_update_step`                                            | Yes     | Yes          | Rejected: carries no RNG      | Pure-loss variant                                     |
| `make_master_update_step`, `make_plan_master_update_step`                        | Yes     | No           | Rejected: carries no RNG      | No finite-value handling                              |
| `make_stateful_sam_step`                                                         | No      | No           | Yes                           | Updates the supplied parameters                       |
| `make_update_step`, `make_plan_update_step`, `make_sam_step`, `LossBundle` steps | No      | No           | Rejected                      | Warn when the bundle keeps masters or scales the loss |

The helpers that take masters and loss-scale state as separate arguments return `None` for masters when the policy keeps
none. When the policy keeps masters or scales the loss, pass those states from the first call: a `None` argument is
filled in on that call, which changes the argument structure, retraces a jitted step, and no longer matches a checkpoint
template built before training. `init_finetune_step_state` creates every component up front, so the composite steps keep
one structure from the start.

A rejected update leaves parameters, masters, optimizer state, and model state unchanged, and does not advance the
forward or stochastic-rounding RNG streams. `StatefulStepInfo.grads_finite` reports gradient overflow separately from
`update_applied`, so a finite gradient whose update leaves the model dtype's range shows as a rejected update.

## Loss scaling

FP16 gradients need loss scaling to stay above the FP16 subnormal range. `loss_scale="static"` multiplies the loss by
`static_loss_scale`. `loss_scale="dynamic"` starts there, multiplies the scale by `backoff_factor` after a gradient
overflow, and by `growth_factor` after `growth_interval` consecutive successful steps; with accumulation, each
microbatch call is a step. The scale stays within `[min_loss_scale, max_loss_scale]`. Only gradient overflow lowers the
scale: a rejected update from finite gradients is not a scaling problem, so it leaves the scale unchanged.

## Gradient accumulation

With `accumulation_steps > 1`, call `make_finetune_update_step` once per microbatch. The optimizer state owns the
window, so `FineTuneStepState.accumulation` tracks only its position. A nonfinite microbatch discards its whole window
at the boundary; this relies on the bundle's finite guard, so accumulation requires `GradientPolicy(nonfinite="skip")`,
the default. `counters.successful_update`, `counters.schedule_step`, and `info.update_applied` count applied updates;
`counters.microstep` counts calls.

`AccumulationConfig.accumulate_dtype` sets the accumulation dtype; `PrecisionConfig.accumulation_dtype` must match it.
The microbatch sums of `make_sam_step` and the `LossBundle` steps also accumulate in this dtype, so BF16 losses and
gradients do not stall: a BF16 running sum of ones stops growing at 256.

## Distributed training

When the gradient policy declares collective axes, every parameter partition reaches the same accept or reject decision,
so one shard's overflow rejects the step on all shards. Stochastic rounding folds the partition axis index into its key:
parameter shards draw independent noise, and replicas share it so replicated parameters stay identical.

## Checkpoints

Save a composite state with `make_step_state_checkpoint` and rebuild it with `step_state_from_checkpoint`; see
[optimizer state and checkpoints](state_and_checkpoints.md#composite-step-state). The checkpoint stores the masters,
loss scale, rounding stream, and counters, so a resumed run continues exactly. `make_state_checkpoint` warns when the
policy keeps masters but the checkpoint omits them, and requires the RNG streams for stochastic parameter rounding.

## Field semantics

| Field                          | Effect                                                                                                |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `expected_model_compute_dtype` | Recorded metadata. Rollfast never casts the model; the trainable tree's dtypes define the model copy. |
| `gradient_dtype`               | Gradients are cast to this dtype after unscaling, before the optimizer.                               |
| `accumulation_dtype`           | Must equal `AccumulationConfig.accumulate_dtype`.                                                     |
| `master_params`                | `"auto"`, `"always"`, or `"never"`.                                                                   |
| `master_param_dtype`           | At least float32 when masters are kept.                                                               |
| `moment_dtype`                 | Moment storage. `report.precision.moment_rounding` states whether BF16 moments round stochastically.  |
| `preconditioner_dtype`         | Kron preconditioner storage; other optimizers reject a non-float32 value.                             |
| `update_compute_dtype`         | Float32 only; Rollfast optimizers compute updates in float32.                                         |
| `cast_back`                    | `"nearest"` only.                                                                                     |
| `parameter_rounding`           | `"nearest"` or `"stochastic"`; `"stochastic"` requires `master_params="never"` and BF16 parameters.   |

Builders accept a full `precision=PrecisionConfig(...)`. Their `moment_dtype` and `preconditioner_dtype` arguments are
shorthands for the same fields and must agree with `precision` when both are given.

## Diagnostics

- `optimizer_state_memory_summary(optim, state, master_params=state.master_params)` and
  `estimate_optimizer_state_memory(plan, optim, include_master_params=True)` count masters under `"master_params"`.
- `low_precision_update_metrics(params, updates, storage_dtype=None)` returns parameter and update RMS, their ratio, and
  `below_half_spacing_fraction`: the fraction of nonzero updates that round-to-nearest storage would discard. It is
  JIT-compatible.

[`examples/finetuning/low_precision.py`](../../examples/finetuning/low_precision.py) compares the BF16 modes on a small
problem and resumes a run from a step-state checkpoint.

## Troubleshooting

| Symptom                                                                   | Cause and fix                                                                                                                   |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| BF16 loss plateaus early                                                  | The report says `master_free_nearest`. Keep masters or use `bf16_master_free_precision()`; check `below_half_spacing_fraction`. |
| `parameter_rounding='stochastic' ... carries no RNG`                      | The helper has no RNG streams. Use `make_finetune_update_step` or another RNG-aware step.                                       |
| `FineTuneStepState.master_params is None but ... keeps master parameters` | Build the state with `init_finetune_step_state` instead of replacing its masters with `None`.                                   |
| Warning that the checkpoint omits master parameters                       | Use `make_step_state_checkpoint`, pass `master_params`, or set `master_params="never"` for direct loops.                        |
| `accumulation requires GradientPolicy(nonfinite='skip')`                  | The composite steps discard nonfinite windows through the finite guard.                                                         |
| `PrecisionConfig.accumulation_dtype must match`                           | Set the accumulation dtype on `AccumulationConfig`, and the same value on `PrecisionConfig`.                                    |
| FP16 loss scale stays at `min_loss_scale`                                 | Gradients overflow even unscaled. Lower the learning rate, clip gradients, or train in BF16.                                    |
