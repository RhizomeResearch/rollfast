# Fine-Tuning Configuration Support

Rollfast rejects fine-tuning configuration values that do not have runtime semantics. Unsupported values raise
`NotImplementedError` when their config object is constructed, including through `from_dict`. Serialized field names and
schema version 1 remain unchanged.

| Field                                          | Supported                                                  | Unsupported                          |
| ---------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ |
| `ScheduleConfig.kind`                          | `constant`, `warmup_cosine`, `wsd`, `linear`, `polynomial` | `custom`                             |
| `ScheduleConfig.step_counter`                  | `optimizer`, `attempt`                                     | `micro`                              |
| `AccumulationConfig.remainder`                 | `error`                                                    | `drop`, `apply_with_true_normalizer` |
| `AccumulationConfig.reduce_after_accumulation` | `True`                                                     | `False`                              |
| `AccumulationConfig.finite_policy`             | `discard_window`                                           | `error`                              |
| `PrecisionConfig.cast_back`                    | `nearest`                                                  | `stochastic`                         |
| `PrecisionConfig.update_compute_dtype`         | `float32`                                                  | Other dtypes                         |

With `remainder="error"`, callers must finish training on a complete accumulation window. Rollfast does not currently
expose a finalization API for flushing or dropping a partial window, so the caller owns that lifecycle.
`finite_policy="discard_window"` discards an accumulated window when it contains nonfinite values.

Use `bundle.update_with_status(grads, state, params)` when the training loop needs to record update outcomes. It returns
`(updates, next_state, status)`; `bundle.update` keeps the Optax-compatible two-value result. Exactly one of the JAX
scalar flags `status.accumulating`, `status.update_applied`, and `status.window_discarded` is true. The status also
reports `microsteps_in_window`. A completed but discarded window does not advance the optimizer's moments, weight decay,
or EMA/SWA averages, and under the default schedule clock it does not advance the schedule either. Optax
`MultiSteps.gradient_step` counts completed windows, including discarded ones; use `status.update_applied` or an
applied-update counter for training progress. A legitimate zero-gradient update can still apply AdamW weight decay, so
do not run an update for a batch that has no valid training targets; record it as a skipped attempt (see
[schedule clocks and skipped attempts](#schedule-clocks-and-skipped-attempts)).

Call `validate_accumulation_complete(state, accumulation_config)` at the end of an epoch or run. It accepts a compiled
optimizer state (including EMA/SWA wrappers), a `FineTuneStepState`, or an explicit `AccumulationState`, and raises
`ValueError` with the pending and required microstep counts for an incomplete window. It does not apply, drop, or flush
that window. Save an incomplete window in the optimizer checkpoint when resuming it later; validate completion only when
training is meant to finish.

Ordinary `MultiSteps` accumulation averages microbatch gradients under `normalization="examples"`. For batches with
different valid counts, use `LossBundle` sums and normalizers with `make_accumulating_loss_bundle_update_step` to obtain
the declared global weighted mean. Its `info.update_applied` reports the actual optimizer outcome, including overflow or
finite-guard rejection. The existing checkpoint schema and state structure are unchanged; previously stored averages
from discarded updates cannot be corrected retroactively.

`cast_back="nearest"` describes how float32 master parameters are cast back to the model dtype; the masters keep the
residual updates, so the cast needs no stochastic rounding. For BF16 parameter storage without masters, set
`parameter_rounding="stochastic"` with `master_params="never"`. [Low-precision fine-tuning](precision.md) describes the
precision modes, the remaining `PrecisionConfig` validation, and which step helpers honor each field.

`schedule_free_adam_from_plan` defaults to a WSD schedule. When a caller supplies any supported `ScheduleConfig`
explicitly, the builder uses and reports that schedule without replacing it.

`ScheduleConfig(kind="wsd")` accepts `warmup_shape` and `decay_shape` (`linear`, `cosine`, `power`, or `exponential`),
with `warmup_power`, `decay_power`, and `exponential_rate`. They default to linear and are ignored by other schedule
kinds. Checkpoints written before these fields existed restore under strict validation as linear WSD schedules.

WSD slot counts follow `wsd_schedule`:

- Warmup covers counts `0` through `warmup_steps`, which is `warmup_steps + 1` slots at
  `(count + 1) / (warmup_steps + 1)`. A ramp of `w` slots that reaches the peak at count `w - 1` needs
  `warmup_steps=w - 1`.
- The first decay slot, count `total_steps - decay_steps`, is still at the peak, and the last slot reaches the floor. So
  `decay_steps=d` gives `d - 1` slots below the peak.
- A decay whose first slot is already below the peak needs `decay_steps=d + 1`.

## Schedule clocks and skipped attempts

`ScheduleConfig.step_counter` selects the clock that indexes the learning-rate schedule:

- `"optimizer"` (the default) counts applied updates. A rejected update, a discarded window, or a skipped batch leaves
  the schedule where it was.
- `"attempt"` counts update attempts. Every attempt consumes one schedule slot, whether its update is applied, rejected
  as nonfinite, or skipped, so fits that see empty batches at different attempts still share one learning rate per
  attempt and reach the schedule's floor at attempt `total_steps - 1`. With accumulation, one attempt is one optimizer
  window: microbatches inside a window do not move the clock, and a discarded window consumes one slot. Adam moments,
  Adam's bias-correction count, and weight decay still follow applied updates only.

The attempt clock is the optimizer state's existing schedule count, so it needs no new state and is saved and restored
with the optimizer state. `counters.schedule_step` reports it in `FineTuneStepState`. `apollo_adamw_from_plan`,
`galore_adamw_from_plan`, and `schedule_free_adam_from_plan` read the learning rate from a count that other optimizer
state shares, so they accept only `"optimizer"`.

For a batch without training targets, call the step from `make_skip_attempt_step(bundle)` instead of the update step. It
computes no loss or gradient and leaves parameters, optimizer moments, model state, loss scale, and RNG streams
unchanged. `counters.microstep` and `counters.attempted_update` advance and `counters.successful_update` does not; under
`"attempt"` the skip also consumes its schedule slot. The skip step requires `accumulation.steps=1`.

```python
update = jax.jit(rfft.make_finetune_update_step(plan, loss_fn, bundle))
skip = jax.jit(rfft.make_skip_attempt_step(bundle))

for batch in batches:
    if has_targets(batch):
        params, step_state, info = update(params, step_state, batch)
    else:
        step_state = skip(step_state)
```

Under `"attempt"`, `make_accumulating_loss_bundle_update_step` also consumes a slot for a window whose normalizer is
zero or whose values are nonfinite.
