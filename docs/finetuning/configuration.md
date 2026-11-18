# Fine-Tuning Configuration Support

Rollfast rejects fine-tuning configuration values that do not have runtime semantics. Unsupported values raise
`NotImplementedError` when their config object is constructed, including through `from_dict`. Serialized field names and
schema version 1 remain unchanged.

| Field                                          | Supported                                                                      | Unsupported                          |
| ---------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------ |
| `ScheduleConfig.kind`                          | `constant`, `warmup_cosine`, `cosine_annealing`, `wsd`, `linear`, `polynomial` | Other kinds                          |
| `ScheduleConfig.step_counter`                  | `optimizer`, `attempt`                                                         | `micro`                              |
| `AccumulationConfig.remainder`                 | `error`                                                                        | `drop`, `apply_with_true_normalizer` |
| `AccumulationConfig.reduce_after_accumulation` | `True`                                                                         | `False`                              |
| `AccumulationConfig.finite_policy`             | `discard_window`                                                               | `error`                              |
| `PrecisionConfig.cast_back`                    | `nearest`                                                                      | `stochastic`                         |
| `PrecisionConfig.update_compute_dtype`         | `float32`                                                                      | Other dtypes                         |

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

## Cosine annealing and per-interval schedules

`ScheduleConfig(kind="cosine_annealing")` follows PyTorch's `CosineAnnealingLR` after the shared linear warmup:

```text
lr(i) = end_lr + (peak_lr - end_lr) * (1 + cos(pi * i / t_max)) / 2,   end_lr = end_lr_ratio * peak_lr
```

where `i` counts slots after warmup and `t_max` defaults to `total_steps - warmup_steps`. Unlike `warmup_cosine`, which
reaches `end_lr` on the run's last slot, this kind reaches it only at slot `t_max`, so with `t_max = total_steps` the
last slot is still at `(1 + cos(pi (t_max - 1) / t_max)) / 2` of the way from the floor to the peak. Warmup
(`warmup_steps` or `warmup_fraction`, default 5%) and `end_lr_ratio` (default 0.01) mean the same as for every other
kind; set `warmup_fraction=0.0` and `end_lr_ratio=0.0` to match PyTorch's defaults. Set `t_max` to anneal over a period
other than the run. `after_end="clamp"` (the default) holds `end_lr` after `t_max`; `after_end="cycle"` keeps following
the cosine back up, as PyTorch does. `t_max` and `after_end="cycle"` apply only to this kind.

Rollfast evaluates PyTorch's closed form, which PyTorch's scheduler computes recursively, so values can differ from
PyTorch's in the last float bits. The float32 evaluation uses the equivalent `sin(pi * (1 - i / t_max) / 2) ** 2` form,
which keeps its relative accuracy in the last slots, where the factor is small.

`steps_per_interval=k` (default 1) holds the learning rate for `k` counts of the schedule clock and applies to every
kind: the schedule is indexed by `count // k`, and `total_steps`, `warmup_steps`, `decay_steps`, and `t_max` count
intervals. Set `k` to the number of updates per epoch to step a schedule once per epoch, as PyTorch training loops
usually do. `ScheduleConfig.total_updates` is `total_steps * k`; the SWA start fraction, `OptimizerReport.total_steps`,
and the default `preview_schedule` steps use it. The builders' `total_steps` argument fills `ScheduleConfig.total_steps`
and therefore also counts intervals. With a short last epoch (`drop_last=False`), pass the number of batches the loader
yields per epoch, `ceil(examples / batch_size)`.

The interval index divides whichever clock `step_counter` selects. Under `"attempt"`, a skipped or rejected attempt
still moves toward the next interval; with accumulation, one count is one optimizer window. `rollfast.schedules` exposes
the same building blocks for optimizers used without a plan: `cosine_annealing_schedule(...)` and
`interval_schedule(schedule, steps_per_interval)`.

## Schedule clocks and skipped attempts

`ScheduleConfig.step_counter` selects the clock that indexes the learning-rate schedule:

- `"optimizer"` (the default) counts applied updates. A rejected update, a discarded window, or a skipped batch leaves
  the schedule where it was.
- `"attempt"` counts update attempts. Every attempt consumes one schedule slot, whether its update is applied, rejected
  as nonfinite, or skipped, so fits that see empty batches at different attempts still share one learning rate per
  attempt and reach the schedule's last value at attempt `total_updates - 1`. With accumulation, one attempt is one
  optimizer window: microbatches inside a window do not move the clock, and a discarded window consumes one slot. Adam
  moments, Adam's bias-correction count, and weight decay still follow applied updates only.

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
