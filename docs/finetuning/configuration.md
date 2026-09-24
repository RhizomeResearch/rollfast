# Fine-Tuning Configuration Support

Rollfast rejects fine-tuning configuration values that do not have runtime semantics. Unsupported values raise
`NotImplementedError` when their config object is constructed, including through `from_dict`. Serialized field names and
schema version 1 remain unchanged.

| Field                                          | Supported                                                  | Unsupported                          |
| ---------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ |
| `ScheduleConfig.kind`                          | `constant`, `warmup_cosine`, `wsd`, `linear`, `polynomial` | `custom`                             |
| `ScheduleConfig.step_counter`                  | `optimizer`                                                | `micro`                              |
| `AccumulationConfig.remainder`                 | `error`                                                    | `drop`, `apply_with_true_normalizer` |
| `AccumulationConfig.reduce_after_accumulation` | `True`                                                     | `False`                              |
| `AccumulationConfig.finite_policy`             | `discard_window`                                           | `error`                              |
| `PrecisionConfig.cast_back`                    | `nearest`                                                  | `stochastic`                         |

With `remainder="error"`, callers must finish training on a complete accumulation window. Rollfast does not currently
expose a finalization API for flushing or dropping a partial window, so the caller owns that lifecycle.
`finite_policy="discard_window"` discards an accumulated window when it contains nonfinite values.

Use `bundle.update_with_status(grads, state, params)` when the training loop needs to record update outcomes. It returns
`(updates, next_state, status)`; `bundle.update` keeps the Optax-compatible two-value result. Exactly one of the JAX
scalar flags `status.accumulating`, `status.update_applied`, and `status.window_discarded` is true. The status also
reports `microsteps_in_window`. A completed but discarded window does not advance the optimizer's schedule, weight
decay, or EMA/SWA averages. Optax `MultiSteps.gradient_step` counts completed windows, including discarded ones; use
`status.update_applied` or an applied-update counter for training progress. A legitimate zero-gradient update can still
apply AdamW weight decay. If a batch has no valid training targets, record the skip and omit the update invocation.

Call `validate_accumulation_complete(state, accumulation_config)` at the end of an epoch or run. It accepts a compiled
optimizer state (including EMA/SWA wrappers) or an explicit `AccumulationState`, and raises `ValueError` with the
pending and required microstep counts for an incomplete window. It does not apply, drop, or flush that window. Save an
incomplete window in the optimizer checkpoint when resuming it later; validate completion only when training is meant to
finish.

Ordinary `MultiSteps` accumulation averages microbatch gradients under `normalization="examples"`. For batches with
different valid counts, use `LossBundle` sums and normalizers with `make_accumulating_loss_bundle_update_step` to obtain
the declared global weighted mean. Its `info.update_applied` reports the actual optimizer outcome, including overflow or
finite-guard rejection. The existing checkpoint schema and state structure are unchanged; previously stored averages
from discarded updates cannot be corrected retroactively.

`cast_back="nearest"` describes the fine-tuning optimizer's deterministic cast-back behavior. For stochastic BF16
parameter updates, use the separate [`apply_updates(..., stochastic=True)` helpers](../usage.md#low-precision-updates)
with a PRNG key.

`schedule_free_adam_from_plan` defaults to a WSD schedule. When a caller supplies any supported `ScheduleConfig`
explicitly, the builder uses and reports that schedule without replacing it.
