# Fine-Tuning Configuration Support

Rollfast rejects fine-tuning configuration values that do not have runtime
semantics. Unsupported values raise `NotImplementedError` when their config
object is constructed, including through `from_dict`. Serialized field names
and schema version 1 remain unchanged.

| Field | Supported | Unsupported |
| --- | --- | --- |
| `ScheduleConfig.kind` | `constant`, `warmup_cosine`, `wsd`, `linear`, `polynomial` | `custom` |
| `ScheduleConfig.step_counter` | `optimizer` | `micro` |
| `AccumulationConfig.remainder` | `error` | `drop`, `apply_with_true_normalizer` |
| `AccumulationConfig.reduce_after_accumulation` | `True` | `False` |
| `AccumulationConfig.finite_policy` | `discard_window` | `error` |
| `PrecisionConfig.cast_back` | `nearest` | `stochastic` |

With `remainder="error"`, callers must finish training on a complete
accumulation window. Rollfast does not currently expose a finalization API for
flushing or dropping a partial window, so the caller owns that lifecycle.
`finite_policy="discard_window"` discards an accumulated window when it contains
nonfinite values.

`cast_back="nearest"` describes the fine-tuning optimizer's deterministic
cast-back behavior. For stochastic BF16 parameter updates, use the separate
[`apply_updates(..., stochastic=True)` helpers](../usage.md#low-precision-updates)
with a PRNG key.

`schedule_free_adam_from_plan` defaults to a WSD schedule. When a caller supplies
any supported `ScheduleConfig` explicitly, the builder uses and reports that
schedule without replacing it.
