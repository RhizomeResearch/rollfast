# Changelog

All notable changes to Rollfast are documented here. Rollfast follows [Semantic Versioning](https://semver.org/) for the
supported public API defined in [`docs/stability.md`](./docs/stability.md).

## Unreleased

## [1.3.0] - 2026-10-15

Low-precision fine-tuning: explicit BF16/FP16 precision modes, master-free stochastic parameter rounding, and composite
steps that honor the whole precision policy. Detailed release notes are available in
[`docs/releases/1.3.0.md`](./docs/releases/1.3.0.md).

### Added

- Add `PrecisionConfig.parameter_rounding="stochastic"` for master-free BF16 parameter storage. The composite and
  stateful steps draw rounding keys from `RNGStreams.stochastic_rounding`, which advances only with applied updates, and
  give each parameter partition independent noise.
- Report the resolved precision mode, master and moment storage, rounding, and parameter bytes in
  `OptimizerReport.precision` (`PrecisionReport`) and the manifest's `resolved_precision`.
- Add the `bf16_mixed_precision`, `bf16_master_free_precision`, and `fp16_mixed_precision` presets, and a `precision=`
  argument on every `*_from_plan` builder.
- Support gradient accumulation in `make_finetune_update_step`, and add `make_finetune_sam_step` for SAM/ASAM with
  master parameters, loss scaling, and stochastic parameter rounding.
- Register `FineTuneStepState`, `RNGStreams`, and `StepCounters` as PyTrees so composite steps run under `jax.jit` and
  `jax.lax.scan`.
- Add `make_step_state_checkpoint` and `step_state_from_checkpoint` for exact resume of a composite step state.
- Add `PrecisionConfig.min_loss_scale` and `max_loss_scale` bounds for dynamic loss scaling.
- Add `low_precision_update_metrics`, and master-parameter accounting in `optimizer_state_memory_summary` and
  `estimate_optimizer_state_memory`.
- Report `grads_finite` and `update_applied` in stateful step info.
- Document the precision modes, helper capabilities, and troubleshooting in
  [`docs/finetuning/precision.md`](./docs/finetuning/precision.md), with a runnable example.

### Changed

- With declared collective axes, every parameter partition reaches the same accept or reject decision.
- Dynamic loss scaling lowers the scale only on gradient overflow and stays within its configured bounds.
- The perturbed pass of `make_sam_step` evaluates the model at the trainable dtype, and its microbatch sums, like those
  of the `LossBundle` steps, accumulate in the accumulation dtype.
- `init_finetune_step_state` validates supplied master parameters, and its default accumulation record holds no
  gradient-sized buffer.
- Complex trainable leaves keep complex master parameters instead of being rejected.
- Direct-parameter helpers warn when they ignore the bundle's master parameters or loss scaling, and reject stochastic
  parameter rounding. `make_state_checkpoint` warns when a policy that keeps masters omits them.
- Reject precision settings that had no effect: non-floating dtypes, a non-float32 `update_compute_dtype`, sub-float32
  master dtypes, an `accumulation_dtype` that differs from `AccumulationConfig.accumulate_dtype`, a non-float32
  `preconditioner_dtype` outside `kron_adam`, and quantized moments whose `moment_dtype` differs from the fallback
  dtype. `hybrid_kron_adam_from_plan` now honors `precision.preconditioner_dtype`.
- `validate_accumulation_complete` accepts a `FineTuneStepState`.

### Fixed

- Keep the state structure of master-aware steps stable when the policy keeps no masters, such as float32 weights under
  `master_params="auto"`. They return `None` for master parameters instead of a copy of the visible parameters, which
  made a jitted step recompile on its second call and stopped checkpoint templates from matching. Policies that keep
  masters still need them, and any loss-scale state, passed from the first call; the composite steps create both up
  front.
- Reject updates that overflow the visible dtype even when the gradients are finite.
- Compute global-norm clipping and SAM norms by promoting to float32 before squaring, so FP16 gradients above 256 no
  longer overflow, and apply the clip scale in float32.
- Project and square BF16 gradients in float32 in GaLore and APOLLO.
- Keep every NaN a NaN under BF16 stochastic rounding.
- Keep the precision of float64 and complex leaves in `make_master_params`.

## [1.2.1] - 2026-09-30

### Fixed

- Make plan and model-state structural fingerprints deterministic across Python processes, including supported static
  callables, partial arguments, dataclasses, and dtype metadata. Reject unsupported static metadata explicitly.

### Changed

- Introduce deterministic fingerprint format v1 and require its version during strict optimizer checkpoint restore. Old
  unversioned checkpoints are incompatible and must be regenerated; checkpoint schema and array format remain v1.

## [1.2.0] - 2026-09-24

Fine-tuning state, checkpoint, and update coordination improvements. Detailed release notes are available in
[`docs/releases/1.2.0.md`](./docs/releases/1.2.0.md).

### Added

- Add bounded, non-executable array checkpoints with checksum and layout validation, including distributed saves and
  restores on a shared filesystem that store each replicated shard once. Restores follow the template's placement,
  accept shape-only `jax.ShapeDtypeStruct` templates, and assemble arrays across a changed mesh or partition.
- Add checkpoint-bound optimizer and EMA evaluation views, explicit update outcomes for accumulation windows, and a
  host-side check for incomplete accumulation.

### Changed

- Check logical checkpoint state structure and component presence, effective optimizer groups, and optional model
  identity and update position during restore. Existing schema-version-1 checkpoints remain readable using a matching
  restore template when needed.
- Qualify full-encoder and frozen-encoder query-model plans, stateful updates, and checkpoint replay with Equimo 2.4.0;
  require Equimo `>=2.4.0,<3.0.0` for the optional development integration.
- Document plan compilation, checkpoint migration, loss normalization, and evaluation-view ownership.

### Fixed

- Reject required group rules that select no trainable parameters, and report discarded accumulation windows without
  advancing their applied-update counters.
- Restore typed PRNG keys from array checkpoints on the minimum supported JAX version.
- Preserve model normalization state on rejected stateful steps, including SAM, and reject gradients that overflow when
  cast to the configured precision.

## [1.1.0] - 2026-09-05

Performance and maintenance improvements with the existing v1 public API, optimizer defaults, and checkpoint layouts
preserved. Detailed release notes are available in [`docs/releases/1.1.0.md`](./docs/releases/1.1.0.md).

### Performance

- Compute GaLore SVD bases and transport projected moments only when a basis refresh is due; regenerate APOLLO
  projections only at their refresh cadence.
- Skip EMA/SWA parameter casting and averaging, and AdaLoRA rank allocation, outside their configured update boundaries.
- Stop Aurora conjugate-gradient matrix products after convergence while retaining differentiation through the solver.
- Use sequential JAX scans for SAM and `LossBundle` microbatch accumulation to reduce tracing and compilation work while
  retaining accumulation order.
- Cache Adam8 codebooks, construct quantized zero moments directly, and avoid unused PRNG splits while preserving
  stochastic-rounding key progression.
- Skip pure optimizer computations on rejected loss-scaled and stateful SAM steps, retaining collective participation
  and custom optimizer effects.
- Evaluate factorized AdamW learning-rate schedules once per parameter group and remove redundant PSGD preconditioner
  selection.
- Batch eligible optimizer-state transfers to the host.

### Changed

- Consolidate shared optimizer, schedule, compiler, update, and state traversal helpers, and use public Optax symbols
  where equivalent APIs are available.
- Consolidate duplicate test fixtures and cases, replace selected forwarding mocks with numerical references, and record
  readable v1 public API snapshots.
- Add regressions for conditional execution, microbatch totals, PRNG sequences, low-precision state, differentiation,
  and multi-device CPU sharding.
- Document the production and test simplification audits.

## [1.0.0] - 2026-07-13

The first stable API release. Detailed release notes are available in
[`docs/releases/1.0.0.md`](./docs/releases/1.0.0.md), and 0.x users should read
[`docs/migrating-to-1.0.md`](./docs/migrating-to-1.0.md).

### Added

- Plan-aware fine-tuning compilation, grouping, schedules, update-step helpers, averaging, state migration, diagnostics,
  and checkpoint manifests under `rollfast.finetune`.
- AdamW with blockwise 8-bit optimizer state.
- Muon, NorMuon/ContraMuon, Pion, RMNP, TrasMuon, SODA, GaLore, APOLLO, SAM, and Hyperball optimizer primitives and
  wrappers.
- Dimension-number helpers, mixed-precision policies, complex-parameter validation, partition-aware reductions, and
  optional Equimo integration.
- Public-import and optimizer-signature compatibility tests.

### Changed

- Python 3.12 or newer is required.
- Optax support is constrained to `>=0.2.8,<0.3.0` because Rollfast uses selected Optax implementation helpers.
- Public API, optimizer state, defaults, and checkpoint compatibility now follow the v1 stability policy.

### Security

- Pickle checkpoint loading is fail-closed unless callers explicitly pass `trusted=True`; non-executable backend-native
  formats remain recommended across trust boundaries.

[1.0.0]: https://gitlab.com/rhizome-labs/public/rollfast/-/releases/v1.0.0
[1.1.0]: https://gitlab.com/rhizome-labs/public/rollfast/-/releases/v1.1.0
[1.2.0]: https://gitlab.com/rhizome-labs/public/rollfast/-/releases/v1.2.0
