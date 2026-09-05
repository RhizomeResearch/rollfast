# Changelog

All notable changes to Rollfast are documented here. Rollfast follows
[Semantic Versioning](https://semver.org/) for the supported public API defined
in [`docs/stability.md`](./docs/stability.md).

## [1.1.0] - 2026-09-05

Performance and maintenance improvements with the existing v1 public API,
optimizer defaults, and checkpoint layouts preserved. Detailed release notes
are available in [`docs/releases/1.1.0.md`](./docs/releases/1.1.0.md).

### Performance

- Compute GaLore SVD bases and transport projected moments only when a basis
  refresh is due; regenerate APOLLO projections only at their refresh cadence.
- Skip EMA/SWA parameter casting and averaging, and AdaLoRA rank allocation,
  outside their configured update boundaries.
- Stop Aurora conjugate-gradient matrix products after convergence while
  retaining differentiation through the solver.
- Use sequential JAX scans for SAM and `LossBundle` microbatch accumulation
  to reduce tracing and compilation work while retaining accumulation order.
- Cache Adam8 codebooks, construct quantized zero moments directly, and avoid
  unused PRNG splits while preserving stochastic-rounding key progression.
- Skip pure optimizer computations on rejected loss-scaled and stateful SAM
  steps, retaining collective participation and custom optimizer effects.
- Evaluate factorized AdamW learning-rate schedules once per parameter group
  and remove redundant PSGD preconditioner selection.
- Batch eligible optimizer-state transfers to the host.

### Changed

- Consolidate shared optimizer, schedule, compiler, update, and state traversal
  helpers, and use public Optax symbols where equivalent APIs are available.
- Consolidate duplicate test fixtures and cases, replace selected forwarding
  mocks with numerical references, and record readable v1 public API snapshots.
- Add regressions for conditional execution, microbatch totals, PRNG sequences,
  low-precision state, differentiation, and multi-device CPU sharding.
- Document the production and test simplification audits.

## [1.0.0] - 2026-07-13

The first stable API release. Detailed release notes are available in
[`docs/releases/1.0.0.md`](./docs/releases/1.0.0.md), and 0.x users should read
[`docs/migrating-to-1.0.md`](./docs/migrating-to-1.0.md).

### Added

- Plan-aware fine-tuning compilation, grouping, schedules, update-step helpers,
  averaging, state migration, diagnostics, and checkpoint manifests under
  `rollfast.finetune`.
- AdamW with blockwise 8-bit optimizer state.
- Muon, NorMuon/ContraMuon, Pion, RMNP, TrasMuon, SODA, GaLore, APOLLO, SAM,
  and Hyperball optimizer primitives and wrappers.
- Dimension-number helpers, mixed-precision policies, complex-parameter
  validation, partition-aware reductions, and optional Equimo integration.
- Public-import and optimizer-signature compatibility tests.

### Changed

- Python 3.12 or newer is required.
- Optax support is constrained to `>=0.2.8,<0.3.0` because Rollfast uses
  selected Optax implementation helpers.
- Public API, optimizer state, defaults, and checkpoint compatibility now
  follow the v1 stability policy.

### Security

- Pickle checkpoint loading is fail-closed unless callers explicitly pass
  `trusted=True`; non-executable backend-native formats remain recommended
  across trust boundaries.

[1.1.0]: https://gitlab.com/rhizome-labs/public/rollfast/-/releases/v1.1.0
[1.0.0]: https://gitlab.com/rhizome-labs/public/rollfast/-/releases/v1.0.0
