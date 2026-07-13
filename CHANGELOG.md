# Changelog

All notable changes to Rollfast are documented here. Rollfast follows
[Semantic Versioning](https://semver.org/) for the supported public API defined
in [`docs/stability.md`](./docs/stability.md).

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

[1.0.0]: https://gitlab.com/rhizome-labs/public/rollfast/-/releases/v1.0.0
