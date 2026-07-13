# Stability and Support Policy

Rollfast 1.x provides a stable software contract for a library of experimental
optimization algorithms. “Stable” refers to API and compatibility guarantees;
it does not claim that every research optimizer has equivalent convergence,
throughput, or reference-profile fidelity on every workload.

## Stable Public Surface

The following are public API when exported through the package's `__all__`:

- `rollfast`;
- `rollfast.optim`;
- `rollfast.schedules`;
- `rollfast.finetune`;
- `rollfast.integrations`.

Modules and names beginning with `_` are private. Direct imports from other
implementation modules are not covered unless the same object is also exported
through one of the public surfaces above.

Within a 1.x release, Rollfast preserves:

- public import paths, call signatures, and documented defaults;
- the Optax `init` and `update` contract;
- optimizer state-tree structure and counter ownership;
- dtype, complex-number, masking, sharding, and PRNG semantics;
- fine-tuning manifest and checkpoint schema compatibility.

An incompatible change to these contracts requires a major release. Additive
parameters and exports may be introduced in a minor release.

## Deprecations

Public API is deprecated before removal. A deprecation identifies the
replacement and the earliest major release in which removal may occur. Private
implementation details may change without deprecation.

## Research Profile Maturity

Some optimizer reports expose `profile_fidelity="experimental"`,
`reference_validated=False`, known deviations, or benchmark warnings. These
fields describe scientific/reference maturity, not API stability. Their
constructors, report fields, Optax behavior, and serialized schema remain
covered by the v1 contract, while task quality and paper parity require
workload-specific validation.

## Supported Runtime

The supported runtime range is declared in `pyproject.toml`. CI exercises both
the minimum supported JAX/Optax versions and the current resolver on Python
3.12. Optional Equinox and Equimo integration is tested separately from the
minimal base import.

Collective behavior is tested with multiple CPU devices. GPU-specific kernels,
performance, memory use, and production sharding must be validated on the
target accelerator; they are not implied by a green CPU test suite.

## Checkpoints

`rollfast.finetune.SCHEMA_VERSION` identifies the manifest/checkpoint schema.
Compatible restores validate the schema, plan fingerprint, method metadata,
state structure, and sharding policy. Raw Optax state created by a different
Rollfast major version is not automatically portable.

Pickle helpers are for trusted local files only. Use a backend-native,
non-executable array checkpoint format across trust boundaries.
