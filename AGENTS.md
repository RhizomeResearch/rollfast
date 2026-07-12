# Rollfast Agent Guide

## Project overview

Rollfast is a Python 3.12+ library of experimental JAX optimizers, schedules,
and fine-tuning utilities built around Optax transformations. Keep changes
compatible with JAX PyTrees, JIT compilation, mixed precision, and the public
Optax `init`/`update` contract.

## Repository map

- `src/rollfast/optim/`: optimizer transformations and public wrappers.
- `src/rollfast/schedules/`: learning-rate and schedule-free utilities.
- `src/rollfast/finetune/`: plan compilation, update steps, state migration,
  serialization, diagnostics, and fine-tuning policies.
- `src/rollfast/integrations/`: optional third-party integrations.
- `tests/`: unit, compatibility, integration, and fine-tuning tests. Mirror the
  source area when adding regression coverage.
- `examples/`, `benchmarks/`, and `docs/`: runnable usage, performance tooling,
  and user-facing documentation.
- `pyproject.toml`: package metadata, dependencies, and tool configuration.
- `.gitlab-ci.yml`: authoritative CI commands and release checks.

Do not edit generated or local-environment directories such as `.venv/`,
`.devenv/`, `dist/`, or tool caches.

## Environment and commands

Use `uv` and the Python version declared in `.python-version` and
`pyproject.toml`.

```bash
# Install the package and development dependencies.
uv sync --group dev

# Run a focused test while iterating.
uv run pytest tests/path/to/test_file.py -q

# Run the full test suite.
uv run pytest

# Run the static checks used by CI.
uv run ruff check .
uv run ruff format --check .
uv run ty check

# Build the distribution when packaging or release behavior changes.
uv build
```

`devenv shell` is an alternative bootstrap path and performs `uv sync` on
entry. Before finishing a code change, run the most focused relevant test,
then the full applicable checks above. Report any check that could not be run.

## Implementation conventions

- Match the existing typed Python style and let Ruff format the code. Do not
  hand-format around Ruff or add new tool configuration without a concrete need.
- Keep optimizer logic functional: parameters, gradients, updates, and state
  are PyTrees; avoid hidden mutation and host-side behavior in traced paths.
- Assume models and optimizer state may be sharded across multiple GPUs.
  Preserve input sharding, avoid accidental host transfers or full-tensor
  materialization, and use the existing collective-axis and sharding helpers for
  global statistics and distributed reductions.
- Assume parameters, gradients, and stored optimizer state may use low precision
  such as bf16 or lower. Perform numerically sensitive work—such as reductions,
  norms, moment updates, bias correction, and matrix iterations—in a stable
  working dtype when needed, normally float32, then cast or store results
  according to the API's precision policy. Do not promote unrelated operations
  without a numerical reason.
- Preserve dtype, complex-number, masking, sharding, and PRNG semantics. Reuse
  the shared helpers in `src/rollfast/utils.py` and
  `src/rollfast/optim/_matrix_runtime.py` instead of duplicating tree logic.
- Validate static user-facing arguments at construction time where practical.
  Keep update paths compatible with `jax.jit`; do not convert traced values to
  Python scalars or booleans.
- Maintain Optax-compatible behavior for transformation `init` and `update`
  functions, including the handling of optional `params`.
- Keep optional integrations optional. Core imports must work with only the
  dependencies in `[project.dependencies]`; do not pull Equimo or Equinox into
  the base import path.
- Public functions and types need useful docstrings and type annotations.
  Comments should explain non-obvious numerical or compatibility decisions,
  not restate the code.
- Prefer the smallest change that solves the task. Avoid unrelated refactors,
  broad formatting churn, and speculative abstractions.

## Public API and compatibility

- Treat exports from `rollfast`, `rollfast.optim`, and `rollfast.finetune` as
  public API. When intentionally adding an API, update the appropriate
  `__init__.py` imports and `__all__` declaration and add compatibility tests.
- Do not silently change optimizer defaults, state-tree structure, counter
  ownership, serialization formats, or checkpoint compatibility. Such changes
  require explicit tests and matching documentation.
- If the package version changes, keep `pyproject.toml` and
  `src/rollfast/__init__.py` synchronized; CI checks them for exact equality.

## Testing expectations

- Add a regression test for every bug fix and tests for both accepted and
  rejected inputs when changing validation.
- Compare against Optax or a mathematical reference when an equivalent exists;
  use numerical tolerances appropriate to the dtype rather than exact equality.
- Exercise eager and JIT-compiled behavior when changing traced optimizer paths.
  Add dtype, complex, PyTree, sharding, or optional-integration cases when those
  semantics are affected.
- When changing reductions, collectives, or partition-aware logic, verify
  multi-device behavior when suitable hardware is available. A single-device
  test does not establish multi-GPU correctness; report when multi-device
  verification could not be run.
- When changing numerically sensitive logic, add representative low-precision
  coverage and check that updates and state remain finite and sufficiently close
  to a higher-precision reference.
- Tests marked `integration` may require optional development dependencies;
  tests marked `slow` should remain narrowly scoped. Do not weaken or remove a
  test merely to make a change pass.
- Update README, `docs/`, and examples when public behavior or usage changes.
  Run benchmarks only when performance-sensitive code is modified; do not treat
  benchmark output as a unit-test assertion.

## Security and repository hygiene

- Never commit credentials, tokens, private checkpoints, model weights, or
  generated training data.
- Do not use untrusted pickle/checkpoint input in tests or examples. Preserve
  explicit trust boundaries in serialization code.
- Keep dependency changes minimal and justified. Verify the minimal base import
  with `uv run --no-dev python -c "import rollfast"` when dependencies or
  optional integrations change.
- Preserve unrelated working-tree changes and do not rewrite history, publish a
  package, or create a release unless explicitly requested.

Treat this file as living documentation: update it when commands, structure, or
project-wide conventions change.
