# Rollfast Source (`src`) Blueprint

## System Overview

Source root (`src/`) for the `rollfast` Python library of experimental JAX/Optax gradient transformations, learning-rate
schedules, Schedule-Free wrappers, plan-driven fine-tuning pipelines, and optional third-party plan integrations. All
functional code resides under the [`rollfast`](./rollfast/README.md) package hierarchy and operates functionally over
JAX PyTrees under `jax.jit`, mixed precision, and multi-device SPMD sharding.

## Subdirectories & System Boundaries

- `rollfast`: [`./rollfast/README.md`](./rollfast/README.md) - Top-level package re-exporting public optimizers,
  schedules, `apply_updates` / `apply_updates_prefix`, and shared PyTree, sharding, collective, and stochastic-rounding
  utilities (`utils.py`).
  - `rollfast/optim`: [`./rollfast/optim/README.md`](./rollfast/optim/README.md) - First-order (`adam`, `adam8`),
    subspace-projected (`galore`, `apollo`), spectral/orthogonalized matrix (`muon`, `normuon`, `trasmuon`, `pion`,
    `prism`, `aurora`, `rmnp`, `orthogonalization`), Kronecker-preconditioned (`psgd`), norm-projected (`hyperball`),
    initialization-anchored (`soda`), sharpness-aware (`sam`), and momentum-masked (`magma`) Optax transformations.
  - `rollfast/schedules`: [`./rollfast/schedules/README.md`](./rollfast/schedules/README.md) - Learning-rate schedules
    (`wsd_schedule`, `power_decay_schedule`, `cosine_annealing_schedule`, `interval_schedule`) and Schedule-Free /
    Schedule-Free+ iterate-averaging wrappers (`schedulefree.py`).
  - `rollfast/finetune`: [`./rollfast/finetune/README.md`](./rollfast/finetune/README.md) - Plan validation,
    parameter-group rules, recipes, optimizer/step compilation, AdaLoRA controllers, EMA/SWA/Schedule-Free evaluation
    views, memory/precision diagnostics, host offload, state migration, and single-file pickle or multi-host NPY array
    checkpoints.
  - `rollfast/integrations`: [`./rollfast/integrations/README.md`](./rollfast/integrations/README.md) - Optional
    duck-typed third-party plan adapters (`equimo.py`) isolated from core imports.

## Global Invariants

- Optax & JAX PyTree Purity (`[rollfast/utils.py]`, `[rollfast/optim/_matrix_runtime.py]`,
  `[rollfast/finetune/step.py]`): Optimizer `init` and `update` paths remain pure functions over JAX PyTrees compatible
  with `jax.jit`, preserving `None` and `optax.MaskedNode` leaves, preserving input shardings
  (`with_reference_sharding`), and validating static arguments at construction time without converting traced arrays to
  Python scalars/booleans.
- Numerical Precision & Stochastic Rounding (`[rollfast/utils.py:apply_updates]`, `[rollfast/finetune/_precision.py]`,
  `[rollfast/optim/aurora.py:_simple_quintic_polar]`): Numerically sensitive reductions, norms, moment updates, bias
  corrections, and matrix iterations execute in `float32` (or `complex64` where complex parameters are supported;
  Aurora's polar polynomial defaults to `polar_compute_dtype=jnp.bfloat16` with `float32` norm and `float32` output)
  before casting to target storage dtypes; `bfloat16` stochastic rounding requires explicit PRNG key ownership
  (`key is not None` when `stochastic=True`).
- CUDA-Graph-Safe Branching (`[rollfast/utils.py:_run_if]`, `[rollfast/finetune/transforms.py:always_skip_nonfinite]`):
  Traced update paths avoid top-level `jax.lax.cond`, using `jnp.where` for cheap elementwise selection and shared
  `_run_if` zero-or-one-trip `while_loop` gates for expensive periodic or probabilistic work.
- Optional Dependency & Trust Boundaries (`[rollfast/__init__.py]`, `[rollfast/integrations/equimo.py]`,
  `[rollfast/finetune/serialization.py]`, `[rollfast/finetune/array_storage.py]`): Base imports depend only on `jax` and
  `optax`; `load_state_checkpoint` rejects pickle deserialization unless `trusted=True`, and
  `load_array_state_checkpoint` enforces resource limits, `O_NOFOLLOW` regular-file checks, SHA-256 verification, and
  template-guided PyTree reconstruction without instantiating schema-named classes.

## Global Side Effects

- Global JAX PyTree Registration (`[rollfast/optim/adam8.py]`, `[rollfast/optim/apollo.py]`,
  `[rollfast/optim/galore.py]`, `[rollfast/finetune/config.py]`): Registers `QuantizedBlocks`, `APOLLOLeafState`,
  `GaLoreLeafState`, `LossBundle`, `AccumulationState`, `StepCounters`, `RNGStreams`, and `FineTuneStepState` in JAX's
  process-wide PyTree registry at module import time.
- SPMD & Multi-Host Collectives (`[rollfast/utils.py:dist_reduce]`, `[rollfast/optim/magma.py:apply_magma_internal]`,
  `[rollfast/schedules/schedulefree.py]`, `[rollfast/finetune/step.py]`, `[rollfast/finetune/array_storage.py]`):
  Executes cross-device reductions (`psum`, `pmean`, `pmax`, `pmin`) and multi-process barrier/broadcast collectives
  (`process_allgather`, `broadcast_one_to_all`) when distributed axes or multi-host array checkpoints are active.
- Filesystem Checkpoint I/O & Device-Host Synchronization (`[rollfast/finetune/serialization.py]`,
  `[rollfast/finetune/array_storage.py]`, `[rollfast/finetune/offload.py]`): Reads/writes/fsyncs checkpoint files and
  shard directories in `rollfast.finetune`, and performs explicit `jax.device_get` / `jax.device_put` transfers during
  host offload, checkpointing, and host-side diagnostics/validation.

## System Verification

- Run: `uv run pytest`
