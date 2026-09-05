# Fine-Tuning Optimization Audit

Status: compatibility baseline with grouped AdamW, Schedule-Free Adam, and
structured hybrid wrappers for `rollfast.finetune`.

## Public Surface Kept

These top-level imports remain supported and are not reorganized by the
fine-tuning layer:

| Existing symbol | Keep | Wrap | Move internally | Deprecate |
|---|---:|---:|---:|---:|
| `adamw` | yes | yes | no | no |
| `wsd_schedule` | yes | yes | no | no |
| `schedule_free_adam` | yes | later | no | no |
| `schedule_free_prism` | yes | later | no | no |
| `schedule_free_kron` | yes | later | no | no |
| `schedule_free_aurora` | yes | later | no | no |
| `aurora`, `riemannian_aurora` | yes | later | no | no |
| `prism` | yes | later | no | no |
| `kron` | yes | later | no | no |
| `pion`, `rmnp` | yes | no | no | no |
| Hyperball wrappers | yes | no | no | no |
| SODA wrappers | yes | no | no | no |

## Defaults Recorded

Fine-tuning wrappers use new task-oriented defaults without changing primitive
optimizer defaults:

- `adamw`: existing primitive remains the implementation used by grouped AdamW.
- `ScheduleConfig(kind="warmup_cosine")`: warmup fraction `0.05`,
  end-LR ratio `0.01`.
- `GradientPolicy`: global norm clip `1.0`, non-finite policy `skip`.
- `AccumulationConfig`: `steps=1`, mean reduction, fp32 accumulator.
- `PrecisionConfig`: bf16 compute metadata and fp32 optimizer moments.

## Private Optax Internals

Production modules use public Optax types, transformation constructors,
composition, counters, and `MaskedNode`. These are the same objects as the
former private imports in the supported Optax 0.2.8 runtime.

Two private helpers remain because Optax has no public equivalents:

- `optax._src.numerics.abs_sq`, used for real and complex squared magnitudes.
- `optax._src.utils.canonicalize_dtype`, used by optimizer dtype policies.

The `rollfast.finetune` modules use public Optax entry points and delegate
optimizer-specific behavior to Rollfast primitives.

## Fine-Tuning Compiler Baseline

Implemented:

- structural plan protocols with no Equimo import;
- strict trainable/label PyTree validation;
- normalized group metadata and stable structure fingerprints;
- constant, warmup-cosine, WSD, linear, and polynomial schedules;
- grouped AdamW using existing `rollfast.optim.adam.adamw`;
- grouped blockwise 8-bit AdamW state for eligible moment leaves;
- grouped Schedule-Free Adam with schedule-free evaluation-parameter extraction;
- grouped Aurora/PRISM/Kron hybrid builders using existing Rollfast primitives;
- LoRA+ `lora_B` LR ratio;
- global-norm clipping with optional named-axis reduction;
- Optax finite guard and `MultiSteps` accumulation;
- EMA and SWA evaluation views, with the accumulation caveat recorded below;
- SAM/ASAM two-pass step helpers with exact microbatch accumulation;
- AdaLoRA fixed-shape budget/rank-mask controller utilities with Equimo
  rank-pattern application;
- measured optimizer-state memory summaries, including Kron preconditioner factors;
- static optimizer-state memory estimates, including Kron preconditioner factors
  and Lipschitz auxiliaries before optimizer initialization;
- optimizer reports and JSON-friendly manifests.
- backend-neutral optimizer-state checkpoints with strict fingerprint restore.
- staged optimizer-state migration with explicit counter policy and shape-checked
  structured-preconditioner preservation.
- runnable benchmark smoke harnesses for state memory, tiny-step throughput, and
  toy AdamW/AdamW8 convergence with environment metadata.

Stable-release gates run the benchmark harnesses as CPU smoke tests. Target
hardware, target Equimo models, real batch shapes, and production sharding must
still be measured before making task-quality or accelerator-specific performance
claims; they are not prerequisites for the CPU-validated API release.

## Simplification audit (2026-09-05)

Shared compiler helpers now own clipping, finite guards, accumulation, and
averaging assembly. Diagnostics and migration share state traversal and path
encoding; their classification policies remain separate. Loss-scaled master
steps share the parameter/optimizer-state commit, while their callers retain
model-state, RNG, and loss-scale ownership.

GaLore and APOLLO each share moment updates and finalization between projected
and full-state leaves. PRISM's private inverse-root iteration is specialized to
the fourth root used by bidirectional PRISM. PSGD shares its seeded norm
iteration while keeping the SPD and skew-Hermitian normalization distinct.
Dynamic 8-bit codebooks retain their exact v1 values and identifiers.

Against the untouched `2c50003d` baseline, 861 comparison records matched
exactly: updates, full state trees, dtypes, counters, PRNG keys, compiler
manifests, evaluation views, schedules, and codebooks. Representative lowered
JAX programs for GaLore, APOLLO, PRISM, and both PSGD norm estimates also matched
exactly on CPU. Public exports, state classes, checkpoint schemas, defaults,
and dependency versions are unchanged.

Two existing behavior gaps were reproduced in both versions with JAX 0.10.2
and Optax 0.2.8 and left for separate fixes:

- With `accumulation_steps=2` and EMA/SWA enabled, two finite microsteps followed
  by two nonfinite microsteps leave parameters unchanged during the rejected
  update, but advance both averaging counts from 1 to 2. `_update_applied`
  observes the outer `MultiSteps.gradient_step` before the inner finite guard.
  Averaging therefore includes a duplicate iterate on rejected accumulated
  updates.
- A Schedule-Free plan compiled with `key=jax.random.key(0)` fails on its first
  update: the finite guard calls `astype` on a typed PRNG key and raises
  `NotImplementedError`. Legacy `jax.random.PRNGKey(0)` works. The preservation
comparisons use legacy keys for these compiler paths.

Final validation on Python 3.12 / CPU:

- `uv run --frozen --no-sync pytest -q --tb=short`: 692 passed, 2 skipped.
- An isolated JAX 0.6.2 / Optax 0.2.8 / Equinox 0.13.6 environment ran the CI
  minimum-version suite, excluding the optional Equimo integrations:
  674 passed, 2 skipped. The base import also passed before installing Equinox.
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run --frozen --no-sync
  pytest tests/test_multidevice_cpu.py -q`: 2 passed.
- Ruff lint, Ruff format checks, Ty, version consistency, and all six CI
  example/benchmark smoke commands passed. Existing test functions were kept
  unchanged; 45 additional parameterized cases cover the refactored contracts.

GPU execution and accelerator throughput were not verified; no GPU was available.
