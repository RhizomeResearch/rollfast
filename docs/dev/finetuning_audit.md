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

Existing production modules still use private Optax internals:

- `src/rollfast/optim/adam.py`: `optax._src.base`, `combine`, `numerics`,
  `transform`, `utils`; `optax.transforms._masking`.
- `src/rollfast/optim/psgd.py`: `optax._src.base`, `numerics`, `transform`,
  `combine`, `utils`; `optax.transforms._masking`.
- `src/rollfast/optim/prism.py`, `aurora.py`, `hyperball.py`, `magma.py`,
  and schedule wrappers also use selected private helpers.

The new `rollfast.finetune` modules prefer public Optax entry points except
where they delegate to existing Rollfast primitives.

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
- EMA and SWA evaluation views that advance only on applied optimizer steps;
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

Release validation should still rerun the benchmark harnesses on target
hardware, target Equimo models, real batch shapes, and production sharding before
making task-quality performance claims.
