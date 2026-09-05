# Test suite simplification audit

Date: 2026-09-05. Scope: the complete repository test suite.

The audit started after the production refactor, with 692 passing tests and two
expected device-dependent skips. Existing working-copy changes were preserved.
No production source, dependency, lockfile, runner configuration, or CI policy
was changed in this test pass.

## Protected behavior and audit coverage

| Area | Evidence retained |
| --- | --- |
| Optimizer primitives and wrappers (`tests/test_*.py`) | Mathematical references, fallback routing, masks, dimensions, defaults, moment state, complex values, low precision, finite guards, PRNG and schedule semantics. Primitive, wrapper, and integration tests remain distinct boundaries. |
| Fine-tuning compilers and policies (`tests/finetune/`) | Group rules, plan validation, quantization eligibility, schedules, clipping, accumulation, SAM, stateful updates, master parameters, loss scaling, averaging, and AdaLoRA. |
| Persistence and state management (`tests/finetune/`) | Checkpoint trust, schema and metadata compatibility, next-update equivalence, migration clocks, rollback, memory accounting, offload, and sharding. |
| Public compatibility (`tests/compat/`, `tests/test_api.py`) | Exact exports, attribute availability, aliases, required signature parameters, and safe key defaults. |
| Optional integrations (`tests/test_equimo_integration.py`, `tests/integrations/`) | All 14 ViT optimizer configurations and the four Equimo plan integration cases. |
| Distributed behavior | Logical partition/replica axes, named sharding, and physical CPU device collectives. |

## Changes and preservation evidence

| Change | Surviving evidence and proof |
| --- | --- |
| Consolidate fine-tuning helpers | Nine identical ones-tree helpers, two zeros-tree helpers, two array-tree comparisons, and two RNG comparisons now share implementations in `tests/finetune/helpers.py`. AST comparison established identical bodies. Every caller retains its inputs, assertions, tolerances, and fresh per-call state. |
| Move shared quantization plans out of a test module | `large_plan` and `leaf_estimation_plan` moved unchanged into `helpers.py`. Compiler and memory-estimation tests use the same arrays, labels, group policies, and threshold cases without importing another test module. |
| Remove obsolete ViT fixture branches | The sole fixture parameter was `vit`; the iFormer and ReduceFormer branches were unreachable. The active ViT constructor, inputs, seeds, and function scope are unchanged. |
| Consolidate ViT optimizer steps | Thirteen identical test bodies became named constructor cases. AST extraction preserved every keyword argument and the init/update/apply/combine boundary, finite loss/update assertions, and seeds. The callable weight-decay-mask test remains separate. |
| Replace mocked collective calls | Three norm, clipping, and SAM tests now run JIT-compiled named-axis `vmap` collectives across two replicas and four model shards. Expected values follow the global norm of eight ones. The SAM test checks the actual perturbation as well as its reported norm. Mutation results are below. |
| Replace Hyperball forwarding mocks | Named Muon/RMNP cases compare projection with a NumPy global-sphere reference, using the public unprojected path for the optimizer direction. PRISM/Aurora fallback cases compare two nonconstant-gradient steps against Optax Adam with Nesterov both enabled and disabled. Mutation results are below. |
| Make public export compatibility readable | `tests/compat/public_api_v1.json` records the exact sorted export lists previously represented by five hashes. Every list was checked against its old hash before replacement. Comparing lists preserves duplicate detection; attribute checks preserve availability. Four overlapping subset-presence tests were removed only after all 72 named occurrences were confirmed in the locked lists. Alias identity tests remain separate. |
| Consolidate signature checks | All 13 function/required-parameter combinations are preserved as named cases. The three hybrid builders share one expected parameter tuple. The array-valued-key-default test remains separate. |
| Consolidate BF16 update application | Both public application functions remain independent named cases for all eight optimizers, with the same gradients, seeds, dtype assertions, and fresh optimizer state. The seven moment-dtype cases now reuse the same constructor data; Schedule-Free Kron remains explicitly excluded from that unsupported argument. |
| Consolidate NorMuon, Hyperball, and SODA cases | AST comparison established identical execution after extracting the constructor, configuration, or shape. Three NorMuon shape cases, both Hyperball vector fallbacks, and four SODA wrapper cases remain independently named. The SODA missing-parameter check uses `pytest.raises` with the same exception type and literal message substring. |
| Remove the standalone Kron Hyperball smoke duplicate | Its constructor arguments, inputs, init/update calls, and shape/norm assertions were identical to the existing Kron row in `test_hyperball_optimizer_wrappers`. The extra-gradient-argument test remains separate. |

## Discriminating mutations

Mutations ran only in a temporary copy of the current production source. The
copy imported its own `rollfast` package. Each original source file was restored
between trials. All nine mutations failed at the intended surviving assertions;
all corresponding unmutated selections passed.

| Temporary defect | Surviving test | Observed failure |
| --- | --- | --- |
| Replace distributed sum with identity in `utils.dist_reduce` | `test_global_l2_norm_reduces_only_partition_axes` | Wrong numerical norm. |
| Stop filtering replicated axes in `resolve_partition_norm_axis_name` | `test_global_l2_norm_reduces_only_partition_axes` | Replicas counted in the norm. |
| Return the legacy axes instead of explicit partition axes | Clipping and SAM partition-axis tests | Both numerical update/perturbation checks failed. |
| Pass `axis_name=None` to terminal Hyperball projection | Muon/RMNP global projection cases | Both differed from global-sphere projection. |
| Force the Adam fallback's Nesterov flag to `True` | PRISM/Aurora plain Adam reference cases | Both update comparisons failed. |
| Force that flag to `False` | PRISM/Aurora Nesterov reference cases | Both update comparisons failed. |
| Remove `sam_perturbation` from root `__all__` | `test_v1_public_api[rollfast]` | Export list mismatch. |
| Delete the root `sam_perturbation` attribute | `test_v1_public_api[rollfast]` | Missing attribute. |
| Duplicate its root `__all__` entry | `test_v1_public_api[rollfast]` | Export list mismatch. |

These trials establish detection of the named defects, not exhaustive mutation
coverage. Pure consolidations used static equivalence of execution, inputs,
assertions, metadata, and isolation rather than coverage percentages.

## Retained candidates

- Persistence, migration, rollback, finite-guard, accumulation, and stateful SAM
  regressions protect distinct state transitions and were not merged away.
- Serialization and SAM tree comparisons retain their different handling of
  `None`, non-array leaves, and tolerances. The tiny typing helper remains useful
  for static checking without introducing runtime behavior.
- Muon's custom orthogonalization callback test protects the public callback's
  ordered and truncated coefficient input. Hyperball's Magma shape/key ownership
  and SODA's key forwarding checks protect exact handoff contracts; those mocks
  remain intentional.
- Quantizer codebook hashes remain checkpoint-compatibility evidence. Public
  export lists are small enough to review directly, whereas these hashes protect
  exact numeric byte sequences.
- Primitive, compiler, ViT, and real Equimo-plan tests are not interchangeable.
  Model fixtures remain function-scoped to preserve isolation.
- Named-axis `vmap` tests do not replace physical device tests. GPU verification
  was unavailable; the existing CPU device checks remain in CI.
- No remaining material candidate had sufficient justification for another
  merge or removal. Further cross-optimizer unification would mix distinct
  numerical paths or hide the inputs and assertions that identify each failure.

## Validation

Main-environment commands used `uv run --frozen --no-sync ...` to avoid
dependency synchronization. Minimum-version compatibility used a separate
temporary environment with the same Python 3.12 and pytest 9.1.1.

- Baseline: `pytest -q --durations=25` — 692 passed, 2 skipped in 182.45 seconds.
- Full suite after changes: the same command — 696 passed, 2 skipped in
  184.82 seconds. These single observations do not establish a runtime improvement.
- Focused checks passed after each conceptual change: fine-tuning (286),
  collectives (6), ViT integration (14), export/config compatibility (29),
  signatures (14), rounding (23), NorMuon (28), Hyperball (26), and SODA (25).
- Minimum JAX/JAXlib 0.6.2, Optax 0.2.8, Equinox 0.13.6:
  `pytest -q --ignore=tests/test_equimo_integration.py --ignore=tests/integrations`
  — 678 passed, 2 skipped in 86.55 seconds.
- Physical CPU devices:
  `XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run --frozen --no-sync pytest tests/test_multidevice_cpu.py -q`
  — 2 passed in 1.41 seconds.
- `ruff check .`, `ruff format --check .`, and `ty check` passed.
- SHA-256 comparison confirmed all 47 production Python files unchanged, and
  the temporary mutation sources restored exactly. The final diff was reviewed;
  dependencies and build policy are unchanged. Temporary validation copies were
  removed after recording these results.
