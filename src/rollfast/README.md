# `rollfast`

## Overview

Top-level Python package (`rollfast`) re-exporting experimental JAX/Optax gradient transformations, full optimizer
wrappers, learning-rate schedules, Schedule-Free wrappers, and shared low-precision/sharding/PyTree runtime helpers,
while organizing plan-based fine-tuning and third-party adapters into dedicated subpackages.

## Interface

- Top-Level Package Exports (`[__init__.py:__all__]`):
  - `__version__` (`[__init__.py:__version__]`): Package version string (`"1.7.0"`).
  - Parameter Update Application (`[utils.py:apply_updates]`, `[utils.py:apply_updates_prefix]`):
    - `apply_updates(params, updates, key=None, stochastic=True) -> optax.Params`
    - `apply_updates_prefix(model, updates, key=None, stochastic=True) -> optax.Params`
  - Re-exported Optimizer Symbols (`[__init__.py:__all__]` from `./optim`): `MUON_NS_COEFFS`, `AuroraDimensionNumbers`,
    `GradClipMode`, `MatrixDimensionNumbers`, `MomentumAccumulator`, `MuonDimensionNumbers`, `MuonNsCoeffs`,
    `MuonState`, `NsCoeffs`, `PreconditionerMode`, `PrismDimensionNumbers`, `WeightDimNumOrFn`, `adamw`, `adamw8`,
    `adamw_hyperball`, `apply_hyperball`, `aurora`, `aurora_hyperball`, `contramuon`, `contranormuon`,
    `get_equinox_aurora_spec`, `get_equinox_prism_spec`, `global_l2_norm`, `hyperball_riemannian_aurora`, `kron`,
    `kron_hyperball`, `muon`, `muon_hyperball`, `normuon`, `orthogonalize_via_newton_schulz`, `pion`,
    `polar_express_coeffs`, `precond_update_prob_schedule`, `prism`, `prism_hyperball`, `resolve_ns_coeffs`,
    `riemannian_aurora`, `riemannian_aurora_hyperball`, `rmnp`, `rmnp_hyperball`, `sam_perturbation`, `scale_by_adam`,
    `scale_by_aurora`, `scale_by_hyperball`, `scale_by_kron`, `scale_by_muon`, `scale_by_muon_shape`,
    `scale_by_normuon`, `scale_by_normuon_shape`, `scale_by_pion`, `scale_by_prism`, `scale_by_riemannian_aurora`,
    `scale_by_rmnp`, `scale_by_rmnp_shape`, `scale_by_trasmuon`, `soda`, `soda_adam`, `soda_kron`, `soda_muon`,
    `soda_prism`, `soda_rmnp`, `trasmuon`.
  - Re-exported Schedule Symbols (`[__init__.py:__all__]` from `./schedules`): `WeightingMode`,
    `cosine_annealing_schedule`, `interval_schedule`, `power_decay_schedule`, `schedule_free`, `schedule_free_adam`,
    `schedule_free_aurora`, `schedule_free_eval_params`, `schedule_free_kron`, `schedule_free_prism`, `wsd_schedule`.
- Shared Runtime Utilities (`[utils.py]`):
  - `MomentumAccumulator`, `MaskOrFn`, `AxisName`, `FirstMomentRuntime` (`[utils.py:MomentumAccumulator]`,
    `[utils.py:MaskOrFn]`, `[utils.py:AxisName]`, `[utils.py:FirstMomentRuntime]`).
  - `add_tiny`, `with_reference_sharding`, `astype_preserving_sharding`, `zeros_like_preserving_sharding`,
    `dist_reduce`, `resolve_partition_norm_axis_name` (`[utils.py:add_tiny]`, `[utils.py:with_reference_sharding]`,
    `[utils.py:astype_preserving_sharding]`, `[utils.py:zeros_like_preserving_sharding]`, `[utils.py:dist_reduce]`,
    `[utils.py:resolve_partition_norm_axis_name]`).
- Subpackages:
  - [`finetune`](./finetune/README.md): Plan validation, parameter-group rules, recipes, optimizer/step compilation,
    AdaLoRA, evaluation views, diagnostics, host offload, state migration, and checkpoints.
  - [`integrations`](./integrations/README.md): Structural duck-typed adapters for optional third-party fine-tuning
    plans (`equimo`).
  - [`optim`](./optim/README.md): First-order, 8-bit quantized, subspace-projected, spectral/orthogonalized matrix,
    Kronecker-preconditioned, norm-projected, initialization-anchored, and SAM optimizers.
  - [`schedules`](./schedules/README.md): Learning-rate schedules (`wsd`, `cosine_annealing`, `power_decay`, `interval`)
    and Schedule-Free iterate-averaging wrappers.

## Invariants

- Minimal base import path (`[__init__.py:__all__]`): Importing `rollfast` loads only core `jax`/`optax`-backed
  `.optim`, `.schedules`, and `.utils` symbols; `.finetune`, `.integrations`, and optional `equinox`/`equimo`
  dependencies are never pulled into the base import path.
- Explicit stochastic-rounding PRNG ownership (`[utils.py:apply_updates]`, `[utils.py:apply_updates_prefix]`,
  `[utils.py:_stochastic_round_bf16]`): `apply_updates` and `apply_updates_prefix` raise `ValueError` when
  `stochastic=True` and `key is None`, skip `jax.random.split` when `stochastic=False`, accumulate `bfloat16` parameter
  updates in `float32`, and apply integer-only `uint32` stochastic rounding that preserves infinities and sets the quiet
  NaN bit (`0x00400000`) so payload-only NaNs never truncate to infinity.
- Prefix-tree and auxiliary-leaf preservation (`[utils.py:apply_updates_prefix]`, `[utils.py:_is_aux_leaf]`,
  `[utils.py:_zeros_like_tree]`, `[utils.py:_cast_state_tree]`): `None` and `optax.MaskedNode` leaves pass through
  untouched; `apply_updates_prefix` maps with `updates` as the leading tree so `None`/`MaskedNode` update leaves
  preserve entire model subtrees (including callable Equinox modules); complex leaves promote to `jnp.complex64` rather
  than truncating imaginary parts when cast with a real target dtype.
- Sharding and distributed-axis invariants (`[utils.py:with_reference_sharding]`, `[utils.py:dist_reduce]`,
  `[utils.py:resolve_partition_norm_axis_name]`): Sharding helpers preserve existing `value.sharding` /
  `reference.sharding` via `jax.lax.with_sharding_constraint`; `dist_reduce` accepts only `op in ("mean", "max", "sum")`
  (`ValueError`); `resolve_partition_norm_axis_name` rejects overlapping `partition_axis_names` and
  `replicated_axis_names` (`ValueError`) and strips replicated axes from global norm reductions.
- CUDA-graph-safe conditional skip & Optax counter typing (`[utils.py:_run_if]`, `[utils.py:_safe_increment]`):
  `_run_if` gates expensive conditional work with a 0-or-1-trip `jax.lax.while_loop` (requiring identical input/output
  PyTree structure and dtypes) instead of `jax.lax.cond`; `_safe_increment` wraps `optax.safe_increment` with
  `jnp.asarray` to guarantee a `jax.Array` across Optax versions.

## Side Effects

- Import-time submodule loading (`[__init__.py:__all__]`): Importing `rollfast` imports the referenced `.optim` and
  `.schedules` modules (which register `QuantizedBlocks`, `APOLLOLeafState`, and `GaLoreLeafState` as JAX PyTree nodes
  and construct default probability schedule closures); `utils.py` itself has no module-level side effects
  (`[utils.py]`).
- SPMD collectives (`[utils.py:dist_reduce]`): Invokes `jax.lax.pmean`, `jax.lax.pmax`, or `jax.lax.psum` across
  `axis_name` when `axis_name is not None`.

## Verification

- Run: `uv run pytest tests/test_api.py tests/test_utils.py tests/test_stochastic_rounding.py`
