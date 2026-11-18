# `rollfast.optim`

## Overview

Experimental and structured JAX/Optax gradient transformations and full optimizer wrappers, including first-order
(`adam`, `adam8`), subspace-projected (`galore`, `apollo`), spectral/orthogonalized matrix (`muon`, `normuon`,
`trasmuon`, `pion`, `prism`, `aurora`, `rmnp`, `orthogonalization`), Kronecker-preconditioned (`psgd`), norm-constrained
(`hyperball`), initialization-anchored (`soda`), sharpness-aware (`sam`), and momentum-masked (`magma`) optimizers.

## Interface

- Exported Submodule Namespaces (`[__init__.py:__all__]`):
  - `adam`, `adam8`, `apollo`, `aurora`, `dimension_numbers`, `galore`, `hyperball`, `magma`, `muon`, `normuon`,
    `orthogonalization`, `pion`, `prism`, `psgd`, `rmnp`, `sam`, `soda`, `trasmuon`.
- Dimension Routing & Orthogonalization (`[dimension_numbers.py]`, `[orthogonalization.py]`):
  - `MatrixDimensionNumbers(reduction_axis=0, output_axis=1)` (`[dimension_numbers.py:MatrixDimensionNumbers]`),
    `DimNumsTree`, `WeightDimNumOrFn`, `MaskOrFn`, `ReshapeFn`.
  - `MUON_NS_COEFFS`, `DION_NS_COEFFS`, `NS_COEFFS_PRESETS`, `MuonPreconditioning`, `NsCoeffTriple`, `NsCoeffs`,
    `MuonNsCoeffs`, `OrthogonalizeFn`, `polar_express_coeffs`, `resolve_ns_coeffs`, `quintic_newton_schulz`,
    `orthogonalize_via_newton_schulz` (`[orthogonalization.py]`).
- First-Order, Quantized & Subspace-Projected Optimizers (`[adam.py]`, `[adam8.py]`, `[galore.py]`, `[apollo.py]`):
  - `ScaleByAdamState`, `scale_by_adam`, `adamw` (`[adam.py]`).
  - `DYNAMIC_SIGNED_CODEBOOK_ID`, `DYNAMIC_UNSIGNED_CODEBOOK_ID`, `SYMMETRIC_INT8_CODEBOOK_ID`, `BlockLayout`,
    `CodebookQuantizer`, `QuantizedBlocks`, `ScaleByAdam8State`, `quantize_blocks`, `dequantize_blocks`,
    `quantized_nbytes`, `tree_state_nbytes`, `estimate_quantized_moment_bytes`, `scale_by_adam8`, `adamw8`
    (`[adam8.py]`).
  - `GaLoreLeafState`, `ScaleByGaLoreState`, `galore_adamw`, `projected_state_nbytes` (`[galore.py:__all__]`), plus
    module type aliases `Projection` and `StateOnRefresh` (`[galore.py:Projection]`, `[galore.py:StateOnRefresh]`).
  - `APOLLOLeafState`, `ScaleByAPOLLOState`, `apollo_adamw`, `apollo_state_nbytes` (`[apollo.py:__all__]`), plus module
    aliases/constants `Scaling`, `Orientation`, `REFERENCE_EPS`, and `SCALING_FACTOR_EPS` (`[apollo.py:Scaling]`,
    `[apollo.py:Orientation]`, `[apollo.py:REFERENCE_EPS]`, `[apollo.py:SCALING_FACTOR_EPS]`).
- Matrix, Spectral & Preconditioned Optimizers (`[muon.py]`, `[normuon.py]`, `[trasmuon.py]`, `[pion.py]`, `[prism.py]`,
  `[aurora.py]`, `[rmnp.py]`, `[psgd.py]`):
  - `MomentumAccumulator`, `MuonState`, `MuonDimensionNumbers`, `MuonNsCoeffs`, `scale_by_muon`, `scale_by_muon_shape`
    (`scale_by_shape`), `muon`, plus re-exported `orthogonalize_via_newton_schulz`, `polar_express_coeffs`, and
    `resolve_ns_coeffs` (`[muon.py:__all__]`).
  - `NorMuonNormalizationAxis`, `NorMuonRescale`, `ScaleByNorMuonState`, `scale_by_normuon`, `scale_by_normuon_shape`,
    `normuon`, `contramuon`, `contranormuon` (`[normuon.py:__all__]`).
  - `ScaleByTrasMuonState`, `scale_by_trasmuon`, `trasmuon` (`[trasmuon.py:__all__]`).
  - `ScaleByPionState`, `scale_by_pion`, `pion` (`[pion.py]`).
  - `PrismDimensionNumbers`, `ScaleByPrismState`, `get_equinox_prism_spec`, `scale_by_prism`, `prism` (`[prism.py]`).
  - `AuroraDimensionNumbers`, `AuroraWeightDimNumOrFn`, `ScaleByAuroraState`, `get_equinox_aurora_spec`,
    `scale_by_aurora`, `scale_by_riemannian_aurora`, `aurora`, `riemannian_aurora` (`[aurora.py]`).
  - `ScaleByRmnpState`, `scale_by_rmnp`, `scale_by_rmnp_shape`, `rmnp` (`[rmnp.py:__all__]`).
  - `PreconditionerMode`, `GradClipMode`, `KronState`, `precond_update_prob_schedule`, `scale_by_kron`, `kron`
    (`[psgd.py]`).
- Wrappers & Perturbation Utilities (`[hyperball.py]`, `[soda.py]`, `[sam.py]`, `[magma.py]`):
  - `HyperballState`, `apply_hyperball` (`scale_by_hyperball`), `adamw_hyperball`, `muon_hyperball`, `rmnp_hyperball`,
    `kron_hyperball`, `prism_hyperball`, `aurora_hyperball`, `riemannian_aurora_hyperball`
    (`hyperball_riemannian_aurora`) (`[hyperball.py]`).
  - `SodaState`, `soda`, `soda_adam`, `soda_prism`, `soda_kron`, `soda_muon`, `soda_rmnp` (`[soda.py:__all__]`).
  - `global_l2_norm`, `sam_perturbation`, `add_perturbation` (`[sam.py:__all__]`).
  - `validate_magma_args`, `apply_magma_internal` (`[magma.py]`).

## Invariants

- Matrix routing and complex exclusion (`[dimension_numbers.py:_validate_matrix_operand]`,
  `[dimension_numbers.py:_get_dimension_numbers]`, `[dimension_numbers.py:_resolve_update_dimension_numbers]`,
  `[dimension_numbers.py:_make_matrix_labels]`): Direct `scale_by_*` matrix transforms (`normuon`, `trasmuon`, `prism`,
  `aurora`, `rmnp`, and `muon`) validate non-`None`, non-`MaskedNode` leaves as real with `ndim >= 2` (though
  `scale_by_muon` with `shape_updates=False` resets rank-2 `dimension_numbers` to `(0, 1)` inside
  `orthogonalize_via_newton_schulz`, and `scale_by_pion.update` checks `d is None` without calling
  `_validate_matrix_operand`). Passing a bare `MatrixDimensionNumbers` instance is restricted to single-leaf `params`
  (`ValueError` on structured PyTrees); callable `weight_dimension_numbers` requires `params is not None` on `update`.
  Default `weight_dimension_numbers=None` assigns `MatrixDimensionNumbers(0, 1)` to rank-2 leaves, while
  `_make_matrix_labels` / `_make_dimension_numbers_mask` additionally exclude complex leaves from matrix branches in
  partitioned wrappers (`scale_by_*_shape` leaves `ndim != 2` untouched without filtering complex rank-2 inputs).
  `adamw8`, `apollo_adamw`, `galore_adamw`, `apply_hyperball`, and `soda` reject complex PyTrees outright
  (`ValueError`).
- CUDA-graph-safe control flow (`[galore.py:_refresh_projected_leaves]`, `[apollo.py:_refresh_projections]`,
  `[pion.py:_alternate_sides]`, `[psgd.py:_norm_lower_bound]`, `[psgd.py:scale_by_kron]`,
  `[aurora.py:_solve_row_norm_multipliers]`): Update paths avoid top-level `jax.lax.cond`, using `jnp.where` for cheap
  elementwise selections (`_norm_lower_bound`), shared `_run_if` zero-or-one-trip `while_loop` gates for periodic
  refreshes (`galore`, `apollo`), alternating sides (`pion`), and probabilistic preconditioner updates
  (`scale_by_kron`), and keeping `lax.cond` inside unrolled-false `fori_loop` bodies (`aurora`).
- Working precision and low-precision state (`[_matrix_runtime.py]`, `[adam.py:scale_by_adam]`,
  `[adam8.py:quantize_blocks]`, `[adam8.py:scale_by_adam8]`, `[psgd.py:scale_by_kron]`,
  `[orthogonalization.py:quintic_newton_schulz]`, `[aurora.py:_simple_quintic_polar]`): Reductions, norms, moment
  updates, bias correction, Kronecker updates, and `quintic_newton_schulz` / `_newton_schulz_spd` run in `float32`,
  while Aurora's `_simple_quintic_polar` defaults to `polar_compute_dtype=jnp.bfloat16` (with `float32` Frobenius norm
  and `polar_output_dtype=jnp.float32`). In `scale_by_adam`, `nu` uses `float32` when `mu_dtype` is `None`, `float32`,
  or complex, and uses `mu_dtype` when `mu_dtype` is real low precision (`bfloat16`/`float16`, after `init_fn` allocates
  `complex64` on complex `params`). Storing `bfloat16` moments/preconditioners uses stochastic rounding across `adam`,
  `psgd`, `pion`, `muon`, `normuon`, `trasmuon`, `rmnp`, `prism`, and `aurora` (while `galore` and `apollo` cast
  deterministically via `.astype(mu_dtype)`); direct `quantize_blocks(..., stochastic_rounding=True)` requires a
  non-`None` `key` (`[adam8.py:_round_scaled]`, `[adam8.py:_quantize_to_codebook]`), whereas `scale_by_adam8` and
  `adamw8` seed `key=None` automatically via `_fresh_prng_key`.
- Weight-decay, `params`, and terminal-wrapper ownership (`[adam.py:adamw]`, `[muon.py:scale_by_muon]`,
  `[muon.py:_build_unscaled_muon_branch]`, `[prism.py:_build_unscaled_prism_branch]`,
  `[aurora.py:_build_unscaled_aurora_branch]`, `[psgd.py:kron]`, `[pion.py:scale_by_pion]`, `[pion.py:pion]`,
  `[galore.py:galore_adamw]`, `[apollo.py:apollo_adamw]`, `[hyperball.py:apply_hyperball]`,
  `[hyperball.py:_partitioned_matrix_hyperball]`, `[soda.py:soda]`, `[sam.py:sam_perturbation]`): When `use_magma=True`,
  weight decay is fused inside `scale_by_*` prior to Magma masking (`adamw8` raises `NotImplementedError` for
  `use_magma=True`); when `use_magma=False`, `scale_by_muon` rejects nonzero/scheduled `weight_decay` (`ValueError`) and
  outer wrappers chain `optax.add_decayed_weights` once, while `galore_adamw` and `apollo_adamw` fuse weight decay and
  `-learning_rate` per leaf (`params` required whenever weight decay is nonzero/scheduled). `scale_by_pion.update`,
  `apply_hyperball.update` (and all `*_hyperball` wrappers), `soda.update`, and `sam_perturbation(adaptive=True)` always
  require `params is not None`. `apply_hyperball` and `scale_by_pion` are terminal transforms returning scaled parameter
  deltas; `pion` omits weight decay on the Pion branch; `*_hyperball` wrappers disable inner branch weight decay so
  `apply_hyperball` owns sphere-projected decay; `soda_*` wrappers take `(learning_rate, total_steps, ...)` to build
  internal WSD schedules and force `weight_decay=0.0` and `use_magma=False`.
- Optional Equinox specs & PSGD tensor rank (`[aurora.py:get_equinox_aurora_spec]`, `[prism.py:get_equinox_prism_spec]`,
  `[psgd.py:_init_Q_exprs]`): `get_equinox_aurora_spec` and `get_equinox_prism_spec` raise `ImportError` if optional
  `equinox` is not installed; PSGD Kron requires tensor `ndim <= 13` (`ValueError`).

## Side Effects

- Eager submodule imports, schedule closures & PyTree/cache registration (`[__init__.py:__all__]`,
  `[adam8.py:QuantizedBlocks]`, `[apollo.py:APOLLOLeafState]`, `[galore.py:GaLoreLeafState]`,
  `[psgd.py:_DEFAULT_SCALE_BY_KRON_SCHEDULE]`, `[psgd.py:_DEFAULT_KRON_SCHEDULE]`,
  `[hyperball.py:_DEFAULT_KRON_HYPERBALL_SCHEDULE]`, `[soda.py:_DEFAULT_SODA_KRON_SCHEDULE]`): Importing
  `rollfast.optim` imports all 18 submodules (including optional `equinox` in `aurora.py`/`prism.py` when installed),
  evaluates default `precond_update_prob_schedule()` closures, and registers `QuantizedBlocks`, `APOLLOLeafState`, and
  `GaLoreLeafState` in JAX's global PyTree node registry; `adam8` populates `@lru_cache` tables on first call to
  `[adam8.py:_dynamic_codebook_values]` and `[adam8.py:_zero_code_for_quantizer]`.
- Distributed SPMD collectives (`[_matrix_runtime.py:_tree_global_norm]`, `[magma.py:apply_magma_internal]`,
  `[psgd.py:scale_by_kron]`, `[hyperball.py:_leaf_l2_norm]`, `[sam.py:global_l2_norm]`): Emits cross-device reductions
  (`dist_reduce` / `jax.lax.psum` / `jax.lax.pmean` / `jax.lax.pmax` / `jax.lax.pmin`) across `axis_name` or
  partition/replicated axes when configured.
- Console logging (`[psgd.py:scale_by_kron]`, `[psgd.py:kron]`): Prints learning-rate guidance and
  preconditioner/momentum byte summaries to `stdout` on `jax.process_index() == 0` when `verbose=True`.

## Verification

- Run:
  `uv run pytest tests/test_adam.py tests/test_adam8.py tests/test_aurora.py tests/test_hyperball.py tests/test_magma.py tests/test_matrix_runtime.py tests/test_muon.py tests/test_normuon.py tests/test_optimizer_performance.py tests/test_orthogonalization.py tests/test_pion.py tests/test_prism.py tests/test_psgd.py tests/test_rmnp.py tests/test_sam.py tests/test_soda.py tests/test_trasmuon.py`
