# `rollfast.finetune`

## Overview

Plan-driven fine-tuning subsystem compiling parameter-group rules, learning-rate schedules, mixed-precision/quantization
policies, EMA/SWA/Schedule-Free evaluation views, stateful and SAM/ASAM update steps, AdaLoRA rank controllers,
optimizer state migration, diagnostics, host offloading, and single-file or multi-host array checkpoints.

## Interface

- Protocols & Plan Validation (`[_protocols.py]`, `[validation.py]`, `[groups.py]`):
  - `PyTree`, `GroupSpecProtocol`, `FineTunePlanProtocol`, `CombinableFineTunePlanProtocol` (`[_protocols.py:__all__]`).
  - `NormalizedPlan`, `validate_plan`, `plan_fingerprint` (`[validation.py:__all__]`).
  - `compile_groups`, `matches_rule`, `preview_groups`, `unmatched_rule_warnings` (`[groups.py:__all__]`).
- Configuration, Recipes & Schedules (`[config.py]`, `[recipes.py]`, `[schedules.py]`):
  - `SCHEMA_VERSION`, `OptimizerConfig`, `OptimizerProfile`, `ScheduleConfig`, `SchedulePoint`, `GradientPolicy`,
    `AccumulationConfig`, `PrecisionConfig`, `PrecisionReport`, `StateQuantizationConfig`, `GroupRule`, `PlanGroup`,
    `CompiledGroup`, `EMAConfig`, `SWAConfig`, `SAMConfig`, `ASAMConfig`, `AdaLoRAControllerConfig`, `GaLoreConfig`,
    `APOLLOConfig`, `MuonConfig`, `LoRAPlusConfig`, `ShardingPolicy`, `StateOffloadPolicy`, `LossBundle`, `AuxLossRule`,
    `NormalizedLeaf`, `CompiledPolicyTrees`, `AlgorithmSemantics`, `OptimizerReport`, `OptimizerBundle`,
    `AccumulationState`, `LossScaleState`, `StepCounters`, `RNGStreams`, `FineTuneStepState`, `AveragingSourceView`
    (`[config.py:__all__]`, `[__init__.py:__all__]`), plus module `Literal` type aliases `ScheduleKind`, `StepCounter`,
    `NonFinitePolicy`, `Normalization`, `SAMNorm`, `ParameterRounding`, `PrecisionMode`, `OptimizerName`, and
    `ProfileFidelity` (`[config.py]`).
  - `DEFAULT_NO_DECAY_TAGS`, `FineTuneOptimizerRecipe`, `no_decay_rules`, `discriminative_adamw_rules`,
    `head_backbone_adamw`, `full_ft_adamw`, `partial_ft_adamw`, `linear_probe_adamw`, `lora_adamw`, `adapters_adamw`,
    `prompts_adamw`, `bitfit_adamw`, `bf16_mixed_precision`, `bf16_master_free_precision`, `fp16_mixed_precision`
    (`[recipes.py:__all__]`).
  - `build_schedule`, `preview_schedule` (`[schedules.py:__all__]`).
- Optimizer Compilation, Transforms & Step Builders (`[builders.py]`, `[transforms.py]`, `[step.py]`,
  `[update_status.py]`):
  - `compile_optimizer`, `optimizer_from_plan`, `adamw_from_plan`, `adamw8_from_plan`, `galore_adamw_from_plan`,
    `apollo_adamw_from_plan`, `schedule_free_adam_from_plan`, `hybrid_aurora_adam_from_plan`,
    `hybrid_prism_adam_from_plan`, `hybrid_kron_adam_from_plan`, `muon_adam_from_plan`, `estimate_optimizer_state`
    (`[builders.py:__all__]`).
  - `AlwaysSkipNonFiniteState`, `always_skip_nonfinite`, `clip_by_global_norm` (`[transforms.py:__all__]`).
  - `init_finetune_step_state`, `init_accumulation_state`, `init_loss_scale_state` (`[step.py:init_loss_scale_state]`,
    re-exported in `[__init__.py:__all__]`), `init_rng_streams`, `init_step_counters`, `make_master_params`,
    `make_update_step`, `make_plan_update_step`, `make_master_update_step`, `make_plan_master_update_step`,
    `make_loss_scaled_master_update_step`, `make_plan_loss_scaled_master_update_step`,
    `make_stateful_loss_scaled_master_update_step`, `make_finetune_update_step`, `make_skip_attempt_step`,
    `make_loss_bundle_update_step`, `make_plan_loss_bundle_update_step`, `make_accumulating_loss_bundle_update_step`,
    `make_sam_step`, `make_stateful_sam_step`, `make_finetune_sam_step`, `sam_cost_report`, `ModelStateAggregator`,
    `SAMStepInfo`, `StatefulStepInfo`, `StatefulSAMStepInfo`, `AccumulatingLossBundleInfo` (`[step.py:__all__]`).
  - `UpdateStatus`, `validate_accumulation_complete` (`[update_status.py:__all__]`).
- Controllers, Evaluation, Diagnostics, Offload & Checkpoints (`[adalora.py]`, `[averaging.py]`, `[evaluation.py]`,
  `[diagnostics.py]`, `[offload.py]`, `[serialization.py]`, `[array_storage.py]`, `[state_migration.py]`,
  `[reconfigure.py]`):
  - `AdaLoRAController`, `AdaLoRAState`, `make_adalora_controller`, `adalora_triplet_importance`, `allocate_rank_mask`
    (`[adalora.py:__all__]`).
  - `AveragingState` (`[__init__.py:__all__]`), plus submodule exports `averaging_enabled`, `default_eval_view`,
    `eval_views`, `make_averaging_eval_fn`, `wrap_with_averaging` (`[averaging.py:__all__]`) and `EvalFn`
    (`[averaging.py:EvalFn]`); `EvaluationView`, `bind_evaluation_view` (`[evaluation.py:__all__]`).
  - `StateLeafSummary`, `OptimizerStateMemorySummary`, `OptimizerStateMemoryEstimate`, `optimizer_state_memory_summary`,
    `estimate_optimizer_state_memory`, `low_precision_update_metrics` (`[diagnostics.py:__all__]`).
  - `with_state_offload`, `offload_optimizer_state`, `restore_offloaded_optimizer_state`, `state_offload_manifest`
    (`[offload.py:__all__]`).
  - `OptimizerStateCheckpoint`, `OptimizerStateRestoreError`, `state_manifest`, `make_state_checkpoint`,
    `make_step_state_checkpoint`, `step_state_from_checkpoint`, `restore_state_checkpoint`, `save_state_checkpoint`,
    `load_state_checkpoint` (`[serialization.py:__all__]`).
  - `ArrayCheckpointLimits`, `save_array_state_checkpoint`, `load_array_state_checkpoint`
    (`[array_storage.py:__all__]`).
  - `StatePolicy`, `CounterPolicy`, `OptimizerMigrationReport`, `StateTransferReport`, `reconfigure_optimizer`,
    `transfer_optimizer_state` (`[state_migration.py:__all__]`, re-exported by `[reconfigure.py:__all__]`).

## Invariants

- Plan alignment and group-spec rules (`[validation.py:validate_plan]`, `[groups.py:compile_groups]`,
  `[builders.py:_validate_complex_plan_compatibility]`): `trainable`, `frozen`, `labels`, and `identities` must share an
  identical PyTree structure (`None` as leaf); every active trainable leaf must be `jnp.inexact` with `frozen is None`,
  a non-`"frozen"` group label in `group_specs`, and a unique `logical_id`; `*_no_decay` specs require
  `weight_decay == False` (`GroupRule` cannot re-enable decay on undecayed plan groups).
  `_validate_complex_plan_compatibility` rejects complex trainable leaves in `adamw8`, `galore_adamw`, `apollo_adamw`,
  and `schedule_free_adam`, whereas `adamw` and hybrid builders (`aurora_adam`, `prism_adam`, `kron_adam`, `muon_adam`)
  accept complex leaves (routing complex leaves to Adam fallback in matrix/Adam hybrids or handling them in `kron`).
- Schedule clocks, fail-closed config variants, and evaluation views (`[config.py:ScheduleConfig]`,
  `[builders.py:_SCHEDULE_CLOCK_STATES]`, `[builders.py:_require_optimizer_schedule_clock]`,
  `[builders.py:_apply_gradient_policy]`, `[averaging.py:_require_initialized]`): `ScheduleConfig.total_steps` counts
  intervals of `steps_per_interval` updates while `total_updates` (`total_steps * steps_per_interval`) drives
  `OptimizerReport.total_steps`, SWA start steps, and default previews; `step_counter="attempt"` advances only
  `ScaleByScheduleState` and `_FactorizedAdamWState` counts on rejected/skipped attempts and is forbidden
  (`NotImplementedError`) for `schedule_free_adam`, `galore_adamw`, and `apollo_adamw`. Several `Literal`-typed config
  options fail closed (`ScheduleConfig(step_counter="micro")`, `GradientPolicy(nonfinite="raise")`, non-default
  `AccumulationConfig` remainder/reduction/finite policies, `PrecisionConfig` with `update_compute_dtype != float32` or
  `cast_back != "nearest"`, `GaLoreConfig(basis_method != "svd")`, and non-`"adamw"` `fallback_optimizer` on
  `APOLLOConfig`/`MuonConfig`). `wrap_with_averaging` requires `params is not None` on `update`, and resolving `"ema"`
  or `"swa"` views via `eval_params` requires host execution outside `jax.jit` (`TypeError` on tracers) with `count > 0`
  (`ValueError`).
- Precision, accumulation, and CUDA-graph-safe step commits (`[_precision.py:validate_precision]`,
  `[transforms.py:always_skip_nonfinite]`, `[step.py:_check_accumulation_guard]`, `[step.py:_commit_optimizer_result]`,
  `[step.py:_make_sam_core]`): `PrecisionConfig.accumulation_dtype` must equal `AccumulationConfig.accumulate_dtype`;
  `parameter_rounding="stochastic"` requires `master_params="never"`, `bfloat16` trainable leaves, and `RNGStreams`.
  Loss-scaled and stateful master step builders (`make_loss_scaled_master_update_step`,
  `make_plan_loss_scaled_master_update_step`, `make_stateful_loss_scaled_master_update_step`,
  `make_finetune_update_step`) require `gradient_policy.nonfinite="skip"` when `accumulation.steps > 1`
  (`NotImplementedError`); SAM builders, `make_skip_attempt_step`, and `make_accumulating_loss_bundle_update_step`
  require `optimizer.accumulation_config.steps == 1` (with `make_accumulating_loss_bundle_update_step` taking its own
  `accumulation.steps > 1` config). Step commits, `always_skip_nonfinite`, and EMA/SWA updates select via `jnp.where`
  without `jax.lax.cond`, and declared collective axes require unanimous finiteness/acceptance agreement
  (`[step.py:_agree_across_axes]`).
- Checkpoint trust boundaries and schema compatibility (`[serialization.py:load_state_checkpoint]`,
  `[serialization.py:make_step_state_checkpoint]`, `[serialization.py:_compatibility_view]`,
  `[_checkpoint_tree.py:_FIELD_SCHEMA_DATACLASSES]`, `[array_storage.py:_prepare_array_load]`,
  `[array_storage.py:_reject_symlinked_parents]`, `[array_storage.py:_open_checkpoint_file]`): `load_state_checkpoint`
  refuses to unpickle unless `trusted=True` (`OptimizerStateRestoreError`); `make_step_state_checkpoint` rejects
  non-`None` `ema_state`, `swa_state`, `adalora_state`, `projection_state`, or `schedule_free_state` (`ValueError`);
  `RNGStreams` and `StepCounters` retain `"dataclass"` schema kind in `_FIELD_SCHEMA_DATACLASSES`; strict manifest
  restore backfills `_PRECISION_FIELDS_ADDED_IN_1_3` and `_SCHEDULE_FIELDS_ADDED_AFTER_1_4`; array checkpoint I/O
  rejects symlinked parent directories (`_reject_symlinked_parents`) and opens checkpoint files with `O_NOFOLLOW` +
  `S_ISREG`, enforces `ArrayCheckpointLimits` and SHA-256 checksums before NPY header parsing, requires gapless shard
  grids, and enforces cross-process generation agreement.

## Side Effects

- Import-time PyTree registration (`[config.py:LossBundle]`, `[config.py:AccumulationState]`,
  `[config.py:StepCounters]`, `[config.py:RNGStreams]`, `[config.py:FineTuneStepState]`): Registers frozen dataclasses
  in JAX's global PyTree node registry at import time.
- Filesystem checkpoint I/O (`[serialization.py:save_state_checkpoint]`, `[serialization.py:load_state_checkpoint]`,
  `[array_storage.py:save_array_state_checkpoint]`, `[array_storage.py:load_array_state_checkpoint]`): Writes/fsyncs
  atomic pickle files or multi-rank `.{name}.generations/{generation}/rank-*/*.npy` shard directories and JSON
  manifests, and reads checkpoint files from disk.
- Device-host transfers, collectives & warnings (`[offload.py:offload_optimizer_state]`,
  `[offload.py:restore_offloaded_optimizer_state]`, `[evaluation.py:bind_evaluation_view]`,
  `[averaging.py:_require_initialized]`, `[update_status.py:validate_accumulation_complete]`,
  `[schedules.py:preview_schedule]`, `[adalora.py:AdaLoRAController.rank_report]`, `[step.py:_agree_across_axes]`,
  `[array_storage.py:save_array_state_checkpoint]`, `[array_storage.py:load_array_state_checkpoint]`,
  `[step.py:_check_direct_precision]`, `[serialization.py:make_state_checkpoint]`): Transfers arrays/counters between
  device and host via `jax.device_get` / `jax.device_put`, runs SPMD and multi-host collectives (`jax.lax.psum`,
  `process_allgather`, `broadcast_one_to_all`), queries `importlib.metadata.version("rollfast")` in
  `OptimizerBundle.manifest()`, and emits `UserWarning` via `warnings.warn` when master parameters or loss scaling are
  bypassed.

## Verification

- Run: `uv run pytest tests/finetune`
