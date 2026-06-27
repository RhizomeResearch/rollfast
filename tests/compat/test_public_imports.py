import rollfast
import rollfast.finetune as rfft


def test_existing_public_imports_remain_available():
    for name in (
        "adamw",
        "adamw8",
        "aurora",
        "riemannian_aurora",
        "prism",
        "kron",
        "schedule_free_adam",
        "schedule_free_prism",
        "schedule_free_kron",
        "schedule_free_aurora",
        "wsd_schedule",
        "global_l2_norm",
        "sam_perturbation",
        "apply_updates",
        "apply_updates_prefix",
    ):
        assert hasattr(rollfast, name), name


def test_finetune_public_imports_are_available():
    for name in (
        "FineTunePlanProtocol",
        "GroupSpecProtocol",
        "OptimizerConfig",
        "ScheduleConfig",
        "GradientPolicy",
        "AccumulationConfig",
        "PrecisionConfig",
        "AdaLoRAController",
        "AdaLoRAState",
        "DEFAULT_NO_DECAY_TAGS",
        "GroupRule",
        "OptimizerBundle",
        "OptimizerReport",
        "OptimizerMigrationReport",
        "OptimizerStateMemoryEstimate",
        "OptimizerStateMemorySummary",
        "validate_plan",
        "compile_optimizer",
        "adamw_from_plan",
        "adamw8_from_plan",
        "hybrid_aurora_adam_from_plan",
        "hybrid_prism_adam_from_plan",
        "hybrid_kron_adam_from_plan",
        "make_sam_step",
        "make_adalora_controller",
        "allocate_rank_mask",
        "make_update_step",
        "sam_cost_report",
        "reconfigure_optimizer",
        "estimate_optimizer_state_memory",
        "optimizer_state_memory_summary",
        "preview_schedule",
        "no_decay_rules",
        "discriminative_adamw_rules",
        "head_backbone_adamw",
        "state_manifest",
        "make_state_checkpoint",
        "restore_state_checkpoint",
        "save_state_checkpoint",
        "load_state_checkpoint",
    ):
        assert hasattr(rfft, name), name
