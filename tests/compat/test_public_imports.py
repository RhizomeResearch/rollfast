import hashlib

import rollfast
import rollfast.finetune as rfft
import rollfast.integrations as integrations
import rollfast.optim as optim
import rollfast.schedules as schedules


V1_PUBLIC_API_HASHES = {
    "rollfast": "cac29233ba13192662bff946d4533860aedda3ef4a6f8b49f40eb7b092921b4f",
    "rollfast.optim": "02bedaa3841d45e2895e639b977573a2608a09cfee50e24b40dd53cf2a6e041b",
    "rollfast.schedules": "36f678ee4d1e0ccd06e7c2d941ee845951073569ce059d9ac114963d5697fb06",
    "rollfast.finetune": "b41853750772225bfdf38160c2919caa35685218d58612c9d24532f5f04b4c46",
    "rollfast.integrations": "e2b7f315709dcffde971ed945d275bee58e4233675076984b2d185c6ad977703",
}


def test_v1_public_api_exports_remain_available():
    for module in (rollfast, optim, schedules, rfft, integrations):
        for name in module.__all__:
            assert hasattr(module, name), f"{module.__name__}.{name}"


def test_v1_public_api_export_sets_change_deliberately():
    for module in (rollfast, optim, schedules, rfft, integrations):
        exports = "\n".join(sorted(module.__all__)).encode()
        digest = hashlib.sha256(exports).hexdigest()
        assert digest == V1_PUBLIC_API_HASHES[module.__name__], (
            f"{module.__name__} public API changed; review compatibility and update "
            "the v1 baseline only for an intentional release change"
        )


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
