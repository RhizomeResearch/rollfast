import inspect

import rollfast
import rollfast.finetune as rfft


def test_adamw_signature_keeps_core_parameters():
    signature = inspect.signature(rollfast.adamw)

    for name in (
        "learning_rate",
        "b1",
        "b2",
        "eps",
        "mu_dtype",
        "weight_decay",
        "weight_decay_mask",
        "axis_name",
    ):
        assert name in signature.parameters


def test_adamw8_signature_exposes_quantized_state_controls():
    signature = inspect.signature(rollfast.adamw8)

    for name in (
        "learning_rate",
        "b1",
        "b2",
        "eps",
        "weight_decay",
        "block_size",
        "min_size",
        "scale_dtype",
        "fallback_dtype",
        "stochastic_rounding",
    ):
        assert name in signature.parameters


def test_finetune_adamw_from_plan_signature_has_plan_controls():
    signature = inspect.signature(rfft.adamw_from_plan)

    for name in (
        "plan",
        "total_steps",
        "base_lr",
        "schedule",
        "weight_decay",
        "clip_global_norm",
        "accumulation_steps",
        "moment_dtype",
        "lora_b_lr_ratio",
        "axis_name",
        "ema",
        "swa",
    ):
        assert name in signature.parameters


def test_finetune_adamw8_from_plan_signature_has_plan_controls():
    signature = inspect.signature(rfft.adamw8_from_plan)

    for name in (
        "plan",
        "total_steps",
        "base_lr",
        "schedule",
        "weight_decay",
        "clip_global_norm",
        "accumulation_steps",
        "lora_b_lr_ratio",
        "axis_name",
        "state_quantization",
        "ema",
        "swa",
    ):
        assert name in signature.parameters


def test_finetune_schedule_free_adam_from_plan_signature_has_plan_controls():
    signature = inspect.signature(rfft.schedule_free_adam_from_plan)

    for name in (
        "plan",
        "total_steps",
        "base_lr",
        "schedule",
        "weight_decay",
        "clip_global_norm",
        "accumulation_steps",
        "moment_dtype",
        "state_dtype",
        "lora_b_lr_ratio",
        "axis_name",
        "weighting_mode",
        "sf_b1",
        "schedule_free_plus",
        "ema",
        "swa",
    ):
        assert name in signature.parameters


def test_finetune_hybrid_signatures_have_plan_controls():
    for builder_name in (
        "hybrid_aurora_adam_from_plan",
        "hybrid_prism_adam_from_plan",
        "hybrid_kron_adam_from_plan",
    ):
        signature = inspect.signature(getattr(rfft, builder_name))
        for name in (
            "plan",
            "total_steps",
            "base_lr",
            "schedule",
            "weight_decay",
            "clip_global_norm",
            "accumulation_steps",
            "moment_dtype",
            "axis_name",
            "ema",
            "swa",
        ):
            assert name in signature.parameters


def test_make_sam_step_signature_has_two_pass_controls():
    signature = inspect.signature(rfft.make_sam_step)

    for name in (
        "plan",
        "base_optimizer",
        "config",
        "loss_fn",
        "has_aux",
        "microbatch_axis",
        "microbatch_count",
        "microbatch_reduction",
    ):
        assert name in signature.parameters


def test_make_adalora_controller_signature_has_rank_controls():
    signature = inspect.signature(rfft.make_adalora_controller)

    for name in (
        "rank_groups",
        "total_steps",
        "config",
    ):
        assert name in signature.parameters


def test_reconfigure_optimizer_signature_has_migration_controls():
    signature = inspect.signature(rfft.reconfigure_optimizer)

    for name in (
        "old_plan",
        "old_bundle",
        "old_state",
        "new_plan",
        "new_recipe",
        "new_bundle",
        "state_policy",
        "counter_policy",
    ):
        assert name in signature.parameters


def test_optimizer_state_memory_summary_signature_has_state_inputs():
    signature = inspect.signature(rfft.optimizer_state_memory_summary)

    for name in (
        "bundle",
        "state",
    ):
        assert name in signature.parameters


def test_estimate_optimizer_state_memory_signature_has_plan_inputs():
    signature = inspect.signature(rfft.estimate_optimizer_state_memory)

    for name in (
        "plan",
        "bundle",
        "preconditioner_dtype",
    ):
        assert name in signature.parameters
