import inspect

import jax
import rollfast
import rollfast.finetune as rfft
import rollfast.finetune.builders as builders_module
import rollfast.optim.adam as adam_module
import rollfast.optim.adam8 as adam8_module
import rollfast.optim.aurora as aurora_module
import rollfast.optim.hyperball as hyperball_module
import rollfast.optim.muon as muon_module
import rollfast.optim.normuon as normuon_module
import rollfast.optim.pion as pion_module
import rollfast.optim.prism as prism_module
import rollfast.optim.psgd as psgd_module
import rollfast.optim.rmnp as rmnp_module
import rollfast.optim.soda as soda_module
import rollfast.optim.trasmuon as trasmuon_module
import rollfast.schedules.schedulefree as schedulefree_module


def test_key_defaults_are_not_jax_arrays():
    modules = (
        adam_module,
        adam8_module,
        aurora_module,
        builders_module,
        hyperball_module,
        muon_module,
        normuon_module,
        pion_module,
        prism_module,
        psgd_module,
        rmnp_module,
        schedulefree_module,
        soda_module,
        trasmuon_module,
    )

    for module in modules:
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not fn.__module__.startswith("rollfast."):
                continue
            parameter = inspect.signature(fn).parameters.get("key")
            if parameter is None or parameter.default is inspect.Parameter.empty:
                continue
            assert not isinstance(parameter.default, jax.Array), (
                f"{module.__name__}.{name} captures a JAX array as its key default"
            )


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
