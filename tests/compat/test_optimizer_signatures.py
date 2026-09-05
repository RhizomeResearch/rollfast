import inspect

import pytest

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


HYBRID_PLAN_PARAMETERS = (
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
)


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


@pytest.mark.parametrize(
    "function, required_parameters",
    [
        pytest.param(
            rollfast.adamw,
            (
                "learning_rate",
                "b1",
                "b2",
                "eps",
                "mu_dtype",
                "weight_decay",
                "weight_decay_mask",
                "axis_name",
            ),
            id="rollfast.adamw",
        ),
        pytest.param(
            rollfast.adamw8,
            (
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
            ),
            id="rollfast.adamw8",
        ),
        pytest.param(
            rfft.adamw_from_plan,
            (
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
            ),
            id="rfft.adamw_from_plan",
        ),
        pytest.param(
            rfft.adamw8_from_plan,
            (
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
            ),
            id="rfft.adamw8_from_plan",
        ),
        pytest.param(
            rfft.schedule_free_adam_from_plan,
            (
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
            ),
            id="rfft.schedule_free_adam_from_plan",
        ),
        pytest.param(
            rfft.hybrid_aurora_adam_from_plan,
            HYBRID_PLAN_PARAMETERS,
            id="rfft.hybrid_aurora_adam_from_plan",
        ),
        pytest.param(
            rfft.hybrid_prism_adam_from_plan,
            HYBRID_PLAN_PARAMETERS,
            id="rfft.hybrid_prism_adam_from_plan",
        ),
        pytest.param(
            rfft.hybrid_kron_adam_from_plan,
            HYBRID_PLAN_PARAMETERS,
            id="rfft.hybrid_kron_adam_from_plan",
        ),
        pytest.param(
            rfft.make_sam_step,
            (
                "plan",
                "base_optimizer",
                "config",
                "loss_fn",
                "has_aux",
                "microbatch_axis",
                "microbatch_count",
                "microbatch_reduction",
            ),
            id="rfft.make_sam_step",
        ),
        pytest.param(
            rfft.make_adalora_controller,
            ("rank_groups", "total_steps", "config"),
            id="rfft.make_adalora_controller",
        ),
        pytest.param(
            rfft.reconfigure_optimizer,
            (
                "old_plan",
                "old_bundle",
                "old_state",
                "new_plan",
                "new_recipe",
                "new_bundle",
                "state_policy",
                "counter_policy",
            ),
            id="rfft.reconfigure_optimizer",
        ),
        pytest.param(
            rfft.optimizer_state_memory_summary,
            ("bundle", "state"),
            id="rfft.optimizer_state_memory_summary",
        ),
        pytest.param(
            rfft.estimate_optimizer_state_memory,
            ("plan", "bundle", "preconditioner_dtype"),
            id="rfft.estimate_optimizer_state_memory",
        ),
    ],
)
def test_public_signature_keeps_required_parameters(function, required_parameters):
    parameters = inspect.signature(function).parameters
    for name in required_parameters:
        assert name in parameters, f"{function.__name__} is missing {name}"
