import jax.numpy as jnp
import pytest

import rollfast.finetune as rfft


@pytest.mark.parametrize(
    "kind",
    ("constant", "warmup_cosine", "wsd", "linear", "polynomial"),
)
def test_supported_schedule_kinds_round_trip(kind):
    config = rfft.ScheduleConfig(kind=kind, total_steps=100)

    assert rfft.ScheduleConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("kind", ("custom", "typo"))
def test_unsupported_schedule_kinds_are_rejected(kind):
    with pytest.raises(NotImplementedError, match="ScheduleConfig.kind"):
        rfft.ScheduleConfig(kind=kind)  # type: ignore[arg-type]


@pytest.mark.parametrize("step_counter", ("micro", "typo"))
def test_unsupported_schedule_step_counters_are_rejected(step_counter):
    with pytest.raises(NotImplementedError, match="ScheduleConfig.step_counter"):
        rfft.ScheduleConfig(step_counter=step_counter)  # type: ignore[arg-type]


def test_supported_accumulation_config_round_trips():
    config = rfft.AccumulationConfig(
        remainder="error",
        reduce_after_accumulation=True,
        finite_policy="discard_window",
    )

    assert rfft.AccumulationConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("remainder", "drop"),
        ("remainder", "apply_with_true_normalizer"),
        ("remainder", "typo"),
        ("reduce_after_accumulation", False),
        ("reduce_after_accumulation", "typo"),
        ("finite_policy", "error"),
        ("finite_policy", "typo"),
    ),
)
def test_unsupported_accumulation_config_is_rejected(field, value):
    with pytest.raises(NotImplementedError, match=field):
        rfft.AccumulationConfig(**{field: value})


@pytest.mark.parametrize("cast_back", ("stochastic", "typo"))
def test_unsupported_precision_cast_back_is_rejected(cast_back):
    with pytest.raises(NotImplementedError, match="PrecisionConfig.cast_back"):
        rfft.PrecisionConfig(cast_back=cast_back)  # type: ignore[arg-type]


def test_legacy_unsupported_config_dictionary_has_a_clear_error():
    with pytest.raises(NotImplementedError, match="AccumulationConfig.remainder"):
        rfft.AccumulationConfig.from_dict({"remainder": "drop"})


def test_config_round_trips_are_deterministic():
    schedule = rfft.ScheduleConfig(kind="warmup_cosine", total_steps=100)
    assert rfft.ScheduleConfig.from_dict(schedule.to_dict()) == schedule

    optimizer = rfft.OptimizerConfig(base_lr=2e-4, weight_decay=0.01)
    assert rfft.OptimizerConfig.from_dict(optimizer.to_dict()) == optimizer

    precision = rfft.PrecisionConfig(
        expected_model_compute_dtype=jnp.bfloat16,
        moment_dtype=jnp.float32,
    )
    assert rfft.PrecisionConfig.from_dict(precision.to_dict()) == precision

    gradient_policy = rfft.GradientPolicy(
        axis_name=("data", "model"),
        partition_axis_names=("model",),
        replicated_axis_names=("data",),
    )
    assert rfft.GradientPolicy.from_dict(gradient_policy.to_dict()) == gradient_policy
    raise_policy = rfft.GradientPolicy.from_dict(
        {"nonfinite": "raise", "max_consecutive_nonfinite": 3}
    )
    assert raise_policy.nonfinite == "raise"
    assert raise_policy.max_consecutive_nonfinite == 3

    asam = rfft.ASAMConfig(
        rho=0.5,
        eta=0.01,
        axis_name=("data", "model"),
        partition_axis_names=("model",),
        replicated_axis_names=("data",),
    )
    assert rfft.ASAMConfig.from_dict(asam.to_dict()) == asam

    apollo = rfft.APOLLOConfig(
        rank=8,
        mini=True,
        scale=128.0,
        scale_front=True,
        disable_norm_growth_limiter=True,
    )
    assert rfft.APOLLOConfig.from_dict(apollo.to_dict()) == apollo

    sharding = rfft.ShardingPolicy(
        mesh_axes=("data", "model"),
        parameter_axes=("model",),
        state_placement="replicate_small_follow_large",
        small_state_threshold=128,
    )
    assert rfft.ShardingPolicy.from_dict(sharding.to_dict()) == sharding


def test_config_validation_rejects_invalid_values():
    with pytest.raises(ValueError, match="base_lr"):
        rfft.OptimizerConfig(base_lr=0.0)
    with pytest.raises(ValueError, match="warmup_fraction"):
        rfft.ScheduleConfig(warmup_fraction=1.5)
    with pytest.raises(ValueError, match="steps"):
        rfft.AccumulationConfig(steps=0)
    with pytest.raises(ValueError, match="lr_multiplier"):
        rfft.GroupRule(label="head", lr_multiplier=0.0)
    with pytest.raises(ValueError, match="weight_decay_value"):
        rfft.GroupRule(label="head", weight_decay_value=-0.1)
    with pytest.raises(ValueError, match="weight_decay_value"):
        rfft.GroupRule(label="head", weight_decay_value=float("inf"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        rfft.GroupRule(label="head", weight_decay=True, weight_decay_value=0.01)
    with pytest.raises(ValueError, match="EMA decay"):
        rfft.EMAConfig(decay=1.0)
    with pytest.raises(ValueError, match="EMA source_view"):
        rfft.EMAConfig(source_view="swa")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="SWA frequency"):
        rfft.SWAConfig(frequency=0)
    with pytest.raises(ValueError, match="SWA source_view"):
        rfft.SWAConfig(source_view="ema")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="SAM rho"):
        rfft.SAMConfig(rho=0.0)
    with pytest.raises(ValueError, match="block_layout"):
        rfft.StateQuantizationConfig(block_layout="per_device")
    with pytest.raises(ValueError, match="APOLLO scale"):
        rfft.APOLLOConfig(scale=0.0)
    with pytest.raises(ValueError, match="disjoint"):
        rfft.GradientPolicy(
            partition_axis_names=("model",),
            replicated_axis_names=("model",),
        )


def test_public_import_surface_contains_core_builders():
    assert hasattr(rfft, "FineTunePlanProtocol")
    assert hasattr(rfft, "compile_optimizer")
    assert hasattr(rfft, "adamw_from_plan")
    assert hasattr(rfft, "make_update_step")
    assert hasattr(rfft, "make_sam_step")
    assert hasattr(rfft, "ASAMConfig")
