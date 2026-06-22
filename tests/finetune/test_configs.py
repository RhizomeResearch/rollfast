import jax.numpy as jnp
import pytest

import rollfast.finetune as rfft


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

    asam = rfft.ASAMConfig(rho=0.5, eta=0.01, axis_name="data")
    assert rfft.ASAMConfig.from_dict(asam.to_dict()) == asam


def test_config_validation_rejects_invalid_values():
    with pytest.raises(ValueError, match="base_lr"):
        rfft.OptimizerConfig(base_lr=0.0)
    with pytest.raises(ValueError, match="warmup_fraction"):
        rfft.ScheduleConfig(warmup_fraction=1.5)
    with pytest.raises(ValueError, match="steps"):
        rfft.AccumulationConfig(steps=0)
    with pytest.raises(ValueError, match="lr_multiplier"):
        rfft.GroupRule(label="head", lr_multiplier=0.0)
    with pytest.raises(ValueError, match="EMA decay"):
        rfft.EMAConfig(decay=1.0)
    with pytest.raises(ValueError, match="SWA frequency"):
        rfft.SWAConfig(frequency=0)
    with pytest.raises(ValueError, match="SAM rho"):
        rfft.SAMConfig(rho=0.0)


def test_public_import_surface_contains_core_builders():
    assert hasattr(rfft, "FineTunePlanProtocol")
    assert hasattr(rfft, "compile_optimizer")
    assert hasattr(rfft, "adamw_from_plan")
    assert hasattr(rfft, "make_update_step")
    assert hasattr(rfft, "make_sam_step")
    assert hasattr(rfft, "ASAMConfig")
