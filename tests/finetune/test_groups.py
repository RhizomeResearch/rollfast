import pytest

import rollfast.finetune as rfft

from .helpers import tiny_lora_plan, tiny_plan


def test_compile_groups_applies_llrd_and_rollfast_rules_once():
    normalized = rfft.validate_plan(tiny_plan())
    groups = rfft.compile_groups(
        normalized.groups,
        rfft.OptimizerConfig(base_lr=1e-3, weight_decay=0.05),
        (rfft.GroupRule(label="head_decay", lr_multiplier=3.0),),
    )
    by_label = {group.source_label: group for group in groups}

    assert by_label["block_00_decay"].effective_lr == pytest.approx(5e-4)
    assert by_label["block_01_decay"].effective_lr == pytest.approx(1e-3)
    assert by_label["head_decay"].effective_lr == pytest.approx(6e-3)
    assert by_label["block_00_no_decay"].weight_decay_value == 0.0
    assert by_label["block_00_decay"].weight_decay_value == pytest.approx(0.05)


def test_lora_b_ratio_applies_only_to_lora_b_labels():
    normalized = rfft.validate_plan(tiny_lora_plan())
    groups = rfft.compile_groups(
        normalized.groups,
        rfft.OptimizerConfig(base_lr=2e-4, weight_decay=0.0, lora_b_lr_ratio=16.0),
    )
    by_label = {group.source_label: group for group in groups}

    assert by_label["lora_A_decay"].effective_lr == pytest.approx(2e-4)
    assert by_label["lora_B_decay"].effective_lr == pytest.approx(3.2e-3)


def test_conflicting_equal_priority_rules_raise():
    normalized = rfft.validate_plan(tiny_plan())

    with pytest.raises(ValueError, match="conflicting weight_decay"):
        rfft.compile_groups(
            normalized.groups,
            rfft.OptimizerConfig(),
            (
                rfft.GroupRule(role="head", weight_decay=True, priority=1),
                rfft.GroupRule(role="head", weight_decay=False, priority=1),
            ),
        )
