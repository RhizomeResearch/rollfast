import pytest

import rollfast.finetune as rfft

from .helpers import tiny_lora_plan, tiny_plan


def test_compile_groups_applies_llrd_and_rollfast_rules_once():
    normalized = rfft.validate_plan(tiny_plan())
    groups = rfft.compile_groups(
        normalized.groups,
        rfft.OptimizerConfig(base_lr=1e-3, weight_decay=0.05),
        (rfft.GroupRule(label="head_decay", lr_multiplier=3.0, name="head_lr"),),
    )
    by_label = {group.source_label: group for group in groups}

    assert by_label["block_00_decay"].effective_lr == pytest.approx(5e-4)
    assert by_label["block_01_decay"].effective_lr == pytest.approx(1e-3)
    assert by_label["head_decay"].effective_lr == pytest.approx(6e-3)
    assert by_label["head_decay"].matched_rule_names == ("head_lr",)
    assert by_label["block_00_no_decay"].weight_decay_value == 0.0
    assert by_label["block_00_decay"].weight_decay_value == pytest.approx(0.05)


def test_compile_groups_supports_wd_ft_weight_decay_values():
    normalized = rfft.validate_plan(tiny_plan())
    groups = rfft.compile_groups(
        normalized.groups,
        rfft.OptimizerConfig(base_lr=1e-3, weight_decay=0.123),
        (
            rfft.GroupRule(role="backbone", weight_decay_value=0.05),
            rfft.GroupRule(role="head", weight_decay_value=0.01),
            rfft.GroupRule(tag="bias", weight_decay_value=0.0, priority=1),
        ),
    )
    by_label = {group.source_label: group for group in groups}

    assert by_label["block_00_decay"].weight_decay_value == pytest.approx(0.05)
    assert by_label["block_01_decay"].weight_decay_value == pytest.approx(0.05)
    assert by_label["head_decay"].weight_decay_value == pytest.approx(0.01)
    assert by_label["block_00_no_decay"].weight_decay_value == 0.0
    assert by_label["block_01_no_decay"].weight_decay_value == 0.0
    assert by_label["block_00_decay"].weight_decay is True
    assert by_label["block_00_no_decay"].weight_decay is False


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


def test_conflicting_equal_priority_weight_decay_values_raise():
    normalized = rfft.validate_plan(tiny_plan())

    with pytest.raises(ValueError, match="conflicting weight_decay"):
        rfft.compile_groups(
            normalized.groups,
            rfft.OptimizerConfig(),
            (
                rfft.GroupRule(role="backbone", weight_decay_value=0.05, priority=1),
                rfft.GroupRule(tag="block", weight_decay_value=0.01, priority=1),
            ),
        )


def test_preview_groups_includes_matched_rule_names():
    normalized = rfft.validate_plan(tiny_plan())
    groups = rfft.compile_groups(
        normalized.groups,
        rfft.OptimizerConfig(),
        (rfft.GroupRule(label="head_decay", lr_multiplier=2.0, name="head_lr"),),
    )
    rows = {row["label"]: row for row in rfft.preview_groups(groups)}

    assert rows["head_decay"]["matched_rules"] == ("head_lr",)


def test_required_group_rule_accepts_populated_match_from_iterator():
    plan = tiny_plan()
    rules = iter((rfft.GroupRule(label="head_decay", required=True, name="head"),))

    groups = rfft.compile_groups(
        rfft.validate_plan(plan).groups,
        rfft.OptimizerConfig(),
        rules,
    )

    assert {group.source_label for group in groups} >= {"head_decay"}
    assert next(
        group for group in groups if group.source_label == "head_decay"
    ).matched_rule_names == ("head",)


def test_required_group_rule_rejects_missing_and_empty_matches():
    plan = tiny_plan()

    with pytest.raises(ValueError, match="required group rule 'missing'"):
        rfft.compile_optimizer(
            plan,
            total_steps=2,
            group_rules=(
                rfft.GroupRule(label="missing", required=True, name="missing"),
            ),
        )

    groups = rfft.validate_plan(plan).groups
    empty_group = rfft.PlanGroup(
        label="empty_decay",
        role="head",
        depth=None,
        lr_multiplier=1.0,
        weight_decay=True,
        tags=frozenset(),
    )
    with pytest.raises(ValueError, match="required group rule 'empty'"):
        rfft.compile_groups(
            {**groups, "empty_decay": empty_group},
            rfft.OptimizerConfig(),
            (rfft.GroupRule(label="empty_decay", required=True, name="empty"),),
        )
