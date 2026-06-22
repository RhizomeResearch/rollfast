from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

import rollfast.finetune as rfft

from .helpers import TinyGroup, TinyPlan, tiny_plan


def test_validate_tiny_plan_counts_groups_and_fingerprint():
    plan = tiny_plan()
    normalized = rfft.validate_plan(plan)

    assert normalized.groups["block_00_decay"].param_count == 4
    assert normalized.groups["block_00_no_decay"].param_count == 2
    assert normalized.groups["head_decay"].param_count == 2
    assert len(normalized.fingerprint) == 64


def test_validation_rejects_missing_label():
    plan = tiny_plan()
    bad_labels = {
        **plan.labels,
        "head": {"w": None},
    }
    bad = replace(plan, labels=bad_labels)

    with pytest.raises(ValueError, match="identical PyTree structure"):
        rfft.validate_plan(bad)


def test_validation_rejects_unknown_label():
    plan = tiny_plan()
    bad_labels = {
        **plan.labels,
        "head": {"w": "unknown"},
    }
    bad = replace(plan, labels=bad_labels)

    with pytest.raises(ValueError, match="missing from plan.group_specs"):
        rfft.validate_plan(bad)


def test_validation_rejects_unused_group_by_default():
    plan = tiny_plan()
    groups = {
        **plan.group_specs,
        "unused_decay": TinyGroup(
            "unused_decay",
            role="head",
            depth=None,
            lr_multiplier=1.0,
            weight_decay=True,
        ),
    }
    bad = replace(plan, group_specs=groups)

    with pytest.raises(ValueError, match="unused group_specs"):
        rfft.validate_plan(bad)


def test_validation_rejects_frozen_label():
    trainable = {"w": jnp.ones((2,), dtype=jnp.float32)}
    labels = {"w": "frozen"}
    groups = {
        "frozen": TinyGroup(
            "frozen",
            role="frozen",
            depth=None,
            lr_multiplier=1.0,
            weight_decay=False,
        )
    }

    with pytest.raises(ValueError, match="not labeled 'frozen'"):
        rfft.validate_plan(TinyPlan(trainable, labels, groups))


def test_validation_rejects_non_inexact_leaf():
    trainable = {"w": jnp.ones((2,), dtype=jnp.int32)}
    labels = {"w": "w_decay"}
    groups = {
        "w_decay": TinyGroup(
            "w_decay",
            role="head",
            depth=None,
            lr_multiplier=1.0,
            weight_decay=True,
        )
    }

    with pytest.raises(ValueError, match="inexact"):
        rfft.validate_plan(TinyPlan(trainable, labels, groups))


def test_empty_all_frozen_plan_is_valid_with_warning():
    plan = TinyPlan(trainable={"w": None}, labels={"w": None}, group_specs={})

    normalized = rfft.validate_plan(plan)

    assert normalized.groups == {}
    assert normalized.warnings == ("plan has no trainable array leaves.",)


def test_fingerprint_ignores_values_but_changes_labels():
    plan = tiny_plan()
    normalized = rfft.validate_plan(plan)
    changed_values = replace(
        plan,
        trainable=jax.tree.map(
            lambda x: x * 7.0 if x is not None else None,
            plan.trainable,
            is_leaf=lambda x: x is None,
        ),
    )
    changed_labels = replace(
        plan,
        labels={**plan.labels, "head": {"w": "block_01_decay"}},
    )

    assert rfft.validate_plan(changed_values).fingerprint == normalized.fingerprint
    assert (
        rfft.validate_plan(changed_labels, allow_empty_groups=True).fingerprint
        != normalized.fingerprint
    )
