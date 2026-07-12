from types import SimpleNamespace

import pytest

from rollfast.integrations import equimo

from tests.finetune.helpers import tiny_plan


def test_equimo_integration_accepts_structural_plan_without_importing_equimo():
    bundle = equimo.adamw_from_equimo_plan(
        tiny_plan(),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )

    assert bundle.report.trainable_params == 14


def test_equimo_compiler_integration_does_not_require_combine():
    source = tiny_plan()
    plan = SimpleNamespace(
        trainable=source.trainable,
        frozen=source.frozen,
        labels=source.labels,
        group_specs=source.group_specs,
        identities=source.identities,
    )

    bundle = equimo.adamw_from_equimo_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )

    assert bundle.report.trainable_params == 14


def test_equimo_update_integration_requires_combine():
    source = tiny_plan()
    plan = SimpleNamespace(
        trainable=source.trainable,
        frozen=source.frozen,
        labels=source.labels,
        group_specs=source.group_specs,
        identities=source.identities,
    )

    optimizer = equimo.adamw_from_equimo_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )

    with pytest.raises(TypeError, match="combine"):
        equimo.make_equimo_update_step(plan, lambda _: 0.0, optimizer)
