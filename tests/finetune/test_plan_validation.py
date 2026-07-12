from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import rollfast.finetune as rfft

from .helpers import TinyGroup, TinyPlan, tiny_plan


@dataclass(frozen=True)
class _IdentityWithSharding:
    logical_id: str
    alias_group: str | None = None
    layout: str | None = None
    sharding_fingerprint: str | None = None


@dataclass(frozen=True)
class _MinimalCompilerPlan:
    trainable: Any
    frozen: Any
    labels: Any
    group_specs: dict[str, TinyGroup]
    identities: Any


class _CombinablePlan:
    def __init__(self, plan: _MinimalCompilerPlan) -> None:
        self.trainable = plan.trainable
        self.frozen = plan.frozen
        self.labels = plan.labels
        self.group_specs = plan.group_specs
        self.identities = plan.identities
        self.combine_calls = 0

    def combine(self, trainable):
        self.combine_calls += 1
        return {"params": trainable}


def _minimal_compiler_plan() -> _MinimalCompilerPlan:
    trainable = {"w": jnp.ones((2,), dtype=jnp.float32)}
    return _MinimalCompilerPlan(
        trainable=trainable,
        frozen={"w": None},
        labels={"w": "w_decay"},
        group_specs={
            "w_decay": TinyGroup(
                "w_decay",
                role="head",
                depth=None,
                lr_multiplier=1.0,
                weight_decay=True,
            )
        },
        identities={"w": _IdentityWithSharding(logical_id="w")},
    )


def test_minimal_compiler_plan_requires_only_five_fields():
    plan = _minimal_compiler_plan()

    normalized = rfft.validate_plan(plan)

    assert normalized.logical_id_table_hash
    assert isinstance(plan, rfft.FineTunePlanProtocol)
    assert not isinstance(plan, rfft.CombinableFineTunePlanProtocol)


@pytest.mark.parametrize(
    "missing_field",
    ("trainable", "frozen", "labels", "group_specs", "identities"),
)
def test_validation_names_each_missing_required_field(missing_field):
    plan = _minimal_compiler_plan()
    fields = {
        name: getattr(plan, name)
        for name in ("trainable", "frozen", "labels", "group_specs", "identities")
        if name != missing_field
    }

    with pytest.raises(TypeError, match=missing_field):
        rfft.validate_plan(SimpleNamespace(**fields))


def test_plan_update_step_calls_combine():
    plan = _CombinablePlan(_minimal_compiler_plan())
    optimizer = rfft.adamw_from_plan(
        plan,
        total_steps=1,
        schedule="constant",
        clip_global_norm=None,
    )
    step = rfft.make_plan_update_step(
        plan,
        lambda model: jnp.sum(model["params"]["w"] ** 2),
        optimizer,
    )

    step(plan.trainable, optimizer.init(plan.trainable))

    assert plan.combine_calls == 1


def test_validation_rejects_identity_without_logical_id():
    plan = _minimal_compiler_plan()
    bad = replace(plan, identities={"w": object()})

    with pytest.raises(ValueError, match="logical_id"):
        rfft.validate_plan(bad)


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

    with pytest.raises(ValueError, match="non-None label"):
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


def test_fingerprint_changes_when_identity_sharding_changes():
    trainable = {"w": jnp.ones((2,), dtype=jnp.float32)}
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
    plan_a = TinyPlan(
        trainable,
        labels,
        groups,
        identities={
            "w": _IdentityWithSharding(
                logical_id="w",
                layout="dense",
                sharding_fingerprint="mesh-a",
            )
        },
    )
    plan_b = replace(
        plan_a,
        identities={
            "w": _IdentityWithSharding(
                logical_id="w",
                layout="dense",
                sharding_fingerprint="mesh-b",
            )
        },
    )

    assert (
        rfft.validate_plan(plan_a).fingerprint != rfft.validate_plan(plan_b).fingerprint
    )


def test_complex_plan_rejects_adamw8_before_state_initialization():
    plan = replace(
        _minimal_compiler_plan(),
        trainable={"w": jnp.ones((2,), dtype=jnp.complex64)},
    )

    with pytest.raises(ValueError, match=r"adamw8.*complex.*w"):
        rfft.compile_optimizer(
            plan,
            optimizer=rfft.OptimizerConfig(name="adamw8"),
            state_quantization=rfft.StateQuantizationConfig(enabled=True),
            total_steps=1,
        )


def test_complex_plan_rejects_real_master_parameter_cast():
    plan = replace(
        _minimal_compiler_plan(),
        trainable={"w": jnp.ones((2,), dtype=jnp.complex64)},
    )

    with pytest.raises(ValueError, match=r"AdamW.*real master-parameter.*w"):
        rfft.compile_optimizer(
            plan,
            precision=rfft.PrecisionConfig(
                master_params="always",
                master_param_dtype=jnp.float32,
            ),
            total_steps=1,
        )
