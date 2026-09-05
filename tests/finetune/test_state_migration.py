import jax
import jax.numpy as jnp
import pytest
from typing import NamedTuple

import rollfast.finetune as rfft
from rollfast.finetune import state_migration as migration_module

from .helpers import ones_like_trainable, TinyGroup, TinyPlan, tiny_plan


def _head_only_plan() -> TinyPlan:
    full = tiny_plan()
    return TinyPlan(
        trainable={
            "embed": None,
            "blocks": (
                {"w": None, "b": None},
                {"w": None, "b": None},
            ),
            "head": {"w": full.trainable["head"]["w"]},
        },
        labels={
            "embed": None,
            "blocks": (
                {"w": None, "b": None},
                {"w": None, "b": None},
            ),
            "head": {"w": "head_decay"},
        },
        group_specs={"head_decay": full.group_specs["head_decay"]},
    )


def _renamed_groups_plan() -> TinyPlan:
    full = tiny_plan()
    mapping = {label: f"stage2_{label}" for label in full.group_specs}

    def rename_label(label):
        return None if label is None else mapping[label]

    labels = jax.tree.map(
        rename_label,
        full.labels,
        is_leaf=lambda x: x is None,
    )
    groups = {
        mapping[label]: TinyGroup(
            mapping[label],
            role=group.role,
            depth=group.depth,
            lr_multiplier=group.lr_multiplier,
            weight_decay=group.weight_decay,
            tags=group.tags,
        )
        for label, group in full.group_specs.items()
    }
    return TinyPlan(
        trainable=full.trainable,
        labels=labels,
        group_specs=groups,
    )


def _path_text(path):
    parts = []
    for part in path:
        if hasattr(part, "name"):
            parts.append(f"attr:{part.name}")
        elif hasattr(part, "key"):
            parts.append(f"key:{part.key}")
        elif hasattr(part, "idx"):
            parts.append(f"idx:{part.idx}")
        else:
            parts.append(repr(part))
    return "/".join(parts)


def _state_leaf(state, *needles):
    for path, leaf in jax.tree_util.tree_flatten_with_path(
        state,
        is_leaf=lambda x: x is None,
    )[0]:
        text = _path_text(path)
        if all(needle in text for needle in needles) and hasattr(leaf, "shape"):
            return leaf
    raise AssertionError(f"state leaf not found for {needles!r}")


def _count_leaves(state, group_label):
    leaves = []
    global_leaves = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(
        state,
        is_leaf=lambda x: x is None,
    )[0]:
        text = _path_text(path)
        if text.endswith("attr:count") and hasattr(leaf, "shape"):
            if group_label in text:
                leaves.append(leaf)
            elif "_FactorizedAdamWState" not in text:
                global_leaves.append(leaf)
    if leaves:
        return leaves
    del group_label
    return global_leaves
    return leaves


def _counter_leaves(state):
    leaves = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(
        state,
        is_leaf=lambda x: x is None,
    )[0]:
        tokens = tuple(_path_text((part,)) for part in path)
        owner = migration_module._counter_owner(tokens)
        if owner is not None and hasattr(leaf, "shape"):
            leaves[_path_text(path)] = (owner, leaf)
    return leaves


def _advance_clock_fixture(bundle, params):
    state = bundle.init(params)
    finite_grads = ones_like_trainable(params)
    nonfinite_grads = jax.tree.map(
        lambda x: None if x is None else jnp.full_like(x, jnp.nan),
        params,
        is_leaf=lambda x: x is None,
    )
    for grads in (
        finite_grads,
        finite_grads,
        nonfinite_grads,
        nonfinite_grads,
        finite_grads,
    ):
        updates, state = bundle.update(grads, state, params)
        params = jax.tree.map(
            lambda param, update: None if param is None else param + update,
            params,
            updates,
            is_leaf=lambda x: x is None,
        )
    return state


class _MysteryCounterState(NamedTuple):
    mystery_count: jax.Array


def test_reconfigure_preserves_shared_head_moments_and_initializes_backbone():
    old_plan = _head_only_plan()
    new_plan = tiny_plan()
    old_bundle = rfft.adamw_from_plan(
        old_plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    new_bundle = rfft.adamw_from_plan(
        new_plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
    )
    old_state = old_bundle.init(old_plan.trainable)
    grads = ones_like_trainable(old_plan.trainable)
    _, old_state = old_bundle.update(grads, old_state, old_plan.trainable)

    _, migrated_state, migration = rfft.reconfigure_optimizer(
        old_plan=old_plan,
        old_bundle=old_bundle,
        old_state=old_state,
        new_plan=new_plan,
        new_bundle=new_bundle,
        state_policy="preserve_shared",
        counter_policy="restart_schedule",
    )

    assert jnp.allclose(
        _state_leaf(migrated_state, "attr:mu", "key:head", "key:w"),
        _state_leaf(old_state, "attr:mu", "key:head", "key:w"),
    )
    assert jnp.allclose(
        _state_leaf(migrated_state, "attr:mu", "key:blocks", "idx:0", "key:w"),
        0.0,
    )
    assert any("key:head/key:w" in path for path in migration.preserved_state_leaves)
    assert any(
        "key:blocks/idx:0/key:w" in path for path in migration.initialized_state_leaves
    )
    assert "logical/head.w" in migration.preserved_param_leaves
    assert "logical/blocks.0.w" in migration.initialized_param_leaves
    assert migration.schedule_counter_behavior.startswith("all known clocks reset")
    assert migration.new_state_bytes >= migration.old_state_bytes


def test_transfer_optimizer_state_reports_new_preserved_and_warnings():
    old_plan = _head_only_plan()
    new_plan = tiny_plan()
    old_bundle = rfft.adamw_from_plan(
        old_plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    new_bundle = rfft.adamw_from_plan(
        new_plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
    )
    old_state = old_bundle.init(old_plan.trainable)
    _, old_state = old_bundle.update(
        ones_like_trainable(old_plan.trainable),
        old_state,
        old_plan.trainable,
    )

    _, migrated_state, transfer = rfft.transfer_optimizer_state(
        old_plan=old_plan,
        old_bundle=old_bundle,
        old_state=old_state,
        new_plan=new_plan,
        new_bundle=new_bundle,
        state_policy="preserve_shared",
        counter_policy="restart_schedule",
    )

    assert not transfer.exact
    assert "logical/head.w" in transfer.preserved_ids
    assert "logical/blocks.0.w" in transfer.new_ids
    assert transfer.dropped_ids == ()
    assert transfer.reset_ids == ()
    assert transfer.source_state_bytes > 0
    assert transfer.target_state_bytes >= transfer.source_state_bytes
    assert transfer.counter_policy["optimizer"] == "restart_schedule"
    assert transfer.counter_policy["selected_policy"] == "restart_schedule"
    assert transfer.counter_policy["optimizer_algorithm_schedule"] == "reset"
    assert transfer.counter_policy["finite_guard"] == "reset"
    assert transfer.counter_policy["accumulation"] == "reset"
    assert transfer.counter_policy["averaging"] == "reset"
    assert transfer.to_dict()["counter_policy"] == dict(transfer.counter_policy)
    assert any("new trainable parameters" in warning for warning in transfer.warnings)
    assert jnp.allclose(
        _state_leaf(migrated_state, "attr:mu", "key:head", "key:w"),
        _state_leaf(old_state, "attr:mu", "key:head", "key:w"),
    )


def test_transfer_optimizer_state_can_be_exact_for_identical_stage():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    _, state = bundle.update(ones_like_trainable(plan.trainable), state, plan.trainable)

    _, migrated_state, transfer = rfft.transfer_optimizer_state(
        old_plan=plan,
        old_bundle=bundle,
        old_state=state,
        new_plan=plan,
        new_bundle=bundle,
        state_policy="preserve_exact_group",
        counter_policy="continue_global_step",
    )

    assert transfer.exact
    assert transfer.converted_ids == ()
    assert transfer.new_ids == ()
    assert transfer.dropped_ids == ()
    assert transfer.reset_ids == ()
    assert transfer.source_state_bytes == transfer.target_state_bytes
    assert transfer.to_dict()["exact"] is True
    assert jnp.allclose(
        _state_leaf(migrated_state, "attr:mu", "key:head", "key:w"),
        _state_leaf(state, "attr:mu", "key:head", "key:w"),
    )


def test_transfer_optimizer_state_reports_group_conversion():
    old_plan = tiny_plan()
    new_plan = _renamed_groups_plan()
    old_bundle = rfft.hybrid_kron_adam_from_plan(
        old_plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        preconditioner_update_probability=1.0,
    )
    new_bundle = rfft.hybrid_kron_adam_from_plan(
        new_plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
        preconditioner_update_probability=1.0,
    )
    old_state = old_bundle.init(old_plan.trainable)
    _, old_state = old_bundle.update(
        ones_like_trainable(old_plan.trainable),
        old_state,
        old_plan.trainable,
    )

    _, migrated_state, transfer = rfft.transfer_optimizer_state(
        old_plan=old_plan,
        old_bundle=old_bundle,
        old_state=old_state,
        new_plan=new_plan,
        new_bundle=new_bundle,
        state_policy="preserve_by_path_and_shape",
        counter_policy="restart_schedule",
    )

    assert not transfer.exact
    assert "logical/blocks.0.w" in transfer.converted_ids
    assert "logical/blocks.0.w" in transfer.preserved_ids
    assert any("deliberately transferred" in warning for warning in transfer.warnings)
    assert jnp.allclose(
        _state_leaf(
            migrated_state,
            "attr:Qs_preconditioners",
            "key:blocks",
            "idx:0",
            "key:w",
            "idx:0",
        ),
        _state_leaf(
            old_state,
            "attr:Qs_preconditioners",
            "key:blocks",
            "idx:0",
            "key:w",
            "idx:0",
        ),
    )


def test_reconfigure_reset_all_initializes_shared_state():
    old_plan = _head_only_plan()
    new_plan = tiny_plan()
    old_bundle = rfft.adamw_from_plan(
        old_plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    new_bundle = rfft.adamw_from_plan(
        new_plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
    )
    old_state = old_bundle.init(old_plan.trainable)
    _, old_state = old_bundle.update(
        ones_like_trainable(old_plan.trainable),
        old_state,
        old_plan.trainable,
    )

    _, migrated_state, migration = rfft.reconfigure_optimizer(
        old_plan=old_plan,
        old_bundle=old_bundle,
        old_state=old_state,
        new_plan=new_plan,
        new_bundle=new_bundle,
        state_policy="reset_all",
    )

    assert jnp.allclose(
        _state_leaf(migrated_state, "attr:mu", "key:head", "key:w"),
        0.0,
    )
    assert migration.preserved_state_leaves == ()


def test_reconfigure_counter_policy_is_explicit():
    old_plan = _head_only_plan()
    new_plan = tiny_plan()
    old_bundle = rfft.adamw_from_plan(
        old_plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    new_bundle = rfft.adamw_from_plan(
        new_plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
    )
    old_state = old_bundle.init(old_plan.trainable)
    _, old_state = old_bundle.update(
        ones_like_trainable(old_plan.trainable),
        old_state,
        old_plan.trainable,
    )

    _, restart_state, _ = rfft.reconfigure_optimizer(
        old_plan=old_plan,
        old_bundle=old_bundle,
        old_state=old_state,
        new_plan=new_plan,
        new_bundle=new_bundle,
        counter_policy="restart_schedule",
    )
    _, continued_state, _ = rfft.reconfigure_optimizer(
        old_plan=old_plan,
        old_bundle=old_bundle,
        old_state=old_state,
        new_plan=new_plan,
        new_bundle=new_bundle,
        counter_policy="continue_global_step",
    )

    assert all(
        jnp.all(count == 0) for count in _count_leaves(restart_state, "head_decay")
    )
    assert any(
        jnp.all(count > 0) for count in _count_leaves(continued_state, "head_decay")
    )


@pytest.mark.parametrize(
    ("counter_policy", "preserved_owners"),
    (
        ("restart_schedule", frozenset()),
        (
            "continue_global_step",
            frozenset(
                {
                    "optimizer_algorithm_schedule",
                    "finite_guard",
                    "accumulation",
                    "averaging",
                }
            ),
        ),
        (
            "continue_optimizer_step_with_new_schedule",
            frozenset({"optimizer_algorithm_schedule"}),
        ),
    ),
)
def test_reconfigure_applies_clock_policy_by_owner(counter_policy, preserved_owners):
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=1.0,
        accumulation_steps=2,
        ema=rfft.EMAConfig(enabled=True, decay=0.5),
        swa=rfft.SWAConfig(enabled=True, start_step=0),
    )
    old_state = _advance_clock_fixture(bundle, plan.trainable)
    initial_state = bundle.init(plan.trainable)

    _, migrated_state, report = rfft.reconfigure_optimizer(
        old_plan=plan,
        old_bundle=bundle,
        old_state=old_state,
        new_plan=plan,
        new_bundle=bundle,
        counter_policy=counter_policy,
    )

    old_counters = _counter_leaves(old_state)
    initial_counters = _counter_leaves(initial_state)
    migrated_counters = _counter_leaves(migrated_state)
    assert old_counters.keys() == initial_counters.keys() == migrated_counters.keys()
    assert set(owner for owner, _ in old_counters.values()) == {
        "optimizer_algorithm_schedule",
        "finite_guard",
        "accumulation",
        "averaging",
    }
    for path, (owner, migrated) in migrated_counters.items():
        expected = (
            old_counters[path][1]
            if owner in preserved_owners
            else initial_counters[path][1]
        )
        assert jnp.array_equal(migrated, expected), path
    assert report.clock_behavior == {
        owner: "preserved" if owner in preserved_owners else "reset"
        for owner in (
            "optimizer_algorithm_schedule",
            "finite_guard",
            "accumulation",
            "averaging",
        )
    }
    serialized = report.to_dict()
    assert serialized["counter_policy"] == counter_policy
    assert serialized["clock_behavior"] == report.clock_behavior


@pytest.mark.parametrize(
    ("counter_policy", "preserved"),
    (
        ("restart_schedule", False),
        ("continue_global_step", True),
        ("continue_optimizer_step_with_new_schedule", True),
    ),
)
def test_reconfigure_schedule_free_step_count_is_optimizer_clock(
    counter_policy, preserved
):
    plan = tiny_plan()
    bundle = rfft.schedule_free_adam_from_plan(
        plan,
        total_steps=10,
        schedule="wsd",
        clip_global_norm=None,
    )
    old_state = bundle.init(plan.trainable)
    updates, old_state = bundle.update(
        ones_like_trainable(plan.trainable), old_state, plan.trainable
    )
    del updates

    _, migrated_state, report = rfft.reconfigure_optimizer(
        old_plan=plan,
        old_bundle=bundle,
        old_state=old_state,
        new_plan=plan,
        new_bundle=bundle,
        counter_policy=counter_policy,
    )

    old_step = _state_leaf(old_state, "attr:step_count")
    migrated_step = _state_leaf(migrated_state, "attr:step_count")
    assert int(old_step) == 1
    assert int(migrated_step) == (1 if preserved else 0)
    if counter_policy == "continue_optimizer_step_with_new_schedule":
        assert "new schedule is evaluated at the continued optimizer step" in (
            report.schedule_counter_behavior
        )


def test_strict_migration_rejects_unknown_counter_name():
    state = _MysteryCounterState(jnp.asarray(1, dtype=jnp.int32))

    with pytest.raises(ValueError, match=r"unclassified.*attr:mystery_count"):
        migration_module._migrate_state_tree(
            state,
            state,
            state_policy="preserve_shared",
            counter_policy="continue_global_step",
            strict=True,
        )


def test_counter_classification_covers_supported_legacy_names():
    assert migration_module._counter_owner(("attr:notfinite_count",)) == (
        "finite_guard"
    )
    assert migration_module._counter_owner(("attr:total_notfinite",)) == (
        "finite_guard"
    )


def test_reconfigure_preserves_kron_preconditioners_across_group_relabel():
    old_plan = tiny_plan()
    new_plan = _renamed_groups_plan()
    old_bundle = rfft.hybrid_kron_adam_from_plan(
        old_plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        preconditioner_update_probability=1.0,
    )
    new_bundle = rfft.hybrid_kron_adam_from_plan(
        new_plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
        preconditioner_update_probability=1.0,
    )
    old_state = old_bundle.init(old_plan.trainable)
    _, old_state = old_bundle.update(
        ones_like_trainable(old_plan.trainable),
        old_state,
        old_plan.trainable,
    )

    _, migrated_state, migration = rfft.reconfigure_optimizer(
        old_plan=old_plan,
        old_bundle=old_bundle,
        old_state=old_state,
        new_plan=new_plan,
        new_bundle=new_bundle,
        state_policy="preserve_by_path_and_shape",
        counter_policy="restart_schedule",
    )

    assert jnp.allclose(
        _state_leaf(
            migrated_state,
            "attr:Qs_preconditioners",
            "key:blocks",
            "idx:0",
            "key:w",
            "idx:0",
        ),
        _state_leaf(
            old_state,
            "attr:Qs_preconditioners",
            "key:blocks",
            "idx:0",
            "key:w",
            "idx:0",
        ),
    )
    assert any(
        "attr:Qs_preconditioners" in path for path in migration.preserved_state_leaves
    )
    assert "logical/blocks.0.w" in migration.changed_group_leaves


def test_reconfigure_rejects_incompatible_shared_parameter_shape():
    old_plan = TinyPlan(
        trainable={"head": {"w": jnp.ones((3, 1), dtype=jnp.float32)}},
        labels={"head": {"w": "head_decay"}},
        group_specs={
            "head_decay": TinyGroup(
                "head_decay",
                role="head",
                depth=None,
                lr_multiplier=1.0,
                weight_decay=True,
            )
        },
    )
    new_plan = TinyPlan(
        trainable={"head": {"w": jnp.ones((2, 1), dtype=jnp.float32)}},
        labels={"head": {"w": "head_decay"}},
        group_specs=old_plan.group_specs,
    )
    old_bundle = rfft.adamw_from_plan(
        old_plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    new_bundle = rfft.adamw_from_plan(
        new_plan,
        total_steps=20,
        schedule="constant",
        clip_global_norm=None,
    )

    with pytest.raises(ValueError, match="incompatible shared parameter"):
        rfft.reconfigure_optimizer(
            old_plan=old_plan,
            old_bundle=old_bundle,
            old_state=old_bundle.init(old_plan.trainable),
            new_plan=new_plan,
            new_bundle=new_bundle,
        )
