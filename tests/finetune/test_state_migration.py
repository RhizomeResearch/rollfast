import jax
import jax.numpy as jnp
import pytest

import rollfast.finetune as rfft

from .helpers import TinyGroup, TinyPlan, tiny_plan


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


def _ones_like_trainable(tree):
    return jax.tree.map(
        lambda x: jnp.ones_like(x) if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
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
    for path, leaf in jax.tree_util.tree_flatten_with_path(
        state,
        is_leaf=lambda x: x is None,
    )[0]:
        text = _path_text(path)
        if group_label in text and text.endswith("attr:count") and hasattr(leaf, "shape"):
            leaves.append(leaf)
    return leaves


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
    grads = _ones_like_trainable(old_plan.trainable)
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
    assert any("key:blocks/idx:0/key:w" in path for path in migration.initialized_state_leaves)
    assert "key:head/key:w" in migration.preserved_param_leaves
    assert "key:blocks/idx:0/key:w" in migration.initialized_param_leaves
    assert migration.schedule_counter_behavior.startswith("initialized count")
    assert migration.new_state_bytes >= migration.old_state_bytes


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
        _ones_like_trainable(old_plan.trainable),
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
        _ones_like_trainable(old_plan.trainable),
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

    assert all(jnp.all(count == 0) for count in _count_leaves(restart_state, "head_decay"))
    assert any(jnp.all(count > 0) for count in _count_leaves(continued_state, "head_decay"))


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
        _ones_like_trainable(old_plan.trainable),
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
        "attr:Qs_preconditioners" in path
        for path in migration.preserved_state_leaves
    )
    assert "key:blocks/idx:0/key:w" in migration.changed_group_leaves


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
