from types import SimpleNamespace

import equimo.finetune as eqft
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from rollfast.integrations import equimo
from tests.finetune.helpers import tiny_plan


@pytest.mark.integration
def test_real_equimo_plan_compiles_and_updates_finitely():
    key = jax.random.PRNGKey(0)
    model = eqx.nn.MLP(4, 2, 8, 2, key=key)
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(mode="full"),
    )
    optimizer = equimo.adamw_from_equimo_plan(
        plan,
        total_steps=10,
        base_lr=1e-3,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
    )
    opt_state = optimizer.init(plan.trainable)
    x = jnp.ones((4,), dtype=jnp.float32)

    def loss_fn(trainable):
        prediction = plan.combine(trainable)(x)
        return jnp.mean(jnp.square(prediction))

    loss, grads = jax.value_and_grad(loss_fn)(plan.trainable)
    updates, _ = optimizer.update(grads, opt_state, plan.trainable)
    updated = optax.apply_updates(plan.trainable, updates)

    assert jnp.isfinite(loss)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(updated))


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
