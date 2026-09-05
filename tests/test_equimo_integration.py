import equimo.vision.models as em
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from rollfast.optim.adam import adamw
from rollfast.optim.aurora import aurora, get_equinox_aurora_spec
from rollfast.optim.hyperball import (
    adamw_hyperball,
    aurora_hyperball,
    kron_hyperball,
    prism_hyperball,
    riemannian_aurora_hyperball,
)
from rollfast.optim.prism import get_equinox_prism_spec, prism
from rollfast.optim.psgd import kron
from rollfast.schedules.schedulefree import (
    schedule_free_adam,
    schedule_free_aurora,
    schedule_free_kron,
    schedule_free_prism,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


@pytest.fixture
def vit_and_data():
    key = jr.PRNGKey(42)
    x = jr.normal(key, (3, 64, 64))
    y = jnp.ones((10,))
    model = em.VisionTransformer(  # ty: ignore[unresolved-attribute]
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=16,
        num_heads=[2],
        depths=[2],
        num_classes=10,
        key=key,
    )
    return model, x, y


def run_opt_step(model, tx, x, y, key):
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = tx.init(params)

    @eqx.filter_value_and_grad
    def loss_fn(m, x_in, y_in, k):
        preds = m(x_in, key=k)
        if isinstance(preds, tuple):
            preds = preds[0]
        return jnp.mean((preds - y_in) ** 2)

    loss, grads = loss_fn(model, x, y, key)
    updates, opt_state = tx.update(grads, opt_state, params)
    del opt_state
    updated_params = optax.apply_updates(params, updates)
    model = eqx.combine(updated_params, static)

    update_leaves = jax.tree.leaves(updates, is_leaf=lambda x: x is None)
    finite_updates = [
        jnp.all(jnp.isfinite(update)) for update in update_leaves if update is not None
    ]
    assert jnp.all(jnp.asarray(finite_updates))
    assert jnp.all(jnp.isfinite(loss))
    return model, loss


def get_equinox_weight_decay_mask(model):
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    return jax.tree.map(
        lambda p: False if p is None else getattr(p, "ndim", 0) >= 2,
        params,
        is_leaf=lambda p: p is None,
    )


@pytest.mark.parametrize(
    "optimizer_fn, kwargs",
    [
        pytest.param(adamw, {"use_magma": True}, id="adamw"),
        pytest.param(
            prism,
            {"prism_weight_dimension_numbers": get_equinox_prism_spec},
            id="prism",
        ),
        pytest.param(
            aurora,
            {
                "aurora_weight_dimension_numbers": get_equinox_aurora_spec,
                "polar_ns_iters": 2,
            },
            id="aurora",
        ),
        pytest.param(kron, {"preconditioner_update_probability": 1.0}, id="kron"),
        pytest.param(adamw_hyperball, {"use_magma": True}, id="adamw_hyperball"),
        pytest.param(
            prism_hyperball,
            {"prism_weight_dimension_numbers": get_equinox_prism_spec},
            id="prism_hyperball",
        ),
        pytest.param(
            aurora_hyperball,
            {
                "aurora_weight_dimension_numbers": get_equinox_aurora_spec,
                "polar_ns_iters": 2,
            },
            id="aurora_hyperball",
        ),
        pytest.param(
            riemannian_aurora_hyperball,
            {
                "aurora_weight_dimension_numbers": get_equinox_aurora_spec,
                "outer_steps": 1,
                "cg_steps": 2,
                "retraction_steps": 1,
                "polar_ns_iters": 2,
            },
            id="riemannian_aurora_hyperball",
        ),
        pytest.param(
            kron_hyperball,
            {"preconditioner_update_probability": 1.0},
            id="kron_hyperball",
        ),
        pytest.param(
            schedule_free_prism,
            {
                "total_steps": 10,
                "prism_weight_dimension_numbers": get_equinox_prism_spec,
            },
            id="schedule_free_prism",
        ),
        pytest.param(
            schedule_free_aurora,
            {
                "total_steps": 10,
                "aurora_weight_dimension_numbers": get_equinox_aurora_spec,
                "polar_ns_iters": 2,
            },
            id="schedule_free_aurora",
        ),
        pytest.param(
            schedule_free_kron,
            {"total_steps": 10, "preconditioner_update_probability": 1.0},
            id="schedule_free_kron",
        ),
        pytest.param(schedule_free_adam, {"total_steps": 10}, id="schedule_free_adam"),
    ],
)
def test_optimizer_step_on_vit(vit_and_data, optimizer_fn, kwargs):
    model, x, y = vit_and_data
    tx = optimizer_fn(learning_rate=1e-3, **kwargs)
    run_opt_step(model, tx, x, y, jr.PRNGKey(1))


def test_aurora_hyperball_with_equinox_weight_decay_mask(vit_and_data):
    model, x, y = vit_and_data
    weight_decay_mask = get_equinox_weight_decay_mask(model)
    tx = aurora_hyperball(
        learning_rate=lambda step: jnp.asarray(1e-3),
        weight_decay=0.01,
        weight_decay_mask=weight_decay_mask,
        adam_learning_rate=lambda step: jnp.asarray(1e-3),
        aurora_weight_dimension_numbers=get_equinox_prism_spec,
        polar_ns_iters=2,
    )

    assert callable(weight_decay_mask)
    run_opt_step(model, tx, x, y, jr.PRNGKey(1))
