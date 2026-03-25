import jax.numpy as jnp
import jax.random as jr
import pytest

try:
    import equimo.models as em
    import equinox as eqx

    EQUINOX_EQUIMO_AVAILABLE = True
except ImportError:
    EQUINOX_EQUIMO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not EQUINOX_EQUIMO_AVAILABLE, reason="equinox or equimo not available"
)

if EQUINOX_EQUIMO_AVAILABLE:
    from rollfast.optim.adam import adamw
    from rollfast.optim.prism import get_equinox_prism_spec, prism
    from rollfast.optim.psgd import kron
    from rollfast.schedules.schedulefree import (
        schedule_free_adam,
        schedule_free_kron,
        schedule_free_prism,
    )


@pytest.fixture(params=["iformer", "vit", "reduceformer"])
def model_and_data(request):
    key = jr.PRNGKey(42)
    model_name = request.param

    if model_name == "iformer":
        x = jr.normal(key, (3, 64, 64))
        y = jnp.ones((10,))
        model = em.iformer_t(in_channels=3, num_classes=10, key=key)
    elif model_name == "vit":
        x = jr.normal(key, (3, 64, 64))
        y = jnp.ones((10,))
        model = em.VisionTransformer(
            img_size=64,
            in_channels=3,
            dim=32,
            patch_size=16,
            num_heads=[2],
            depths=[2],
            num_classes=10,
            key=key,
        )
    elif model_name == "reduceformer":
        x = jr.normal(key, (3, 64, 64))
        y = jnp.ones((10,))
        model = em.reduceformer_backbone_b1(in_channels=3, num_classes=10, key=key)
    return model, x, y


def run_opt_step(model, tx, x, y, key):
    import equinox as eqx

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
    model = eqx.combine(updates, static)
    return model, loss


def test_adamw(model_and_data):
    model, x, y = model_and_data
    tx = adamw(learning_rate=1e-3, use_magma=True)
    run_opt_step(model, tx, x, y, jr.PRNGKey(1))


def test_prism(model_and_data):
    model, x, y = model_and_data
    tx = prism(
        learning_rate=1e-3,
        prism_weight_dimension_numbers=get_equinox_prism_spec,
    )
    run_opt_step(model, tx, x, y, jr.PRNGKey(1))


def test_kron(model_and_data):
    model, x, y = model_and_data
    tx = kron(learning_rate=1e-3, preconditioner_update_probability=1.0)
    run_opt_step(model, tx, x, y, jr.PRNGKey(1))


def test_schedule_free_prism(model_and_data):
    model, x, y = model_and_data
    tx = schedule_free_prism(
        learning_rate=1e-3,
        total_steps=10,
        prism_weight_dimension_numbers=get_equinox_prism_spec,
    )
    run_opt_step(model, tx, x, y, jr.PRNGKey(1))


def test_schedule_free_kron(model_and_data):
    model, x, y = model_and_data
    tx = schedule_free_kron(
        learning_rate=1e-3, total_steps=10, preconditioner_update_probability=1.0
    )
    run_opt_step(model, tx, x, y, jr.PRNGKey(1))


def test_schedule_free_adam(model_and_data):
    model, x, y = model_and_data
    tx = schedule_free_adam(learning_rate=1e-3, total_steps=10)
    run_opt_step(model, tx, x, y, jr.PRNGKey(1))
