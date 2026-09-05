import jax
import jax.numpy as jnp
import pytest

from rollfast.optim.adam import adamw
from rollfast.optim.aurora import aurora
from rollfast.optim.prism import prism
from rollfast.optim.psgd import kron
from rollfast.schedules.schedulefree import (
    schedule_free_adam,
    schedule_free_aurora,
    schedule_free_kron,
    schedule_free_prism,
)
from rollfast.utils import apply_updates, apply_updates_prefix
from tests._typing import as_array_dict

all_optimizers = [
    (adamw, {"learning_rate": 0.001}),
    (aurora, {"learning_rate": 0.001, "polar_ns_iters": 2}),
    (prism, {"learning_rate": 0.001, "ns_iters": 2}),
    (kron, {"learning_rate": 0.001, "preconditioner_update_probability": 1.0}),
    (schedule_free_adam, {"learning_rate": 0.001, "total_steps": 10}),
    (
        schedule_free_aurora,
        {"learning_rate": 0.001, "total_steps": 10, "polar_ns_iters": 2},
    ),
    (schedule_free_prism, {"learning_rate": 0.001, "total_steps": 10, "ns_iters": 2}),
    (
        schedule_free_kron,
        {
            "learning_rate": 0.001,
            "total_steps": 10,
            "preconditioner_update_probability": 1.0,
        },
    ),
]

# Schedule-Free Kron deliberately has no public mu_dtype argument.
mu_dtype_optimizers = [
    (fn, kwargs) for fn, kwargs in all_optimizers if fn is not schedule_free_kron
]


@pytest.mark.parametrize(
    "optimizer_fn, kwargs",
    mu_dtype_optimizers,
    ids=[fn.__name__ for fn, _ in mu_dtype_optimizers],
)
def test_mu_dtype_bf16(optimizer_fn, kwargs):
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}

    # Enable mu_dtype = bf16
    kwargs_copy = kwargs.copy()
    kwargs_copy["mu_dtype"] = jnp.bfloat16
    tx = optimizer_fn(**kwargs_copy)

    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)

    assert "w" in updates
    assert updates["w"].shape == (4, 4)


@pytest.mark.parametrize(
    "apply_fn", [apply_updates, apply_updates_prefix], ids=["full_tree", "prefix"]
)
@pytest.mark.parametrize(
    "optimizer_fn, kwargs",
    all_optimizers,
    ids=[fn.__name__ for fn, _ in all_optimizers],
)
def test_pure_bf16_apply_updates(optimizer_fn, kwargs, apply_fn):
    # Model in pure BF16
    params = {"w": jnp.ones((4, 4), dtype=jnp.bfloat16)}
    # Gradients in BF16
    grads = {"w": jnp.ones((4, 4), dtype=jnp.bfloat16) * 0.1}

    tx = optimizer_fn(**kwargs)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)

    key = jax.random.PRNGKey(0)

    # Stochastic rounding
    new_params = apply_fn(params, updates, key, stochastic=True)
    new_params = as_array_dict(new_params)

    assert "w" in new_params
    assert new_params["w"].dtype == jnp.bfloat16
