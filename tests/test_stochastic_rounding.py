import jax
import jax.numpy as jnp
import pytest

from rollfast.optim.adam import adamw
from rollfast.optim.prism import prism
from rollfast.optim.psgd import kron
from rollfast.schedules.schedulefree import (
    schedule_free_adam,
    schedule_free_kron,
    schedule_free_prism,
)
from rollfast.utils import apply_updates, apply_updates_prefix

# Base optimizers and their kwargs
base_optimizers = [
    (adamw, {"learning_rate": 1e-3}),
    (prism, {"learning_rate": 1e-3, "ns_iters": 2}),
    (kron, {"learning_rate": 1e-3, "preconditioner_update_probability": 1.0}),
]

# Schedule-free optimizers and their kwargs
schedule_free_optimizers = [
    (schedule_free_adam, {"learning_rate": 1e-3, "total_steps": 10}),
    (schedule_free_prism, {"learning_rate": 1e-3, "total_steps": 10, "ns_iters": 2}),
    (
        schedule_free_kron,
        {
            "learning_rate": 1e-3,
            "total_steps": 10,
            "preconditioner_update_probability": 1.0,
        },
    ),
]

# All optimizers combined
all_optimizers = base_optimizers + schedule_free_optimizers

# Optimizers that support mu_dtype directly
mu_dtype_optimizers = [
    (adamw, {"learning_rate": 1e-3}),
    (prism, {"learning_rate": 1e-3, "ns_iters": 2}),
    (kron, {"learning_rate": 1e-3, "preconditioner_update_probability": 1.0}),
    (schedule_free_adam, {"learning_rate": 1e-3, "total_steps": 10}),
    (schedule_free_prism, {"learning_rate": 1e-3, "total_steps": 10, "ns_iters": 2}),
]


@pytest.mark.parametrize("optimizer_fn, kwargs", mu_dtype_optimizers)
def test_mu_dtype_bf16(optimizer_fn, kwargs):
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}

    # Enable mu_dtype = bf16
    kwargs_copy = kwargs.copy()
    kwargs_copy["mu_dtype"] = jnp.bfloat16
    tx = optimizer_fn(**kwargs_copy)

    state = tx.init(params)
    updates, state = tx.update(grads, state, params)

    assert "w" in updates
    assert updates["w"].shape == (4, 4)


@pytest.mark.parametrize("optimizer_fn, kwargs", all_optimizers)
def test_pure_bf16_apply_updates(optimizer_fn, kwargs):
    # Model in pure BF16
    params = {"w": jnp.ones((4, 4), dtype=jnp.bfloat16)}
    # Gradients in BF16
    grads = {"w": jnp.ones((4, 4), dtype=jnp.bfloat16) * 0.1}

    tx = optimizer_fn(**kwargs)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)

    key = jax.random.PRNGKey(0)

    # Stochastic rounding
    new_params = apply_updates(params, updates, key, stochastic=True)

    assert "w" in new_params
    assert new_params["w"].dtype == jnp.bfloat16


@pytest.mark.parametrize("optimizer_fn, kwargs", all_optimizers)
def test_pure_bf16_apply_updates_prefix(optimizer_fn, kwargs):
    # Model in pure BF16
    params = {"w": jnp.ones((4, 4), dtype=jnp.bfloat16)}
    # Gradients in BF16
    grads = {"w": jnp.ones((4, 4), dtype=jnp.bfloat16) * 0.1}

    tx = optimizer_fn(**kwargs)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)

    key = jax.random.PRNGKey(0)

    # Stochastic rounding with prefix
    new_params = apply_updates_prefix(params, updates, key, stochastic=True)

    assert "w" in new_params
    assert new_params["w"].dtype == jnp.bfloat16
