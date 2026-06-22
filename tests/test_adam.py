import jax
import jax.numpy as jnp
import optax

from rollfast.optim.adam import adamw, scale_by_adam
from tests._typing import as_array_dict


def _assert_tree_allclose(left, right, *, atol=1e-6):
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        assert jnp.allclose(left_leaf, right_leaf, atol=atol)


def test_scale_by_adam():
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.ones((2, 2)) * 0.1}
    tx = scale_by_adam()
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert updates["w"].shape == (2, 2)


def test_scale_by_adam_matches_optax_direction():
    params = {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}
    grads = {"w": jnp.array([0.1, -0.2], dtype=jnp.float32)}
    rollfast_tx = scale_by_adam(b1=0.9, b2=0.999, eps=1e-8)
    optax_tx = optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8)
    rollfast_state = rollfast_tx.init(params)
    optax_state = optax_tx.init(params)

    for _ in range(3):
        rollfast_updates, rollfast_state = rollfast_tx.update(
            grads,
            rollfast_state,
            params,
        )
        optax_updates, optax_state = optax_tx.update(grads, optax_state, params)
        _assert_tree_allclose(rollfast_updates, optax_updates)


def test_adamw():
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.ones((2, 2)) * 0.1}
    tx = adamw(learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert updates["w"].shape == (2, 2)


def test_adamw_matches_optax_masked_decoupled_weight_decay():
    params = {
        "w": jnp.array([1.0, -2.0], dtype=jnp.float32),
        "b": jnp.array([0.5], dtype=jnp.float32),
    }
    grads = {
        "w": jnp.array([0.1, -0.2], dtype=jnp.float32),
        "b": jnp.array([0.0], dtype=jnp.float32),
    }
    mask = {"w": True, "b": False}
    kwargs = {
        "learning_rate": 0.01,
        "b1": 0.9,
        "b2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    rollfast_tx = adamw(**kwargs, weight_decay_mask=mask)
    optax_tx = optax.adamw(**kwargs, mask=mask)
    rollfast_state = rollfast_tx.init(params)
    optax_state = optax_tx.init(params)

    for _ in range(3):
        rollfast_updates, rollfast_state = rollfast_tx.update(
            grads,
            rollfast_state,
            params,
        )
        optax_updates, optax_state = optax_tx.update(grads, optax_state, params)
        _assert_tree_allclose(rollfast_updates, optax_updates)
        params = optax.apply_updates(params, rollfast_updates)
