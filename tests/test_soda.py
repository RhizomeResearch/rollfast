import jax
import jax.numpy as jnp
import optax

from rollfast.schedules.soda import (
    soda,
    soda_adam,
    soda_kron,
    soda_muon,
    soda_prism,
    soda_rmnp,
)
from tests._typing import as_array_dict


def test_soda_adds_initialization_anchor():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones((2, 2), dtype=jnp.float32) * 0.1}
    base = optax.sgd(0.01)
    tx = soda(base)
    state = tx.init(params)

    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert jnp.allclose(updates["w"], -0.001)

    params = optax.apply_updates(params, updates)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    expected_anchor = (1.0 - 0.999) / 3.0
    assert jnp.allclose(updates["w"], -0.001 + expected_anchor)


def test_soda_requires_params():
    tx = soda(optax.sgd(0.01))
    state = tx.init({"w": jnp.ones((2, 2))})
    try:
        tx.update({"w": jnp.ones((2, 2))}, state)
    except ValueError as err:
        assert "`params` must be provided" in str(err)
    else:
        raise AssertionError("soda update should require params")


def test_soda_adam():
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.ones((2, 2)) * 0.1}
    tx = soda_adam(learning_rate=0.01, total_steps=100)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert updates["w"].shape == (2, 2)
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_soda_prism():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    tx = soda_prism(learning_rate=0.01, total_steps=100)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert updates["w"].shape == (4, 4)


def test_soda_kron():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    tx = soda_kron(
        learning_rate=0.01,
        total_steps=100,
        preconditioner_update_probability=1.0,
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert updates["w"].shape == (4, 4)


def test_soda_muon():
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = soda_muon(learning_rate=0.01, total_steps=100, ns_steps=2)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.all(jnp.isfinite(updates["b"]))


def test_soda_rmnp():
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = soda_rmnp(learning_rate=0.01, total_steps=100)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.all(jnp.isfinite(updates["b"]))
