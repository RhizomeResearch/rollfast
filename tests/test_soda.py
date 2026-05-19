import jax
import jax.numpy as jnp
import optax
import pytest
from optax._src import base

import rollfast
import rollfast.optim.soda as soda_module
import rollfast.schedules as schedules
from rollfast.optim.soda import (
    soda,
    soda_adam,
    soda_kron,
    soda_muon,
    soda_prism,
    soda_rmnp,
)
from tests._typing import as_array_dict


def test_public_soda_wrappers_are_exported_from_optim_soda():
    assert rollfast.soda is soda_module.soda
    assert rollfast.soda_adam is soda_module.soda_adam
    assert rollfast.soda_kron is soda_module.soda_kron
    assert rollfast.soda_prism is soda_module.soda_prism
    assert rollfast.soda_muon is soda_module.soda_muon
    assert rollfast.soda_rmnp is soda_module.soda_rmnp
    assert "soda_adam" in rollfast.__all__
    assert "soda_kron" in rollfast.__all__
    assert "soda_adam" in soda_module.__all__
    assert "soda_kron" in soda_module.__all__
    assert not hasattr(schedules, "soda")
    assert not hasattr(schedules, "soda_adam")
    assert not hasattr(schedules, "soda_kron")
    assert not hasattr(schedules, "soda_prism")
    assert not hasattr(schedules, "soda_muon")
    assert not hasattr(schedules, "soda_rmnp")


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


def test_soda_plain_base_optimizer_ignores_extra_args():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = soda(optax.sgd(0.01))
    updates, _ = tx.update(grads, tx.init(params), params, batch_stats={})
    updates = as_array_dict(updates)

    assert updates["w"].shape == params["w"].shape


def test_soda_does_not_swallow_base_optimizer_type_errors():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}

    def init_fn(params):
        del params
        return ()

    def update_fn(updates, state, params=None, **extra_args):
        del updates, state, params, extra_args
        raise TypeError("inner optimizer bug")

    tx = soda(base.GradientTransformationExtraArgs(init_fn, update_fn))

    with pytest.raises(TypeError, match="inner optimizer bug"):
        tx.update(grads, tx.init(params), params, batch_stats={})


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


def test_soda_prism_accepts_preconditioning_selector():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    tx = soda_prism(
        learning_rate=0.01,
        total_steps=100,
        ns_iters=2,
        preconditioning="spectral",
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_soda_prism_accepts_heavy_ball_momentum_accumulator():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    tx = soda_prism(
        learning_rate=0.01,
        total_steps=100,
        ns_iters=2,
        momentum_accumulator="heavy_ball",
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


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


def test_soda_muon_accepts_heavy_ball_momentum_accumulator():
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = soda_muon(
        learning_rate=0.01,
        total_steps=100,
        ns_steps=2,
        momentum_accumulator="heavy_ball",
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == (4, 4)
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_soda_muon_forwards_key(monkeypatch):
    captured = {}
    key = jax.random.PRNGKey(123)

    def fake_muon(**kwargs):
        captured.update(kwargs)
        return optax.identity()

    monkeypatch.setattr(soda_module, "muon", fake_muon)
    tx = soda_muon(learning_rate=0.01, total_steps=100, key=key)
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    tx.init(params)

    assert jnp.array_equal(captured["key"], key)


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


def test_soda_rmnp_accepts_heavy_ball_momentum_accumulator():
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = soda_rmnp(
        learning_rate=0.01,
        total_steps=100,
        momentum_accumulator="heavy_ball",
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == (4, 4)
    assert jnp.all(jnp.isfinite(updates["w"]))


@pytest.mark.parametrize(
    "make_tx",
    [
        lambda: soda_adam(learning_rate=0.01, total_steps=20, mu_dtype=jnp.bfloat16),
        lambda: soda_prism(
            learning_rate=0.01,
            total_steps=20,
            ns_iters=2,
            mu_dtype=jnp.bfloat16,
        ),
        lambda: soda_kron(
            learning_rate=0.01,
            total_steps=20,
            preconditioner_update_probability=1.0,
            mu_dtype=jnp.bfloat16,
        ),
        lambda: soda_muon(
            learning_rate=0.01,
            total_steps=20,
            ns_steps=2,
            mu_dtype=jnp.bfloat16,
        ),
        lambda: soda_rmnp(
            learning_rate=0.01,
            total_steps=20,
            mu_dtype=jnp.bfloat16,
        ),
    ],
)
def test_soda_wrappers_accept_bf16_mu_dtype(make_tx):
    params = {"w": jnp.ones((4, 4), dtype=jnp.bfloat16)}
    grads = {"w": jnp.ones_like(params["w"]) * jnp.asarray(0.1, jnp.bfloat16)}
    tx = make_tx()
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)
    next_params = rollfast.apply_updates(
        params,
        updates,
        key=jax.random.PRNGKey(0),
    )
    next_params = as_array_dict(next_params)

    assert updates["w"].shape == params["w"].shape
    assert next_params["w"].dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(updates["w"]))
