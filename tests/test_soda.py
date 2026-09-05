import jax
import jax.numpy as jnp
import optax
import pytest
from optax._src import base

import rollfast
import rollfast.optim.soda as soda_module
from rollfast import schedules
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


@pytest.mark.parametrize("jit_update", [False, True])
def test_soda_preserves_implicit_anchor_dtypes_and_zero_updates(jit_update):
    params = {
        "bf16": jnp.asarray([1.0], dtype=jnp.bfloat16),
        "fp32": jnp.asarray([1.001], dtype=jnp.float32),
        "none": None,
    }
    grads = jax.tree.map(
        lambda x: jnp.zeros_like(x) if x is not None else None,
        params,
        is_leaf=lambda x: x is None,
    )
    tx = soda(optax.set_to_zero())
    state = tx.init(params)

    assert state.z0["bf16"].dtype == params["bf16"].dtype
    assert state.z0["fp32"].dtype == params["fp32"].dtype
    assert state.z0["none"] is None

    update_fn = jax.jit(tx.update) if jit_update else tx.update
    updates, _ = update_fn(grads, state, params)

    assert updates["bf16"].dtype == params["bf16"].dtype
    assert updates["fp32"].dtype == params["fp32"].dtype
    assert jnp.array_equal(updates["bf16"], jnp.zeros_like(params["bf16"]))
    assert jnp.array_equal(updates["fp32"], jnp.zeros_like(params["fp32"]))
    assert updates["none"] is None


def test_soda_explicit_state_dtype_applies_to_every_anchor_leaf():
    params = {
        "bf16": jnp.asarray([1.0], dtype=jnp.bfloat16),
        "fp32": jnp.asarray([1.001], dtype=jnp.float32),
        "none": None,
    }
    state = soda(optax.set_to_zero(), state_dtype=jnp.bfloat16).init(params)

    assert state.z0["bf16"].dtype == jnp.bfloat16
    assert state.z0["fp32"].dtype == jnp.bfloat16
    assert state.z0["none"] is None


def test_soda_mixed_dtypes_remain_finite_across_two_updates():
    params = {
        "bf16": jnp.asarray([1.0], dtype=jnp.bfloat16),
        "fp32": jnp.asarray([1.0], dtype=jnp.float32),
    }
    grads = {
        "bf16": jnp.asarray([0.125], dtype=jnp.bfloat16),
        "fp32": jnp.asarray([0.125], dtype=jnp.float32),
    }
    tx = soda(optax.scale(-0.125))
    state = tx.init(params)

    for _ in range(2):
        updates, state = tx.update(grads, state, params)
        for name, param in params.items():
            assert updates[name].dtype == param.dtype
            assert jnp.all(jnp.isfinite(updates[name]))
        params = optax.apply_updates(params, updates)


def test_soda_implicit_fp32_state_matches_explicit_fp32_reference():
    params = {"w": jnp.asarray([1.001], dtype=jnp.float32)}
    grads = {"w": jnp.asarray([0.125], dtype=jnp.float32)}
    implicit_tx = soda(optax.scale(-0.125))
    explicit_tx = soda(optax.scale(-0.125), state_dtype=jnp.float32)
    implicit_state = implicit_tx.init(params)
    explicit_state = explicit_tx.init(params)
    implicit_params = params
    explicit_params = params

    for _ in range(2):
        implicit_updates, implicit_state = implicit_tx.update(
            grads, implicit_state, implicit_params
        )
        explicit_updates, explicit_state = explicit_tx.update(
            grads, explicit_state, explicit_params
        )
        assert jnp.array_equal(implicit_updates["w"], explicit_updates["w"])
        implicit_params = optax.apply_updates(implicit_params, implicit_updates)
        explicit_params = optax.apply_updates(explicit_params, explicit_updates)


def test_soda_requires_params():
    tx = soda(optax.sgd(0.01))
    state = tx.init({"w": jnp.ones((2, 2))})
    with pytest.raises(ValueError, match="`params` must be provided"):
        tx.update({"w": jnp.ones((2, 2))}, state)


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


@pytest.mark.parametrize(
    "optimizer_fn, kwargs",
    [
        pytest.param(soda_muon, {"ns_steps": 2}, id="muon"),
        pytest.param(soda_rmnp, {}, id="rmnp"),
    ],
)
def test_soda_matrix_wrapper(optimizer_fn, kwargs):
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = optimizer_fn(**kwargs, learning_rate=0.01, total_steps=100)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.all(jnp.isfinite(updates["b"]))


@pytest.mark.parametrize(
    "optimizer_fn, kwargs",
    [
        pytest.param(soda_muon, {"ns_steps": 2}, id="muon"),
        pytest.param(soda_rmnp, {}, id="rmnp"),
    ],
)
def test_soda_matrix_wrapper_accepts_heavy_ball_momentum_accumulator(
    optimizer_fn, kwargs
):
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = optimizer_fn(
        **kwargs,
        learning_rate=0.01,
        total_steps=100,
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
