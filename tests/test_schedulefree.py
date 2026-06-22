import jax.numpy as jnp
import optax
import pytest

from rollfast.schedules.schedulefree import (
    schedule_free,
    schedule_free_aurora,
    schedule_free_eval_params,
    schedule_free_prism,
    schedule_free_kron,
)
from tests._typing import as_array_dict


def test_schedule_free():
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.ones((2, 2)) * 0.1}
    base_opt = optax.sgd(0.01)
    tx = schedule_free(base_opt, learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates

    eval_params = schedule_free_eval_params(state, params)
    eval_params = as_array_dict(eval_params)
    assert "w" in eval_params


def test_schedule_free_prism():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    tx = schedule_free_prism(learning_rate=0.01, total_steps=100)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates


def test_schedule_free_aurora():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    tx = schedule_free_aurora(
        learning_rate=0.01,
        total_steps=100,
        polar_ns_iters=2,
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates


def test_schedule_free_kron():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    tx = schedule_free_kron(
        learning_rate=0.01, total_steps=100, preconditioner_update_probability=1.0
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    (
        (schedule_free_prism, {"ns_iters": 1}),
        (schedule_free_aurora, {"polar_ns_iters": 1}),
    ),
)
def test_schedule_free_matrix_wrappers_keep_tree_lr_for_adamc(factory, kwargs):
    params = {"matrix": jnp.ones((2, 2)), "vector": jnp.ones((2,))}
    grads = {"matrix": jnp.zeros((2, 2)), "vector": jnp.zeros((2,))}
    tx = factory(
        learning_rate=0.01,
        adam_learning_rate=0.02,
        total_steps=100,
        warmup_fraction=0.0,
        decay_fraction=0.0,
        use_adamc=True,
        weight_decay=0.1,
        grad_clip_max_amps=None,
        **kwargs,
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)

    assert state.scheduled_lr["matrix"] == pytest.approx(0.01)
    assert state.scheduled_lr["vector"] == pytest.approx(0.02)
    assert jnp.allclose(updates["matrix"], -1e-5, rtol=5e-3, atol=2e-8)
    assert jnp.allclose(updates["vector"], -4e-5, rtol=5e-3, atol=2e-8)


def test_schedule_free_kron_records_wsd_lr_for_all_leaves():
    params = {"matrix": jnp.ones((2, 2)), "vector": jnp.ones((2,))}
    grads = {"matrix": jnp.zeros((2, 2)), "vector": jnp.zeros((2,))}
    tx = schedule_free_kron(
        learning_rate=0.01,
        total_steps=100,
        warmup_fraction=0.0,
        decay_fraction=0.0,
        preconditioner_update_probability=1.0,
    )
    state = tx.init(params)
    _, state = tx.update(grads, state, params)

    assert state.scheduled_lr["matrix"] == pytest.approx(0.01)
    assert state.scheduled_lr["vector"] == pytest.approx(0.01)
