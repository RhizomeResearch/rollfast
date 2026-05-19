import jax.numpy as jnp
from rollfast.optim.psgd import scale_by_kron, kron
from tests._typing import as_array_dict


def test_scale_by_kron():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    # precond_update_prob needs to be 1.0 to ensure update code runs without error
    tx = scale_by_kron(preconditioner_update_probability=1.0)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert updates["w"].shape == (4, 4)


def test_kron():
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    grads = {"w": jnp.ones((4, 4)) * 0.1, "b": jnp.ones((4,)) * 0.1}
    tx = kron(learning_rate=0.01, preconditioner_update_probability=1.0)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert "b" in updates
    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)


def test_scale_by_kron_magma_weight_decay_accepts_array_mask():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.zeros_like(params["w"])}
    mask = {"w": jnp.array([[True, False], [False, True]])}
    tx = scale_by_kron(
        b1=0.9,
        preconditioner_update_probability=0.0,
        grad_clip_max_amps=(1e9, 1e9),
        use_magma=True,
        magma_p=1.0,
        weight_decay=0.2,
        weight_decay_mask=mask,
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert jnp.all(updates["w"][mask["w"]] > 0)
    assert jnp.allclose(updates["w"][jnp.logical_not(mask["w"])], 0.0)
