import jax.numpy as jnp
from rollfast.optim.adam import scale_by_adam, adamw
from tests._typing import as_array_dict


def test_scale_by_adam():
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.ones((2, 2)) * 0.1}
    tx = scale_by_adam()
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert updates["w"].shape == (2, 2)


def test_adamw():
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.ones((2, 2)) * 0.1}
    tx = adamw(learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert updates["w"].shape == (2, 2)


def test_adamw_magma_applies_weight_decay_before_masking():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.zeros_like(params["w"])}
    tx = adamw(
        learning_rate=1.0,
        weight_decay=0.2,
        use_magma=True,
        magma_p=0.0,
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert jnp.allclose(updates["w"], jnp.zeros_like(updates["w"]))


def test_scale_by_adam_weight_decay_accepts_array_mask():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.zeros_like(params["w"])}
    mask = {"w": jnp.array([[True, False], [False, True]])}
    tx = scale_by_adam(
        b1=0.0,
        b2=0.0,
        weight_decay=0.2,
        weight_decay_mask=mask,
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    expected = jnp.array([[0.2, 0.0], [0.0, 0.2]], dtype=jnp.float32)
    assert jnp.allclose(updates["w"], expected)
