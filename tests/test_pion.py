import jax
import jax.numpy as jnp
import optax

from rollfast.optim.pion import pion, scale_by_pion
from rollfast.optim.prism import PrismDimensionNumbers


def test_scale_by_pion():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    tx = scale_by_pion(learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert updates["w"].shape == (4, 4)
    assert state.m_in["w"].shape == (1, 4, 4)


def test_pion_partitions_vectors_to_adam():
    params = {
        "w": jnp.eye(4, dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = pion(learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.all(jnp.isfinite(updates["b"]))


def test_pion_bilateral_mode():
    params = {"w": jnp.eye(3, dtype=jnp.float32)}
    grads = {"w": jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 10.0}
    tx = pion(learning_rate=0.01, alternating=False)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert updates["w"].shape == (3, 3)
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_pion_custom_dimension_numbers_for_conv_kernel():
    params = {"kernel": jnp.ones((8, 3, 3, 4), dtype=jnp.float32)}
    grads = {"kernel": jnp.ones_like(params["kernel"]) * 0.01}
    specs = {
        "kernel": PrismDimensionNumbers(
            reduction_axis=(1, 2, 3),
            output_axis=0,
        )
    }
    tx = pion(learning_rate=0.01, pion_weight_dimension_numbers=specs)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert updates["kernel"].shape == params["kernel"].shape


def test_pion_update_preserves_spectrum_to_second_order():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grad = jnp.array(
        [
            [0.0, 0.2, -0.1, 0.3],
            [-0.4, 0.0, 0.5, -0.2],
            [0.3, -0.1, 0.0, 0.4],
            [-0.2, 0.1, -0.3, 0.0],
        ],
        dtype=jnp.float32,
    )
    tx = pion(learning_rate=1e-3)
    state = tx.init(params)
    updates, state = tx.update({"w": grad}, state, params)
    new_w = optax.apply_updates(params, updates)["w"]
    assert jnp.allclose(jnp.linalg.svd(new_w, compute_uv=False), jnp.ones(4), atol=1e-3)
