from typing import cast

import jax
import jax.numpy as jnp
import optax
import pytest

from rollfast.optim.pion import ScaleByPionState, pion, scale_by_pion
from rollfast.optim.dimension_numbers import MatrixDimensionNumbers


def test_scale_by_pion():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    tx = scale_by_pion(learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = cast(dict[str, jax.Array], updates)
    state = cast(ScaleByPionState, state)
    m_in = cast(dict[str, jax.Array], state.m_in)
    assert updates["w"].shape == (4, 4)
    assert m_in["w"].shape == (1, 4, 4)


def test_scale_by_pion_schedule_uses_current_count_before_increment():
    params = {"w": jnp.eye(3, dtype=jnp.float32)}
    grads = {"w": jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 10.0}
    tx = scale_by_pion(
        learning_rate=lambda count: jnp.where(count == 0, 0.0, 1.0),
        b1=0.0,
        b2=0.0,
    )
    state = tx.init(params)

    first_updates, state = tx.update(grads, state, params)
    second_updates, state = tx.update(grads, state, params)
    first_updates = cast(dict[str, jax.Array], first_updates)
    second_updates = cast(dict[str, jax.Array], second_updates)

    assert jnp.allclose(first_updates["w"], jnp.zeros_like(params["w"]))
    assert not jnp.allclose(second_updates["w"], jnp.zeros_like(params["w"]))


def test_scale_by_pion_heavy_ball_momentum_accumulator():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grad = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 10.0
    grads = {"w": grad}
    tx = scale_by_pion(
        learning_rate=0.01,
        b1=0.5,
        b2=0.0,
        momentum_accumulator="heavy_ball",
    )
    _, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByPionState, state)
    m_in = cast(dict[str, jax.Array], state.m_in)
    expected = grad - grad.T

    assert jnp.allclose(m_in["w"][0], expected)


def test_scale_by_pion_bf16_state():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 10.0}
    tx = scale_by_pion(
        learning_rate=0.01,
        b1=0.0,
        b2=0.0,
        mu_dtype=jnp.bfloat16,
    )
    _, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByPionState, state)
    m_in = cast(dict[str, jax.Array], state.m_in)
    v_in = cast(dict[str, jax.Array], state.v_in)
    m_out = cast(dict[str, jax.Array], state.m_out)
    v_out = cast(dict[str, jax.Array], state.v_out)

    assert m_in["w"].dtype == jnp.bfloat16
    assert v_in["w"].dtype == jnp.bfloat16
    assert m_out["w"].dtype == jnp.bfloat16
    assert v_out["w"].dtype == jnp.bfloat16


def test_scale_by_pion_rejects_direct_fallback_leaves():
    params = {
        "w": jnp.eye(4, dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = scale_by_pion(learning_rate=0.01)

    with pytest.raises(ValueError, match="public `pion` wrapper"):
        tx.update(grads, tx.init(params), params)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"b1": -0.1},
        {"b1": 1.0},
        {"b2": -0.1},
        {"b2": 1.0},
        {"eps": 0.0},
        {"rms_constant": 0.0},
    ],
)
def test_scale_by_pion_rejects_invalid_static_parameters(kwargs):
    with pytest.raises(ValueError):
        scale_by_pion(learning_rate=0.01, **kwargs)


def test_pion_partitions_vectors_to_adam():
    params = {
        "w": jnp.eye(4, dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = pion(learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = cast(dict[str, jax.Array], updates)
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
    updates = cast(dict[str, jax.Array], updates)
    assert updates["w"].shape == (3, 3)
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_pion_custom_dimension_numbers_for_conv_kernel():
    params = {"kernel": jnp.ones((8, 3, 3, 4), dtype=jnp.float32)}
    grads = {"kernel": jnp.ones_like(params["kernel"]) * 0.01}
    specs = {
        "kernel": MatrixDimensionNumbers(
            reduction_axis=(1, 2, 3),
            output_axis=0,
        )
    }
    tx = pion(learning_rate=0.01, pion_weight_dimension_numbers=specs)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = cast(dict[str, jax.Array], updates)
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
    new_params = cast(dict[str, jax.Array], optax.apply_updates(params, updates))
    new_w = new_params["w"]
    assert jnp.allclose(jnp.linalg.svd(new_w, compute_uv=False), jnp.ones(4), atol=1e-3)
