from typing import cast

import jax
import jax.numpy as jnp
import pytest

from rollfast.optim.adam import ScaleByAdamState, scale_by_adam, adamw
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


def test_scale_by_adam_preserves_non_bf16_state_dtype():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_adam(mu_dtype=jnp.float16)
    _, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByAdamState, state)
    mu = as_array_dict(state.mu)
    nu = as_array_dict(state.nu)

    assert mu["w"].dtype == jnp.float16
    assert nu["w"].dtype == jnp.float16


def test_scale_by_adam_preserves_complex_first_moment_and_real_second_moment():
    params = {"w": jnp.ones((2,), dtype=jnp.complex64)}
    grads = {"w": jnp.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=jnp.complex64)}
    tx = scale_by_adam(b1=0.0, b2=0.0, eps=1e-8)

    updates, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByAdamState, state)
    updates = as_array_dict(updates)
    mu = as_array_dict(state.mu)
    nu = as_array_dict(state.nu)

    assert updates["w"].dtype == jnp.complex64
    assert mu["w"].dtype == jnp.complex64
    assert nu["w"].dtype == jnp.float32
    assert jnp.allclose(mu["w"], grads["w"])
    assert jnp.allclose(nu["w"], jnp.array([5.0, 25.0], dtype=jnp.float32))


def test_scale_by_adam_complex_ignores_real_bf16_state_dtype_for_first_moment():
    params = {"w": jnp.ones((2,), dtype=jnp.complex64)}
    grads = {"w": jnp.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=jnp.complex64)}
    tx = scale_by_adam(b1=0.0, b2=0.0, mu_dtype=jnp.bfloat16)

    _, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByAdamState, state)
    mu = as_array_dict(state.mu)
    nu = as_array_dict(state.nu)

    assert mu["w"].dtype == jnp.complex64
    assert nu["w"].dtype == jnp.bfloat16


def test_scale_by_adam_requires_params_for_weight_decay():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = jax.tree.map(jnp.zeros_like, params)
    tx = scale_by_adam(weight_decay=0.1)

    with pytest.raises(ValueError, match="params.*scale_by_adam"):
        tx.update(grads, tx.init(params))


def test_scale_by_adam_validates_magma_args():
    with pytest.raises(ValueError, match="magma_p"):
        scale_by_adam(use_magma=True, magma_p=1.1)
    with pytest.raises(ValueError, match="magma_tau"):
        scale_by_adam(use_magma=True, magma_tau=0.0)
