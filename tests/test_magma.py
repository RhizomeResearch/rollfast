import jax.numpy as jnp
import pytest

from rollfast.optim.adam import adamw
from rollfast.optim.aurora import aurora, scale_by_aurora
from rollfast.optim.magma import apply_magma_internal
from rollfast.optim.prism import scale_by_prism
from rollfast.optim.psgd import scale_by_kron
from tests._typing import as_array_dict


def test_magma_via_adam():
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.ones((2, 2)) * 0.1}
    tx = adamw(learning_rate=0.01, use_magma=True)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert updates["w"].shape == (2, 2)


def test_magma_via_aurora():
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.ones((2, 2)) * 0.1}
    tx = aurora(learning_rate=0.01, use_magma=True, polar_ns_iters=2)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert updates["w"].shape == (2, 2)


def test_magma_complex_leaves_use_real_hermitian_alignment():
    raw_gradients = {"w": jnp.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=jnp.complex64)}
    first_moments = {"w": jnp.array([2.0 - 1.0j, 1.0 + 0.5j], dtype=jnp.complex64)}
    base_updates = {"w": jnp.ones((2,), dtype=jnp.complex64)}
    magma_s = {"w": jnp.array(0.5, dtype=jnp.float32)}

    updates, next_s = apply_magma_internal(
        raw_gradients,
        first_moments,
        base_updates,
        magma_s,
        key=jnp.array([0, 0], dtype=jnp.uint32),
        p=1.0,
    )
    updates = as_array_dict(updates)
    next_s = as_array_dict(next_s)

    assert updates["w"].dtype == jnp.complex64
    assert next_s["w"].dtype == jnp.float32
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.all(jnp.isfinite(next_s["w"]))


def test_magma_rejects_mismatched_input_trees():
    with pytest.raises(ValueError, match=r"first_moments.*raw_gradients"):
        apply_magma_internal(
            raw_gradients={"w": jnp.ones((2,), dtype=jnp.float32)},
            first_moments={
                "w": jnp.ones((2,), dtype=jnp.float32),
                "b": jnp.ones((2,), dtype=jnp.float32),
            },
            base_updates={"w": jnp.ones((2,), dtype=jnp.float32)},
            magma_s_prev={"w": jnp.array(0.5, dtype=jnp.float32)},
            key=jnp.array([0, 0], dtype=jnp.uint32),
        )


@pytest.mark.parametrize("make_tx", [scale_by_aurora, scale_by_prism, scale_by_kron])
def test_magma_enabled_transforms_validate_probability(make_tx):
    with pytest.raises(ValueError, match="magma_p"):
        make_tx(use_magma=True, magma_p=-0.1)
