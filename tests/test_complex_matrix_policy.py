import jax.numpy as jnp
import pytest

from rollfast.optim.aurora import aurora, scale_by_aurora
from rollfast.optim.dimension_numbers import MatrixDimensionNumbers
from rollfast.optim.muon import muon, scale_by_muon
from rollfast.optim.normuon import normuon, scale_by_normuon
from rollfast.optim.pion import pion, scale_by_pion
from rollfast.optim.prism import prism, scale_by_prism
from rollfast.optim.rmnp import rmnp, scale_by_rmnp
from rollfast.optim.trasmuon import scale_by_trasmuon, trasmuon


@pytest.mark.parametrize(
    "make_tx",
    [
        lambda: scale_by_muon(ns_steps=2),
        lambda: scale_by_prism(ns_iters=2, grad_clip_max_amps=None),
        lambda: scale_by_aurora(polar_ns_iters=2, grad_clip_max_amps=None),
        lambda: scale_by_rmnp(beta=0.0, nesterov=False),
        lambda: scale_by_normuon(beta1=0.0, beta2=0.0, nesterov=False, ns_iters=2),
        lambda: scale_by_trasmuon(beta1=0.0, beta2=0.0, ns_iters=2),
        lambda: scale_by_pion(learning_rate=0.1),
    ],
)
def test_direct_matrix_transforms_reject_complex_matrix_leaves(make_tx):
    params = {"w": jnp.eye(2, dtype=jnp.complex64)}
    grads = {"w": jnp.ones((2, 2), dtype=jnp.complex64)}
    tx = make_tx()

    with pytest.raises(ValueError, match="complex"):
        state = tx.init(params)
        tx.update(grads, state, params)


@pytest.mark.parametrize(
    "make_tx",
    [
        lambda: muon(learning_rate=0.01, ns_steps=2),
        lambda: prism(learning_rate=0.01, ns_iters=2, grad_clip_max_amps=None),
        lambda: aurora(learning_rate=0.01, polar_ns_iters=2, grad_clip_max_amps=None),
        lambda: rmnp(learning_rate=0.01, beta=0.0, nesterov=False),
        lambda: normuon(
            learning_rate=0.01, beta1=0.0, beta2=0.0, nesterov=False, ns_iters=2
        ),
        lambda: trasmuon(learning_rate=0.01, beta1=0.0, beta2=0.0, ns_iters=2),
        lambda: pion(learning_rate=0.01),
    ],
)
def test_public_matrix_wrappers_route_complex_matrices_to_adam(make_tx):
    params = {
        "complex": jnp.eye(2, dtype=jnp.complex64),
        "real": jnp.eye(2, dtype=jnp.float32),
    }
    grads = {
        "complex": jnp.ones((2, 2), dtype=jnp.complex64) * (1.0 + 2.0j),
        "real": jnp.ones((2, 2), dtype=jnp.float32),
    }
    tx = make_tx()

    updates, _ = tx.update(grads, tx.init(params), params)

    assert jnp.issubdtype(updates["complex"].dtype, jnp.complexfloating)
    assert jnp.all(jnp.isfinite(updates["complex"]))
    assert jnp.any(jnp.imag(updates["complex"]) != 0.0)
    assert not jnp.issubdtype(updates["real"].dtype, jnp.complexfloating)
    assert jnp.all(jnp.isfinite(updates["real"]))


def test_explicit_matrix_spec_still_routes_complex_public_leaf_to_adam():
    params = {"w": jnp.eye(2, dtype=jnp.complex64)}
    grads = {"w": jnp.ones((2, 2), dtype=jnp.complex64) * (1.0 + 2.0j)}
    tx = muon(
        learning_rate=0.01,
        ns_steps=2,
        muon_weight_dimension_numbers={"w": MatrixDimensionNumbers()},
    )

    updates, _ = tx.update(grads, tx.init(params), params)

    assert jnp.any(jnp.imag(updates["w"]) != 0.0)
