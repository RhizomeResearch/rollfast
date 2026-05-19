import jax.numpy as jnp
import pytest

from rollfast.optim.aurora import scale_by_aurora
from rollfast.optim.muon import scale_by_muon
from rollfast.optim.normuon import scale_by_normuon
from rollfast.optim.pion import scale_by_pion
from rollfast.optim.prism import scale_by_prism
from rollfast.optim.rmnp import scale_by_rmnp
from rollfast.optim.trasmuon import scale_by_trasmuon


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
