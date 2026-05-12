import jax.numpy as jnp
from rollfast.optim.adam import adamw
from rollfast.optim.aurora import aurora

def test_magma_via_adam():
    params = {'w': jnp.ones((2, 2))}
    grads = {'w': jnp.ones((2, 2)) * 0.1}
    tx = adamw(learning_rate=0.01, use_magma=True)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert 'w' in updates
    assert updates['w'].shape == (2, 2)

def test_magma_via_aurora():
    params = {'w': jnp.ones((2, 2))}
    grads = {'w': jnp.ones((2, 2)) * 0.1}
    tx = aurora(learning_rate=0.01, use_magma=True, polar_ns_iters=2)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert 'w' in updates
    assert updates['w'].shape == (2, 2)
