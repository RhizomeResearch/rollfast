import jax
import jax.numpy as jnp
import optax
from rollfast.optim.adam import adamw

def test_magma_via_adam():
    params = {'w': jnp.ones((2, 2))}
    grads = {'w': jnp.ones((2, 2)) * 0.1}
    tx = adamw(learning_rate=0.01, use_magma=True)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert 'w' in updates
    assert updates['w'].shape == (2, 2)
