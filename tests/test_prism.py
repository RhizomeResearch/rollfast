import jax
import jax.numpy as jnp
import optax
from rollfast.optim.prism import scale_by_prism, prism

def test_scale_by_prism():
    params = {'w': jnp.ones((4, 4))}
    grads = {'w': jnp.ones((4, 4)) * 0.1}
    tx = scale_by_prism()
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert 'w' in updates
    assert updates['w'].shape == (4, 4)

def test_prism():
    params = {'w': jnp.ones((4, 4)), 'b': jnp.ones((4,))}
    grads = {'w': jnp.ones((4, 4)) * 0.1, 'b': jnp.ones((4,)) * 0.1}
    tx = prism(learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert 'w' in updates
    assert 'b' in updates
    assert updates['w'].shape == (4, 4)
    assert updates['b'].shape == (4,)
