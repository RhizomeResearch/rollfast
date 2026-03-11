import jax
import jax.numpy as jnp
import optax
from rollfast.optim.psgd import scale_by_kron, kron

def test_scale_by_kron():
    params = {'w': jnp.ones((4, 4))}
    grads = {'w': jnp.ones((4, 4)) * 0.1}
    # precond_update_prob needs to be 1.0 to ensure update code runs without error
    tx = scale_by_kron(preconditioner_update_probability=1.0)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert 'w' in updates
    assert updates['w'].shape == (4, 4)

def test_kron():
    params = {'w': jnp.ones((4, 4)), 'b': jnp.ones((4,))}
    grads = {'w': jnp.ones((4, 4)) * 0.1, 'b': jnp.ones((4,)) * 0.1}
    tx = kron(learning_rate=0.01, preconditioner_update_probability=1.0)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert 'w' in updates
    assert 'b' in updates
    assert updates['w'].shape == (4, 4)
    assert updates['b'].shape == (4,)
