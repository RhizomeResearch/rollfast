import jax
import jax.numpy as jnp
import optax
from rollfast.schedules.schedulefree import schedule_free, schedule_free_prism, schedule_free_kron, schedule_free_eval_params

def test_schedule_free():
    params = {'w': jnp.ones((2, 2))}
    grads = {'w': jnp.ones((2, 2)) * 0.1}
    base_opt = optax.sgd(0.01)
    tx = schedule_free(base_opt, learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert 'w' in updates
    
    eval_params = schedule_free_eval_params(state, params)
    assert 'w' in eval_params

def test_schedule_free_prism():
    params = {'w': jnp.ones((4, 4))}
    grads = {'w': jnp.ones((4, 4)) * 0.1}
    tx = schedule_free_prism(learning_rate=0.01, total_steps=100)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert 'w' in updates

def test_schedule_free_kron():
    params = {'w': jnp.ones((4, 4))}
    grads = {'w': jnp.ones((4, 4)) * 0.1}
    tx = schedule_free_kron(learning_rate=0.01, total_steps=100, preconditioner_update_probability=1.0)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert 'w' in updates
