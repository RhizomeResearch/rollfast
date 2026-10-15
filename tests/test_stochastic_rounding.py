import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rollfast.optim.adam import adamw
from rollfast.optim.aurora import aurora
from rollfast.optim.prism import prism
from rollfast.optim.psgd import kron
from rollfast.schedules.schedulefree import (
    schedule_free_adam,
    schedule_free_aurora,
    schedule_free_kron,
    schedule_free_prism,
)
from rollfast.utils import _stochastic_round_bf16, apply_updates, apply_updates_prefix
from tests._typing import as_array_dict

all_optimizers = [
    (adamw, {"learning_rate": 0.001}),
    (aurora, {"learning_rate": 0.001, "polar_ns_iters": 2}),
    (prism, {"learning_rate": 0.001, "ns_iters": 2}),
    (kron, {"learning_rate": 0.001, "preconditioner_update_probability": 1.0}),
    (schedule_free_adam, {"learning_rate": 0.001, "total_steps": 10}),
    (
        schedule_free_aurora,
        {"learning_rate": 0.001, "total_steps": 10, "polar_ns_iters": 2},
    ),
    (schedule_free_prism, {"learning_rate": 0.001, "total_steps": 10, "ns_iters": 2}),
    (
        schedule_free_kron,
        {
            "learning_rate": 0.001,
            "total_steps": 10,
            "preconditioner_update_probability": 1.0,
        },
    ),
]

# Schedule-Free Kron deliberately has no public mu_dtype argument.
mu_dtype_optimizers = [
    (fn, kwargs) for fn, kwargs in all_optimizers if fn is not schedule_free_kron
]


@pytest.mark.parametrize(
    "optimizer_fn, kwargs",
    mu_dtype_optimizers,
    ids=[fn.__name__ for fn, _ in mu_dtype_optimizers],
)
def test_mu_dtype_bf16(optimizer_fn, kwargs):
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}

    # Enable mu_dtype = bf16
    kwargs_copy = kwargs.copy()
    kwargs_copy["mu_dtype"] = jnp.bfloat16
    tx = optimizer_fn(**kwargs_copy)

    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)

    assert "w" in updates
    assert updates["w"].shape == (4, 4)


@pytest.mark.parametrize(
    "apply_fn", [apply_updates, apply_updates_prefix], ids=["full_tree", "prefix"]
)
@pytest.mark.parametrize(
    "optimizer_fn, kwargs",
    all_optimizers,
    ids=[fn.__name__ for fn, _ in all_optimizers],
)
def test_pure_bf16_apply_updates(optimizer_fn, kwargs, apply_fn):
    # Model in pure BF16
    params = {"w": jnp.ones((4, 4), dtype=jnp.bfloat16)}
    # Gradients in BF16
    grads = {"w": jnp.ones((4, 4), dtype=jnp.bfloat16) * 0.1}

    tx = optimizer_fn(**kwargs)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)

    key = jax.random.PRNGKey(0)

    # Stochastic rounding
    new_params = apply_fn(params, updates, key, stochastic=True)
    new_params = as_array_dict(new_params)

    assert "w" in new_params
    assert new_params["w"].dtype == jnp.bfloat16


def _float32_from_bits(bits):
    return jax.lax.bitcast_convert_type(
        jnp.asarray(bits, dtype=jnp.uint32), jnp.float32
    )


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_stochastic_round_bf16_keeps_exact_and_special_values(jit):
    round_bf16 = jax.jit(_stochastic_round_bf16) if jit else _stochastic_round_bf16
    exact = jnp.asarray([0.0, -0.0, 1.0, -2.5, jnp.inf, -jnp.inf], dtype=jnp.float32)
    # The second NaN's payload lies only in the bits that rounding discards.
    nans = jnp.stack([jnp.float32(jnp.nan), _float32_from_bits(0x7F800001)])
    for seed in range(16):
        key = jax.random.key(seed)
        rounded = round_bf16(exact, key)
        assert rounded.dtype == jnp.bfloat16
        np.testing.assert_array_equal(rounded.astype(jnp.float32), exact)
        np.testing.assert_array_equal(jnp.signbit(rounded), jnp.signbit(exact))
        assert bool(jnp.all(jnp.isnan(round_bf16(nans, key))))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_stochastic_round_bf16_rounds_up_in_proportion_to_the_remainder(sign):
    # A quarter of the BF16 spacing (2**-7) above one.
    value = sign * (1.0 + 2.0**-9)
    x = jnp.full((40_000,), value, dtype=jnp.float32)
    rounded = jax.jit(_stochastic_round_bf16)(x, jax.random.key(0)).astype(jnp.float32)
    away = rounded == sign * (1.0 + 2.0**-7)
    assert bool(jnp.all(jnp.logical_or(away, rounded == sign)))
    np.testing.assert_allclose(jnp.mean(away), 0.25, atol=0.01)
    np.testing.assert_allclose(jnp.mean(rounded), value, rtol=1e-4)


def test_stochastic_round_bf16_may_round_past_the_largest_finite_value():
    largest_bf16 = float(jnp.finfo(jnp.bfloat16).max)
    x = jnp.full((1_000,), jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    rounded = _stochastic_round_bf16(x, jax.random.key(0)).astype(jnp.float32)
    assert bool(jnp.all((rounded == largest_bf16) | (rounded == jnp.inf)))


def test_stochastic_apply_updates_keeps_small_updates_in_expectation():
    params = {"w": jnp.ones((40_000,), dtype=jnp.bfloat16)}
    updates = {"w": jnp.full((40_000,), 2.0**-9, dtype=jnp.float32)}
    nearest = apply_updates(params, updates, stochastic=False)
    stochastic = apply_updates(params, updates, jax.random.key(1), stochastic=True)
    assert stochastic["w"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(nearest["w"], params["w"])
    np.testing.assert_allclose(
        jnp.mean(stochastic["w"].astype(jnp.float32)), 1.0 + 2.0**-9, rtol=1e-4
    )
