"""Equivalence and executed-work regressions for optimizer fast paths."""

import importlib
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rollfast.optim.adam8 import _init_moment_leaf, quantize_blocks, scale_by_adam8
from rollfast.optim.psgd import _init_Q_exprs, _precond_grad, scale_by_kron


def _assert_exact(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("projection", ["left", "right", "two_sided"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_galore_only_computes_basis_on_refresh(monkeypatch, projection, dtype):
    module = importlib.import_module("rollfast.optim.galore")
    grad = jax.random.normal(jax.random.PRNGKey(7), (6, 4))
    state = (
        module.galore_adamw(
            0.01, rank=2, projection=projection, basis_dtype=dtype, min_matrix_size=0
        )
        .init(grad)
        .leaves
    )
    basis = module._basis_from_grad(grad, state.orientation, state.mu.shape)
    expected = tuple(x.astype(dtype) for x in basis)
    state = replace(state, basis_left=expected[0], basis_right=expected[1])
    calls = []
    original = module._basis_from_grad

    def observed(*args):
        jax.debug.callback(lambda: calls.append(True))
        return original(*args)

    monkeypatch.setattr(module, "_basis_from_grad", observed)
    refresh = jax.jit(
        lambda g, s, flag: module._refresh_basis(
            g, s, should_refresh=flag, basis_dtype=dtype
        )
    )
    _assert_exact(jax.block_until_ready(refresh(grad, state, False)), expected)
    assert not calls
    _assert_exact(jax.block_until_ready(refresh(grad, state, True)), expected)
    assert calls == [True]


@pytest.mark.parametrize("projection", ["left", "right", "two_sided"])
def test_galore_transport_matches_reconstructed_first_moment(projection):
    module = importlib.import_module("rollfast.optim.galore")
    grad = jax.random.normal(jax.random.PRNGKey(7), (6, 4))
    state = (
        module.galore_adamw(0.01, rank=2, projection=projection, min_matrix_size=0)
        .init(grad)
        .leaves
    )
    old_basis = module._basis_from_grad(grad, state.orientation, state.mu.shape)
    new_basis = module._basis_from_grad(grad[::-1], state.orientation, state.mu.shape)
    state = replace(
        state,
        basis_left=old_basis[0],
        basis_right=old_basis[1],
        mu=jnp.arange(state.mu.size, dtype=jnp.float32).reshape(state.mu.shape),
        nu=jnp.ones_like(state.nu),
    )
    full_moment = module._reconstruct(state.mu, *old_basis, state.orientation)
    expected = module._project(full_moment, *new_basis, state.orientation)
    for update in (
        module._transport_projected_moments,
        jax.jit(module._transport_projected_moments),
    ):
        _assert_exact(
            update(state, *new_basis, should_refresh=False), (state.mu, state.nu)
        )
        mu, nu = update(state, *new_basis, should_refresh=True)
        np.testing.assert_allclose(mu, expected, rtol=1e-5, atol=1e-6)
        assert jnp.all(nu >= 0)


def test_apollo_projection_refresh_retains_seed_and_cadence(monkeypatch):
    module = importlib.import_module("rollfast.optim.apollo")
    state = module.apollo_adamw(0.01, rank=2).init(jnp.ones((6, 4))).leaves
    original = module._make_projection
    calls = []

    def observed(**kwargs):
        jax.debug.callback(lambda: calls.append(True))
        return original(**kwargs)

    monkeypatch.setattr(module, "_make_projection", observed)
    refresh = jax.jit(
        lambda s, count: module._refresh_projection(
            s, count=count, rank=2, projection_seed=0, projection_refresh_interval=3
        )
    )
    reference = jax.jit(
        lambda count: original(
            projection_seed=0,
            leaf_index=state.leaf_index,
            step=count,
            shape=state.projection.shape,
            dtype=state.projection.dtype,
        )
    )
    for count in range(5):
        actual = jax.block_until_ready(refresh(state, count))
        expected = reference(count) if count % 3 == 0 else state.projection
        _assert_exact(actual, expected)
    assert len(calls) == 2


def test_aurora_cg_stops_matvecs_after_convergence(monkeypatch):
    module = importlib.import_module("rollfast.optim.aurora")
    u = jnp.full((6, 2), 0.05)
    calls = []
    original = jnp.matmul

    def observed(*args, **kwargs):
        jax.debug.callback(lambda: calls.append(True))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.jnp, "matmul", observed)
    result = jax.jit(module._solve_row_norm_multipliers)(
        u, jnp.array(0.5), jnp.zeros(6)
    )
    np.testing.assert_array_equal(jax.block_until_ready(result), np.zeros(6))
    assert len(calls) == 2


def test_aurora_cg_matches_dense_solve_and_remains_differentiable():
    module = importlib.import_module("rollfast.optim.aurora")
    u = jax.random.normal(jax.random.PRNGKey(2), (6, 2)) * 0.05
    b = jnp.arange(1, 7, dtype=jnp.float32)
    r = jnp.asarray(0.5)
    u_np = np.asarray(u)
    reg = max(np.max(np.sum(u_np**2, axis=-1) ** 2) - float(r) + 1e-3, 0.0)
    matrix = (float(r) + reg) * np.eye(6) - (u_np @ u_np.T) ** 2
    solve = lambda rhs: module._solve_row_norm_multipliers(u, r, rhs)
    np.testing.assert_allclose(jax.jit(solve)(b), np.linalg.solve(matrix, b), rtol=1e-5)
    jacobian = jax.jit(jax.jacrev(solve))(b)
    np.testing.assert_allclose(jacobian, np.linalg.inv(matrix), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_kron_uses_correct_preconditioner_generation(first, probability):
    params = jnp.ones((2, 3))
    grads = jnp.arange(1, 7, dtype=jnp.float32).reshape(2, 3) * 0.01
    tx = scale_by_kron(
        preconditioner_update_probability=probability,
        preconditioner_init_scale=1.0,
        preconditioner_mode="EQ",
        update_preconditioner_first=first,
        grad_clip_max_amps=(1e9, 1e9),
        precond_update_precision="float32",
    )
    state = tx.init(params)
    updates, next_state = jax.jit(tx.update)(grads, state, params)
    factors = next_state.Qs_preconditioners if first else state.Qs_preconditioners
    exprs = _init_Q_exprs(
        grads, 1.0, 8192, 1.0, 2, None, jnp.float32, existing_Q=factors
    )
    expected = _precond_grad(factors, grads, exprs)
    np.testing.assert_allclose(updates, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("shape", [(0,), (7,), (8, 4)])
@pytest.mark.parametrize("layout", ["shard_local", "logical_global"])
@pytest.mark.parametrize("quantizer", ["dynamic_signed", "dynamic_unsigned"])
@pytest.mark.parametrize("scale_dtype", [jnp.float32, jnp.bfloat16])
def test_direct_quantized_zero_init_matches_quantization(
    shape, layout, quantizer, scale_dtype
):
    param = jnp.ones(shape, dtype=jnp.bfloat16)
    expected = quantize_blocks(
        jnp.zeros_like(param, dtype=jnp.float32),
        block_size=8,
        block_layout=layout,
        quantizer=quantizer,
        scale_dtype=scale_dtype,
    )
    init = lambda p: _init_moment_leaf(
        p,
        block_size=8,
        min_size=0,
        scale_dtype=scale_dtype,
        fallback_dtype=jnp.float32,
        block_layout=layout,
        quantize=True,
        quantizer=quantizer,
    )
    for fn in (init, jax.jit(init)):
        _assert_exact(fn(param), expected)


@pytest.mark.parametrize("stochastic", [False, True])
def test_adam8_preserves_parent_and_stochastic_leaf_keys(monkeypatch, stochastic):
    params = {"a": None, "b": jnp.arange(32, dtype=jnp.float32), "c": jnp.ones(3)}
    key = jax.random.PRNGKey(13)
    tx = scale_by_adam8(
        block_size=8, min_size=8, key=key, stochastic_rounding=stochastic
    )
    state = tx.init(params)
    _assert_exact(state.key, jax.random.split(key, 3)[2])
    assert state.mu["a"] is None
    assert state.mu["c"].dtype == jnp.float32
    original = jax.random.split
    splits = []

    def observed(key, num=2):
        splits.append(num)
        return original(key, num)

    monkeypatch.setattr(jax.random, "split", observed)
    updates, next_state = tx.update(params, state)
    assert len(splits) == (3 if stochastic else 1)
    _assert_exact(next_state.key, original(state.key, 3)[2])
    for moment, parent in (
        (next_state.mu, original(state.key, 3)[0]),
        (next_state.nu, original(state.key, 3)[1]),
    ):
        child_key = original(parent, 3)[1]
        value = (
            0.1 * params["b"] if moment is next_state.mu else 0.001 * params["b"] ** 2
        )
        reference = quantize_blocks(
            value,
            block_size=8,
            stochastic_rounding=stochastic,
            key=child_key,
            quantizer="dynamic_signed"
            if moment is next_state.mu
            else "dynamic_unsigned",
        )
        _assert_exact(moment["b"], reference)
    jit_updates, jit_state = jax.jit(tx.update)(params, state)
    _assert_exact(jit_state, next_state)
    for actual, expected in zip(
        jax.tree.leaves(jit_updates), jax.tree.leaves(updates), strict=True
    ):
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
