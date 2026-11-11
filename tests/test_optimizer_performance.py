"""Equivalence and executed-work regressions for optimizer fast paths."""

import importlib
from dataclasses import replace

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest

from rollfast.optim.adam import scale_by_adam
from rollfast.optim.adam8 import _init_moment_leaf, quantize_blocks, scale_by_adam8
from rollfast.optim.apollo import apollo_adamw
from rollfast.optim.galore import galore_adamw
from rollfast.optim.hyperball import apply_hyperball
from rollfast.optim.psgd import _init_Q_exprs, _precond_grad, scale_by_kron


def _assert_exact(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _direct_eqns(jaxpr, primitive):
    return [eqn for eqn in jaxpr.eqns if eqn.primitive.name == primitive]


def _conds_outside_loops(jaxpr):
    """Return cond equations that XLA may capture into a CUDA graph.

    Loops stay outside CUDA graphs under default XLA flags, so a cond inside a
    while body or a non-unrolled scan body is not captured.
    """

    found = []
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "cond":
            found.append(eqn)
        if name == "while" or (name == "scan" and eqn.params.get("unroll", 1) == 1):
            continue
        for sub in jax.extend.core.jaxprs_in_params(eqn.params):
            found.extend(_conds_outside_loops(sub))
    return found


def _optimizers():
    import rollfast as r

    sf = {"learning_rate": 0.01, "total_steps": 10}
    return {
        "adamw": lambda: r.adamw(0.01),
        "adamw8": lambda: r.adamw8(0.01),
        "aurora": lambda: r.aurora(0.01),
        "riemannian_aurora": lambda: r.riemannian_aurora(0.01),
        "contramuon": lambda: r.contramuon(0.01),
        "kron": lambda: r.kron(learning_rate=0.01),
        "muon_aol": lambda: r.muon(0.01, preconditioning="aol"),
        "muon_schatten": lambda: r.muon(0.01, preconditioning="schatten"),
        "normuon": lambda: r.normuon(0.01),
        "pion": lambda: r.pion(0.01),
        "prism": lambda: r.prism(0.01),
        "rmnp": lambda: r.rmnp(0.01),
        "trasmuon": lambda: r.trasmuon(0.01),
        "apollo": lambda: apollo_adamw(0.01, rank=2, projection_refresh_interval=3),
        "galore": lambda: galore_adamw(0.01, rank=2, min_matrix_size=0),
        "schedule_free_kron": lambda: r.schedule_free_kron(**sf),
        "soda_muon": lambda: r.soda_muon(**sf),
    }


@pytest.mark.parametrize("name", list(_optimizers()))
def test_optimizer_updates_keep_conds_out_of_cuda_graphs(name):
    params = {"w": jnp.ones((8, 6)), "b": jnp.ones(6)}
    tx = _optimizers()[name]()

    graph = jax.make_jaxpr(tx.update)(params, tx.init(params), params).jaxpr

    assert not _conds_outside_loops(graph)


def test_finetune_update_paths_keep_conds_out_of_cuda_graphs():
    import rollfast.finetune as rfft
    from tests.finetune.helpers import tiny_plan

    plan = tiny_plan()
    averaged = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        ema=rfft.EMAConfig(enabled=True),
        swa=rfft.SWAConfig(enabled=True, start_step=0),
    )

    def bundle_loss(params, value):
        loss = sum(jnp.sum((leaf - value) ** 2) for leaf in jax.tree.leaves(params))
        return rfft.LossBundle(
            loss_sum=loss,
            normalizer=jnp.asarray(1.0),
            metrics_sums={},
            metric_normalizers={},
            new_model_state=None,
        )

    accumulation = rfft.AccumulationConfig(steps=2)
    accumulate = rfft.make_accumulating_loss_bundle_update_step(
        bundle_loss, averaged, accumulation=accumulation
    )
    controller = rfft.make_adalora_controller({"a": 4, "b": 4}, total_steps=10)
    params = plan.trainable

    graphs = [
        jax.make_jaxpr(averaged.update)(params, averaged.init(params), params),
        jax.make_jaxpr(accumulate)(
            params,
            averaged.init(params),
            rfft.init_accumulation_state(params, accumulation),
            jnp.asarray(0.5),
        ),
        jax.make_jaxpr(controller.update)(controller.init(), jnp.ones((2, 4))),
    ]

    for graph in graphs:
        assert not _conds_outside_loops(graph.jaxpr)


def test_kron_skips_only_the_preconditioner_update_without_a_cond():
    params = jnp.ones((2, 3), dtype=jnp.float32)
    tx = scale_by_kron(preconditioner_update_probability=0.5)

    graph = jax.make_jaxpr(tx.update)(params, tx.init(params), params).jaxpr

    assert not _direct_eqns(graph, "cond")
    loops = _direct_eqns(graph, "while")
    assert len(loops) == 1
    update_body = loops[0].params["body_jaxpr"].jaxpr
    assert not _direct_eqns(update_body, "cond")
    assert not _direct_eqns(update_body, "while")


@pytest.mark.parametrize("use_jit", [False, True])
@pytest.mark.parametrize(
    "build_tx",
    [
        lambda: scale_by_adam(weight_decay=0.0),
        lambda: galore_adamw(0.01, rank=1, weight_decay=0.0, min_matrix_size=0),
        lambda: apollo_adamw(0.01, rank=1, weight_decay=0.0),
        lambda: apply_hyperball(
            0.01,
            weight_decay=0.0,
            hyperball_mask={"w": False},
            fallback_weight_decay=True,
        ),
    ],
    ids=["adam", "galore", "apollo", "hyperball"],
)
def test_static_zero_weight_decay_skips_parameter_work(build_tx, use_jit):
    params = {"w": jnp.ones((4, 2), dtype=jnp.float32)}
    grads = {"w": jnp.full((4, 2), 1e-3, dtype=jnp.float32)}
    nan_params = {"w": jnp.full((4, 2), jnp.nan, dtype=jnp.float32)}
    tx = build_tx()
    update = jax.jit(tx.update) if use_jit else tx.update

    updates, _ = update(grads, tx.init(params), nan_params)

    assert bool(jnp.all(jnp.isfinite(updates["w"])))


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

    def refresh_basis(grad, state, flag):
        state = module._refresh_projected_leaves(
            grad, state, should_refresh=flag, transport=False, basis_dtype=dtype
        )
        return state.basis_left, state.basis_right

    refresh = jax.jit(refresh_basis)
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

    def transport(flag):
        refreshed = module._refresh_projected_leaves(
            grad[::-1],
            state,
            should_refresh=flag,
            transport=True,
            basis_dtype=jnp.float32,
        )
        return refreshed.mu, refreshed.nu

    for update in (transport, jax.jit(transport)):
        _assert_exact(update(False), (state.mu, state.nu))
        mu, nu = update(True)
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
        lambda s, count: (
            module._refresh_projections(
                s, count=count, projection_seed=0, projection_refresh_interval=3
            ).projection
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
