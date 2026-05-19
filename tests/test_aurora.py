from typing import cast

import jax
import jax.numpy as jnp
from rollfast.optim.dimension_numbers import MatrixDimensionNumbers
from rollfast.optim.aurora import (
    ScaleByAuroraState,
    aurora,
    scale_by_aurora,
    scale_by_riemannian_aurora,
)
from tests._typing import as_array_dict


def test_scale_by_aurora():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    tx = scale_by_aurora()
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert updates["w"].shape == (4, 4)


def test_scale_by_aurora_heavy_ball_momentum_accumulator():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_aurora(
        b1=0.5,
        nesterov=False,
        grad_clip_max_amps=None,
        momentum_accumulator="heavy_ball",
    )
    _, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByAuroraState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert jnp.allclose(mu["w"], grads["w"])


def test_scale_by_aurora_bf16_momentum_state():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_aurora(
        mu_dtype=jnp.bfloat16,
        polar_ns_iters=2,
        grad_clip_max_amps=None,
    )
    _, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByAuroraState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert mu["w"].dtype == jnp.bfloat16


def test_scale_by_riemannian_aurora_magma_bf16_raw_clip():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"])}
    tx = scale_by_riemannian_aurora(
        b1=0.0,
        outer_steps=1,
        cg_steps=2,
        retraction_steps=1,
        polar_ns_iters=2,
        nesterov=False,
        mu_dtype=jnp.bfloat16,
        raw_global_grad_clip=0.25,
        permissive_spike_protection=True,
        grad_clip_max_amps=None,
        use_magma=True,
        magma_p=1.0,
    )
    updates, state = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)
    state = cast(ScaleByAuroraState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert mu["w"].dtype == jnp.bfloat16


def test_scale_by_aurora_spike_skip_preserves_momentum_and_magma_state():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"])}
    tx = scale_by_aurora(
        raw_global_grad_clip=0.01,
        permissive_spike_protection=False,
        polar_ns_iters=2,
        grad_clip_max_amps=None,
        use_magma=True,
        magma_p=1.0,
    )
    state0 = cast(ScaleByAuroraState, tx.init(params))
    updates, state1 = tx.update(grads, state0, params)
    state1 = cast(ScaleByAuroraState, state1)
    updates = as_array_dict(updates)
    mu0 = cast(dict[str, jax.Array], state0.mu)
    mu1 = cast(dict[str, jax.Array], state1.mu)
    magma0 = cast(dict[str, jax.Array], state0.magma_s)
    magma1 = cast(dict[str, jax.Array], state1.magma_s)

    assert jnp.allclose(updates["w"], jnp.zeros_like(updates["w"]))
    assert jnp.allclose(mu1["w"], mu0["w"])
    assert jnp.allclose(magma1["w"], magma0["w"])


def test_scale_by_aurora_magma_weight_decay_mask():
    params = {
        "decay": jnp.ones((4, 4), dtype=jnp.float32),
        "skip": jnp.ones((4, 4), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.zeros_like, params)
    tx = scale_by_aurora(
        b1=0.0,
        nesterov=False,
        polar_ns_iters=2,
        grad_clip_max_amps=None,
        use_magma=True,
        magma_p=1.0,
        weight_decay=0.2,
        weight_decay_mask={"decay": True, "skip": False},
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert jnp.linalg.norm(updates["decay"]) > 0
    assert jnp.allclose(updates["skip"], jnp.zeros_like(updates["skip"]))


def test_aurora_magma_wrapper_masks_matrix_weight_decay_and_keeps_adam_finite():
    params = {
        "decay": jnp.ones((4, 4), dtype=jnp.float32),
        "skip": jnp.ones((4, 4), dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.zeros_like, params)
    tx = aurora(
        learning_rate=1.0,
        b1=0.0,
        nesterov=False,
        polar_ns_iters=2,
        grad_clip_max_amps=None,
        use_magma=True,
        magma_p=0.0,
        weight_decay=0.2,
        weight_decay_mask={"decay": True, "skip": False, "b": False},
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert jnp.allclose(updates["decay"], jnp.zeros_like(updates["decay"]))
    assert jnp.allclose(updates["skip"], jnp.zeros_like(updates["skip"]))
    assert updates["b"].shape == params["b"].shape
    assert jnp.all(jnp.isfinite(updates["b"]))


def test_aurora_adam_fallback_honors_nesterov():
    params = {"b": jnp.ones((4,), dtype=jnp.float32)}
    grads = {"b": jnp.ones_like(params["b"])}
    plain_tx = aurora(
        learning_rate=1.0,
        adam_b1=0.5,
        adam_b2=0.0,
        nesterov=False,
    )
    nesterov_tx = aurora(
        learning_rate=1.0,
        adam_b1=0.5,
        adam_b2=0.0,
        nesterov=True,
    )

    plain_updates, _ = plain_tx.update(grads, plain_tx.init(params), params)
    nesterov_updates, _ = nesterov_tx.update(grads, nesterov_tx.init(params), params)
    plain_updates = as_array_dict(plain_updates)
    nesterov_updates = as_array_dict(nesterov_updates)

    assert not jnp.allclose(plain_updates["b"], nesterov_updates["b"])
    assert jnp.all(jnp.isfinite(nesterov_updates["b"]))


def test_scale_by_aurora_explicit_high_rank_dimension_spec():
    params = {"w": jnp.ones((2, 3, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_aurora(
        polar_ns_iters=2,
        grad_clip_max_amps=None,
        weight_dimension_numbers={
            "w": MatrixDimensionNumbers(reduction_axis=1, output_axis=2)
        },
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_aurora():
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    grads = {"w": jnp.ones((4, 4)) * 0.1, "b": jnp.ones((4,)) * 0.1}
    tx = aurora(learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert "b" in updates
    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)
