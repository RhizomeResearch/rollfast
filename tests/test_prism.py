import jax.numpy as jnp
from typing import cast

import jax

from rollfast.optim.dimension_numbers import MatrixDimensionNumbers
from rollfast.optim.prism import ScaleByPrismState, scale_by_prism, prism
from tests._typing import as_array_dict


def test_scale_by_prism():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    tx = scale_by_prism()
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert updates["w"].shape == (4, 4)


def test_prism():
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    grads = {"w": jnp.ones((4, 4)) * 0.1, "b": jnp.ones((4,)) * 0.1}
    tx = prism(learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert "b" in updates
    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)


def test_scale_by_prism_accepts_polar_express_coeffs():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    tx = scale_by_prism(ns_coeffs="polar_express", ns_iters=3)
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_scale_by_prism_accepts_preconditioning_selector():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    tx = scale_by_prism(
        ns_iters=2,
        preconditioning="spectral",
        grad_clip_max_amps=None,
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_scale_by_prism_heavy_ball_momentum_accumulator():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    tx = scale_by_prism(
        b1=0.5,
        nesterov=False,
        grad_clip_max_amps=None,
        momentum_accumulator="heavy_ball",
    )
    _, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByPrismState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert jnp.allclose(mu["w"], grads["w"])


def test_scale_by_prism_bf16_momentum_state():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_prism(mu_dtype=jnp.bfloat16, ns_iters=2, grad_clip_max_amps=None)
    _, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByPrismState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert mu["w"].dtype == jnp.bfloat16


def test_scale_by_prism_bidirectional_magma_bf16_raw_clip():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"])}
    tx = scale_by_prism(
        mode="bidirectional",
        inv_steps=2,
        b1=0.0,
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
    state = cast(ScaleByPrismState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert mu["w"].dtype == jnp.bfloat16


def test_scale_by_prism_spike_skip_preserves_momentum_and_magma_state():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"])}
    tx = scale_by_prism(
        raw_global_grad_clip=0.01,
        permissive_spike_protection=False,
        grad_clip_max_amps=None,
        use_magma=True,
        magma_p=1.0,
    )
    state0 = cast(ScaleByPrismState, tx.init(params))
    updates, state1 = tx.update(grads, state0, params)
    state1 = cast(ScaleByPrismState, state1)
    updates = as_array_dict(updates)
    mu0 = cast(dict[str, jax.Array], state0.mu)
    mu1 = cast(dict[str, jax.Array], state1.mu)
    magma0 = cast(dict[str, jax.Array], state0.magma_s)
    magma1 = cast(dict[str, jax.Array], state1.magma_s)

    assert jnp.allclose(updates["w"], jnp.zeros_like(updates["w"]))
    assert jnp.allclose(mu1["w"], mu0["w"])
    assert jnp.allclose(magma1["w"], magma0["w"])


def test_scale_by_prism_magma_weight_decay_mask():
    params = {
        "decay": jnp.ones((4, 4), dtype=jnp.float32),
        "skip": jnp.ones((4, 4), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.zeros_like, params)
    tx = scale_by_prism(
        b1=0.0,
        nesterov=False,
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


def test_prism_magma_wrapper_masks_matrix_weight_decay_and_keeps_adam_finite():
    params = {
        "decay": jnp.ones((4, 4), dtype=jnp.float32),
        "skip": jnp.ones((4, 4), dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.zeros_like, params)
    tx = prism(
        learning_rate=1.0,
        b1=0.0,
        nesterov=False,
        ns_iters=2,
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


def test_scale_by_prism_explicit_high_rank_dimension_spec():
    params = {"w": jnp.ones((2, 3, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_prism(
        ns_iters=2,
        grad_clip_max_amps=None,
        weight_dimension_numbers={
            "w": MatrixDimensionNumbers(reduction_axis=1, output_axis=2)
        },
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_prism_ordered_coeff_schedule_changes_original_path():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    schedule = (
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    )
    default_tx = scale_by_prism(ns_iters=2, grad_clip_max_amps=None)
    scheduled_tx = scale_by_prism(
        ns_iters=2,
        ns_coeffs=schedule,
        grad_clip_max_amps=None,
    )

    default_updates, _ = default_tx.update(grads, default_tx.init(params), params)
    scheduled_updates, _ = scheduled_tx.update(grads, scheduled_tx.init(params), params)
    default_updates = as_array_dict(default_updates)
    scheduled_updates = as_array_dict(scheduled_updates)

    assert not jnp.allclose(default_updates["w"], scheduled_updates["w"])
