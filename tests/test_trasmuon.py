from typing import cast

import jax
import jax.numpy as jnp
import optax

from rollfast.optim.dimension_numbers import MatrixDimensionNumbers
from rollfast.optim.trasmuon import ScaleByTrasMuonState, scale_by_trasmuon, trasmuon


def test_scale_by_trasmuon_calibrates_matrix_update_norm():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) + 1.0}
    tx = scale_by_trasmuon(
        beta1=0.0,
        beta2=0.0,
        clip_beta=0.0,
        trigger=1e6,
        ns_iters=2,
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.allclose(jnp.linalg.norm(updates["w"]), 4.0, rtol=1e-5, atol=1e-5)


def test_scale_by_trasmuon_damps_high_energy_columns():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {
        "w": jnp.array(
            [
                [100.0, 1.0, 1.0, 1.0],
                [100.0, 1.0, 1.0, 1.0],
                [100.0, 1.0, 1.0, 1.0],
                [100.0, 1.0, 1.0, 1.0],
            ],
            dtype=jnp.float32,
        )
    }
    tx = scale_by_trasmuon(
        beta1=0.0,
        beta2=0.0,
        clip_alpha=10.0,
        clip_beta=0.0,
        clip_min=0.05,
        trigger=1.0,
        ns_iters=2,
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    state = cast(ScaleByTrasMuonState, state)
    updates = cast(dict[str, jax.Array], updates)

    clip_last = cast(dict[str, jax.Array], state.clip_last)
    assert clip_last["w"][0, 0, 0] < clip_last["w"][0, 0, 1]


def test_scale_by_trasmuon_accepts_preconditioning_selector():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) + 1.0}
    tx = scale_by_trasmuon(
        beta1=0.0,
        beta2=0.0,
        ns_iters=2,
        preconditioning="spectral",
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_scale_by_trasmuon_heavy_ball_momentum_accumulator():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_trasmuon(
        beta1=0.5,
        beta2=0.0,
        ns_iters=2,
        momentum_accumulator="heavy_ball",
    )
    state = tx.init(params)
    _, state = tx.update(grads, state, params)
    state = cast(ScaleByTrasMuonState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert jnp.allclose(mu["w"], grads["w"])


def test_scale_by_trasmuon_bf16_momentum_state():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_trasmuon(
        beta1=0.0,
        beta2=0.0,
        ns_iters=2,
        mu_dtype=jnp.bfloat16,
    )
    _, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByTrasMuonState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert mu["w"].dtype == jnp.bfloat16


def test_trasmuon_partitions_vectors_to_adam():
    params = {
        "w": jnp.ones((4, 4), dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = trasmuon(learning_rate=0.01, ns_iters=2)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.all(jnp.isfinite(updates["b"]))
    assert jnp.all(updates["w"] < 0.0)


def test_trasmuon_custom_dimension_numbers_for_conv_kernel():
    params = {"kernel": jnp.ones((8, 3, 3, 4), dtype=jnp.float32)}
    grads = {"kernel": jnp.ones_like(params["kernel"]) * 0.01}
    specs = {
        "kernel": MatrixDimensionNumbers(
            reduction_axis=(1, 2, 3),
            output_axis=0,
        )
    }
    tx = trasmuon(
        learning_rate=0.01,
        ns_iters=2,
        trasmuon_weight_dimension_numbers=specs,
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["kernel"].shape == params["kernel"].shape
    assert jnp.all(jnp.isfinite(updates["kernel"]))


def test_trasmuon_applies_weight_decay_to_matrix_branch():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.zeros_like(params["w"])}
    tx = trasmuon(
        learning_rate=0.1,
        weight_decay=0.2,
        beta1=0.0,
        beta2=0.0,
        ns_iters=2,
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    next_params = cast(dict[str, jax.Array], optax.apply_updates(params, updates))

    assert jnp.all(next_params["w"] < params["w"])


def test_trasmuon_accepts_ordered_coeff_schedule():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    coeffs = (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
    )
    tx = scale_by_trasmuon(
        beta1=0.0,
        beta2=0.0,
        ns_iters=2,
        ns_coeffs=coeffs,
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))
