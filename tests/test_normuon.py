from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
import pytest

from rollfast.optim.dimension_numbers import MatrixDimensionNumbers
from rollfast.optim.normuon import (
    ScaleByNorMuonState,
    contramuon,
    contranormuon,
    normuon,
    scale_by_normuon,
    scale_by_normuon_shape,
)


def test_scale_by_normuon_tracks_row_second_moment_for_tall_matrix():
    params = {"w": jnp.ones((4, 2), dtype=jnp.float32)}
    grads = {"w": jnp.arange(8, dtype=jnp.float32).reshape(4, 2) + 1.0}
    tx = scale_by_normuon(beta1=0.0, beta2=0.0, nesterov=False, ns_iters=2)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    state = cast(ScaleByNorMuonState, state)
    updates = cast(dict[str, jax.Array], updates)
    nu = cast(dict[str, jax.Array], state.nu)

    assert updates["w"].shape == params["w"].shape
    assert nu["w"].shape == (1, 4, 1)
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_scale_by_normuon_defaults_to_row_second_moment_for_wide_matrix():
    params = {"w": jnp.ones((2, 4), dtype=jnp.float32)}
    grads = {"w": jnp.arange(8, dtype=jnp.float32).reshape(2, 4) + 1.0}
    tx = scale_by_normuon(beta1=0.0, beta2=0.0, nesterov=False, ns_iters=2)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    state = cast(ScaleByNorMuonState, state)
    updates = cast(dict[str, jax.Array], updates)
    nu = cast(dict[str, jax.Array], state.nu)

    assert updates["w"].shape == params["w"].shape
    assert nu["w"].shape == (1, 2, 1)
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_scale_by_normuon_auto_tracks_column_second_moment_for_wide_matrix():
    params = {"w": jnp.ones((2, 4), dtype=jnp.float32)}
    grads = {"w": jnp.arange(8, dtype=jnp.float32).reshape(2, 4) + 1.0}
    tx = scale_by_normuon(
        beta1=0.0,
        beta2=0.0,
        nesterov=False,
        ns_iters=2,
        normalization_axis="auto",
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    state = cast(ScaleByNorMuonState, state)
    updates = cast(dict[str, jax.Array], updates)
    nu = cast(dict[str, jax.Array], state.nu)

    assert updates["w"].shape == params["w"].shape
    assert nu["w"].shape == (1, 1, 4)
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_scale_by_normuon_preserves_muon_update_norm_by_default():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) + 1.0}
    muon_tx = scale_by_normuon(
        beta1=0.0,
        beta2=None,
        nesterov=False,
        ns_iters=2,
    )
    normuon_tx = scale_by_normuon(
        beta1=0.0,
        beta2=0.0,
        nesterov=False,
        ns_iters=2,
    )

    muon_updates, _ = muon_tx.update(grads, muon_tx.init(params), params)
    normuon_updates, _ = normuon_tx.update(grads, normuon_tx.init(params), params)
    muon_updates = cast(dict[str, jax.Array], muon_updates)
    normuon_updates = cast(dict[str, jax.Array], normuon_updates)

    assert jnp.allclose(
        jnp.linalg.norm(normuon_updates["w"]),
        jnp.linalg.norm(muon_updates["w"]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_scale_by_normuon_fixed_rms_rescale():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) + 1.0}
    tx = scale_by_normuon(
        beta1=0.0,
        beta2=0.0,
        nesterov=False,
        ns_iters=2,
        normalization_rescale="fixed_rms",
        normalization_rms=0.2,
    )

    updates, _ = tx.update(grads, tx.init(params), params)
    updates = cast(dict[str, jax.Array], updates)

    assert jnp.allclose(
        jnp.sqrt(jnp.mean(jnp.square(updates["w"]))),
        0.2,
        rtol=1e-5,
        atol=1e-5,
    )


def test_scale_by_normuon_rejects_nonpositive_fixed_rms_scalar():
    with pytest.raises(ValueError, match="normalization_rms"):
        scale_by_normuon(
            normalization_rescale="fixed_rms",
            normalization_rms=0.0,
        )


def test_scale_by_normuon_accepts_preconditioning_selector():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) + 1.0}
    tx = scale_by_normuon(
        beta1=0.0,
        beta2=0.0,
        nesterov=False,
        ns_iters=2,
        preconditioning="spectral",
    )

    updates, _ = tx.update(grads, tx.init(params), params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_scale_by_normuon_heavy_ball_momentum_accumulator():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_normuon(
        beta1=0.5,
        beta2=0.0,
        nesterov=False,
        ns_iters=2,
        momentum_accumulator="heavy_ball",
    )
    state = tx.init(params)
    _, state = tx.update(grads, state, params)
    state = cast(ScaleByNorMuonState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert jnp.allclose(mu["w"], grads["w"])


def test_scale_by_normuon_bf16_momentum_state():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_normuon(beta1=0.0, beta2=0.0, ns_iters=2, mu_dtype=jnp.bfloat16)
    _, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByNorMuonState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert mu["w"].dtype == jnp.bfloat16


def test_scale_by_normuon_rejects_unknown_rescale():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"])}
    tx = scale_by_normuon(
        beta1=0.0,
        beta2=0.0,
        nesterov=False,
        ns_iters=2,
        normalization_rescale=cast(Any, "bad"),
    )

    with pytest.raises(ValueError, match="normalization_rescale"):
        tx.update(grads, tx.init(params), params)


def test_scale_by_normuon_shape_matches_muon_width_scaling():
    updates = {"w": jnp.ones((2, 8), dtype=jnp.float32)}
    tx = scale_by_normuon_shape()
    state = tx.init(updates)
    scaled, state = tx.update(updates, state)
    scaled = cast(dict[str, jax.Array], scaled)

    assert jnp.allclose(scaled["w"], jnp.ones((2, 8)) * 2.0)


def test_contramuon_changes_muon_direction():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.diag(jnp.array([4.0, 3.0, 2.0, 1.0], dtype=jnp.float32))}
    muon_tx = scale_by_normuon(
        beta1=0.0,
        beta2=None,
        nesterov=False,
        ns_iters=2,
        contra_coeff=0.0,
    )
    contra_tx = scale_by_normuon(
        beta1=0.0,
        beta2=None,
        nesterov=False,
        ns_iters=2,
        contra_coeff=1.0,
    )

    muon_updates, _ = muon_tx.update(grads, muon_tx.init(params), params)
    contra_updates, _ = contra_tx.update(grads, contra_tx.init(params), params)
    muon_updates = cast(dict[str, jax.Array], muon_updates)
    contra_updates = cast(dict[str, jax.Array], contra_updates)

    assert not jnp.allclose(muon_updates["w"], contra_updates["w"])
    assert jnp.all(jnp.isfinite(contra_updates["w"]))


def test_contramuon_preserves_pre_contra_update_norm():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.diag(jnp.array([4.0, 3.0, 2.0, 1.0], dtype=jnp.float32))}
    muon_tx = scale_by_normuon(
        beta1=0.0,
        beta2=None,
        nesterov=False,
        ns_iters=2,
        contra_coeff=0.0,
    )
    contra_tx = scale_by_normuon(
        beta1=0.0,
        beta2=None,
        nesterov=False,
        ns_iters=2,
        contra_coeff=0.4,
        contra_power_iters=5,
    )

    muon_updates, _ = muon_tx.update(grads, muon_tx.init(params), params)
    contra_updates, _ = contra_tx.update(grads, contra_tx.init(params), params)
    muon_updates = cast(dict[str, jax.Array], muon_updates)
    contra_updates = cast(dict[str, jax.Array], contra_updates)

    assert jnp.allclose(
        jnp.linalg.norm(muon_updates["w"]),
        jnp.linalg.norm(contra_updates["w"]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_normuon_partitions_vectors_to_adam():
    params = {
        "w": jnp.ones((4, 4), dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = normuon(learning_rate=0.01, ns_iters=2)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.all(jnp.isfinite(updates["b"]))
    assert jnp.all(updates["w"] < 0.0)


def test_contramuon_and_contranormuon_run():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}

    for optimizer in (
        contramuon(learning_rate=0.01, ns_iters=2),
        contranormuon(learning_rate=0.01, ns_iters=2),
    ):
        state = optimizer.init(params)
        updates, state = optimizer.update(grads, state, params)
        updates = cast(dict[str, jax.Array], updates)
        assert updates["w"].shape == params["w"].shape
        assert jnp.all(jnp.isfinite(updates["w"]))


def test_normuon_custom_dimension_numbers_for_conv_kernel():
    params = {"kernel": jnp.ones((8, 3, 3, 4), dtype=jnp.float32)}
    grads = {"kernel": jnp.ones_like(params["kernel"]) * 0.01}
    specs = {
        "kernel": MatrixDimensionNumbers(
            reduction_axis=(1, 2, 3),
            output_axis=0,
        )
    }
    tx = normuon(
        learning_rate=0.01,
        ns_iters=2,
        normuon_weight_dimension_numbers=specs,
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["kernel"].shape == params["kernel"].shape
    assert jnp.all(jnp.isfinite(updates["kernel"]))


def test_normuon_applies_weight_decay_to_matrix_branch():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.zeros_like(params["w"])}
    tx = normuon(
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


def test_normuon_fallback_adam_inherits_weight_decay_by_default():
    params = {
        "w": jnp.ones((2, 2), dtype=jnp.float32),
        "b": jnp.ones((2,), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.zeros_like, params)

    inherit_tx = normuon(
        learning_rate=1.0,
        weight_decay=0.2,
        beta1=0.0,
        beta2=0.0,
        nesterov=False,
        ns_iters=2,
        weight_decay_mask={"w": False, "b": True},
    )
    override_tx = normuon(
        learning_rate=1.0,
        weight_decay=0.2,
        adam_weight_decay=0.0,
        beta1=0.0,
        beta2=0.0,
        nesterov=False,
        ns_iters=2,
        weight_decay_mask={"w": False, "b": True},
    )

    inherit_updates, _ = inherit_tx.update(grads, inherit_tx.init(params), params)
    override_updates, _ = override_tx.update(grads, override_tx.init(params), params)
    inherit_updates = cast(dict[str, jax.Array], inherit_updates)
    override_updates = cast(dict[str, jax.Array], override_updates)

    assert jnp.allclose(inherit_updates["b"], -0.2 * params["b"])
    assert jnp.allclose(override_updates["b"], jnp.zeros_like(params["b"]))


def test_normuon_accepts_polar_express_coeffs():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    tx = normuon(
        learning_rate=0.01,
        ns_coeffs="polar_express",
        ns_iters=3,
        beta1=0.0,
        beta2=0.0,
        nesterov=False,
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))
