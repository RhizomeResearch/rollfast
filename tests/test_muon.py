from typing import cast

import jax
import jax.numpy as jnp
import pytest

from rollfast.optim.muon import (
    MuonDimensionNumbers,
    MuonState,
    muon,
    orthogonalize_via_newton_schulz,
    polar_express_coeffs,
    scale_by_muon,
    scale_by_muon_shape,
)
from tests._typing import as_array_dict


def _params():
    return {
        "w": jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4) / 10.0,
        "b": jnp.ones((4,), dtype=jnp.float32),
    }


def _grads(params):
    return jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)


def test_scale_by_muon_orthogonalizes_isotropic_matrix_direction():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.eye(4, dtype=jnp.float32)}
    tx = scale_by_muon(
        beta=0.0,
        nesterov=False,
        ns_steps=5,
        weight_dimension_numbers={"w": MuonDimensionNumbers()},
    )

    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)
    singular_values = jnp.linalg.svd(updates["w"], compute_uv=False)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.allclose(singular_values, singular_values[0], rtol=1e-6, atol=1e-6)


def test_muon_partitions_vectors_to_rollfast_adamw():
    params = _params()
    grads = _grads(params)
    tx = muon(learning_rate=0.01, ns_steps=2)
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.all(jnp.isfinite(updates["b"]))
    assert jnp.all(updates["w"] < 0.0)


def test_muon_dimension_numbers_for_conv_kernel():
    params = {"kernel": jnp.ones((8, 3, 3, 4), dtype=jnp.float32)}
    grads = {"kernel": jnp.ones_like(params["kernel"]) * 0.01}
    specs = {
        "kernel": MuonDimensionNumbers(
            reduction_axis=(1, 2, 3),
            output_axis=0,
        )
    }
    tx = muon(
        learning_rate=0.01,
        ns_steps=2,
        muon_weight_dimension_numbers=specs,
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["kernel"].shape == params["kernel"].shape
    assert jnp.all(jnp.isfinite(updates["kernel"]))


def test_scale_by_muon_shape_matches_width_and_consistent_rms_scaling():
    updates = {"w": jnp.ones((2, 8), dtype=jnp.float32)}
    scaled, _ = scale_by_muon_shape().update(
        updates, scale_by_muon_shape().init(updates)
    )
    scaled = as_array_dict(scaled)
    assert jnp.allclose(scaled["w"], 2.0)

    tx = scale_by_muon_shape(consistent_rms=0.2)
    rms_scaled, _ = tx.update(updates, tx.init(updates))
    rms_scaled = as_array_dict(rms_scaled)
    assert jnp.allclose(rms_scaled["w"], jnp.sqrt(8.0) * 0.2)


def test_scale_by_muon_shape_callable_spec_requires_params():
    updates = {"w": jnp.ones((2, 8), dtype=jnp.float32)}
    tx = scale_by_muon_shape(weight_dimension_numbers=lambda params: params)

    with pytest.raises(ValueError, match="scale_by_muon_shape"):
        tx.update(updates, tx.init(updates))


def test_scale_by_muon_rejects_direct_fallback_leaves():
    params = {
        "w": jnp.ones((2, 2), dtype=jnp.float32),
        "b": jnp.ones((2,), dtype=jnp.float32),
    }
    grads = _grads(params)
    tx = scale_by_muon(ns_steps=2)

    with pytest.raises(ValueError, match="scale_by_muon.*matrix dimension specs"):
        tx.update(grads, tx.init(params), params)


def test_muon_bf16_momentum_storage_uses_bf16_state_with_finite_updates():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    tx = scale_by_muon(mu_dtype=jnp.bfloat16, ns_steps=2)
    updates, state = tx.update(grads, tx.init(params), params)
    state = cast(MuonState, state)
    updates = as_array_dict(updates)
    mu_state = as_array_dict(state.mu)

    assert mu_state["w"].dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_muon_magma_smoke_changes_update_with_p_one():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    base_tx = scale_by_muon(beta=0.0, nesterov=False, ns_steps=2)
    magma_tx = scale_by_muon(
        beta=0.0,
        nesterov=False,
        ns_steps=2,
        use_magma=True,
        magma_p=1.0,
    )

    base_updates, _ = base_tx.update(grads, base_tx.init(params), params)
    magma_updates, _ = magma_tx.update(grads, magma_tx.init(params), params)
    base_updates = as_array_dict(base_updates)
    magma_updates = as_array_dict(magma_updates)

    assert magma_updates["w"].shape == base_updates["w"].shape
    assert jnp.all(jnp.isfinite(magma_updates["w"]))
    assert not jnp.allclose(magma_updates["w"], base_updates["w"])


def test_scale_by_muon_magma_weight_decay_mask():
    params = {
        "decay": jnp.ones((4, 4), dtype=jnp.float32),
        "skip": jnp.ones((4, 4), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.zeros_like, params)
    tx = scale_by_muon(
        beta=0.0,
        nesterov=False,
        ns_steps=2,
        use_magma=True,
        magma_p=1.0,
        weight_decay=0.2,
        weight_decay_mask={"decay": True, "skip": False},
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert jnp.linalg.norm(updates["decay"]) > 0
    assert jnp.allclose(updates["skip"], jnp.zeros_like(updates["skip"]))


def test_scale_by_muon_magma_scheduled_weight_decay_uses_pre_increment_count():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = jax.tree.map(jnp.zeros_like, params)
    tx = scale_by_muon(
        beta=0.0,
        nesterov=False,
        ns_steps=2,
        use_magma=True,
        magma_p=1.0,
        weight_decay=lambda count: jnp.where(count == 0, 0.2, 0.9),
    )

    updates, state = tx.update(grads, tx.init(params), params)
    next_updates, _ = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    next_updates = as_array_dict(next_updates)

    assert jnp.allclose(updates["w"], jnp.ones_like(params["w"]) * 0.1)
    assert jnp.allclose(next_updates["w"], jnp.ones_like(params["w"]) * 0.45)


def test_scale_by_muon_requires_params_for_magma_weight_decay():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = jax.tree.map(jnp.zeros_like, params)
    tx = scale_by_muon(use_magma=True, weight_decay=0.1, ns_steps=2)

    with pytest.raises(ValueError, match="params.*scale_by_muon"):
        tx.update(grads, tx.init(params))


def test_muon_magma_wrapper_masks_matrix_and_adam_fallback_weight_decay():
    params = {
        "w": jnp.ones((4, 4), dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.zeros_like, params)
    tx = muon(
        learning_rate=1.0,
        ns_steps=2,
        beta=0.0,
        nesterov=False,
        use_magma=True,
        magma_p=0.0,
        weight_decay=0.2,
        adam_weight_decay=0.2,
        weight_decay_mask={"w": True, "b": True},
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert jnp.allclose(updates["w"], jnp.zeros_like(updates["w"]))
    assert jnp.allclose(updates["b"], jnp.zeros_like(updates["b"]))


def test_muon_fallback_adam_inherits_weight_decay_by_default():
    params = {
        "w": jnp.ones((4, 4), dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.zeros_like, params)

    inherit_tx = muon(
        learning_rate=1.0,
        ns_steps=2,
        beta=0.0,
        nesterov=False,
        weight_decay=0.2,
        weight_decay_mask={"w": False, "b": True},
    )
    override_tx = muon(
        learning_rate=1.0,
        ns_steps=2,
        beta=0.0,
        nesterov=False,
        weight_decay=0.2,
        adam_weight_decay=0.0,
        weight_decay_mask={"w": False, "b": True},
    )

    inherit_updates, _ = inherit_tx.update(grads, inherit_tx.init(params), params)
    override_updates, _ = override_tx.update(grads, override_tx.init(params), params)
    inherit_updates = as_array_dict(inherit_updates)
    override_updates = as_array_dict(override_updates)

    assert jnp.allclose(inherit_updates["b"], -0.2 * params["b"])
    assert jnp.allclose(override_updates["b"], jnp.zeros_like(params["b"]))


def test_muon_momentum_accumulator_modes_differ_after_first_step():
    params = {"w": jnp.eye(4, dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    ema_tx = scale_by_muon(beta=0.5, nesterov=False, ns_steps=2)
    heavy_tx = scale_by_muon(
        beta=0.5,
        nesterov=False,
        ns_steps=2,
        momentum_accumulator="heavy_ball",
    )

    ema_state = ema_tx.init(params)
    heavy_state = heavy_tx.init(params)
    _, ema_state = ema_tx.update(grads, ema_state, params)
    _, heavy_state = heavy_tx.update(grads, heavy_state, params)
    _, ema_state = ema_tx.update(grads, ema_state, params)
    _, heavy_state = heavy_tx.update(grads, heavy_state, params)

    ema_state = cast(MuonState, ema_state)
    heavy_state = cast(MuonState, heavy_state)
    ema_mu = as_array_dict(ema_state.mu)
    heavy_mu = as_array_dict(heavy_state.mu)
    assert not jnp.allclose(ema_mu["w"], heavy_mu["w"])


def test_polar_express_coeffs_match_pr_1613_reference_without_safety():
    expected = [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
        (1.875, -1.25, 0.375),
    ]
    computed = polar_express_coeffs(
        l=1e-3,
        num_iters=8,
        safety_factor_eps=0.0,
        cushion=0.02407327424182761,
    )
    assert jnp.allclose(jnp.asarray(computed), jnp.asarray(expected), rtol=1e-8)


def test_two_dimensional_coefficients_are_used_in_order():
    coeffs = ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    captured = {}

    def hook(x, ns_coeffs, ns_steps, preconditioning, eps, dimension_numbers):
        del x, ns_steps, preconditioning, eps, dimension_numbers
        captured["coeffs"] = ns_coeffs
        return jnp.ones((2, 2), dtype=jnp.float32)

    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    tx = scale_by_muon(ns_coeffs=coeffs, ns_steps=2, orthogonalize_fn=hook)
    tx.update(grads, tx.init(params), params)

    assert jnp.allclose(
        captured["coeffs"], jnp.asarray(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    )


def test_polar_express_preset_runs():
    params = {"w": jnp.eye(8, dtype=jnp.float32)}
    grads = {"w": jnp.eye(8, dtype=jnp.float32)}
    tx = muon(learning_rate=0.1, ns_coeffs="polar_express", ns_steps=4)
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_orthogonalize_schatten_mode_runs():
    mat = jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4)
    out = orthogonalize_via_newton_schulz(
        mat,
        jnp.asarray(polar_express_coeffs(num_iters=2)),
        ns_steps=2,
        preconditioning="schatten",
        dimension_numbers=MuonDimensionNumbers(),
    )
    assert out.shape == mat.shape
    assert jnp.all(jnp.isfinite(out))


def test_orthogonalize_accepts_plain_2d_matrix_without_dimension_numbers():
    mat = jnp.eye(4, dtype=jnp.float32) * 2.0
    out = orthogonalize_via_newton_schulz(
        mat,
        jnp.asarray((3.4445, -4.7750, 2.0315)),
        ns_steps=2,
    )

    assert out.shape == mat.shape
    assert not jnp.allclose(out, mat)
