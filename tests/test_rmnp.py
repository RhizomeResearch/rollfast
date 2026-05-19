from typing import cast

import jax
import jax.numpy as jnp
import optax
import pytest

from rollfast.optim.dimension_numbers import MatrixDimensionNumbers
from rollfast.optim.rmnp import (
    ScaleByRmnpState,
    rmnp,
    scale_by_rmnp,
    scale_by_rmnp_shape,
)


def test_scale_by_rmnp_row_normalizes_matrix_momentum():
    params = {"w": jnp.ones((2, 3), dtype=jnp.float32)}
    grads = {
        "w": jnp.array(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]],
            dtype=jnp.float32,
        )
    }
    tx = scale_by_rmnp(beta=0.0, nesterov=False)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.allclose(jnp.linalg.norm(updates["w"], axis=1), jnp.ones((2,)))


def test_scale_by_rmnp_shape_matches_muon_width_scaling():
    updates = {"w": jnp.ones((2, 8), dtype=jnp.float32)}
    tx = scale_by_rmnp_shape()
    state = tx.init(updates)
    scaled, state = tx.update(updates, state)
    scaled = cast(dict[str, jax.Array], scaled)

    assert jnp.allclose(scaled["w"], jnp.ones((2, 8)) * 2.0)


def test_scale_by_rmnp_rejects_direct_fallback_leaves():
    params = {
        "w": jnp.ones((2, 2), dtype=jnp.float32),
        "b": jnp.ones((2,), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.ones_like, params)
    tx = scale_by_rmnp(beta=0.0, nesterov=False)

    with pytest.raises(ValueError, match="scale_by_rmnp.*matrix dimension specs"):
        tx.update(grads, tx.init(params), params)


def test_scale_by_rmnp_heavy_ball_momentum_accumulator():
    params = {"w": jnp.ones((2, 3), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_rmnp(
        beta=0.5,
        nesterov=False,
        momentum_accumulator="heavy_ball",
    )
    state = tx.init(params)
    _, state = tx.update(grads, state, params)
    state = cast(ScaleByRmnpState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert jnp.allclose(mu["w"], grads["w"])


def test_scale_by_rmnp_bf16_momentum_state():
    params = {"w": jnp.ones((2, 3), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_rmnp(beta=0.0, nesterov=False, mu_dtype=jnp.bfloat16)
    _, state = tx.update(grads, tx.init(params), params)
    state = cast(ScaleByRmnpState, state)
    mu = cast(dict[str, jax.Array], state.mu)

    assert mu["w"].dtype == jnp.bfloat16


def test_rmnp_partitions_vectors_to_adam():
    params = {
        "w": jnp.ones((4, 4), dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    tx = rmnp(learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.all(jnp.isfinite(updates["b"]))
    assert jnp.all(updates["w"] < 0.0)


def test_rmnp_custom_dimension_numbers_for_conv_kernel():
    params = {"kernel": jnp.ones((8, 3, 3, 4), dtype=jnp.float32)}
    grads = {"kernel": jnp.ones_like(params["kernel"]) * 0.01}
    specs = {
        "kernel": MatrixDimensionNumbers(
            reduction_axis=(1, 2, 3),
            output_axis=0,
        )
    }
    tx = rmnp(learning_rate=0.01, rmnp_weight_dimension_numbers=specs)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = cast(dict[str, jax.Array], updates)

    assert updates["kernel"].shape == params["kernel"].shape
    assert jnp.all(jnp.isfinite(updates["kernel"]))


def test_rmnp_applies_weight_decay_to_matrix_branch():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.zeros_like(params["w"])}
    tx = rmnp(learning_rate=0.1, weight_decay=0.2, beta=0.0, nesterov=False)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    next_params = cast(dict[str, jax.Array], optax.apply_updates(params, updates))

    assert jnp.all(next_params["w"] < params["w"])


def test_rmnp_fallback_adam_inherits_weight_decay_by_default():
    params = {
        "w": jnp.ones((2, 2), dtype=jnp.float32),
        "b": jnp.ones((2,), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.zeros_like, params)

    inherit_tx = rmnp(
        learning_rate=1.0,
        weight_decay=0.2,
        beta=0.0,
        nesterov=False,
        weight_decay_mask={"w": False, "b": True},
    )
    override_tx = rmnp(
        learning_rate=1.0,
        weight_decay=0.2,
        adam_weight_decay=0.0,
        beta=0.0,
        nesterov=False,
        weight_decay_mask={"w": False, "b": True},
    )

    inherit_updates, _ = inherit_tx.update(grads, inherit_tx.init(params), params)
    override_updates, _ = override_tx.update(grads, override_tx.init(params), params)
    inherit_updates = cast(dict[str, jax.Array], inherit_updates)
    override_updates = cast(dict[str, jax.Array], override_updates)

    assert jnp.allclose(inherit_updates["b"], -0.2 * params["b"])
    assert jnp.allclose(override_updates["b"], jnp.zeros_like(params["b"]))
