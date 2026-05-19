from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from optax.transforms import _masking

from rollfast.optim._matrix_runtime import (
    apply_matrix_post_shape_lookahead,
    finish_matrix_runtime_step,
    init_matrix_magma_state,
    init_matrix_momentum_state,
    prepare_matrix_runtime_step,
)


def test_matrix_runtime_aux_leaves_bypass_shared_state_paths():
    masked = _masking.MaskedNode()
    params = {
        "w": jnp.ones((2, 2), dtype=jnp.float32),
        "masked": masked,
        "none": None,
    }
    grads = {
        "w": jnp.ones((2, 2), dtype=jnp.float32),
        "masked": masked,
        "none": None,
    }
    mu = init_matrix_momentum_state(params, jnp.bfloat16)
    magma_s = init_matrix_magma_state(params, use_magma=True)

    runtime = prepare_matrix_runtime_step(
        grads,
        count=jnp.zeros([], dtype=jnp.int32),
        mu=mu,
        key=jax.random.PRNGKey(0),
        beta=0.9,
        nesterov=True,
        shape_nesterov=True,
        bias_correction=True,
        momentum_accumulator="ema",
        mu_dtype=jnp.bfloat16,
        raw_global_grad_clip=0.5,
        permissive_spike_protection=True,
        use_magma=True,
        axis_name=None,
    )
    shaped = {
        "w": jnp.ones((2, 2), dtype=jnp.float32),
        "masked": masked,
        "none": None,
    }

    updates, new_magma_s = finish_matrix_runtime_step(
        shaped,
        runtime,
        params=params,
        magma_s=magma_s,
        use_magma=True,
        magma_p=1.0,
        magma_tau=2.0,
        weight_decay=0.1,
        weight_decay_mask={"w": True, "masked": True, "none": True},
        grad_clip_max_amps=(0.25, 0.5),
        axis_name=None,
    )

    updates = cast(dict[str, object], updates)
    mu_cast = cast(dict[str, object], runtime.mu_cast)
    new_magma_s = cast(dict[str, object], new_magma_s)

    assert updates["none"] is None
    assert mu_cast["none"] is None
    assert new_magma_s["none"] is None
    assert isinstance(updates["masked"], _masking.MaskedNode)
    assert isinstance(mu_cast["masked"], _masking.MaskedNode)
    assert isinstance(new_magma_s["masked"], _masking.MaskedNode)
    assert jnp.all(jnp.isfinite(cast(jax.Array, updates["w"])))


def test_matrix_runtime_rejects_invalid_momentum_accumulator():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    mu = init_matrix_momentum_state(params, jnp.float32)

    with pytest.raises(ValueError, match="momentum_accumulator"):
        prepare_matrix_runtime_step(
            params,
            count=jnp.zeros([], dtype=jnp.int32),
            mu=mu,
            key=jax.random.PRNGKey(0),
            beta=0.9,
            nesterov=False,
            shape_nesterov=False,
            bias_correction=False,
            momentum_accumulator=cast(Any, "bad"),
            mu_dtype=jnp.float32,
            raw_global_grad_clip=None,
            permissive_spike_protection=True,
            use_magma=False,
            axis_name=None,
        )


def test_matrix_runtime_permissive_raw_global_clip_updates_momentum():
    grads = {"w": jnp.array([[3.0, 4.0]], dtype=jnp.float32)}
    mu = init_matrix_momentum_state(grads, jnp.float32)

    runtime = prepare_matrix_runtime_step(
        grads,
        count=jnp.zeros([], dtype=jnp.int32),
        mu=mu,
        key=jax.random.PRNGKey(0),
        beta=0.0,
        nesterov=False,
        shape_nesterov=False,
        bias_correction=False,
        momentum_accumulator="ema",
        mu_dtype=jnp.float32,
        raw_global_grad_clip=1.0,
        permissive_spike_protection=True,
        use_magma=False,
        axis_name=None,
    )
    effective_updates = cast(dict[str, jax.Array], runtime.effective_updates)
    mu_f32 = cast(dict[str, jax.Array], runtime.mu_f32)

    assert bool(runtime.should_skip) is False
    assert jnp.allclose(effective_updates["w"], grads["w"] / 5.0)
    assert jnp.allclose(mu_f32["w"], grads["w"] / 5.0)


def test_matrix_runtime_bias_corrected_nesterov_target_is_value_level():
    grads = {"w": jnp.ones((2, 2), dtype=jnp.float32) * 2.0}
    mu = init_matrix_momentum_state(grads, jnp.float32)

    runtime = prepare_matrix_runtime_step(
        grads,
        count=jnp.zeros([], dtype=jnp.int32),
        mu=mu,
        key=jax.random.PRNGKey(0),
        beta=0.5,
        nesterov=True,
        shape_nesterov=True,
        bias_correction=True,
        momentum_accumulator="ema",
        mu_dtype=jnp.float32,
        raw_global_grad_clip=None,
        permissive_spike_protection=True,
        use_magma=False,
        axis_name=None,
    )
    mu_f32 = cast(dict[str, jax.Array], runtime.mu_f32)
    target_for_shape = cast(dict[str, jax.Array], runtime.target_for_shape)

    assert jnp.allclose(mu_f32["w"], jnp.ones_like(grads["w"]))
    assert jnp.allclose(target_for_shape["w"], jnp.ones_like(grads["w"]) * (8.0 / 3.0))


def test_matrix_runtime_post_shape_lookahead_uses_effective_updates():
    grads = {"w": jnp.ones((2, 2), dtype=jnp.float32) * 2.0}
    mu = init_matrix_momentum_state(grads, jnp.float32)

    runtime = prepare_matrix_runtime_step(
        grads,
        count=jnp.zeros([], dtype=jnp.int32),
        mu=mu,
        key=jax.random.PRNGKey(0),
        beta=0.5,
        nesterov=False,
        shape_nesterov=False,
        bias_correction=False,
        momentum_accumulator="ema",
        mu_dtype=jnp.float32,
        raw_global_grad_clip=None,
        permissive_spike_protection=True,
        use_magma=False,
        axis_name=None,
    )
    shaped = {"w": jnp.ones((2, 2), dtype=jnp.float32) * 10.0}

    looked_ahead = apply_matrix_post_shape_lookahead(
        shaped,
        runtime,
        beta=0.5,
        nesterov=True,
        shape_nesterov=False,
        momentum_accumulator="ema",
    )
    looked_ahead = cast(dict[str, jax.Array], looked_ahead)

    assert jnp.allclose(looked_ahead["w"], jnp.ones_like(grads["w"]) * 6.0)


def test_matrix_runtime_heavy_ball_nesterov_target():
    grads = {"w": jnp.ones((2, 2), dtype=jnp.float32) * 2.0}
    mu = init_matrix_momentum_state(grads, jnp.float32)

    runtime = prepare_matrix_runtime_step(
        grads,
        count=jnp.zeros([], dtype=jnp.int32),
        mu=mu,
        key=jax.random.PRNGKey(0),
        beta=0.5,
        nesterov=True,
        shape_nesterov=True,
        bias_correction=True,
        momentum_accumulator="heavy_ball",
        mu_dtype=jnp.float32,
        raw_global_grad_clip=None,
        permissive_spike_protection=True,
        use_magma=False,
        axis_name=None,
    )
    mu_f32 = cast(dict[str, jax.Array], runtime.mu_f32)
    target_for_shape = cast(dict[str, jax.Array], runtime.target_for_shape)

    assert jnp.allclose(mu_f32["w"], grads["w"])
    assert jnp.allclose(target_for_shape["w"], jnp.ones_like(grads["w"]) * 3.0)


def test_matrix_runtime_scheduled_weight_decay_uses_pre_increment_count():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = jax.tree.map(jnp.zeros_like, params)
    mu = init_matrix_momentum_state(params, jnp.float32)
    runtime = prepare_matrix_runtime_step(
        grads,
        count=jnp.zeros([], dtype=jnp.int32),
        mu=mu,
        key=jax.random.PRNGKey(0),
        beta=0.0,
        nesterov=False,
        shape_nesterov=False,
        bias_correction=False,
        momentum_accumulator="ema",
        mu_dtype=jnp.float32,
        raw_global_grad_clip=None,
        permissive_spike_protection=True,
        use_magma=False,
        axis_name=None,
    )

    updates, _ = finish_matrix_runtime_step(
        grads,
        runtime,
        params=params,
        magma_s=(),
        use_magma=False,
        magma_p=1.0,
        magma_tau=2.0,
        weight_decay=lambda count: jnp.where(count == 0, 0.2, 0.9),
        weight_decay_mask=None,
        grad_clip_max_amps=None,
        axis_name=None,
    )
    updates = cast(dict[str, jax.Array], updates)

    assert jnp.allclose(updates["w"], jnp.ones_like(params["w"]) * 0.2)


def test_matrix_runtime_weight_decay_accepts_array_mask():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = jax.tree.map(jnp.zeros_like, params)
    mu = init_matrix_momentum_state(params, jnp.float32)
    runtime = prepare_matrix_runtime_step(
        grads,
        count=jnp.zeros([], dtype=jnp.int32),
        mu=mu,
        key=jax.random.PRNGKey(0),
        beta=0.0,
        nesterov=False,
        shape_nesterov=False,
        bias_correction=False,
        momentum_accumulator="ema",
        mu_dtype=jnp.float32,
        raw_global_grad_clip=None,
        permissive_spike_protection=True,
        use_magma=False,
        axis_name=None,
    )

    updates, _ = finish_matrix_runtime_step(
        grads,
        runtime,
        params=params,
        magma_s=(),
        use_magma=False,
        magma_p=1.0,
        magma_tau=2.0,
        weight_decay=0.2,
        weight_decay_mask={"w": jnp.array([[True, False], [False, True]])},
        grad_clip_max_amps=None,
        axis_name=None,
    )
    updates = cast(dict[str, jax.Array], updates)

    expected = jnp.array([[0.2, 0.0], [0.0, 0.2]], dtype=jnp.float32)
    assert jnp.allclose(updates["w"], expected)


def test_matrix_runtime_axis_name_global_clip_smoke():
    def run(grad_leaf):
        grads = {"w": grad_leaf}
        mu = init_matrix_momentum_state(grads, jnp.float32)
        runtime = prepare_matrix_runtime_step(
            grads,
            count=jnp.zeros([], dtype=jnp.int32),
            mu=mu,
            key=jax.random.PRNGKey(0),
            beta=0.0,
            nesterov=False,
            shape_nesterov=False,
            bias_correction=False,
            momentum_accumulator="ema",
            mu_dtype=jnp.float32,
            raw_global_grad_clip=1.0,
            permissive_spike_protection=True,
            use_magma=False,
            axis_name="devices",
        )
        effective_updates = cast(dict[str, jax.Array], runtime.effective_updates)
        return effective_updates["w"]

    run = jax.pmap(run, axis_name="devices")
    grads = jnp.ones((jax.local_device_count(), 2, 2), dtype=jnp.float32)
    clipped = run(grads)
    expected_scale = 1.0 / (
        2.0 * jnp.sqrt(jnp.asarray(jax.local_device_count(), dtype=jnp.float32))
    )

    assert jnp.allclose(clipped, grads * expected_scale)
