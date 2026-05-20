from typing import cast

import jax
import jax.numpy as jnp
import pytest

from rollfast.optim.psgd import KronState, kron, scale_by_kron
from tests._typing import as_array_dict


def test_scale_by_kron():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    # precond_update_prob needs to be 1.0 to ensure update code runs without error
    tx = scale_by_kron(preconditioner_update_probability=1.0)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert updates["w"].shape == (4, 4)


def test_scale_by_kron_init_is_quiet_by_default(capsys):
    params = {"w": jnp.ones((4, 4))}
    tx = scale_by_kron(preconditioner_update_probability=1.0)

    tx.init(params)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_kron():
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    grads = {"w": jnp.ones((4, 4)) * 0.1, "b": jnp.ones((4,)) * 0.1}
    tx = kron(learning_rate=0.01, preconditioner_update_probability=1.0)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates
    assert "b" in updates
    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)


def test_scale_by_kron_magma_weight_decay_accepts_array_mask():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.zeros_like(params["w"])}
    mask = {"w": jnp.array([[True, False], [False, True]])}
    tx = scale_by_kron(
        b1=0.9,
        preconditioner_update_probability=0.0,
        grad_clip_max_amps=(1e9, 1e9),
        use_magma=True,
        magma_p=1.0,
        weight_decay=0.2,
        weight_decay_mask=mask,
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert jnp.all(updates["w"][mask["w"]] > 0)
    assert jnp.allclose(updates["w"][jnp.logical_not(mask["w"])], 0.0)


def test_scale_by_kron_rejects_unknown_preconditioner_mode():
    with pytest.raises(ValueError, match="preconditioner_mode"):
        scale_by_kron(preconditioner_mode="typo")


def test_scale_by_kron_requires_params_for_weight_decay():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.zeros_like(params["w"])}
    tx = scale_by_kron(
        b1=0.0,
        preconditioner_update_probability=0.0,
        grad_clip_max_amps=(1e9, 1e9),
        weight_decay=0.1,
    )

    with pytest.raises(ValueError, match=r"params.*scale_by_kron"):
        tx.update(grads, tx.init(params))


def test_scale_by_kron_rejects_negative_weight_decay():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.zeros_like(params["w"])}
    tx = scale_by_kron(
        b1=0.0,
        preconditioner_update_probability=0.0,
        grad_clip_max_amps=(1e9, 1e9),
        weight_decay=-0.1,
    )

    with pytest.raises(ValueError, match="nonnegative"):
        tx.update(grads, tx.init(params), params)


def test_scale_by_kron_rejects_nonpositive_clip_thresholds():
    with pytest.raises(ValueError, match="raw_global_grad_clip"):
        scale_by_kron(raw_global_grad_clip=0.0)
    with pytest.raises(ValueError, match="grad_clip_max_amps"):
        scale_by_kron(grad_clip_max_amps=(-1.0, 10.0))


def test_scale_by_kron_handles_masked_first_leaf_when_updating_preconditioner():
    params = {
        "a": None,
        "w": jnp.ones((2, 2), dtype=jnp.float32),
    }
    grads = {
        "a": None,
        "w": jnp.ones_like(params["w"]),
    }
    tx = scale_by_kron(
        preconditioner_update_probability=1.0,
        grad_clip_max_amps=(1e9, 1e9),
    )

    updates, _ = tx.update(grads, tx.init(params), params)
    updates = cast(dict[str, object], updates)

    assert updates["a"] is None
    assert jnp.all(jnp.isfinite(cast(jax.Array, updates["w"])))


def test_scale_by_kron_spike_skip_preserves_momentum_and_zeroes_update():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"])}
    tx = scale_by_kron(
        b1=0.9,
        preconditioner_update_probability=1.0,
        raw_global_grad_clip=0.01,
        permissive_spike_protection=False,
        grad_clip_max_amps=(1e9, 1e9),
    )
    state0 = cast(KronState, tx.init(params))
    updates, state1 = tx.update(grads, state0, params)
    updates = as_array_dict(updates)
    state1 = cast(KronState, state1)
    mu = cast(dict[str, jax.Array], state1.mu)

    assert jnp.allclose(updates["w"], jnp.zeros_like(updates["w"]))
    assert jnp.allclose(mu["w"], jnp.zeros_like(mu["w"]))


def test_scale_by_kron_magma_spike_skip_handles_masked_first_leaf():
    params = {
        "a": None,
        "w": jnp.ones((2, 2), dtype=jnp.float32),
    }
    grads = {
        "a": None,
        "w": jnp.ones_like(params["w"]),
    }
    tx = scale_by_kron(
        b1=0.9,
        preconditioner_update_probability=1.0,
        raw_global_grad_clip=0.01,
        permissive_spike_protection=False,
        grad_clip_max_amps=(1e9, 1e9),
        use_magma=True,
        magma_p=1.0,
    )

    updates, state = tx.update(grads, tx.init(params), params)
    updates = cast(dict[str, object], updates)
    state = cast(KronState, state)
    mu = cast(dict[str, object], state.mu)

    assert updates["a"] is None
    assert mu["a"] is None
    assert jnp.allclose(cast(jax.Array, updates["w"]), jnp.zeros((2, 2)))
