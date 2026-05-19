from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
import pytest

import rollfast
import rollfast.optim.hyperball as hyperball_module
from rollfast.optim.hyperball import (
    HyperballState,
    adamw_hyperball,
    apply_hyperball,
    aurora_hyperball,
    hyperball_riemannian_aurora,
    kron_hyperball,
    muon_hyperball,
    prism_hyperball,
    riemannian_aurora_hyperball,
    rmnp_hyperball,
    scale_by_hyperball,
)
from tests._typing import ArrayDict, as_array_dict


@jax.tree_util.register_pytree_node_class
class CallableTree:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("callable PyTree masks must not be invoked")

    def tree_flatten(self):
        return (self.w, self.b), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


def _params() -> ArrayDict:
    return {
        "w": jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }


def _grads() -> ArrayDict:
    return {
        "w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1,
        "b": jnp.ones((4,), dtype=jnp.float32) * 0.1,
    }


def _assert_shapes_and_matrix_norm(
    updates: object, params: ArrayDict
) -> tuple[ArrayDict, ArrayDict]:
    updates = as_array_dict(updates)
    next_params = as_array_dict(optax.apply_updates(params, updates))

    assert "w" in updates
    assert "b" in updates
    assert updates["w"].shape == (4, 4)
    assert updates["b"].shape == (4,)
    assert jnp.allclose(jnp.linalg.norm(next_params["w"]), jnp.linalg.norm(params["w"]))

    return updates, next_params


def test_apply_hyperball_default_mask_preserves_matrix_only():
    params = _params()
    directions = _grads()
    tx = apply_hyperball(learning_rate=0.01)

    state = tx.init(params)
    updates, state = tx.update(directions, state, params)
    updates, next_params = _assert_shapes_and_matrix_norm(updates, params)

    state = cast(HyperballState, state)
    assert int(state.count) == 1
    assert jnp.allclose(updates["b"], -0.01 * directions["b"])
    assert not jnp.allclose(
        jnp.linalg.norm(next_params["b"]), jnp.linalg.norm(params["b"])
    )


def test_apply_hyperball_fallback_weight_decay_is_opt_in():
    params = {"b": jnp.ones((4,), dtype=jnp.float32) * 2.0}
    directions = {"b": jnp.ones((4,), dtype=jnp.float32) * 0.25}
    tx = apply_hyperball(
        learning_rate=0.1,
        weight_decay=0.5,
        hyperball_mask={"b": False},
        fallback_weight_decay=True,
    )

    state = tx.init(params)
    updates, state = tx.update(directions, state, params)
    updates = as_array_dict(updates)

    assert jnp.allclose(updates["b"], -0.1 * (directions["b"] + 0.5 * params["b"]))


def test_apply_hyperball_accepts_callable_pytree_masks():
    params = CallableTree(
        w=jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4),
        b=jnp.ones((4,), dtype=jnp.float32),
    )
    directions = CallableTree(
        w=jnp.ones((4, 4), dtype=jnp.float32) * 0.1,
        b=jnp.ones((4,), dtype=jnp.float32) * 0.25,
    )
    mask = CallableTree(w=True, b=False)
    tx = apply_hyperball(
        learning_rate=0.01,
        weight_decay=0.5,
        weight_decay_mask=mask,
        hyperball_mask=mask,
        fallback_weight_decay=True,
    )

    state = tx.init(cast(Any, params))
    updates, _ = tx.update(cast(Any, directions), state, cast(Any, params))
    updates = cast(CallableTree, updates)

    assert updates.w.shape == (4, 4)
    assert updates.b.shape == (4,)
    assert jnp.allclose(updates.b, -0.01 * directions.b)


def test_apply_hyperball_caution_uses_extra_grad_args():
    params = {"w": jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4)}
    directions = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    raw_grads = {"w": -jnp.ones((4, 4), dtype=jnp.float32)}
    tx = apply_hyperball(
        learning_rate=0.01,
        hyperball_mask={"w": True},
        caution=True,
    )

    state = tx.init(params)
    updates, state = tx.update(directions, state, params, grad=raw_grads)
    updates = as_array_dict(updates)

    assert jnp.allclose(updates["w"], jnp.zeros_like(params["w"]))


def test_apply_hyperball_requires_params():
    params = _params()
    tx = apply_hyperball(learning_rate=0.01)
    state = tx.init(params)

    with pytest.raises(ValueError, match="params"):
        tx.update(_grads(), state)


def test_apply_hyperball_rejects_nonpositive_eps():
    with pytest.raises(ValueError, match="eps"):
        apply_hyperball(learning_rate=0.01, eps=0.0)


def test_kron_hyperball():
    params = _params()
    grads = _grads()
    tx = kron_hyperball(
        learning_rate=0.01,
        preconditioner_update_probability=1.0,
    )

    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    _assert_shapes_and_matrix_norm(updates, params)


def test_kron_hyperball_accepts_extra_grad_args():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1}
    tx = kron_hyperball(
        learning_rate=0.01,
        preconditioner_update_probability=1.0,
        caution=True,
    )

    state = tx.init(params)
    updates, state = tx.update(grads, state, params, grad=grads)
    updates = as_array_dict(updates)

    assert updates["w"].shape == (4, 4)


@pytest.mark.parametrize(
    ("optimizer_fn", "kwargs"),
    [
        (adamw_hyperball, {"learning_rate": 0.01}),
        (
            kron_hyperball,
            {"learning_rate": 0.01, "preconditioner_update_probability": 1.0},
        ),
        (
            muon_hyperball,
            {
                "learning_rate": 0.01,
                "ns_steps": 2,
                "momentum_accumulator": "heavy_ball",
                "use_magma": True,
                "magma_p": 1.0,
            },
        ),
        (
            rmnp_hyperball,
            {"learning_rate": 0.01, "momentum_accumulator": "heavy_ball"},
        ),
        (
            prism_hyperball,
            {
                "learning_rate": 0.01,
                "ns_iters": 2,
                "preconditioning": "spectral",
                "momentum_accumulator": "heavy_ball",
            },
        ),
        (
            aurora_hyperball,
            {
                "learning_rate": 0.01,
                "polar_ns_iters": 2,
                "momentum_accumulator": "heavy_ball",
            },
        ),
        (
            riemannian_aurora_hyperball,
            {
                "learning_rate": 0.01,
                "outer_steps": 1,
                "cg_steps": 2,
                "retraction_steps": 1,
                "polar_ns_iters": 2,
                "momentum_accumulator": "heavy_ball",
            },
        ),
    ],
)
def test_hyperball_optimizer_wrappers(optimizer_fn, kwargs):
    params = _params()
    grads = _grads()
    tx = optimizer_fn(**kwargs)

    state = tx.init(params)
    updates, state = tx.update(grads, state, params)

    _assert_shapes_and_matrix_norm(updates, params)


def test_public_aliases():
    assert scale_by_hyperball is apply_hyperball
    assert hyperball_riemannian_aurora is riemannian_aurora_hyperball
    assert rollfast.apply_hyperball is apply_hyperball
    assert rollfast.scale_by_hyperball is apply_hyperball
    assert rollfast.hyperball_riemannian_aurora is riemannian_aurora_hyperball


def test_muon_hyperball_routes_vectors_to_adam_fallback():
    params = _params()
    grads = _grads()
    tx = muon_hyperball(
        learning_rate=0.01,
        ns_steps=2,
        fallback_weight_decay=True,
        weight_decay=0.1,
    )

    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates, next_params = _assert_shapes_and_matrix_norm(updates, params)

    assert not jnp.allclose(
        jnp.linalg.norm(next_params["b"]), jnp.linalg.norm(params["b"])
    )
    assert jnp.all(jnp.isfinite(updates["b"]))


def test_rmnp_hyperball_routes_vectors_to_adam_fallback():
    params = _params()
    grads = _grads()
    tx = rmnp_hyperball(
        learning_rate=0.01,
        fallback_weight_decay=True,
        weight_decay=0.1,
    )

    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates, next_params = _assert_shapes_and_matrix_norm(updates, params)

    assert not jnp.allclose(
        jnp.linalg.norm(next_params["b"]), jnp.linalg.norm(params["b"])
    )
    assert jnp.all(jnp.isfinite(updates["b"]))


def test_partitioned_hyperball_wrappers_forward_axis_name(monkeypatch):
    captured_axis_names = []

    def fake_apply_hyperball(**kwargs):
        captured_axis_names.append(kwargs["axis_name"])
        return optax.identity()

    monkeypatch.setattr(hyperball_module, "apply_hyperball", fake_apply_hyperball)

    hyperball_module.muon_hyperball(
        learning_rate=0.01,
        ns_steps=2,
        axis_name="devices",
    )
    hyperball_module.rmnp_hyperball(
        learning_rate=0.01,
        axis_name="devices",
    )

    assert captured_axis_names == ["devices", "devices"]


def test_muon_hyperball_forwards_magma_axis_and_key_to_partition_branches(
    monkeypatch,
):
    captured_muon = {}
    captured_adam = {}
    shape_calls = []
    key = jax.random.PRNGKey(7)
    key_muon, key_adam = jax.random.split(key, 2)

    def fake_scale_by_muon(**kwargs):
        captured_muon.update(kwargs)
        return optax.identity()

    def fake_scale_by_adam(**kwargs):
        captured_adam.update(kwargs)
        return optax.identity()

    def fake_scale_by_shape(**kwargs):
        shape_calls.append(kwargs)
        return optax.identity()

    monkeypatch.setattr(
        hyperball_module.optax_muon, "scale_by_muon", fake_scale_by_muon
    )
    monkeypatch.setattr(
        hyperball_module.optax_muon, "scale_by_shape", fake_scale_by_shape
    )
    monkeypatch.setattr(hyperball_module, "scale_by_adam", fake_scale_by_adam)

    hyperball_module.muon_hyperball(
        learning_rate=0.01,
        ns_steps=2,
        consistent_rms=0.2,
        use_magma=True,
        magma_p=0.75,
        magma_tau=1.5,
        axis_name="devices",
        key=key,
    )

    assert captured_muon["use_magma"] is True
    assert captured_muon["shape_updates"] is True
    assert captured_muon["consistent_rms"] == 0.2
    assert captured_muon["magma_p"] == 0.75
    assert captured_muon["magma_tau"] == 1.5
    assert captured_muon["axis_name"] == "devices"
    assert jnp.array_equal(captured_muon["key"], key_muon)
    assert shape_calls == []
    assert captured_adam["use_magma"] is True
    assert captured_adam["magma_p"] == 0.75
    assert captured_adam["magma_tau"] == 1.5
    assert captured_adam["axis_name"] == "devices"
    assert jnp.array_equal(captured_adam["key"], key_adam)


def test_prism_and_aurora_hyperball_forward_nesterov_to_adam_fallback(monkeypatch):
    captured_adam = []

    def fake_scale_by_adam(**kwargs):
        captured_adam.append(kwargs)
        return optax.identity()

    monkeypatch.setattr(hyperball_module, "scale_by_adam", fake_scale_by_adam)
    monkeypatch.setattr(
        hyperball_module, "_build_unscaled_prism_branch", lambda **_: optax.identity()
    )
    monkeypatch.setattr(
        hyperball_module, "_build_unscaled_aurora_branch", lambda **_: optax.identity()
    )

    hyperball_module.prism_hyperball(
        learning_rate=0.01,
        ns_iters=2,
        nesterov=False,
    )
    hyperball_module.aurora_hyperball(
        learning_rate=0.01,
        polar_ns_iters=2,
        nesterov=False,
    )

    assert [kwargs["nesterov"] for kwargs in captured_adam] == [False, False]


def test_muon_hyperball_keeps_external_shape_scaling_without_magma(monkeypatch):
    captured_muon = {}
    shape_calls = []

    def fake_scale_by_muon(**kwargs):
        captured_muon.update(kwargs)
        return optax.identity()

    def fake_scale_by_shape(**kwargs):
        shape_calls.append(kwargs)
        return optax.identity()

    monkeypatch.setattr(
        hyperball_module.optax_muon, "scale_by_muon", fake_scale_by_muon
    )
    monkeypatch.setattr(
        hyperball_module.optax_muon, "scale_by_shape", fake_scale_by_shape
    )

    hyperball_module.muon_hyperball(
        learning_rate=0.01,
        ns_steps=2,
        consistent_rms=0.2,
        use_magma=False,
    )

    assert captured_muon["shape_updates"] is False
    assert captured_muon["consistent_rms"] == 0.2
    assert len(shape_calls) == 1
    assert shape_calls[0]["consistent_rms"] == 0.2
