from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
import pytest

from rollfast.optim.hyperball import (
    HyperballState,
    adamw_hyperball,
    apply_hyperball,
    aurora_hyperball,
    hyperball_riemannian_aurora,
    kron_hyperball,
    prism_hyperball,
    riemannian_aurora_hyperball,
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
        (prism_hyperball, {"learning_rate": 0.01, "ns_iters": 2}),
        (aurora_hyperball, {"learning_rate": 0.01, "polar_ns_iters": 2}),
        (
            riemannian_aurora_hyperball,
            {
                "learning_rate": 0.01,
                "outer_steps": 1,
                "cg_steps": 2,
                "retraction_steps": 1,
                "polar_ns_iters": 2,
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
