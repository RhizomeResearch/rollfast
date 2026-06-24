import inspect
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
import pytest
from optax._src import base

import rollfast
import rollfast.schedules as schedules
import rollfast.schedules.schedulefree as schedulefree_module
from rollfast.optim.dimension_numbers import MatrixDimensionNumbers
from rollfast.schedules.schedulefree import (
    ScheduleFreeState,
    schedule_free,
    schedule_free_adam,
    schedule_free_aurora,
    schedule_free_eval_params,
    schedule_free_kron,
    schedule_free_prism,
)
from tests._typing import as_array_dict


def test_schedule_free():
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.ones((2, 2)) * 0.1}
    base_opt = optax.sgd(0.01)
    tx = schedule_free(base_opt, learning_rate=0.01)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates

    eval_params = schedule_free_eval_params(state, params)
    eval_params = as_array_dict(eval_params)
    assert "w" in eval_params


def test_public_schedule_free_exports():
    assert rollfast.schedule_free is schedule_free
    assert schedules.schedule_free is schedule_free
    assert "schedule_free" in schedules.__all__
    assert "schedule_free_adam" in schedulefree_module.__all__
    assert "ScheduleFreeState" in schedulefree_module.__all__


def test_schedule_free_plain_base_optimizer_ignores_extra_args():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    tx = schedule_free(optax.sgd(0.01), learning_rate=0.01)

    updates, _ = tx.update(grads, tx.init(params), params, batch_stats={})
    updates = as_array_dict(updates)

    assert updates["w"].shape == params["w"].shape


def test_schedule_free_requires_params_on_update():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    tx = schedule_free(optax.sgd(0.01), learning_rate=0.01)

    with pytest.raises(ValueError, match=r"params.*schedule_free"):
        tx.update(grads, tx.init(params))


def test_schedule_free_does_not_swallow_base_optimizer_type_errors():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones((2, 2), dtype=jnp.float32)}

    def init_fn(params):
        del params
        return base.EmptyState()

    def update_fn(updates, state, params=None, **extra_args):
        del updates, state, params, extra_args
        raise TypeError("inner optimizer bug")

    tx = schedule_free(
        base.GradientTransformationExtraArgs(init_fn, update_fn),
        learning_rate=0.01,
    )

    with pytest.raises(TypeError, match="inner optimizer bug"):
        tx.update(grads, tx.init(params), params)


def test_schedule_free_does_not_swallow_two_arg_schedule_type_errors():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones((2, 2), dtype=jnp.float32)}

    def bad_schedule(count, params):
        del count, params
        raise TypeError("schedule bug")

    tx = schedule_free(optax.sgd(0.01), learning_rate=bad_schedule)

    with pytest.raises(TypeError, match="schedule bug"):
        tx.update(grads, tx.init(params), params)


def test_schedule_free_rejects_array_valued_learning_rate_leaves():
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"])}

    def array_lr(count, params):
        del count, params
        return {"w": jnp.ones((2, 2), dtype=jnp.float32)}

    tx = schedule_free(optax.sgd(0.01), learning_rate=array_lr)

    with pytest.raises(ValueError, match="scalar"):
        tx.update(grads, tx.init(params), params)


def test_schedule_free_prism():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    tx = schedule_free_prism(learning_rate=0.01, total_steps=100)
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates


def test_schedule_free_wrappers_forward_key_to_outer_state():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    key = jax.random.PRNGKey(123)

    for make_tx in (
        lambda: schedule_free_adam(learning_rate=0.01, total_steps=100, key=key),
        lambda: schedule_free_prism(
            learning_rate=0.01,
            total_steps=100,
            ns_iters=2,
            key=key,
        ),
        lambda: schedule_free_kron(
            learning_rate=0.01,
            total_steps=100,
            preconditioner_update_probability=1.0,
            key=key,
        ),
        lambda: schedule_free_aurora(
            learning_rate=0.01,
            total_steps=100,
            polar_ns_iters=2,
            key=key,
        ),
    ):
        state = cast(ScheduleFreeState, make_tx().init(params))
        assert jnp.array_equal(state.key, key)


def test_schedule_free_kron_does_not_expose_mu_dtype():
    assert "mu_dtype" not in inspect.signature(schedule_free_kron).parameters


def test_schedule_free_kron_rejects_negative_weight_decay():
    with pytest.raises(ValueError, match=r"weight_decay.*nonnegative"):
        schedule_free_kron(learning_rate=0.01, total_steps=100, weight_decay=-0.1)


def test_schedule_free_prism_accepts_shared_ns_coeff_schedule():
    params = {"w": jnp.eye(4, dtype=jnp.float32), "b": jnp.ones((4,))}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1, "b": jnp.ones((4,))}
    tx = schedule_free_prism(
        learning_rate=0.01,
        total_steps=100,
        ns_iters=2,
        ns_coeffs=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == params["w"].shape
    assert updates["b"].shape == params["b"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_schedule_free_prism_accepts_preconditioning_selector():
    params = {"w": jnp.eye(4, dtype=jnp.float32), "b": jnp.ones((4,))}
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32) * 0.1, "b": jnp.ones((4,))}
    tx = schedule_free_prism(
        learning_rate=0.01,
        total_steps=100,
        ns_iters=2,
        preconditioning="spectral",
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_schedule_free_prism_keeps_magma_out_of_public_api():
    assert "use_magma" not in inspect.signature(schedule_free_prism).parameters
    with pytest.raises(TypeError, match="use_magma"):
        cast(Any, schedule_free_prism)(
            learning_rate=0.01,
            total_steps=100,
            use_magma=True,
        )


def test_schedule_free_prism_accepts_high_rank_dimension_specs():
    params = {
        "kernel": jnp.ones((2, 3, 4), dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = {
        "kernel": jnp.ones_like(params["kernel"]) * 0.1,
        "b": jnp.ones_like(params["b"]) * 0.1,
    }
    tx = schedule_free_prism(
        learning_rate=0.01,
        total_steps=100,
        ns_iters=2,
        prism_weight_dimension_numbers={
            "kernel": MatrixDimensionNumbers(reduction_axis=1, output_axis=2),
            "b": None,
        },
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["kernel"].shape == params["kernel"].shape
    assert updates["b"].shape == params["b"].shape
    assert jnp.all(jnp.isfinite(updates["kernel"]))
    assert jnp.all(jnp.isfinite(updates["b"]))


def test_schedule_free_aurora():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    tx = schedule_free_aurora(
        learning_rate=0.01,
        total_steps=100,
        polar_ns_iters=2,
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates


def test_schedule_free_aurora_keeps_magma_out_of_public_api():
    assert "use_magma" not in inspect.signature(schedule_free_aurora).parameters
    with pytest.raises(TypeError, match="use_magma"):
        cast(Any, schedule_free_aurora)(
            learning_rate=0.01,
            total_steps=100,
            use_magma=True,
        )


def test_schedule_free_aurora_accepts_high_rank_dimension_specs():
    params = {
        "kernel": jnp.ones((2, 3, 4), dtype=jnp.float32),
        "b": jnp.ones((4,), dtype=jnp.float32),
    }
    grads = {
        "kernel": jnp.ones_like(params["kernel"]) * 0.1,
        "b": jnp.ones_like(params["b"]) * 0.1,
    }
    tx = schedule_free_aurora(
        learning_rate=0.01,
        total_steps=100,
        polar_ns_iters=2,
        aurora_weight_dimension_numbers={
            "kernel": MatrixDimensionNumbers(reduction_axis=1, output_axis=2),
            "b": None,
        },
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = as_array_dict(updates)

    assert updates["kernel"].shape == params["kernel"].shape
    assert updates["b"].shape == params["b"].shape
    assert jnp.all(jnp.isfinite(updates["kernel"]))
    assert jnp.all(jnp.isfinite(updates["b"]))


def test_schedule_free_kron():
    params = {"w": jnp.ones((4, 4))}
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    tx = schedule_free_kron(
        learning_rate=0.01, total_steps=100, preconditioner_update_probability=1.0
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    updates = as_array_dict(updates)
    assert "w" in updates


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    (
        (schedule_free_prism, {"ns_iters": 1}),
        (schedule_free_aurora, {"polar_ns_iters": 1}),
    ),
)
def test_schedule_free_matrix_wrappers_keep_tree_lr_for_adamc(factory, kwargs):
    params = {"matrix": jnp.ones((2, 2)), "vector": jnp.ones((2,))}
    grads = {"matrix": jnp.zeros((2, 2)), "vector": jnp.zeros((2,))}
    tx = factory(
        learning_rate=0.01,
        adam_learning_rate=0.02,
        total_steps=100,
        warmup_fraction=0.0,
        decay_fraction=0.0,
        use_adamc=True,
        weight_decay=0.1,
        grad_clip_max_amps=None,
        **kwargs,
    )
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)

    assert state.scheduled_lr["matrix"] == pytest.approx(0.01)
    assert state.scheduled_lr["vector"] == pytest.approx(0.02)
    assert jnp.allclose(updates["matrix"], -1e-5, rtol=5e-3, atol=2e-8)
    assert jnp.allclose(updates["vector"], -4e-5, rtol=5e-3, atol=2e-8)


def test_schedule_free_kron_records_wsd_lr_for_all_leaves():
    params = {"matrix": jnp.ones((2, 2)), "vector": jnp.ones((2,))}
    grads = {"matrix": jnp.zeros((2, 2)), "vector": jnp.zeros((2,))}
    tx = schedule_free_kron(
        learning_rate=0.01,
        total_steps=100,
        warmup_fraction=0.0,
        decay_fraction=0.0,
        preconditioner_update_probability=1.0,
    )
    state = tx.init(params)
    _, state = tx.update(grads, state, params)

    assert state.scheduled_lr["matrix"] == pytest.approx(0.01)
    assert state.scheduled_lr["vector"] == pytest.approx(0.01)
