from typing import cast

import jax
import jax.numpy as jnp
import optax
import pytest
from optax.transforms import _masking

from rollfast.utils import (
    _map_non_aux,
    apply_updates,
    apply_updates_prefix,
    dist_reduce,
)
from tests._typing import as_array_dict


def test_apply_updates_skips_none_and_masked_update_leaves():
    masked = _masking.MaskedNode()
    params = {
        "w": jnp.ones((2,), dtype=jnp.float32),
        "none": jnp.ones((2,), dtype=jnp.float32),
        "masked": jnp.ones((2,), dtype=jnp.float32),
    }
    updates = {
        "w": jnp.ones((2,), dtype=jnp.float32),
        "none": None,
        "masked": masked,
    }

    next_params = apply_updates(
        params, updates, key=jax.random.PRNGKey(0), stochastic=False
    )
    next_params = as_array_dict(next_params)

    assert jnp.allclose(next_params["w"], 2.0)
    assert jnp.allclose(next_params["none"], params["none"])
    assert jnp.allclose(next_params["masked"], params["masked"])


def test_apply_updates_allows_missing_key_when_deterministic():
    params = {"w": jnp.ones((2,), dtype=jnp.bfloat16)}
    updates = {"w": jnp.ones((2,), dtype=jnp.float32)}

    next_params = apply_updates(params, updates, stochastic=False)
    next_params = as_array_dict(next_params)

    assert jnp.allclose(next_params["w"], jnp.array([2.0, 2.0], dtype=jnp.bfloat16))


@pytest.mark.parametrize("use_jit", (False, True))
@pytest.mark.parametrize(
    ("param_dtype", "update_dtype"),
    (
        (jnp.float32, jnp.float32),
        (jnp.bfloat16, jnp.bfloat16),
        (jnp.bfloat16, jnp.float32),
        (jnp.float16, jnp.float32),
        (jnp.complex64, jnp.complex64),
    ),
)
def test_deterministic_apply_updates_matches_optax(param_dtype, update_dtype, use_jit):
    values = jnp.asarray([1.0, -2.0, 0.03125], dtype=param_dtype)
    changes = jnp.asarray([2.0**-9, 0.125, -0.01], dtype=update_dtype)
    if jnp.issubdtype(param_dtype, jnp.complexfloating):
        values = values * (1.0 + 0.5j)
        changes = changes * (1.0 - 0.25j)
    params = {"w": values, "nested": (values * 2, None)}
    updates = {"w": changes, "nested": (changes * -2, None)}

    def apply(params, updates):
        return apply_updates(params, updates, stochastic=False)

    actual = (jax.jit(apply) if use_jit else apply)(params, updates)
    reference = jax.jit(optax.apply_updates) if use_jit else optax.apply_updates
    expected = reference(params, updates)

    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert left.dtype == right.dtype
        assert jnp.array_equal(left, right)


def test_apply_updates_prefix_allows_missing_key_when_deterministic():
    params = {"w": jnp.ones((2,), dtype=jnp.bfloat16), "static": "value"}
    updates = {"w": jnp.ones((2,), dtype=jnp.float32), "static": None}

    next_params = apply_updates_prefix(params, updates, stochastic=False)
    next_params = cast(dict[str, object], next_params)

    assert jnp.allclose(
        cast(jax.Array, next_params["w"]),
        jnp.array([2.0, 2.0], dtype=jnp.bfloat16),
    )
    assert next_params["static"] == "value"


def test_apply_updates_prefix_can_skip_entire_subtree():
    params = {
        "train": {
            "w": jnp.ones((2,), dtype=jnp.float32),
            "b": jnp.ones((2,), dtype=jnp.float32) * 3.0,
        },
        "static": {"name": "layer"},
    }
    updates = {
        "train": {
            "w": jnp.ones((2,), dtype=jnp.float32),
            "b": None,
        },
        "static": None,
    }

    next_params = apply_updates_prefix(params, updates, stochastic=False)
    next_params = cast(dict[str, object], next_params)
    train = cast(dict[str, jax.Array], next_params["train"])

    assert jnp.allclose(train["w"], jnp.array([2.0, 2.0], dtype=jnp.float32))
    assert jnp.allclose(train["b"], params["train"]["b"])
    assert next_params["static"] == params["static"]


def test_apply_updates_stochastic_bf16_preserves_sub_ulp_signal():
    params = {"w": jnp.ones((4096,), dtype=jnp.bfloat16)}
    updates = {"w": jnp.full((4096,), 2.0**-9, dtype=jnp.float32)}

    deterministic = apply_updates(params, updates, stochastic=False)
    stochastic = apply_updates(params, updates, key=jax.random.PRNGKey(0))
    deterministic = as_array_dict(deterministic)
    stochastic = as_array_dict(stochastic)

    assert jnp.all(deterministic["w"] == jnp.ones_like(deterministic["w"]))
    assert jnp.any(stochastic["w"] > deterministic["w"])
    assert jnp.mean(stochastic["w"].astype(jnp.float32)) > jnp.mean(
        deterministic["w"].astype(jnp.float32)
    )


def test_dist_reduce_rejects_unknown_op():
    with pytest.raises(ValueError, match="op must be"):
        dist_reduce(jnp.array(1.0), axis_name="devices", op="median")


def test_map_non_aux_skips_none_and_masked_leaves():
    masked = _masking.MaskedNode()
    tree = {
        "w": jnp.ones((2,), dtype=jnp.float32),
        "none": None,
        "masked": masked,
    }

    mapped = _map_non_aux(lambda x: x + 1.0, tree)

    assert jnp.allclose(mapped["w"], jnp.array([2.0, 2.0], dtype=jnp.float32))
    assert mapped["none"] is None
    assert isinstance(mapped["masked"], _masking.MaskedNode)
