from typing import cast

import jax
import jax.numpy as jnp
import pytest
from optax.transforms import _masking

from rollfast.utils import apply_updates, apply_updates_prefix, dist_reduce
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


def test_dist_reduce_rejects_unknown_op():
    with pytest.raises(ValueError, match="op must be"):
        dist_reduce(jnp.array(1.0), axis_name="devices", op="median")
