import jax
import jax.numpy as jnp
import pytest
from optax.transforms import _masking

from rollfast.utils import apply_updates, dist_reduce
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


def test_dist_reduce_rejects_unknown_op():
    with pytest.raises(ValueError, match="op must be"):
        dist_reduce(jnp.array(1.0), axis_name="devices", op="median")
