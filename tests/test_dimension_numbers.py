from typing import Any, cast

import jax
import jax.numpy as jnp
from optax.transforms import _masking

from rollfast.optim.dimension_numbers import (
    MatrixDimensionNumbers,
    _compute_matrix_reshape,
    _get_dimension_numbers,
    _mask_dimension_numbers,
)
from rollfast.optim.prism import PrismDimensionNumbers


def test_prism_dimension_numbers_is_compatibility_alias():
    assert PrismDimensionNumbers is MatrixDimensionNumbers


def test_get_dimension_numbers_defaults_to_rank_two_leaves():
    params = {
        "matrix": jnp.ones((2, 3)),
        "vector": jnp.ones((3,)),
        "none": None,
    }

    dim_nums = cast(dict[str, Any], _get_dimension_numbers(None, params))

    assert dim_nums["matrix"] == MatrixDimensionNumbers()
    assert dim_nums["vector"] is None
    assert dim_nums["none"] is None


def test_mask_dimension_numbers_replaces_none_leaves():
    dim_nums = {"matrix": MatrixDimensionNumbers(), "vector": None}

    masked = cast(dict[str, Any], _mask_dimension_numbers(dim_nums))

    assert masked["matrix"] == MatrixDimensionNumbers()
    assert isinstance(masked["vector"], _masking.MaskedNode)


def test_compute_matrix_reshape_round_trips_high_rank_tensor():
    x = jnp.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    dim_nums = MatrixDimensionNumbers(reduction_axis=(1, 2), output_axis=3)

    reshape_fn, inverse_fn = _compute_matrix_reshape(x, dim_nums)
    matrix = reshape_fn(x)
    restored = inverse_fn(matrix)

    assert matrix.shape == (2, 12, 5)
    assert jax.numpy.array_equal(restored, x)
