from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from optax.transforms import _masking

import rollfast
from rollfast.optim.dimension_numbers import (
    MatrixDimensionNumbers,
    _compute_matrix_reshape,
    _get_dimension_numbers,
    _make_matrix_partition_fns,
    _mask_dimension_numbers,
)
from rollfast.optim.prism import PrismDimensionNumbers
from rollfast.optim.prism import scale_by_prism
from rollfast.optim.aurora import AuroraDimensionNumbers
from rollfast.optim.aurora import scale_by_aurora, scale_by_riemannian_aurora
from rollfast.optim.muon import scale_by_muon


def test_prism_dimension_numbers_is_compatibility_alias():
    assert PrismDimensionNumbers is MatrixDimensionNumbers


def test_public_matrix_dimension_aliases_are_exported():
    assert AuroraDimensionNumbers is MatrixDimensionNumbers
    assert rollfast.PrismDimensionNumbers is MatrixDimensionNumbers
    assert rollfast.AuroraDimensionNumbers is MatrixDimensionNumbers
    assert rollfast.scale_by_prism is scale_by_prism
    assert rollfast.scale_by_aurora is scale_by_aurora
    assert rollfast.scale_by_riemannian_aurora is scale_by_riemannian_aurora


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


def test_matrix_partition_helper_builds_labels_masked_specs_and_default_mask():
    params = {
        "matrix": jnp.ones((2, 3)),
        "vector": jnp.ones((3,)),
        "none": None,
    }
    partition = _make_matrix_partition_fns(None, "matrix_opt")

    labels = cast(dict[str, Any], partition.labels(params))
    masked_specs = cast(dict[str, Any], partition.masked_specs(params))
    default_mask = cast(dict[str, Any], partition.default_mask(params))

    assert labels == {
        "matrix": "matrix_opt",
        "vector": "adam",
        "none": None,
    }
    assert masked_specs["matrix"] == MatrixDimensionNumbers()
    assert isinstance(masked_specs["vector"], _masking.MaskedNode)
    assert default_mask == {"matrix": True, "vector": False, "none": False}


def test_matrix_partition_helper_routes_masked_specs_to_fallback():
    params = {
        "matrix": jnp.ones((2, 3)),
        "vector": jnp.ones((3,)),
    }
    partition = _make_matrix_partition_fns(
        {
            "matrix": _masking.MaskedNode(),
            "vector": None,
        },
        "matrix_opt",
    )

    labels = cast(dict[str, Any], partition.labels(params))
    default_mask = cast(dict[str, Any], partition.default_mask(params))

    assert labels == {"matrix": "adam", "vector": "adam"}
    assert default_mask == {"matrix": False, "vector": False}


def test_compute_matrix_reshape_round_trips_high_rank_tensor():
    x = jnp.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    dim_nums = MatrixDimensionNumbers(reduction_axis=(1, 2), output_axis=3)

    reshape_fn, inverse_fn = _compute_matrix_reshape(x, dim_nums)
    matrix = reshape_fn(x)
    restored = inverse_fn(matrix)

    assert matrix.shape == (2, 12, 5)
    assert jax.numpy.array_equal(restored, x)


@pytest.mark.parametrize(
    ("dim_nums", "match"),
    [
        (
            MatrixDimensionNumbers(reduction_axis=99, output_axis=1),
            "out of bounds",
        ),
        (
            MatrixDimensionNumbers(reduction_axis=-4, output_axis=1),
            "out of bounds",
        ),
        (
            MatrixDimensionNumbers(reduction_axis=(0, -3), output_axis=1),
            "duplicate",
        ),
        (
            MatrixDimensionNumbers(reduction_axis=0, output_axis=(1, -2)),
            "duplicate",
        ),
        (
            MatrixDimensionNumbers(reduction_axis=0, output_axis=-3),
            "disjoint",
        ),
        (
            MatrixDimensionNumbers(reduction_axis=(), output_axis=1),
            "must not be empty",
        ),
        (
            MatrixDimensionNumbers(reduction_axis=0, output_axis=()),
            "must not be empty",
        ),
    ],
)
def test_compute_matrix_reshape_rejects_invalid_axis_specs(dim_nums, match):
    x = jnp.ones((2, 3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match=match):
        _compute_matrix_reshape(x, dim_nums)


def test_explicit_dimension_spec_tree_works_without_params_at_update():
    params = {"w": jnp.ones((2, 3, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}
    tx = scale_by_prism(
        ns_iters=2,
        grad_clip_max_amps=None,
        weight_dimension_numbers={
            "w": MatrixDimensionNumbers(reduction_axis=1, output_axis=2)
        },
    )

    updates, _ = tx.update(grads, tx.init(params))
    updates = cast(dict[str, jax.Array], updates)

    assert updates["w"].shape == params["w"].shape
    assert jnp.all(jnp.isfinite(updates["w"]))


def test_callable_dimension_specs_require_params_at_update():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.ones_like(params["w"]) * 0.1}

    def specs(model):
        return jax.tree.map(lambda _: MatrixDimensionNumbers(), model)

    tx = scale_by_muon(ns_steps=2, weight_dimension_numbers=specs)

    with pytest.raises(ValueError, match="params.*scale_by_muon"):
        tx.update(grads, tx.init(params))
