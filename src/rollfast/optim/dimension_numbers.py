"""Shared matrix dimension-number utilities for optimizer routing."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, cast

import jax
from optax._src import base
from optax.transforms import _masking


class MatrixDimensionNumbers(NamedTuple):
    """Specification for flattening tensor axes into a matrix optimizer layout.

    Matrix optimizers in this package operate on tensors reshaped to
    ``(batch, reduction, output)``. Axes not listed in ``reduction_axis`` or
    ``output_axis`` are treated as batch axes and are processed independently.
    """

    reduction_axis: Sequence[int] | int = 0
    output_axis: Sequence[int] | int = 1


DimNumsTree = base.Params
WeightDimNumOrFn = (
    MatrixDimensionNumbers | DimNumsTree | Callable[[base.Params], DimNumsTree]
)
ReshapeFn = Callable[[jax.Array], jax.Array]


def _is_dimension_numbers_leaf(x: Any) -> bool:
    return (
        x is None
        or isinstance(x, MatrixDimensionNumbers)
        or isinstance(x, _masking.MaskedNode)
    )


def _get_dimension_numbers(
    weight_dimension_numbers: WeightDimNumOrFn | None, params: base.Params
) -> base.Params:
    """Resolve a dimension-number argument into a PyTree aligned with params."""
    if weight_dimension_numbers is None:

        def _get_default_spec(x):
            if isinstance(x, _masking.MaskedNode) or x is None:
                return None
            return (
                MatrixDimensionNumbers() if hasattr(x, "ndim") and x.ndim == 2 else None
            )

        return jax.tree.map(
            _get_default_spec,
            params,
            is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
        )

    if callable(weight_dimension_numbers):
        dim_num_fn = cast(
            Callable[[base.Params], DimNumsTree], weight_dimension_numbers
        )
        return dim_num_fn(params)

    return weight_dimension_numbers


def _mask_dimension_numbers(dim_nums_tree: base.Params) -> base.Params:
    """Replace ``None`` entries with ``MaskedNode`` for partition compatibility."""
    return jax.tree.map(
        lambda d: d if d is not None else _masking.MaskedNode(),
        dim_nums_tree,
        is_leaf=_is_dimension_numbers_leaf,
    )


def _is_standard_2d_spec(dim_nums: MatrixDimensionNumbers) -> bool:
    """Return whether a spec is the ordinary 2D ``(reduction=0, output=1)`` layout."""
    red = dim_nums.reduction_axis
    out = dim_nums.output_axis

    red_norm = (red,) if isinstance(red, int) else tuple(red)
    out_norm = (out,) if isinstance(out, int) else tuple(out)

    return red_norm == (0,) and out_norm == (1,)


def _normalize_axes(
    x: jax.Array, dim_nums: MatrixDimensionNumbers
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Normalize dimension specs to tuples of positive integers."""
    reduction_axis = dim_nums.reduction_axis
    reduction_axis = (
        (reduction_axis,) if isinstance(reduction_axis, int) else reduction_axis
    )
    reduction_axes = tuple(ax % x.ndim for ax in reduction_axis)

    output_axis = dim_nums.output_axis
    output_axis = (output_axis,) if isinstance(output_axis, int) else output_axis
    output_axes = tuple(ax % x.ndim for ax in output_axis)
    return reduction_axes, output_axes


def _compute_matrix_reshape(
    x: jax.Array, dim_nums: MatrixDimensionNumbers
) -> tuple[ReshapeFn, ReshapeFn]:
    """Build forward/inverse reshapes for ``(batch, reduction, output)`` layout."""
    if x.ndim < 2:
        raise ValueError(
            f"Matrix-optimized parameters must have rank >= 2, got {x.ndim=}."
        )

    reduction_axes, output_axes = _normalize_axes(x, dim_nums)
    if set(reduction_axes) & set(output_axes):
        raise ValueError(
            f"Reduction and output axes must be disjoint. Got {reduction_axes} and {output_axes}."
        )

    batch_axes = tuple(
        sorted(set(range(x.ndim)) - set(reduction_axes) - set(output_axes))
    )
    transpose = batch_axes + reduction_axes + output_axes
    inv_transpose = tuple(sorted(range(x.ndim), key=lambda i: transpose[i]))

    axes2shape = lambda axes: tuple(x.shape[ax] for ax in axes)

    flat_shape = (
        math.prod(axes2shape(batch_axes)),
        math.prod(axes2shape(reduction_axes)),
        math.prod(axes2shape(output_axes)),
    )
    unflat_shape = (
        axes2shape(batch_axes) + axes2shape(reduction_axes) + axes2shape(output_axes)
    )

    reshape_fn = lambda x: x.transpose(transpose).reshape(flat_shape)
    inverse_fn = lambda x: x.reshape(unflat_shape).transpose(inv_transpose)
    return reshape_fn, inverse_fn


__all__ = [
    "DimNumsTree",
    "MatrixDimensionNumbers",
    "ReshapeFn",
    "WeightDimNumOrFn",
]
