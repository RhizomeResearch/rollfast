"""Shared matrix dimension-number utilities for optimizer routing."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, TypeAlias, cast

import jax
import jax.numpy as jnp
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
# A bare MatrixDimensionNumbers is intentionally single-leaf only. It is useful
# for direct transforms over one array, but it is not broadcast over PyTrees:
# structured params must use a matching spec tree or callable so routing remains
# explicit for biases, embeddings, convolution kernels, and fallback leaves.
MaskOrFn: TypeAlias = Any | Callable[[base.Params], Any] | None
ReshapeFn = Callable[[jax.Array], jax.Array]


class _MatrixPartitionFns(NamedTuple):
    """Resolved matrix optimizer routing helpers for one branch label."""

    resolve: Callable[[base.Params], base.Params]
    labels: Callable[[base.Params], base.Params]
    masked_specs: Callable[[base.Params], base.Params]
    default_mask: Callable[[base.Params], base.Params]


def _is_dimension_numbers_leaf(x: Any) -> bool:
    return (
        x is None
        or isinstance(x, MatrixDimensionNumbers)
        or isinstance(x, _masking.MaskedNode)
    )


def _is_array_like_leaf(x: Any) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _validate_matrix_operand(
    x: Any,
    dim_nums: MatrixDimensionNumbers | None,
    transform_name: str,
) -> None:
    """Reject leaves that a direct matrix transform cannot safely handle."""
    if x is None or isinstance(x, _masking.MaskedNode):
        return
    if dim_nums is None or isinstance(dim_nums, _masking.MaskedNode):
        raise ValueError(
            f"`{transform_name}` only supports leaves with matrix dimension specs. "
            "Use the public wrapper for Adam fallback leaves, or pass a "
            "`weight_dimension_numbers` tree that marks every updated leaf."
        )
    if jnp.issubdtype(jnp.dtype(x.dtype), jnp.complexfloating):
        raise ValueError(
            f"`{transform_name}` does not support complex matrix leaves. Route "
            "complex parameters to an Adam fallback branch or convert them to a "
            "real-valued representation before using matrix optimizers."
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

    if isinstance(weight_dimension_numbers, MatrixDimensionNumbers):
        if _is_array_like_leaf(params):
            return weight_dimension_numbers
        raise ValueError(
            "A bare MatrixDimensionNumbers can only be used with a single array "
            "leaf. For PyTree params, pass a matching PyTree of specs/None or a "
            "callable dimension-spec function; bare specs are not broadcast."
        )

    return weight_dimension_numbers


def _resolve_update_dimension_numbers(
    weight_dimension_numbers: WeightDimNumOrFn | None,
    *,
    params: base.Params | None,
    updates: base.Updates,
    transform_name: str,
) -> base.Params:
    """Resolve update-time dimension specs with consistent params requirements."""
    if params is None:
        if callable(weight_dimension_numbers):
            raise ValueError(
                f"`params` must be provided to `{transform_name}` when "
                "`weight_dimension_numbers` is callable."
            )
        return _get_dimension_numbers(weight_dimension_numbers, updates)
    return _get_dimension_numbers(weight_dimension_numbers, params)


def _mask_dimension_numbers(dim_nums_tree: base.Params) -> base.Params:
    """Replace ``None`` entries with ``MaskedNode`` for partition compatibility."""
    return jax.tree.map(
        lambda d: d if d is not None else _masking.MaskedNode(),
        dim_nums_tree,
        is_leaf=_is_dimension_numbers_leaf,
    )


def _make_matrix_labels(
    dim_nums_tree: base.Params,
    params: base.Params,
    matrix_label: str,
    fallback_label: str = "adam",
) -> base.Params:
    """Create optimizer branch labels from a resolved dimension-number tree."""
    return jax.tree.map(
        lambda d, p: (
            None
            if p is None
            else (
                matrix_label
                if d is not None and not isinstance(d, _masking.MaskedNode)
                else fallback_label
            )
        ),
        dim_nums_tree,
        params,
        is_leaf=_is_dimension_numbers_leaf,
    )


def _make_dimension_numbers_mask(
    dim_nums_tree: base.Params,
    params: base.Params,
) -> base.Params:
    """Create a bool mask selecting leaves with real matrix dimension specs."""
    return jax.tree.map(
        lambda d, p: (
            False
            if p is None or isinstance(p, _masking.MaskedNode)
            else d is not None and not isinstance(d, _masking.MaskedNode)
        ),
        dim_nums_tree,
        params,
        is_leaf=_is_dimension_numbers_leaf,
    )


def _make_matrix_partition_fns(
    weight_dimension_numbers: WeightDimNumOrFn | None,
    matrix_label: str,
    fallback_label: str = "adam",
) -> _MatrixPartitionFns:
    """Build repeated matrix/fallback partition closures.

    The default resolver preserves existing behavior: rank-2 array leaves route
    to the matrix branch; all other leaves route to the fallback branch.
    """

    def resolve(params: base.Params) -> base.Params:
        return _get_dimension_numbers(weight_dimension_numbers, params)

    def labels(params: base.Params) -> base.Params:
        return _make_matrix_labels(
            resolve(params),
            params,
            matrix_label=matrix_label,
            fallback_label=fallback_label,
        )

    def masked_specs(params: base.Params) -> base.Params:
        return _mask_dimension_numbers(resolve(params))

    def default_mask(params: base.Params) -> base.Params:
        return _make_dimension_numbers_mask(resolve(params), params)

    return _MatrixPartitionFns(
        resolve=resolve,
        labels=labels,
        masked_specs=masked_specs,
        default_mask=default_mask,
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
    reduction_axes = _normalize_axis_group(x, dim_nums.reduction_axis, "reduction_axis")
    output_axes = _normalize_axis_group(x, dim_nums.output_axis, "output_axis")
    return reduction_axes, output_axes


def _normalize_axis_group(
    x: jax.Array,
    axes: Sequence[int] | int,
    name: str,
) -> tuple[int, ...]:
    axes_tuple = (axes,) if isinstance(axes, int) else tuple(axes)
    if not axes_tuple:
        raise ValueError(f"{name} must not be empty.")

    normalized = []
    for axis in axes_tuple:
        if axis < -x.ndim or axis >= x.ndim:
            raise ValueError(
                f"{name} axis {axis} is out of bounds for rank-{x.ndim} parameter."
            )
        normalized.append(axis if axis >= 0 else axis + x.ndim)

    normalized_tuple = tuple(normalized)
    if len(set(normalized_tuple)) != len(normalized_tuple):
        raise ValueError(f"{name} contains duplicate axes: {axes_tuple}.")

    return normalized_tuple


def _compute_matrix_reshape(
    x: jax.Array, dim_nums: MatrixDimensionNumbers
) -> tuple[ReshapeFn, ReshapeFn]:
    """Build forward/inverse reshapes for ``(batch, reduction, output)`` layout."""
    if jnp.issubdtype(jnp.dtype(x.dtype), jnp.complexfloating):
        raise ValueError(
            "Matrix optimizers do not support complex matrix leaves. Route complex "
            "parameters to an Adam fallback branch or convert them to a real-valued "
            "representation before using matrix optimizers."
        )
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
    "MaskOrFn",
    "MatrixDimensionNumbers",
    "ReshapeFn",
    "WeightDimNumOrFn",
]
