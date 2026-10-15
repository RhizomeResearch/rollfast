"""Resolve fine-tuning precision policies against trainable parameters."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .config import (
    AccumulationConfig,
    PrecisionConfig,
    PrecisionMode,
    PrecisionReport,
)

_LOW_PRECISION = frozenset({jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)})


def array_leaves(tree: Any) -> list[Any]:
    """Return the array leaves of ``tree``, skipping ``None`` placeholders."""
    return [
        leaf
        for leaf in jax.tree.leaves(tree, is_leaf=lambda x: x is None)
        if leaf is not None and hasattr(leaf, "dtype")
    ]


def master_leaf_dtype(leaf_dtype: Any, master_dtype: Any) -> jnp.dtype:
    """Return a master dtype that keeps the leaf's precision and imaginary part."""
    leaf_dtype = jnp.dtype(leaf_dtype)
    if not jnp.issubdtype(leaf_dtype, jnp.inexact):
        return leaf_dtype
    return jnp.promote_types(leaf_dtype, master_dtype)


def uses_master_params(trainable: Any, precision: PrecisionConfig) -> bool:
    """Return whether ``precision`` keeps master parameters for ``trainable``."""
    if precision.master_params == "always":
        return True
    if precision.master_params == "never":
        return False
    return any(
        jnp.dtype(leaf.dtype) in _LOW_PRECISION for leaf in array_leaves(trainable)
    )


def validate_precision(
    trainable: Any,
    precision: PrecisionConfig,
    accumulation: AccumulationConfig,
) -> None:
    """Reject precision settings that the trainable tree cannot honor."""
    if jnp.dtype(precision.accumulation_dtype) != jnp.dtype(
        accumulation.accumulate_dtype
    ):
        raise ValueError(
            "PrecisionConfig.accumulation_dtype must match "
            "AccumulationConfig.accumulate_dtype, which controls gradient "
            f"accumulation; got {jnp.dtype(precision.accumulation_dtype).name} and "
            f"{jnp.dtype(accumulation.accumulate_dtype).name}."
        )
    if precision.parameter_rounding != "stochastic":
        return
    dtypes = {jnp.dtype(leaf.dtype) for leaf in array_leaves(trainable)}
    if jnp.dtype(jnp.float16) in dtypes:
        raise ValueError(
            "parameter_rounding='stochastic' supports bfloat16 parameters only; "
            "keep float16 parameters with master_params='always'."
        )
    if jnp.dtype(jnp.bfloat16) not in dtypes:
        raise ValueError(
            "parameter_rounding='stochastic' requires bfloat16 trainable "
            "parameters; wider parameters already keep small updates."
        )


def precision_report(
    trainable: Any,
    precision: PrecisionConfig,
    accumulation: AccumulationConfig,
    *,
    moment_storage: str,
    moment_rounding: str,
) -> PrecisionReport:
    """Describe the resolved storage, rounding, and memory of a precision policy."""
    leaves = array_leaves(trainable)
    dtypes = tuple(dict.fromkeys(jnp.dtype(leaf.dtype).name for leaf in leaves))
    masters = uses_master_params(trainable, precision)
    master_dtypes = [
        master_leaf_dtype(leaf.dtype, precision.master_param_dtype) for leaf in leaves
    ]
    low_precision = any(jnp.dtype(leaf.dtype) in _LOW_PRECISION for leaf in leaves)
    mode: PrecisionMode
    if masters and low_precision:
        mode = "master_mixed_precision"
    elif low_precision and precision.parameter_rounding == "stochastic":
        mode = "master_free_stochastic"
    elif low_precision:
        mode = "master_free_nearest"
    else:
        mode = "full_precision"
    expected = precision.expected_model_compute_dtype
    return PrecisionReport(
        mode=mode,
        trainable_dtypes=dtypes,
        master_params=masters,
        master_param_dtypes=tuple(dict.fromkeys(d.name for d in master_dtypes))
        if masters
        else (),
        cast_back=precision.cast_back if masters else None,
        parameter_rounding=precision.parameter_rounding,
        gradient_dtype=jnp.dtype(precision.gradient_dtype).name,
        accumulation_dtype=jnp.dtype(accumulation.accumulate_dtype).name,
        moment_storage=moment_storage,
        moment_rounding=moment_rounding,
        loss_scale=precision.loss_scale,
        expected_model_compute_dtype=None
        if expected is None
        else jnp.dtype(expected).name,
        visible_param_bytes=sum(
            int(leaf.size * leaf.dtype.itemsize) for leaf in leaves
        ),
        master_param_bytes=sum(
            int(leaf.size * dtype.itemsize)
            for leaf, dtype in zip(leaves, master_dtypes, strict=True)
        )
        if masters
        else 0,
    )
