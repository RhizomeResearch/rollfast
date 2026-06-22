"""Measured diagnostics for fine-tuning optimizer state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from rollfast.optim.adam8 import (
    QuantizedBlocks,
    estimate_quantized_moment_bytes,
    quantized_nbytes,
)

from .config import CompiledGroup, OptimizerBundle, SCHEMA_VERSION, StateQuantizationConfig
from .validation import validate_plan


@dataclass(frozen=True)
class StateLeafSummary:
    """One materialized optimizer-state leaf."""

    path: str
    category: str
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    group: str | None = None
    storage: str = "array"
    placement: str = "unsharded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "category": self.category,
            "shape": self.shape,
            "dtype": self.dtype,
            "bytes": self.bytes,
            "group": self.group,
            "storage": self.storage,
            "placement": self.placement,
        }


@dataclass(frozen=True)
class OptimizerStateMemorySummary:
    """Measured optimizer-state memory report."""

    schema_version: int
    optimizer: str
    total_bytes: int
    estimated_state_bytes: int
    by_category: Mapping[str, int]
    by_group: Mapping[str, int]
    by_placement: Mapping[str, int]
    replicated_bytes: int
    globally_sharded_bytes: int
    unsharded_bytes: int
    preconditioner_bytes: int
    preconditioner_factors: tuple[StateLeafSummary, ...]
    leaves: tuple[StateLeafSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "optimizer": self.optimizer,
            "total_bytes": self.total_bytes,
            "estimated_state_bytes": self.estimated_state_bytes,
            "by_category": dict(self.by_category),
            "by_group": dict(self.by_group),
            "by_placement": dict(self.by_placement),
            "replicated_bytes": self.replicated_bytes,
            "globally_sharded_bytes": self.globally_sharded_bytes,
            "unsharded_bytes": self.unsharded_bytes,
            "preconditioner_bytes": self.preconditioner_bytes,
            "preconditioner_factors": [
                factor.to_dict() for factor in self.preconditioner_factors
            ],
            "leaves": [leaf.to_dict() for leaf in self.leaves],
        }


@dataclass(frozen=True)
class OptimizerStateMemoryEstimate:
    """Static optimizer-state memory estimate before state initialization."""

    schema_version: int
    optimizer: str
    total_bytes: int
    moment_bytes: int
    preconditioner_bytes: int
    preconditioner_aux_bytes: int
    by_category: Mapping[str, int]
    by_group: Mapping[str, int]
    by_placement: Mapping[str, int]
    replicated_bytes: int
    globally_sharded_bytes: int
    unsharded_bytes: int
    preconditioner_factors: tuple[StateLeafSummary, ...]
    leaves: tuple[StateLeafSummary, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "optimizer": self.optimizer,
            "total_bytes": self.total_bytes,
            "moment_bytes": self.moment_bytes,
            "preconditioner_bytes": self.preconditioner_bytes,
            "preconditioner_aux_bytes": self.preconditioner_aux_bytes,
            "by_category": dict(self.by_category),
            "by_group": dict(self.by_group),
            "by_placement": dict(self.by_placement),
            "replicated_bytes": self.replicated_bytes,
            "globally_sharded_bytes": self.globally_sharded_bytes,
            "unsharded_bytes": self.unsharded_bytes,
            "preconditioner_factors": [
                factor.to_dict() for factor in self.preconditioner_factors
            ],
            "leaves": [leaf.to_dict() for leaf in self.leaves],
            "warnings": self.warnings,
        }


def optimizer_state_memory_summary(
    bundle: OptimizerBundle,
    state: Any,
) -> OptimizerStateMemorySummary:
    """Measure optimizer-state storage from an initialized state PyTree."""

    leaves = tuple(
        summary
        for path, leaf in _path_leaves(state)
        if (summary := _summarize_leaf(path, leaf)) is not None
    )
    by_category: dict[str, int] = {}
    by_group: dict[str, int] = {}
    by_placement: dict[str, int] = {}
    for leaf in leaves:
        by_category[leaf.category] = by_category.get(leaf.category, 0) + leaf.bytes
        by_placement[leaf.placement] = by_placement.get(leaf.placement, 0) + leaf.bytes
        if leaf.group is not None:
            by_group[leaf.group] = by_group.get(leaf.group, 0) + leaf.bytes
    preconditioner_factors = tuple(
        leaf for leaf in leaves if leaf.category == "preconditioner"
    )
    preconditioner_bytes = sum(leaf.bytes for leaf in preconditioner_factors)
    return OptimizerStateMemorySummary(
        schema_version=SCHEMA_VERSION,
        optimizer=bundle.optimizer_config.name,
        total_bytes=sum(leaf.bytes for leaf in leaves),
        estimated_state_bytes=bundle.report.estimated_state_bytes,
        by_category=dict(sorted(by_category.items())),
        by_group=dict(sorted(by_group.items())),
        by_placement=dict(sorted(by_placement.items())),
        replicated_bytes=by_placement.get("replicated", 0),
        globally_sharded_bytes=by_placement.get("globally_sharded", 0),
        unsharded_bytes=by_placement.get("unsharded", 0),
        preconditioner_bytes=preconditioner_bytes,
        preconditioner_factors=preconditioner_factors,
        leaves=leaves,
    )


def estimate_optimizer_state_memory(
    plan: Any,
    bundle: OptimizerBundle,
    *,
    preconditioner_dtype: Any | None = None,
) -> OptimizerStateMemoryEstimate:
    """Estimate optimizer-state storage from the plan before ``bundle.init``.

    ``preconditioner_dtype`` should match the value passed to
    ``hybrid_kron_adam_from_plan``. ``None`` matches PSGD/Kron's default fp32
    factor initialization for non-scalar factors.
    """

    normalized = validate_plan(plan)
    if normalized.fingerprint != bundle.report.fingerprint:
        raise ValueError(
            "plan fingerprint does not match optimizer bundle report; "
            "state-memory estimates require the same plan used to build the bundle."
        )

    plan_leaves = _plan_trainable_leaves(
        normalized.trainable,
        normalized.labels,
    )
    leaves = []
    if bundle.optimizer_config.name == "adamw8":
        leaves.extend(_estimate_adamw8_group_moment_leaves(bundle))
    for path, leaf, label in plan_leaves:
        if bundle.optimizer_config.name != "adamw8":
            leaves.extend(_estimate_moment_leaves(path, leaf, label, bundle))
        if bundle.optimizer_config.name == "kron_adam":
            leaves.extend(
                _estimate_kron_leaves(
                    path,
                    leaf,
                    label,
                    preconditioner_dtype=preconditioner_dtype,
                )
            )

    by_category: dict[str, int] = {}
    by_group: dict[str, int] = {}
    by_placement: dict[str, int] = {}
    leaves = [
        _with_estimated_placement(leaf, bundle)
        for leaf in leaves
    ]
    for leaf in leaves:
        by_category[leaf.category] = by_category.get(leaf.category, 0) + leaf.bytes
        by_placement[leaf.placement] = by_placement.get(leaf.placement, 0) + leaf.bytes
        if leaf.group is not None:
            by_group[leaf.group] = by_group.get(leaf.group, 0) + leaf.bytes

    preconditioner_factors = tuple(
        leaf for leaf in leaves if leaf.category == "preconditioner"
    )
    preconditioner_bytes = sum(leaf.bytes for leaf in preconditioner_factors)
    preconditioner_aux_bytes = by_category.get("preconditioner_aux", 0)
    moment_bytes = (
        by_category.get("first_moment", 0)
        + by_category.get("second_moment", 0)
        + by_category.get("moments", 0)
    )
    warnings = [
        "static estimates exclude wrapper counters, finite guards, RNG keys, "
        "schedule-free averaging state, and accumulation buffers."
    ]
    if (
        bundle.optimizer_config.name in {"aurora_adam", "prism_adam", "kron_adam"}
        and bundle.report.estimated_state_bytes != moment_bytes
    ):
        warnings.append(
            "bundle.report.estimated_state_bytes uses the legacy Adam-style "
            "moment estimate; this estimate uses optimizer-family moment state."
        )

    return OptimizerStateMemoryEstimate(
        schema_version=SCHEMA_VERSION,
        optimizer=bundle.optimizer_config.name,
        total_bytes=sum(leaf.bytes for leaf in leaves),
        moment_bytes=moment_bytes,
        preconditioner_bytes=preconditioner_bytes,
        preconditioner_aux_bytes=preconditioner_aux_bytes,
        by_category=dict(sorted(by_category.items())),
        by_group=dict(sorted(by_group.items())),
        by_placement=dict(sorted(by_placement.items())),
        replicated_bytes=by_placement.get("replicated", 0),
        globally_sharded_bytes=by_placement.get("globally_sharded", 0),
        unsharded_bytes=by_placement.get("unsharded", 0),
        preconditioner_factors=preconditioner_factors,
        leaves=tuple(leaves),
        warnings=tuple(warnings),
    )


def _summarize_leaf(path: tuple[Any, ...], leaf: Any) -> StateLeafSummary | None:
    if leaf is None or _is_masked_node(leaf):
        return None
    tokens = _path_tokens(path)
    path_text = _format_tokens(tokens)
    group = _group_from_tokens(tokens)
    category = _category_from_tokens(tokens)
    storage = _storage_from_tokens(tokens, leaf)
    if isinstance(leaf, QuantizedBlocks):
        return StateLeafSummary(
            path=path_text,
            category=category,
            shape=leaf.shape,
            dtype=f"{leaf.values.dtype.name}+scale:{leaf.scales.dtype.name}",
            bytes=quantized_nbytes(leaf),
            group=group,
            storage=storage,
            placement=_placement_from_array(leaf.values),
        )
    if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
        return None
    return StateLeafSummary(
        path=path_text,
        category=category,
        shape=tuple(int(dim) for dim in leaf.shape),
        dtype=leaf.dtype.name,
        bytes=int(leaf.size * leaf.dtype.itemsize),
        group=group,
        storage=storage,
        placement=_placement_from_array(leaf),
    )


def _with_estimated_placement(
    leaf: StateLeafSummary,
    bundle: OptimizerBundle,
) -> StateLeafSummary:
    return StateLeafSummary(
        path=leaf.path,
        category=leaf.category,
        shape=leaf.shape,
        dtype=leaf.dtype,
        bytes=leaf.bytes,
        group=leaf.group,
        storage=leaf.storage,
        placement=_estimated_placement(leaf.bytes, bundle),
    )


def _placement_from_array(value: Any) -> str:
    sharding = getattr(value, "sharding", None)
    if sharding is None:
        return "unsharded"
    if bool(getattr(sharding, "is_fully_replicated", False)):
        return "replicated"
    return "globally_sharded"


def _estimated_placement(bytes_: int, bundle: OptimizerBundle) -> str:
    sharding = bundle.sharding_policy
    if not sharding.mesh_axes:
        return "unsharded"
    if (
        sharding.state_placement == "replicate_small_follow_large"
        and bytes_ <= sharding.small_state_threshold
    ):
        return "replicated"
    return "globally_sharded"


def _plan_trainable_leaves(
    trainable: Any,
    labels: Any,
) -> tuple[tuple[tuple[Any, ...], Any, str], ...]:
    label_leaves = jax.tree_util.tree_leaves(labels)
    return tuple(
        (path, leaf, label)
        for (path, leaf), label in zip(
            jax.tree_util.tree_leaves_with_path(trainable),
            label_leaves,
            strict=True,
        )
        if leaf is not None
    )


def _estimate_adamw8_group_moment_leaves(
    bundle: OptimizerBundle,
) -> tuple[StateLeafSummary, ...]:
    quantization = bundle.quantization_config
    leaves = []
    for group in bundle.report.groups:
        for category, suffix in (
            ("first_moment", "mu"),
            ("second_moment", "nu"),
        ):
            if _quantize_group_state(group, quantization):
                bytes_ = estimate_quantized_moment_bytes(
                    group.param_count,
                    block_size=quantization.block_size,
                    scale_dtype=quantization.scale_dtype,
                )
                dtype = (
                    f"{jnp.dtype(jnp.uint8).name}"
                    f"+scale:{jnp.dtype(quantization.scale_dtype).name}"
                )
                storage = "blockwise_int8"
            else:
                dtype = jnp.dtype(quantization.fallback_dtype).name
                bytes_ = int(group.param_count * jnp.dtype(dtype).itemsize)
                storage = "array"
            leaves.append(
                StateLeafSummary(
                    path=f"group:{group.source_label}/moment:{suffix}",
                    category=category,
                    shape=(group.param_count,),
                    dtype=dtype,
                    bytes=bytes_,
                    group=group.source_label,
                    storage=storage,
                )
            )
    return tuple(leaves)


def _quantize_group_state(
    group: CompiledGroup,
    state_quantization: StateQuantizationConfig,
) -> bool:
    if not state_quantization.enabled:
        return False
    if group.param_count < state_quantization.min_size:
        return False
    keep_tags = {tag.lower() for tag in state_quantization.keep_fp32_tags}
    group_terms = {tag.lower() for tag in group.tags}
    group_terms.update((group.source_label.lower(), group.role.lower()))
    return not any(
        keep_tag in term
        for keep_tag in keep_tags
        for term in group_terms
    )


def _estimate_moment_leaves(
    path: tuple[Any, ...],
    leaf: Any,
    label: str,
    bundle: OptimizerBundle,
) -> tuple[StateLeafSummary, ...]:
    dtype = jnp.dtype(bundle.precision_config.moment_dtype)
    path_text = _format_tokens(_path_tokens(path))
    shape = tuple(int(dim) for dim in leaf.shape)
    leaf_bytes = int(leaf.size * dtype.itemsize)
    optimizer = bundle.optimizer_config.name

    if optimizer in {"aurora_adam", "prism_adam", "kron_adam"}:
        if bundle.optimizer_config.b1 <= 0:
            return ()
        return (
            StateLeafSummary(
                path=f"{path_text}/moment:mu",
                category="first_moment",
                shape=shape,
                dtype=dtype.name,
                bytes=leaf_bytes,
                group=label,
            ),
        )

    return (
        StateLeafSummary(
            path=f"{path_text}/moment:mu",
            category="first_moment",
            shape=shape,
            dtype=dtype.name,
            bytes=leaf_bytes,
            group=label,
        ),
        StateLeafSummary(
            path=f"{path_text}/moment:nu",
            category="second_moment",
            shape=shape,
            dtype=dtype.name,
            bytes=leaf_bytes,
            group=label,
        ),
    )


def _estimate_kron_leaves(
    path: tuple[Any, ...],
    leaf: Any,
    label: str,
    *,
    preconditioner_dtype: Any | None,
) -> tuple[StateLeafSummary, ...]:
    path_text = _format_tokens(_path_tokens(path))
    factor_dtype = _kron_factor_dtype(leaf, preconditioner_dtype)
    leaves: list[StateLeafSummary] = []
    for index, shape in enumerate(_kron_factor_shapes(leaf.shape)):
        storage = "matrix_factor" if len(shape) == 2 else "diagonal_factor"
        size = _shape_size(shape)
        leaves.append(
            StateLeafSummary(
                path=f"{path_text}/preconditioner:{index}",
                category="preconditioner",
                shape=shape,
                dtype=factor_dtype.name,
                bytes=int(size * factor_dtype.itemsize),
                group=label,
                storage=storage,
            )
        )
        leaves.append(
            StateLeafSummary(
                path=f"{path_text}/preconditioner_lipschitz:{index}",
                category="preconditioner_aux",
                shape=(),
                dtype=jnp.dtype(jnp.float32).name,
                bytes=jnp.dtype(jnp.float32).itemsize,
                group=label,
                storage="scalar_lipschitz",
            )
        )
    leaves.append(
        StateLeafSummary(
            path=f"{path_text}/needs_scale_init",
            category="preconditioner_aux",
            shape=(),
            dtype=jnp.dtype(jnp.bool_).name,
            bytes=jnp.dtype(jnp.bool_).itemsize,
            group=label,
            storage="array",
        )
    )
    return tuple(leaves)


def _kron_factor_shapes(
    shape: tuple[int, ...],
    *,
    max_size_triangular: int = 8192,
    max_skew_triangular: float = 1.0,
    min_ndim_triangular: int = 2,
    memory_save_mode: str | None = None,
) -> tuple[tuple[int, ...], ...]:
    if len(shape) == 0:
        return ((),)
    if len(shape) > 13:
        raise ValueError(
            f"Got tensor with dim {len(shape)}; Einstein runs out of letters."
        )
    if memory_save_mode is None:
        dim_diag = [False for _ in shape]
    elif memory_save_mode == "one_diag":
        largest = max(range(len(shape)), key=lambda index: shape[index])
        dim_diag = [index == largest for index in range(len(shape))]
    elif memory_save_mode == "all_diag":
        dim_diag = [True for _ in shape]
    else:
        raise ValueError(
            "memory_save_mode must be one of None, 'one_diag', or 'all_diag'."
        )

    total_numel = _shape_size(shape)
    factors = []
    for size, force_diag in zip(shape, dim_diag, strict=True):
        is_diagonal = (
            size <= 1
            or size > max_size_triangular
            or size**2 > max_skew_triangular * total_numel
            or len(shape) < min_ndim_triangular
            or force_diag
        )
        factors.append((int(size),) if is_diagonal else (int(size), int(size)))
    return tuple(factors)


def _kron_factor_dtype(leaf: Any, preconditioner_dtype: Any | None) -> jnp.dtype:
    if preconditioner_dtype is not None:
        return jnp.dtype(preconditioner_dtype)
    if len(leaf.shape) == 0:
        return jnp.dtype(leaf.dtype)
    return jnp.dtype(jnp.float32)


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def _category_from_tokens(tokens: tuple[str, ...]) -> str:
    if "attr:Qs_preconditioners" in tokens:
        return "preconditioner"
    if "attr:Ls_lipschitz" in tokens:
        return "preconditioner_aux"
    if "attr:mu" in tokens:
        return "first_moment"
    if "attr:nu" in tokens:
        return "second_moment"
    if _is_counter_path(tokens):
        return "counter"
    if tokens and tokens[-1] == "attr:key":
        return "rng"
    if tokens and tokens[-1] == "attr:last_finite":
        return "finite_guard"
    if tokens and tokens[-1] == "attr:needs_scale_init":
        return "preconditioner_aux"
    return "other"


def _storage_from_tokens(tokens: tuple[str, ...], leaf: Any) -> str:
    if isinstance(leaf, QuantizedBlocks):
        return "blockwise_int8"
    if "attr:Qs_preconditioners" in tokens:
        if len(leaf.shape) == 1:
            return "diagonal_factor"
        if len(leaf.shape) == 2:
            return "matrix_factor"
    if "attr:Ls_lipschitz" in tokens:
        return "scalar_lipschitz"
    return "array"


def _group_from_tokens(tokens: tuple[str, ...]) -> str | None:
    for token in tokens:
        if token.startswith("key:"):
            group = token.removeprefix("key:")
            if group.endswith(("_decay", "_no_decay")) or group in {
                "adam",
                "aurora",
                "prism",
            }:
                return group
    return None


def _path_leaves(tree: Any) -> list[tuple[Any, Any]]:
    return jax.tree_util.tree_flatten_with_path(tree, is_leaf=_is_leaf)[0]


def _is_leaf(leaf: Any) -> bool:
    return leaf is None or isinstance(leaf, QuantizedBlocks) or _is_masked_node(leaf)


def _is_masked_node(leaf: Any) -> bool:
    return leaf.__class__.__name__ == "MaskedNode"


def _path_tokens(path: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(_path_token(part) for part in path)


def _path_token(part: Any) -> str:
    if hasattr(part, "name"):
        return f"attr:{part.name}"
    if hasattr(part, "key"):
        return f"key:{part.key}"
    if hasattr(part, "idx"):
        return f"idx:{part.idx}"
    return repr(part)


def _format_tokens(tokens: tuple[str, ...]) -> str:
    return "/".join(tokens)


def _is_counter_path(tokens: tuple[str, ...]) -> bool:
    return tokens and tokens[-1] in {
        "attr:count",
        "attr:notfinite_count",
        "attr:total_notfinite",
    }


__all__ = (
    "OptimizerStateMemoryEstimate",
    "OptimizerStateMemorySummary",
    "StateLeafSummary",
    "estimate_optimizer_state_memory",
    "optimizer_state_memory_summary",
)
