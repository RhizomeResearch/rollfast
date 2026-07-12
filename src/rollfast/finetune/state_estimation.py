"""Private shared helpers for optimizer-state estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from rollfast.optim.adam8 import estimate_quantized_moment_bytes

from .config import CompiledGroup, StateQuantizationConfig


@dataclass(frozen=True)
class AdamW8MomentEstimate:
    """Estimated storage for one moment of one AdamW8 parameter leaf."""

    path: tuple[Any, ...]
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    group: str
    storage: str


def estimate_adamw8_moment_leaves(
    trainable: Any,
    labels: Any,
    groups: tuple[CompiledGroup, ...],
    quantization: StateQuantizationConfig,
) -> tuple[AdamW8MomentEstimate, ...]:
    """Estimate one AdamW8 moment for every trainable parameter leaf."""

    groups_by_label = {group.source_label: group for group in groups}
    label_leaves = jax.tree_util.tree_leaves(labels)
    estimates = []
    for (path, leaf), label in zip(
        jax.tree_util.tree_leaves_with_path(trainable),
        label_leaves,
        strict=True,
    ):
        if leaf is None:
            continue
        group = groups_by_label[label]
        if quantize_group_state(group, quantization) and _quantize_leaf_state(
            leaf, quantization
        ):
            estimates.append(
                AdamW8MomentEstimate(
                    path=path,
                    shape=tuple(int(dim) for dim in leaf.shape),
                    dtype=(
                        f"{jnp.dtype(jnp.uint8).name}"
                        f"+scale:{jnp.dtype(quantization.scale_dtype).name}"
                    ),
                    bytes=estimate_quantized_moment_bytes(
                        int(leaf.size),
                        block_size=quantization.block_size,
                        scale_dtype=quantization.scale_dtype,
                    ),
                    group=label,
                    storage="blockwise_int8",
                )
            )
        else:
            dtype = jnp.dtype(quantization.fallback_dtype)
            estimates.append(
                AdamW8MomentEstimate(
                    path=path,
                    shape=tuple(int(dim) for dim in leaf.shape),
                    dtype=dtype.name,
                    bytes=int(leaf.size * dtype.itemsize),
                    group=label,
                    storage="array",
                )
            )
    return tuple(estimates)


def quantize_group_state(
    group: CompiledGroup,
    quantization: StateQuantizationConfig,
) -> bool:
    """Return whether a compiled group permits AdamW8 moment quantization."""

    if not quantization.enabled:
        return False
    keep_tags = {tag.lower() for tag in quantization.keep_fp32_tags}
    group_terms = {tag.lower() for tag in group.tags}
    group_terms.update((group.source_label.lower(), group.role.lower()))
    return not any(keep_tag in term for keep_tag in keep_tags for term in group_terms)


def _quantize_leaf_state(leaf: Any, quantization: StateQuantizationConfig) -> bool:
    return bool(
        leaf.size >= quantization.min_size and jnp.issubdtype(leaf.dtype, jnp.inexact)
    )
