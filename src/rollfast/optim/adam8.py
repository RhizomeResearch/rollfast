"""AdamW with blockwise 8-bit optimizer-state moments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal, NamedTuple, Optional, Union, cast

import jax
import jax.numpy as jnp
from optax._src import base, combine, numerics, transform, utils
from optax.transforms import _masking

from rollfast.utils import _safe_bias_correction, zeros_like_preserving_sharding


SYMMETRIC_INT8_CODEBOOK_ID = "rollfast.symmetric_int8.neg127_pos127.v1"
DYNAMIC_SIGNED_CODEBOOK_ID = "bitsandbytes.dynamic.signed.8bit.v1"
DYNAMIC_UNSIGNED_CODEBOOK_ID = "bitsandbytes.dynamic.unsigned.8bit.v1"
BlockLayout = Literal["shard_local", "logical_global"]
CodebookQuantizer = Literal["dynamic_signed", "dynamic_unsigned", "symmetric_int8"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QuantizedBlocks:
    """Blockwise fixed-codebook 8-bit representation for one state leaf."""

    values: jax.Array
    scales: jax.Array
    shape: tuple[int, ...]
    size: int
    block_size: int
    quantizer: str = "blockwise_dynamic_signed_8bit"
    codebook_id: str = DYNAMIC_SIGNED_CODEBOOK_ID
    qmin: int = 0
    qmax: int = 255
    zero_point: int = 0
    block_layout: BlockLayout = "shard_local"

    def tree_flatten(self):
        return (self.values, self.scales), (
            self.shape,
            self.size,
            self.block_size,
            self.quantizer,
            self.codebook_id,
            self.qmin,
            self.qmax,
            self.zero_point,
            self.block_layout,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            shape,
            size,
            block_size,
            quantizer,
            codebook_id,
            qmin,
            qmax,
            zero_point,
            block_layout,
        ) = aux_data
        values, scales = children
        return cls(
            values=values,
            scales=scales,
            shape=shape,
            size=size,
            block_size=block_size,
            quantizer=quantizer,
            codebook_id=codebook_id,
            qmin=qmin,
            qmax=qmax,
            zero_point=zero_point,
            block_layout=block_layout,
        )

    def metadata(self) -> dict[str, Any]:
        """Return exact checkpoint metadata for this quantized state leaf."""

        return {
            "quantizer": self.quantizer,
            "codebook_id": self.codebook_id,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "zero_point": self.zero_point,
            "shape": self.shape,
            "size": self.size,
            "block_size": self.block_size,
            "block_layout": self.block_layout,
            "scale_dtype": self.scales.dtype.name,
            "values_dtype": self.values.dtype.name,
        }


class ScaleByAdam8State(NamedTuple):
    """State for Adam with quantized first and second moments."""

    count: jax.Array
    mu: base.Updates
    nu: base.Updates
    key: jax.Array


def quantize_blocks(
    x: jax.Array,
    *,
    block_size: int = 2048,
    scale_dtype: jax.typing.DTypeLike = jnp.float32,
    stochastic_rounding: bool = False,
    key: jax.Array | None = None,
    block_layout: BlockLayout = "shard_local",
    quantizer: CodebookQuantizer = "dynamic_signed",
) -> QuantizedBlocks:
    """Quantize one array with fixed-codebook blockwise dynamic scales."""

    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if block_layout not in {"shard_local", "logical_global"}:
        raise ValueError("block_layout must be 'shard_local' or 'logical_global'.")
    if quantizer not in {"dynamic_signed", "dynamic_unsigned", "symmetric_int8"}:
        raise ValueError(
            "quantizer must be 'dynamic_signed', 'dynamic_unsigned', or "
            "'symmetric_int8'."
        )
    x_f32 = jnp.asarray(x, dtype=jnp.float32)
    shape = tuple(x_f32.shape)
    size = int(x_f32.size)
    blocks_count = max(1, math.ceil(size / block_size))
    padded_size = blocks_count * block_size
    flat = jnp.reshape(x_f32, (-1,))
    padded = jnp.pad(flat, (0, padded_size - size))
    blocks = jnp.reshape(padded, (blocks_count, block_size))
    max_abs = jnp.max(jnp.abs(blocks), axis=1)
    if quantizer == "symmetric_int8":
        qmin, qmax = -127.0, 127.0
        scales = jnp.where(max_abs > 0.0, max_abs / qmax, 1.0)
        scaled = jnp.clip(blocks / scales[:, None], qmin, qmax)
        rounded = _round_scaled(
            scaled,
            stochastic_rounding=stochastic_rounding,
            key=key,
        )
        return QuantizedBlocks(
            values=rounded.astype(jnp.int8),
            scales=scales.astype(scale_dtype),
            shape=shape,
            size=size,
            block_size=block_size,
            quantizer="blockwise_symmetric_int8",
            codebook_id=SYMMETRIC_INT8_CODEBOOK_ID,
            qmin=-127,
            qmax=127,
            zero_point=0,
            block_layout=block_layout,
        )

    codebook = _codebook_for_quantizer(quantizer)
    scales = jnp.where(max_abs > 0.0, max_abs, 1.0)
    normalized = jnp.clip(blocks / scales[:, None], codebook[0], codebook[-1])
    codes = _quantize_to_codebook(
        normalized,
        codebook,
        stochastic_rounding=stochastic_rounding,
        key=key,
    )
    return QuantizedBlocks(
        values=codes.astype(jnp.uint8),
        scales=scales.astype(scale_dtype),
        shape=shape,
        size=size,
        block_size=block_size,
        quantizer=f"blockwise_{quantizer}_8bit",
        codebook_id=_codebook_id_for_quantizer(quantizer),
        qmin=0,
        qmax=255,
        zero_point=_zero_code_for_quantizer(quantizer),
        block_layout=block_layout,
    )


def dequantize_blocks(
    blocks: QuantizedBlocks,
    *,
    dtype: jax.typing.DTypeLike = jnp.float32,
) -> jax.Array:
    """Dequantize a ``QuantizedBlocks`` leaf and remove deterministic padding."""

    if blocks.codebook_id == SYMMETRIC_INT8_CODEBOOK_ID:
        values = blocks.values.astype(jnp.float32)
    else:
        codebook = _codebook_for_id(blocks.codebook_id)
        values = codebook[blocks.values.astype(jnp.uint8).astype(jnp.int32)]
    scales = blocks.scales.astype(jnp.float32)
    flat = jnp.reshape(values * scales[:, None], (-1,))[: blocks.size]
    return jnp.reshape(flat, blocks.shape).astype(dtype)


def quantized_nbytes(blocks: QuantizedBlocks) -> int:
    """Return materialized array bytes used by one quantized state leaf."""

    return int(blocks.values.size * blocks.values.dtype.itemsize) + int(
        blocks.scales.size * blocks.scales.dtype.itemsize
    )


def tree_state_nbytes(tree: Any) -> int:
    """Measure array storage bytes in an optimizer-state PyTree."""

    total = 0
    for leaf in jax.tree.leaves(tree, is_leaf=_is_state_leaf):
        if isinstance(leaf, QuantizedBlocks):
            total += quantized_nbytes(leaf)
        elif hasattr(leaf, "dtype") and hasattr(leaf, "size"):
            total += int(leaf.size * leaf.dtype.itemsize)
    return total


def estimate_quantized_moment_bytes(
    size: int,
    *,
    block_size: int = 2048,
    scale_dtype: jax.typing.DTypeLike = jnp.float32,
) -> int:
    """Estimate bytes for one quantized moment leaf with ``size`` elements."""

    if size <= 0:
        return 0
    blocks_count = max(1, math.ceil(size / block_size))
    return int(blocks_count * block_size) + int(
        blocks_count * jnp.dtype(scale_dtype).itemsize
    )


def scale_by_adam8(
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    *,
    block_size: int = 2048,
    min_size: int = 4096,
    scale_dtype: jax.typing.DTypeLike = jnp.float32,
    fallback_dtype: jax.typing.DTypeLike = jnp.float32,
    stochastic_rounding: bool = False,
    block_layout: BlockLayout = "shard_local",
    quantize: bool = True,
    nesterov: bool = False,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """Rescale gradients by Adam while storing eligible moments as 8-bit blocks."""

    scale_dtype = utils.canonicalize_dtype(scale_dtype)
    fallback_dtype = utils.canonicalize_dtype(fallback_dtype)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if min_size < 0:
        raise ValueError("min_size must be non-negative.")
    if block_layout not in {"shard_local", "logical_global"}:
        raise ValueError("block_layout must be 'shard_local' or 'logical_global'.")

    def init_fn(params):
        key_mu, key_nu, next_key = jax.random.split(key, 3)
        mu = _init_moment_tree(
            params,
            block_size=block_size,
            min_size=min_size,
            scale_dtype=scale_dtype,
            fallback_dtype=fallback_dtype,
            block_layout=block_layout,
            quantize=quantize,
            quantizer="dynamic_signed",
            key=key_mu,
        )
        nu = _init_moment_tree(
            params,
            block_size=block_size,
            min_size=min_size,
            scale_dtype=scale_dtype,
            fallback_dtype=fallback_dtype,
            block_layout=block_layout,
            quantize=quantize,
            quantizer="dynamic_unsigned",
            key=key_nu,
        )
        return ScaleByAdam8State(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            nu=nu,
            key=next_key,
        )

    def update_fn(updates, state, params=None):
        del params
        key_mu, key_nu, next_key = jax.random.split(state.key, 3)
        updates_f32 = jax.tree.map(
            _to_f32_leaf,
            updates,
            is_leaf=_is_passthrough_leaf,
        )
        mu_prev = jax.tree.map(
            _dequantize_state_leaf,
            state.mu,
            is_leaf=_is_state_leaf,
        )
        nu_prev = jax.tree.map(
            _dequantize_state_leaf,
            state.nu,
            is_leaf=_is_state_leaf,
        )
        mu_f32 = jax.tree.map(
            lambda g, m: m if _is_passthrough_leaf(m) else b1 * m + (1.0 - b1) * g,
            updates_f32,
            mu_prev,
            is_leaf=_is_passthrough_leaf,
        )
        nu_f32 = jax.tree.map(
            lambda g, v: (
                v if _is_passthrough_leaf(v) else b2 * v + (1.0 - b2) * jnp.square(g)
            ),
            updates_f32,
            nu_prev,
            is_leaf=_is_passthrough_leaf,
        )
        count_inc = cast(jax.Array, numerics.safe_increment(state.count))

        mu_bc_factor = 1.0 - b1**count_inc
        nu_bc_factor = 1.0 - b2**count_inc

        if nesterov:
            mu_bc_factor_next = 1.0 - b1 ** numerics.safe_increment(count_inc)
            mu_bc = _safe_bias_correction(mu_f32, mu_bc_factor_next)
            g_bc = _safe_bias_correction(updates_f32, mu_bc_factor)
            mu_hat = jax.tree.map(
                lambda m, g: m if _is_passthrough_leaf(m) else b1 * m + (1.0 - b1) * g,
                mu_bc,
                g_bc,
                is_leaf=_is_passthrough_leaf,
            )
        else:
            mu_hat = _safe_bias_correction(mu_f32, mu_bc_factor)
        nu_hat = _safe_bias_correction(nu_f32, nu_bc_factor)

        adam_updates = jax.tree.map(
            lambda m, v: (
                m if _is_passthrough_leaf(m) else m / (jnp.sqrt(v + eps_root) + eps)
            ),
            mu_hat,
            nu_hat,
            is_leaf=_is_passthrough_leaf,
        )
        mu_stored = _store_moment_tree(
            mu_f32,
            state.mu,
            block_size=block_size,
            scale_dtype=scale_dtype,
            fallback_dtype=fallback_dtype,
            block_layout=block_layout,
            stochastic_rounding=stochastic_rounding,
            key=key_mu,
        )
        nu_stored = _store_moment_tree(
            nu_f32,
            state.nu,
            block_size=block_size,
            scale_dtype=scale_dtype,
            fallback_dtype=fallback_dtype,
            block_layout=block_layout,
            stochastic_rounding=stochastic_rounding,
            key=key_nu,
        )

        return adam_updates, ScaleByAdam8State(
            count=count_inc,
            mu=mu_stored,
            nu=nu_stored,
            key=next_key,
        )

    return base.GradientTransformation(init_fn, update_fn)


def adamw8(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    weight_decay: base.ScalarOrSchedule = 1e-4,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    block_size: int = 2048,
    min_size: int = 4096,
    scale_dtype: jax.typing.DTypeLike = jnp.float32,
    fallback_dtype: jax.typing.DTypeLike = jnp.float32,
    stochastic_rounding: bool = False,
    block_layout: BlockLayout = "shard_local",
    quantize: bool = True,
    nesterov: bool = False,
    use_magma: bool = False,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """AdamW whose eligible moment leaves use blockwise 8-bit codebook state."""

    if use_magma:
        raise NotImplementedError("Magma is not supported by adamw8 yet.")
    components = [
        scale_by_adam8(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            block_size=block_size,
            min_size=min_size,
            scale_dtype=scale_dtype,
            fallback_dtype=fallback_dtype,
            stochastic_rounding=stochastic_rounding,
            block_layout=block_layout,
            quantize=quantize,
            nesterov=nesterov,
            key=key,
        )
    ]
    weight_decay_is_nonzero = (
        weight_decay > 0.0 if isinstance(weight_decay, (int, float)) else True
    )
    if weight_decay_is_nonzero:
        components.append(
            transform.add_decayed_weights(weight_decay, weight_decay_mask)
        )
    components.append(transform.scale_by_learning_rate(learning_rate))
    return combine.chain(*components)


def _round_scaled(
    scaled: jax.Array,
    *,
    stochastic_rounding: bool,
    key: jax.Array | None,
) -> jax.Array:
    if not stochastic_rounding:
        return jnp.round(scaled)
    if key is None:
        raise ValueError("stochastic rounding requires a PRNG key.")
    lower = jnp.floor(scaled)
    probability = scaled - lower
    return lower + jax.random.bernoulli(key, probability, scaled.shape)


def _dynamic_codebook(*, signed: bool) -> jax.Array:
    return jnp.asarray(_dynamic_codebook_values(signed=signed), dtype=jnp.float32)


def _dynamic_codebook_values(*, signed: bool) -> tuple[float, ...]:
    data: list[float] = []
    total_bits = 8
    max_exponent_bits = 7
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for index in range(max_exponent_bits):
        exponent_scale = 10 ** (-(max_exponent_bits - 1) + index)
        fraction_items = (
            2 ** (index + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (index + non_sign_bits - max_exponent_bits + 1) + 1
        )
        means = _linspace_bin_means(0.1, 1.0, fraction_items)
        data.extend((exponent_scale * value for value in means))
        if signed:
            data.extend((-exponent_scale * value for value in means))
    if additional_items > 0:
        exponent_scale = 10 ** (-(max_exponent_bits - 1) + max_exponent_bits - 1)
        means = _linspace_bin_means(0.1, 1.0, additional_items + 1)
        data.extend((exponent_scale * value for value in means))
        if signed:
            data.extend((-exponent_scale * value for value in means))
    data.append(0.0)
    data.append(1.0)
    if len(data) != 2**total_bits:
        raise AssertionError("dynamic codebook must contain 256 values.")
    return tuple(sorted(data))


def _linspace_bin_means(start: float, stop: float, points: int) -> list[float]:
    if points <= 1:
        return []
    step = (stop - start) / (points - 1)
    boundaries = [start + step * index for index in range(points)]
    return [
        (left + right) / 2.0
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def _codebook_for_quantizer(quantizer: CodebookQuantizer) -> jax.Array:
    if quantizer == "dynamic_signed":
        return _dynamic_codebook(signed=True)
    if quantizer == "dynamic_unsigned":
        return _dynamic_codebook(signed=False)
    raise ValueError("symmetric_int8 does not use a lookup codebook.")


def _codebook_id_for_quantizer(quantizer: CodebookQuantizer) -> str:
    if quantizer == "dynamic_signed":
        return DYNAMIC_SIGNED_CODEBOOK_ID
    if quantizer == "dynamic_unsigned":
        return DYNAMIC_UNSIGNED_CODEBOOK_ID
    if quantizer == "symmetric_int8":
        return SYMMETRIC_INT8_CODEBOOK_ID
    raise ValueError(f"Unknown quantizer: {quantizer!r}.")


def _codebook_for_id(codebook_id: str) -> jax.Array:
    if codebook_id == DYNAMIC_SIGNED_CODEBOOK_ID:
        return _dynamic_codebook(signed=True)
    if codebook_id == DYNAMIC_UNSIGNED_CODEBOOK_ID:
        return _dynamic_codebook(signed=False)
    raise ValueError(f"Unknown codebook_id: {codebook_id!r}.")


def _zero_code_for_quantizer(quantizer: CodebookQuantizer) -> int:
    if quantizer == "dynamic_signed":
        codebook = _dynamic_codebook_values(signed=True)
    elif quantizer == "dynamic_unsigned":
        codebook = _dynamic_codebook_values(signed=False)
    else:
        return 0
    return min(range(len(codebook)), key=lambda index: abs(codebook[index]))


def _quantize_to_codebook(
    normalized: jax.Array,
    codebook: jax.Array,
    *,
    stochastic_rounding: bool,
    key: jax.Array | None,
) -> jax.Array:
    upper_index = jnp.searchsorted(codebook, normalized, side="left")
    upper_index = jnp.clip(upper_index, 0, codebook.shape[0] - 1)
    lower_index = jnp.clip(upper_index - 1, 0, codebook.shape[0] - 1)
    lower_value = codebook[lower_index]
    upper_value = codebook[upper_index]
    if stochastic_rounding:
        if key is None:
            raise ValueError("stochastic rounding requires a PRNG key.")
        denominator = upper_value - lower_value
        probability = jnp.where(
            denominator > 0,
            (normalized - lower_value) / denominator,
            0.0,
        )
        probability = jnp.clip(probability, 0.0, 1.0)
        choose_upper = jax.random.bernoulli(key, probability, normalized.shape)
    else:
        choose_upper = jnp.abs(upper_value - normalized) < jnp.abs(
            normalized - lower_value
        )
    return jnp.where(choose_upper, upper_index, lower_index).astype(jnp.uint8)


def _init_moment_tree(
    params: Any,
    *,
    block_size: int,
    min_size: int,
    scale_dtype: jax.typing.DTypeLike,
    fallback_dtype: jax.typing.DTypeLike,
    block_layout: BlockLayout,
    quantize: bool,
    quantizer: CodebookQuantizer,
    key: jax.Array,
) -> Any:
    leaves, treedef = jax.tree.flatten(params, is_leaf=_is_passthrough_leaf)
    keys = tuple(jax.random.split(key, len(leaves))) if leaves else ()
    key_tree = treedef.unflatten(keys)
    return jax.tree.map(
        lambda p, k: _init_moment_leaf(
            p,
            block_size=block_size,
            min_size=min_size,
            scale_dtype=scale_dtype,
            fallback_dtype=fallback_dtype,
            block_layout=block_layout,
            quantize=quantize,
            quantizer=quantizer,
            key=k,
        ),
        params,
        key_tree,
        is_leaf=_is_passthrough_leaf,
    )


def _init_moment_leaf(
    param: Any,
    *,
    block_size: int,
    min_size: int,
    scale_dtype: jax.typing.DTypeLike,
    fallback_dtype: jax.typing.DTypeLike,
    block_layout: BlockLayout,
    quantize: bool,
    quantizer: CodebookQuantizer,
    key: jax.Array,
) -> Any:
    if _is_passthrough_leaf(param):
        return param
    zeros = zeros_like_preserving_sharding(param, fallback_dtype)
    if quantize and param.size >= min_size and jnp.issubdtype(param.dtype, jnp.inexact):
        return quantize_blocks(
            zeros,
            block_size=block_size,
            scale_dtype=scale_dtype,
            stochastic_rounding=False,
            key=key,
            block_layout=block_layout,
            quantizer=quantizer,
        )
    return zeros


def _store_moment_tree(
    moment: Any,
    template: Any,
    *,
    block_size: int,
    scale_dtype: jax.typing.DTypeLike,
    fallback_dtype: jax.typing.DTypeLike,
    block_layout: BlockLayout,
    stochastic_rounding: bool,
    key: jax.Array,
) -> Any:
    leaves, treedef = jax.tree.flatten(template, is_leaf=_is_state_leaf)
    keys = tuple(jax.random.split(key, len(leaves))) if leaves else ()
    key_tree = treedef.unflatten(keys)
    return jax.tree.map(
        lambda m, t, k: _store_moment_leaf(
            m,
            t,
            block_size=block_size,
            scale_dtype=scale_dtype,
            fallback_dtype=fallback_dtype,
            block_layout=block_layout,
            stochastic_rounding=stochastic_rounding,
            key=k,
        ),
        moment,
        template,
        key_tree,
        is_leaf=_is_state_leaf,
    )


def _store_moment_leaf(
    moment: Any,
    template: Any,
    *,
    block_size: int,
    scale_dtype: jax.typing.DTypeLike,
    fallback_dtype: jax.typing.DTypeLike,
    block_layout: BlockLayout,
    stochastic_rounding: bool,
    key: jax.Array,
) -> Any:
    if isinstance(template, QuantizedBlocks):
        return quantize_blocks(
            moment,
            block_size=block_size,
            scale_dtype=scale_dtype,
            stochastic_rounding=stochastic_rounding,
            key=key,
            block_layout=block_layout,
            quantizer=_quantizer_from_state_leaf(template),
        )
    if _is_passthrough_leaf(template):
        return template
    return moment.astype(fallback_dtype)


def _dequantize_state_leaf(leaf: Any) -> Any:
    if isinstance(leaf, QuantizedBlocks):
        return dequantize_blocks(leaf, dtype=jnp.float32)
    if _is_passthrough_leaf(leaf):
        return leaf
    return leaf.astype(jnp.float32)


def _to_f32_leaf(leaf: Any) -> Any:
    if _is_passthrough_leaf(leaf):
        return leaf
    return leaf.astype(jnp.float32)


def _is_passthrough_leaf(x: Any) -> bool:
    return isinstance(x, _masking.MaskedNode) or x is None


def _is_state_leaf(x: Any) -> bool:
    return isinstance(x, QuantizedBlocks) or _is_passthrough_leaf(x)


def _quantizer_from_state_leaf(leaf: QuantizedBlocks) -> CodebookQuantizer:
    if leaf.codebook_id == DYNAMIC_SIGNED_CODEBOOK_ID:
        return "dynamic_signed"
    if leaf.codebook_id == DYNAMIC_UNSIGNED_CODEBOOK_ID:
        return "dynamic_unsigned"
    if leaf.codebook_id == SYMMETRIC_INT8_CODEBOOK_ID:
        return "symmetric_int8"
    raise ValueError(f"Unknown codebook_id: {leaf.codebook_id!r}.")


__all__ = (
    "QuantizedBlocks",
    "BlockLayout",
    "CodebookQuantizer",
    "ScaleByAdam8State",
    "DYNAMIC_SIGNED_CODEBOOK_ID",
    "DYNAMIC_UNSIGNED_CODEBOOK_ID",
    "SYMMETRIC_INT8_CODEBOOK_ID",
    "adamw8",
    "dequantize_blocks",
    "estimate_quantized_moment_bytes",
    "quantize_blocks",
    "quantized_nbytes",
    "scale_by_adam8",
    "tree_state_nbytes",
)
