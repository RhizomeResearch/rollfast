import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rollfast.optim.adam import adamw
from rollfast.optim.adam8 import (
    DYNAMIC_SIGNED_CODEBOOK_ID,
    DYNAMIC_UNSIGNED_CODEBOOK_ID,
    SYMMETRIC_INT8_CODEBOOK_ID,
    QuantizedBlocks,
    _dynamic_codebook_values,
    adamw8,
    dequantize_blocks,
    quantize_blocks,
    scale_by_adam8,
)


@pytest.mark.parametrize(
    ("signed", "digest"),
    [
        (False, "02a9b523750e21348029626da435f2adadb851b687d86349cd20752f9255de26"),
        (True, "035c4114538df2a03bbcb4d8a39b9325d26d84a00cf6cf29018d1a6608ab14c4"),
    ],
)
def test_dynamic_v1_codebook_values_are_checkpoint_stable(signed, digest):
    # Stored indices and v1 codebook IDs must continue decoding to these values.
    values = np.asarray(_dynamic_codebook_values(signed=signed), dtype="<f4")
    assert values.shape == (256,)
    assert hashlib.sha256(values.tobytes()).hexdigest() == digest


def test_legacy_symmetric_int8_quantize_dequantize_error_is_bounded_by_half_scale():
    x = jnp.linspace(-3.0, 2.0, 513, dtype=jnp.float32)
    blocks = quantize_blocks(
        x,
        block_size=128,
        stochastic_rounding=False,
        key=jax.random.PRNGKey(0),
        quantizer="symmetric_int8",
    )
    restored = dequantize_blocks(blocks)

    assert isinstance(blocks, QuantizedBlocks)
    assert blocks.values.dtype == jnp.int8
    assert blocks.scales.dtype == jnp.float32
    assert blocks.shape == x.shape
    assert blocks.size == x.size
    assert blocks.quantizer == "blockwise_symmetric_int8"
    assert blocks.codebook_id == SYMMETRIC_INT8_CODEBOOK_ID
    assert blocks.qmin == -127
    assert blocks.qmax == 127
    assert blocks.zero_point == 0
    assert blocks.block_layout == "shard_local"
    assert blocks.metadata()["codebook_id"] == SYMMETRIC_INT8_CODEBOOK_ID
    assert blocks.metadata()["block_layout"] == "shard_local"
    assert jnp.max(jnp.abs(restored - x)) <= jnp.max(blocks.scales) / 2 + 1e-6


def test_default_dynamic_codebook_quantizer_metadata_and_error_bound():
    x = jnp.linspace(-3.0, 2.0, 513, dtype=jnp.float32)
    blocks = quantize_blocks(
        x,
        block_size=128,
        stochastic_rounding=False,
        key=jax.random.PRNGKey(0),
    )
    restored = dequantize_blocks(blocks)

    assert isinstance(blocks, QuantizedBlocks)
    assert blocks.values.dtype == jnp.uint8
    assert blocks.scales.dtype == jnp.float32
    assert blocks.quantizer == "blockwise_dynamic_signed_8bit"
    assert blocks.codebook_id == DYNAMIC_SIGNED_CODEBOOK_ID
    assert blocks.qmin == 0
    assert blocks.qmax == 255
    assert blocks.zero_point == 127
    assert blocks.block_layout == "shard_local"
    assert blocks.metadata()["codebook_id"] == DYNAMIC_SIGNED_CODEBOOK_ID
    assert jnp.max(jnp.abs(restored - x)) <= jnp.max(blocks.scales) * 0.01


def test_dynamic_unsigned_quantizer_keeps_nonnegative_values():
    x = jnp.linspace(0.0, 3.0, 513, dtype=jnp.float32)
    blocks = quantize_blocks(
        x,
        block_size=128,
        stochastic_rounding=False,
        key=jax.random.PRNGKey(0),
        quantizer="dynamic_unsigned",
    )
    restored = dequantize_blocks(blocks)

    assert blocks.values.dtype == jnp.uint8
    assert blocks.quantizer == "blockwise_dynamic_unsigned_8bit"
    assert blocks.codebook_id == DYNAMIC_UNSIGNED_CODEBOOK_ID
    assert blocks.zero_point == 0
    assert jnp.min(restored) >= 0.0
    assert jnp.max(jnp.abs(restored - x)) <= jnp.max(blocks.scales) * 0.01


def test_scale_by_adam8_uses_signed_first_and_unsigned_second_moments():
    params = {
        "w": jnp.linspace(-1.0, 1.0, 4096, dtype=jnp.float32).reshape(64, 64),
    }
    tx = scale_by_adam8(
        block_size=256,
        min_size=1,
        stochastic_rounding=False,
    )
    state = tx.init(params)
    mu_blocks = [
        leaf
        for leaf in jax.tree.leaves(
            state.mu,
            is_leaf=lambda x: isinstance(x, QuantizedBlocks),
        )
        if isinstance(leaf, QuantizedBlocks)
    ]
    nu_blocks = [
        leaf
        for leaf in jax.tree.leaves(
            state.nu,
            is_leaf=lambda x: isinstance(x, QuantizedBlocks),
        )
        if isinstance(leaf, QuantizedBlocks)
    ]

    assert {leaf.codebook_id for leaf in mu_blocks} == {DYNAMIC_SIGNED_CODEBOOK_ID}
    assert {leaf.codebook_id for leaf in nu_blocks} == {DYNAMIC_UNSIGNED_CODEBOOK_ID}
    assert {leaf.values.dtype for leaf in mu_blocks + nu_blocks} == {
        jnp.dtype(jnp.uint8)
    }


def test_adamw8_second_step_is_close_to_fp32_adamw():
    params = {
        "w": jnp.linspace(-1.0, 1.0, 4096, dtype=jnp.float32).reshape(64, 64),
    }
    grads = {"w": jnp.ones((64, 64), dtype=jnp.float32) * 0.125}
    fp32 = adamw(
        learning_rate=1e-3,
        weight_decay=0.01,
        eps=1e-6,
    )
    quantized = adamw8(
        learning_rate=1e-3,
        weight_decay=0.01,
        eps=1e-6,
        block_size=256,
        min_size=1,
        stochastic_rounding=False,
    )
    fp32_state = fp32.init(params)
    q_state = quantized.init(params)
    fp32_updates, fp32_state = fp32.update(grads, fp32_state, params)
    q_updates, q_state = quantized.update(grads, q_state, params)
    fp32_params = optax.apply_updates(params, fp32_updates)
    q_params = optax.apply_updates(params, q_updates)

    fp32_updates, _ = fp32.update(grads, fp32_state, fp32_params)
    q_updates, _ = quantized.update(grads, q_state, q_params)

    assert jnp.allclose(q_updates["w"], fp32_updates["w"], rtol=0.03, atol=3e-4)
