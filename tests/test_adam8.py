import jax
import jax.numpy as jnp
import optax

from rollfast.optim.adam import adamw
from rollfast.optim.adam8 import (
    QuantizedBlocks,
    SYMMETRIC_INT8_CODEBOOK_ID,
    adamw8,
    dequantize_blocks,
    quantize_blocks,
)


def test_blockwise_quantize_dequantize_error_is_bounded_by_half_scale():
    x = jnp.linspace(-3.0, 2.0, 513, dtype=jnp.float32)
    blocks = quantize_blocks(
        x,
        block_size=128,
        stochastic_rounding=False,
        key=jax.random.PRNGKey(0),
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
