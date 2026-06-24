"""Compare fp32 AdamW state bytes with blockwise 8-bit AdamW state bytes."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

import rollfast
from rollfast.optim.adam8 import tree_state_nbytes


def main() -> None:
    params = {
        "w0": jnp.ones((4096, 64), dtype=jnp.float32),
        "w1": jnp.ones((4096, 64), dtype=jnp.float32) * 0.5,
        "bias": jnp.ones((1024,), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.ones_like, params)
    fp32 = rollfast.adamw(learning_rate=1e-3, weight_decay=0.01)
    q8 = rollfast.adamw8(
        learning_rate=1e-3,
        weight_decay=0.01,
        block_size=2048,
        min_size=4096,
        stochastic_rounding=False,
    )

    fp32_state = fp32.init(params)
    q8_state = q8.init(params)
    print(
        {
            "fp32_state_bytes": tree_state_nbytes(fp32_state),
            "adamw8_state_bytes": tree_state_nbytes(q8_state),
        }
    )

    fp32_step = jax.jit(lambda s, p: fp32.update(grads, s, p))
    q8_step = jax.jit(lambda s, p: q8.update(grads, s, p))
    fp32_updates, fp32_state = fp32_step(fp32_state, params)
    q8_updates, q8_state = q8_step(q8_state, params)
    jax.block_until_ready(jax.tree.leaves(fp32_updates)[0])
    jax.block_until_ready(jax.tree.leaves(q8_updates)[0])

    for name, step, state in (
        ("fp32", fp32_step, fp32_state),
        ("adamw8", q8_step, q8_state),
    ):
        started = time.perf_counter()
        updates, next_state = step(state, params)
        jax.block_until_ready(jax.tree.leaves(updates)[0])
        elapsed = time.perf_counter() - started
        print({f"{name}_step_seconds": elapsed})
        del next_state


if __name__ == "__main__":
    main()
