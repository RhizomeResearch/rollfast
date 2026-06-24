"""Emit toy quadratic convergence smoke benchmarks for AdamW and AdamW8."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

import rollfast
from _common import emit, metadata


def _loss(params):
    return jnp.mean((params["w"] - 0.125) ** 2)


def _run(tx, params, *, steps: int):
    state = tx.init(params)
    value_and_grad = jax.value_and_grad(_loss)
    initial = _loss(params)
    for _ in range(steps):
        value, grads = value_and_grad(params)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    return {
        "initial_loss": float(initial),
        "final_loss": float(value),
        "final_param_mean": float(jnp.mean(params["w"])),
    }


def main() -> None:
    steps = 25
    params = {"w": jnp.ones((8192,), dtype=jnp.float32)}
    adamw = rollfast.adamw(learning_rate=1e-2, weight_decay=0.0)
    adamw8 = rollfast.adamw8(
        learning_rate=1e-2,
        weight_decay=0.0,
        block_size=2048,
        min_size=4096,
        stochastic_rounding=False,
    )
    emit(
        {
            "metadata": metadata(warmup_steps=0, measured_steps=steps),
            "scenarios": {
                "adamw": _run(adamw, params, steps=steps),
                "adamw8": _run(adamw8, params, steps=steps),
            },
            "notes": [
                "Toy convex smoke test; not a task-quality convergence benchmark.",
            ],
        }
    )


if __name__ == "__main__":
    main()
