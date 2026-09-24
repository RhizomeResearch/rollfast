"""Apply a few AdamW steps to a small parameter PyTree."""

import jax
import jax.numpy as jnp

from rollfast import adamw, apply_updates


def main() -> None:
    params = {
        "linear": {
            "w": jnp.ones((4, 3)),
            "b": jnp.zeros((4,)),
        }
    }
    grads = jax.tree.map(lambda x: jnp.full_like(x, 0.1), params)
    decay_mask = {"linear": {"w": True, "b": False}}

    optimizer = adamw(
        learning_rate=1e-2,
        weight_decay=0.01,
        weight_decay_mask=decay_mask,
    )
    state = optimizer.init(params)

    for _ in range(3):
        updates, state = optimizer.update(grads, state, params)
        params = apply_updates(params, updates, stochastic=False)

    print("weight_mean:", params["linear"]["w"].mean())
    print("bias_mean:", params["linear"]["b"].mean())


if __name__ == "__main__":
    main()
