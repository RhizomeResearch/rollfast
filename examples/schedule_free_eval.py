"""Train with Schedule-Free Adam and read the averaged eval parameters."""

import jax.numpy as jnp
import optax

from rollfast import schedule_free_adam, schedule_free_eval_params


def main() -> None:
    params = {"w": jnp.ones((2, 2))}
    grads = {"w": jnp.full((2, 2), 0.1)}

    optimizer = schedule_free_adam(
        learning_rate=1e-2,
        total_steps=20,
        warmup_fraction=0.0,
        decay_fraction=0.0,
    )
    state = optimizer.init(params)

    for _ in range(5):
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)

    eval_params = schedule_free_eval_params(state, params)

    print("train_w_mean:", params["w"].mean())
    print("eval_w_mean:", eval_params["w"].mean())


if __name__ == "__main__":
    main()
