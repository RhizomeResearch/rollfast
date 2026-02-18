import jax.numpy as jnp
import optax


def wsd_schedule(
    peak_lr: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
) -> optax.Schedule:
    """Creates a Warmup-Stable-Decay (Trapezoidal) schedule.

    This schedule is recommended for Schedule-Free optimization. It consists of:
    1. Linear warmup.
    2. Constant (stable) phase.
    3. Linear decay (cooldown) phase.

    Args:
        peak_lr: The constant learning rate during the stable phase.
        total_steps: Total training steps T.
        warmup_fraction: Fraction of steps for linear warmup.
        decay_fraction: Fraction of steps for linear decay (cooldown).

    Returns:
        An optax schedule function.
    """
    warmup_steps = int(total_steps * warmup_fraction)
    decay_steps = int(total_steps * decay_fraction)

    T_w = warmup_steps
    T_c = total_steps - decay_steps
    T_final = total_steps - 1

    def schedule(count):
        # Case 1: 0 <= t <= T_w (Warmup)
        # Eq: (t + 1) / (T_w + 1)
        warmup_val = (count + 1.0) / (T_w + 1.0) * peak_lr

        # Case 2: T_w < t <= T_c (Stable)
        stable_val = peak_lr

        # Case 3: T_c < t <= T (Decay)
        # Eq: (T - t + 1) / (T - T_c + 1)
        decay_val = (T_final - count + 1.0) / (T_final - T_c + 1.0) * peak_lr

        is_warmup = count <= T_w
        is_decay = count > T_c

        val = jnp.where(is_warmup, warmup_val, stable_val)
        val = jnp.where(is_decay, decay_val, val)

        # Ensure we don't go negative or process past T_final roughly
        return jnp.maximum(0.0, val)

    return schedule
