import jax.numpy as jnp
import optax


def wsd_schedule(
    peak_lr: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    *,
    warmup_steps: int | None = None,
    decay_steps: int | None = None,
    end_lr_ratio: float = 0.0,
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
        warmup_steps: Explicit warmup step count. Overrides ``warmup_fraction``.
        decay_steps: Explicit decay step count. Overrides ``decay_fraction``.
        end_lr_ratio: Final learning-rate ratio at the last scheduled step.

    Returns:
        An optax schedule function.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("warmup_fraction must satisfy 0 <= value <= 1.")
    if not 0.0 <= decay_fraction <= 1.0:
        raise ValueError("decay_fraction must satisfy 0 <= value <= 1.")
    if end_lr_ratio < 0.0:
        raise ValueError("end_lr_ratio must be non-negative.")

    resolved_warmup_steps = (
        int(total_steps * warmup_fraction)
        if warmup_steps is None
        else int(warmup_steps)
    )
    resolved_decay_steps = (
        int(total_steps * decay_fraction)
        if decay_steps is None
        else int(decay_steps)
    )
    if resolved_warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if resolved_decay_steps < 0:
        raise ValueError("decay_steps must be non-negative.")
    if resolved_warmup_steps + resolved_decay_steps > total_steps:
        raise ValueError("warmup_steps + decay_steps must not exceed total_steps.")

    decay_start = total_steps - resolved_decay_steps
    end_lr = peak_lr * end_lr_ratio

    def schedule(count):
        count = jnp.asarray(count, dtype=jnp.float32)
        # Case 1: 0 <= t < warmup_steps.
        if resolved_warmup_steps > 0:
            warmup_val = (count + 1.0) / float(resolved_warmup_steps) * peak_lr
        else:
            warmup_val = jnp.asarray(peak_lr, dtype=jnp.float32)

        # Case 2: warmup_steps <= t < decay_start.
        stable_val = peak_lr

        # Case 3: decay_start <= t < total_steps.
        if resolved_decay_steps > 0:
            progress = (count - float(decay_start) + 1.0) / float(
                resolved_decay_steps
            )
            progress = jnp.clip(progress, 0.0, 1.0)
            decay_val = peak_lr + (end_lr - peak_lr) * progress
        else:
            decay_val = jnp.asarray(end_lr, dtype=jnp.float32)

        is_warmup = count < resolved_warmup_steps
        is_decay = count >= decay_start

        val = jnp.where(is_warmup, warmup_val, stable_val)
        val = jnp.where(is_decay, decay_val, val)

        return jnp.maximum(0.0, val)

    return schedule
