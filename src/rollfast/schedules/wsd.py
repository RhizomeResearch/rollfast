from typing import Literal

import jax.numpy as jnp
import optax

ScheduleShape = Literal["linear", "cosine", "power", "exponential"]


def _validate_shape(name: str, shape: str) -> None:
    if shape not in ("linear", "cosine", "power", "exponential"):
        raise ValueError(
            f"{name} must be one of 'linear', 'cosine', 'power', or "
            f"'exponential', got {shape!r}."
        )


def _shaped_progress(
    progress,
    shape: ScheduleShape,
    power: float,
    exponential_rate: float,
):
    progress = jnp.clip(progress, 0.0, 1.0)
    if shape == "linear":
        return progress
    if shape == "cosine":
        return 0.5 * (1.0 - jnp.cos(jnp.pi * progress))
    if shape == "power":
        return jnp.power(progress, power)

    rate = jnp.asarray(exponential_rate, dtype=jnp.float32)
    return jnp.expm1(rate * progress) / jnp.expm1(rate)


def wsd_schedule(
    peak_lr: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    warmup_shape: ScheduleShape = "linear",
    decay_shape: ScheduleShape = "linear",
    final_lr_ratio: float = 0.0,
    warmup_power: float = 1.0,
    decay_power: float = 1.0,
    exponential_rate: float = 5.0,
) -> optax.Schedule:
    """Creates a Warmup-Stable-Decay schedule.

    It consists of:
    1. Linear warmup.
    2. Constant (stable) phase.
    3. Configurable decay (cooldown) phase.

    Args:
        peak_lr: The constant learning rate during the stable phase.
        total_steps: Total training steps T.
        warmup_fraction: Fraction of steps for warmup. The schedule uses
            ``warmup_steps = int(total_steps * warmup_fraction)`` and treats
            counts ``0..warmup_steps`` as warmup.
        decay_fraction: Fraction of steps for decay (cooldown). The schedule
            uses ``decay_steps = int(total_steps * decay_fraction)`` and treats
            counts after ``total_steps - decay_steps`` as decay; count
            ``total_steps - 1`` reaches ``final_lr_ratio * peak_lr``.
        warmup_shape: Warmup interpolation shape.
        decay_shape: Decay interpolation shape.
        final_lr_ratio: Final learning-rate ratio relative to ``peak_lr``.
        warmup_power: Exponent used when ``warmup_shape="power"``.
        decay_power: Exponent used when ``decay_shape="power"``.
        exponential_rate: Curvature used by exponential warmup/decay shapes.

    Returns:
        An optax schedule function.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("warmup_fraction must be in [0, 1].")
    if not 0.0 <= decay_fraction <= 1.0:
        raise ValueError("decay_fraction must be in [0, 1].")
    if warmup_fraction + decay_fraction > 1.0:
        raise ValueError("warmup_fraction + decay_fraction must be <= 1.")
    if not 0.0 <= final_lr_ratio <= 1.0:
        raise ValueError("final_lr_ratio must be in [0, 1].")
    if warmup_power <= 0.0:
        raise ValueError("warmup_power must be positive.")
    if decay_power <= 0.0:
        raise ValueError("decay_power must be positive.")
    if exponential_rate == 0.0:
        raise ValueError("exponential_rate must be nonzero.")
    _validate_shape("warmup_shape", warmup_shape)
    _validate_shape("decay_shape", decay_shape)

    warmup_steps = int(total_steps * warmup_fraction)
    decay_steps = int(total_steps * decay_fraction)

    T_w = warmup_steps
    T_c = total_steps - decay_steps
    T_final = total_steps - 1
    decay_denominator = max(T_final - T_c, 1)

    def schedule(count):
        warmup_progress = (count + 1.0) / (T_w + 1.0)
        warmup_val = (
            _shaped_progress(
                warmup_progress,
                warmup_shape,
                warmup_power,
                exponential_rate,
            )
            * peak_lr
        )

        stable_val = peak_lr

        decay_progress = (count - T_c) / decay_denominator
        decay_fraction_done = _shaped_progress(
            decay_progress,
            decay_shape,
            decay_power,
            exponential_rate,
        )
        decay_factor = final_lr_ratio + (1.0 - final_lr_ratio) * (
            1.0 - decay_fraction_done
        )
        decay_val = decay_factor * peak_lr

        is_warmup = count <= T_w
        is_decay = count > T_c

        val = jnp.where(is_warmup, warmup_val, stable_val)
        val = jnp.where(is_decay, decay_val, val)

        return jnp.maximum(0.0, val)

    return schedule


def _make_wsd_schedule_pair(
    *,
    learning_rate: float,
    adam_learning_rate: float | None,
    total_steps: int,
    warmup_fraction: float,
    decay_fraction: float,
) -> tuple[optax.Schedule, optax.Schedule]:
    """Build matrix/fallback WSD schedules with shared defaults."""
    matrix_schedule = wsd_schedule(
        peak_lr=learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )
    if adam_learning_rate is None or adam_learning_rate == learning_rate:
        return matrix_schedule, matrix_schedule
    return matrix_schedule, wsd_schedule(
        peak_lr=adam_learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )


def power_decay_schedule(
    peak_lr: float,
    total_steps: int,
    power: float,
    warmup_fraction: float = 0.0,
    warmup_shape: ScheduleShape = "linear",
    final_lr_ratio: float = 0.0,
    warmup_power: float = 1.0,
    exponential_rate: float = 5.0,
) -> optax.Schedule:
    """Creates a finite-horizon power decay schedule.

    After optional warmup, the learning rate follows
    ``peak_lr * (1 - progress) ** power`` down to ``final_lr_ratio * peak_lr``.

    Args:
        peak_lr: Learning rate reached after warmup.
        total_steps: Total training steps.
        power: Decay exponent.
        warmup_fraction: Fraction of steps for warmup.
        warmup_shape: Warmup interpolation shape.
        final_lr_ratio: Final learning-rate ratio relative to ``peak_lr``.
        warmup_power: Exponent used when ``warmup_shape="power"``.
        exponential_rate: Curvature used by exponential warmup shape.

    Returns:
        An optax schedule function.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    if power <= 0.0:
        raise ValueError("power must be positive.")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("warmup_fraction must be in [0, 1].")
    if not 0.0 <= final_lr_ratio <= 1.0:
        raise ValueError("final_lr_ratio must be in [0, 1].")
    if warmup_power <= 0.0:
        raise ValueError("warmup_power must be positive.")
    if exponential_rate == 0.0:
        raise ValueError("exponential_rate must be nonzero.")
    _validate_shape("warmup_shape", warmup_shape)

    warmup_steps = int(total_steps * warmup_fraction)
    T_w = warmup_steps
    T_final = total_steps - 1
    decay_denominator = max(T_final - T_w, 1)

    def schedule(count):
        warmup_progress = (count + 1.0) / (T_w + 1.0)
        warmup_val = (
            _shaped_progress(
                warmup_progress,
                warmup_shape,
                warmup_power,
                exponential_rate,
            )
            * peak_lr
        )

        decay_progress = jnp.clip((count - T_w) / decay_denominator, 0.0, 1.0)
        decay_factor = jnp.power(1.0 - decay_progress, power)
        decay_factor = final_lr_ratio + (1.0 - final_lr_ratio) * decay_factor
        decay_val = peak_lr * decay_factor

        val = jnp.where(count <= T_w, warmup_val, decay_val)
        return jnp.maximum(0.0, val)

    return schedule


__all__ = [
    "ScheduleShape",
    "power_decay_schedule",
    "wsd_schedule",
]
