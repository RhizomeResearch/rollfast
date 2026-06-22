"""Schedule factory and previews for fine-tuning builders."""

from __future__ import annotations

import jax.numpy as jnp
import optax

from rollfast.schedules.wsd import wsd_schedule

from .config import ScheduleConfig, SchedulePoint


def build_schedule(
    config: ScheduleConfig,
    *,
    peak_lr: float,
    total_steps: int | None = None,
) -> optax.Schedule:
    """Build an Optax schedule from a serializable config."""

    config = config.resolved(total_steps)
    if config.kind == "constant":
        return _constant_schedule(peak_lr)
    if config.total_steps is None:
        raise ValueError(f"{config.kind!r} schedule requires total_steps.")
    if config.kind == "warmup_cosine":
        return _warmup_cosine_schedule(config, peak_lr)
    if config.kind == "wsd":
        return wsd_schedule(
            peak_lr=peak_lr,
            total_steps=config.total_steps,
            warmup_fraction=config.warmup_fraction,
            decay_fraction=config.decay_fraction,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
        )
    if config.kind == "linear":
        return _linear_schedule(config, peak_lr)
    if config.kind == "polynomial":
        return _polynomial_schedule(config, peak_lr)
    raise ValueError(f"Unsupported schedule kind: {config.kind!r}.")


def preview_schedule(
    schedule_or_config: optax.Schedule | ScheduleConfig,
    *,
    peak_lr: float | None = None,
    total_steps: int | None = None,
    steps: tuple[int, ...] | None = None,
) -> tuple[SchedulePoint, ...]:
    """Return numeric schedule values at representative steps."""

    if isinstance(schedule_or_config, ScheduleConfig):
        if peak_lr is None:
            raise ValueError("peak_lr is required when previewing a ScheduleConfig.")
        schedule = build_schedule(
            schedule_or_config,
            peak_lr=peak_lr,
            total_steps=total_steps,
        )
        resolved_total = schedule_or_config.resolved(total_steps).total_steps
    else:
        schedule = schedule_or_config
        resolved_total = total_steps

    if steps is None:
        steps = _default_preview_steps(resolved_total)
    return tuple(
        SchedulePoint(step=int(step), value=float(schedule(jnp.asarray(step))))
        for step in steps
    )


def _default_preview_steps(total_steps: int | None) -> tuple[int, ...]:
    if total_steps is None:
        return (0, 1, 10)
    if total_steps <= 1:
        return (0,)
    candidates = {
        0,
        max(0, total_steps // 20),
        max(0, total_steps // 2),
        max(0, total_steps - 1),
    }
    return tuple(sorted(candidates))


def _constant_schedule(peak_lr: float) -> optax.Schedule:
    def schedule(count):
        del count
        return jnp.asarray(peak_lr, dtype=jnp.float32)

    return schedule


def _warmup_cosine_schedule(config: ScheduleConfig, peak_lr: float) -> optax.Schedule:
    total_steps = int(config.total_steps or 1)
    if total_steps <= 1:
        return _constant_schedule(peak_lr)
    warmup_steps = _resolve_warmup_steps(config, total_steps)
    end_lr = peak_lr * config.end_lr_ratio

    def schedule(count):
        count = jnp.asarray(count, dtype=jnp.float32)
        if warmup_steps > 0:
            warmup = peak_lr * (count + 1.0) / float(warmup_steps)
        else:
            warmup = jnp.asarray(peak_lr, dtype=jnp.float32)

        decay_start = float(warmup_steps)
        decay_span = max(total_steps - warmup_steps - 1, 1)
        progress = jnp.clip((count - decay_start) / float(decay_span), 0.0, 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        decayed = end_lr + (peak_lr - end_lr) * cosine
        return jnp.where(count < warmup_steps, warmup, decayed)

    return schedule


def _linear_schedule(config: ScheduleConfig, peak_lr: float) -> optax.Schedule:
    total_steps = int(config.total_steps or 1)
    if total_steps <= 1:
        return _constant_schedule(peak_lr)
    warmup_steps = _resolve_warmup_steps(config, total_steps)
    end_lr = peak_lr * config.end_lr_ratio

    def schedule(count):
        count = jnp.asarray(count, dtype=jnp.float32)
        if warmup_steps > 0:
            warmup = peak_lr * (count + 1.0) / float(warmup_steps)
        else:
            warmup = jnp.asarray(peak_lr, dtype=jnp.float32)

        decay_span = max(total_steps - warmup_steps - 1, 1)
        progress = jnp.clip((count - float(warmup_steps)) / float(decay_span), 0.0, 1.0)
        decayed = peak_lr + (end_lr - peak_lr) * progress
        return jnp.where(count < warmup_steps, warmup, decayed)

    return schedule


def _polynomial_schedule(config: ScheduleConfig, peak_lr: float) -> optax.Schedule:
    total_steps = int(config.total_steps or 1)
    if total_steps <= 1:
        return _constant_schedule(peak_lr)
    warmup_steps = _resolve_warmup_steps(config, total_steps)
    end_lr = peak_lr * config.end_lr_ratio

    def schedule(count):
        count = jnp.asarray(count, dtype=jnp.float32)
        if warmup_steps > 0:
            warmup = peak_lr * (count + 1.0) / float(warmup_steps)
        else:
            warmup = jnp.asarray(peak_lr, dtype=jnp.float32)

        decay_span = max(total_steps - warmup_steps - 1, 1)
        progress = jnp.clip((count - float(warmup_steps)) / float(decay_span), 0.0, 1.0)
        decayed = end_lr + (peak_lr - end_lr) * (1.0 - progress) ** config.power
        return jnp.where(count < warmup_steps, warmup, decayed)

    return schedule


def _resolve_warmup_steps(config: ScheduleConfig, total_steps: int) -> int:
    if config.warmup_steps is not None:
        return min(config.warmup_steps, total_steps)
    return min(int(total_steps * config.warmup_fraction), total_steps)


__all__ = ("build_schedule", "preview_schedule")
