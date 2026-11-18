from typing import Literal

import jax.numpy as jnp
import optax

AfterEnd = Literal["clamp", "cycle"]


def cosine_annealing_schedule(
    peak_lr: float,
    t_max: int,
    *,
    end_lr_ratio: float = 0.0,
    warmup_steps: int = 0,
    after_end: AfterEnd = "clamp",
) -> optax.Schedule:
    """Creates PyTorch's ``CosineAnnealingLR`` schedule, with optional warmup.

    After ``warmup_steps`` linear warmup slots at ``peak_lr * (count + 1) /
    warmup_steps``, slot ``i = count - warmup_steps`` follows PyTorch's closed
    form::

        end_lr + (peak_lr - end_lr) * (1 + cos(pi * i / t_max)) / 2

    with ``end_lr = end_lr_ratio * peak_lr``. Unlike Rollfast's
    ``warmup_cosine``, which reaches ``end_lr`` on its last slot, the floor is
    reached only at ``i = t_max``: a run of ``t_max`` slots ends at
    ``(1 + cos(pi * (t_max - 1) / t_max)) / 2`` of the way from ``end_lr`` to
    ``peak_lr``. PyTorch's scheduler computes the same values recursively, so
    its floats can differ from this closed form in the last bits.

    Args:
        peak_lr: Learning rate at the first slot after warmup (PyTorch's base
            learning rate).
        t_max: Slots from the peak to the floor (PyTorch's ``T_max``).
        end_lr_ratio: Floor relative to ``peak_lr`` (PyTorch's
            ``eta_min / base_lr``).
        warmup_steps: Linear warmup slots before the cosine starts.
        after_end: ``"clamp"`` holds ``end_lr`` after ``t_max``; ``"cycle"``
            keeps following the cosine back up, as PyTorch does.

    Returns:
        An optax schedule function.
    """
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}.")
    if end_lr_ratio < 0.0:
        raise ValueError("end_lr_ratio must be non-negative.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if after_end not in ("clamp", "cycle"):
        raise ValueError(f"after_end must be 'clamp' or 'cycle', got {after_end!r}.")

    end_lr = end_lr_ratio * peak_lr

    def schedule(count):
        count = jnp.asarray(count, dtype=jnp.float32)
        remaining = (t_max - (count - warmup_steps)) / t_max
        if after_end == "clamp":
            remaining = jnp.clip(remaining, 0.0, 1.0)
        else:
            remaining = jnp.minimum(remaining, 1.0)
        # (1 + cos(pi * x)) / 2 == sin(pi * (1 - x) / 2) ** 2. The sine form keeps
        # float32 relative accuracy in the last slots, where the factor is small
        # and 1 + cos would cancel.
        factor = jnp.square(jnp.sin(0.5 * jnp.pi * remaining))
        decayed = end_lr + (peak_lr - end_lr) * factor
        if warmup_steps == 0:
            return decayed
        warmup = peak_lr * (count + 1.0) / warmup_steps
        return jnp.where(count < warmup_steps, warmup, decayed)

    return schedule


__all__ = [
    "AfterEnd",
    "cosine_annealing_schedule",
]
