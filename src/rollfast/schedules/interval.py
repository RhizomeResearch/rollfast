import jax.numpy as jnp
import optax


def interval_schedule(
    schedule: optax.Schedule, steps_per_interval: int
) -> optax.Schedule:
    """Holds ``schedule`` constant for ``steps_per_interval`` counts at a time.

    The held schedule returns ``schedule(count // steps_per_interval)``, so the
    wrapped schedule is indexed in intervals, such as epochs of
    ``steps_per_interval`` updates. This matches a PyTorch scheduler stepped once
    per epoch. With ``steps_per_interval=1`` the schedule is returned unchanged.

    Args:
        schedule: Schedule indexed by interval.
        steps_per_interval: Counts that share one schedule value.

    Returns:
        An optax schedule function indexed by count.
    """
    if (
        not isinstance(steps_per_interval, int)
        or isinstance(steps_per_interval, bool)
        or steps_per_interval < 1
    ):
        raise ValueError(
            f"steps_per_interval must be a positive integer, got {steps_per_interval!r}."
        )
    if steps_per_interval == 1:
        return schedule

    def held(count):
        return schedule(jnp.asarray(count) // steps_per_interval)

    return held


__all__ = ["interval_schedule"]
