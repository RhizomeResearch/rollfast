"""Applied-update status and accumulation lifecycle checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .config import AccumulationConfig


class UpdateStatus(NamedTuple):
    """Outcome of one optimizer invocation, with JAX scalar fields.

    Exactly one of ``accumulating``, ``update_applied``, and
    ``window_discarded`` is true. A discarded window consumed its scheduled
    microsteps without applying parameter updates or advancing the guarded
    inner optimizer.
    """

    accumulating: jax.Array
    update_applied: jax.Array
    window_discarded: jax.Array
    microsteps_in_window: jax.Array


def _update_status(old_state: Any, new_state: Any) -> UpdateStatus:
    if hasattr(old_state, "gradient_step") and hasattr(new_state, "gradient_step"):
        completed = new_state.gradient_step > old_state.gradient_step
        accepted = _last_finite(new_state.inner_opt_state)
        applied = jnp.logical_and(completed, accepted)
        return UpdateStatus(
            accumulating=jnp.logical_not(completed),
            update_applied=applied,
            window_discarded=jnp.logical_and(completed, jnp.logical_not(accepted)),
            microsteps_in_window=new_state.mini_step,
        )
    if hasattr(new_state, "last_finite"):
        applied = jnp.asarray(new_state.last_finite, dtype=jnp.bool_)
        return UpdateStatus(
            accumulating=jnp.asarray(False),
            update_applied=applied,
            window_discarded=jnp.logical_not(applied),
            microsteps_in_window=jnp.asarray(0, dtype=jnp.int32),
        )
    if hasattr(new_state, "inner_state"):
        return _update_status(old_state.inner_state, new_state.inner_state)
    if hasattr(new_state, "inner_opt_state"):
        return _update_status(old_state.inner_opt_state, new_state.inner_opt_state)
    return UpdateStatus(
        accumulating=jnp.asarray(False),
        update_applied=jnp.asarray(True),
        window_discarded=jnp.asarray(False),
        microsteps_in_window=jnp.asarray(0, dtype=jnp.int32),
    )


def _last_finite(state: Any) -> jax.Array:
    if hasattr(state, "last_finite"):
        return jnp.asarray(state.last_finite, dtype=jnp.bool_)
    if hasattr(state, "inner_state"):
        return _last_finite(state.inner_state)
    if hasattr(state, "inner_opt_state"):
        return _last_finite(state.inner_opt_state)
    return jnp.asarray(True)


def validate_accumulation_complete(
    state: Any, accumulation_config: AccumulationConfig
) -> None:
    """Raise when a host-side lifecycle boundary has pending microsteps.

    ``state`` may be a compiled optimizer state or an explicit
    ``AccumulationState``. Call after the final microbatch and before declaring
    an epoch or run complete. This function does not modify the state.
    """

    if accumulation_config.steps == 1:
        return
    current = state
    while True:
        if hasattr(current, "microsteps_in_window"):
            pending = current.microsteps_in_window
            break
        if hasattr(current, "mini_step"):
            pending = current.mini_step
            break
        if hasattr(current, "inner_state"):
            current = current.inner_state
            continue
        if hasattr(current, "inner_opt_state"):
            current = current.inner_opt_state
            continue
        raise ValueError("state has no accumulation window for the supplied config.")
    pending_count = int(jax.device_get(pending))
    if pending_count:
        raise ValueError(
            f"incomplete accumulation window: {pending_count} of "
            f"{accumulation_config.steps} microsteps; no update was applied."
        )


__all__ = ("UpdateStatus", "validate_accumulation_complete")
