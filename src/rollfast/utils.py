from typing import Optional

import jax
import jax.numpy as jnp


def add_tiny(x):
    """Add smallest normal number to avoid division by zero."""
    return x + jnp.finfo(x.dtype).tiny


def dist_reduce(x: jax.Array, axis_name: Optional[str], op: str = "mean") -> jax.Array:
    """Applies a distributed reduction (pmean, pmax, psum) if an axis name is provided.

    Args:
        x: The input array.
        axis_name: The name of the mapped axis (e.g., 'batch'). If None, returns x unchanged.
        op: The reduction operation ('mean', 'max', 'sum').

    Returns:
        The reduced array across devices.
    """
    if axis_name is None:
        return x
    if op == "mean":
        return jax.lax.pmean(x, axis_name=axis_name)
    elif op == "max":
        return jax.lax.pmax(x, axis_name=axis_name)
    elif op == "sum":
        return jax.lax.psum(x, axis_name=axis_name)
    return x
