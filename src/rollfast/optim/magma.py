from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from optax._src import base
from optax.transforms import _masking

from rollfast.utils import _tree_update_moment_f32, dist_reduce


class MagmaState(NamedTuple):
    """Isolated State for the generic Magma masking algorithm."""

    mu: base.Updates  # Internal EMA of the update stream
    magma_s: base.Updates  # EMA of the alignment score
    key: jax.Array  # Stateful PRNG for Bernoulli masking


def apply_magma_mask(
    magma_tau: float = 2.0,
    beta: float = 0.9,  # Magma needs its own momentum coefficient
    gamma: float = 0.9,  # Decay rate for alignment score s_t
    axis_name: Optional[str] = None,
    key: int = 42,
) -> base.GradientTransformation:
    """
    Generic Momentum-aligned gradient masking (Magma).
    Composes safely after weight_decay to prevent drift of dormant parameters.
    """

    def init_fn(params):
        mu = optax.tree.zeros_like(params)

        def _init_s(x):
            if x is None:
                return None
            if isinstance(x, _masking.MaskedNode):
                return _masking.MaskedNode()
            return jnp.zeros([], jnp.float32)

        magma_s = jax.tree.map(
            _init_s,
            params,
            is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
        )

        return MagmaState(
            mu=mu,
            magma_s=magma_s,
            key=jax.random.PRNGKey(key),
        )

    def update_fn(updates, state, params=None):
        del params
        next_state_key, magma_key = jax.random.split(state.key, 2)

        # Track internal EMA strictly in FP32
        mu_f32 = _tree_update_moment_f32(updates, state.mu, beta)

        def _compute_magma_state(u, m_u, s_old):
            if u is None or isinstance(u, _masking.MaskedNode):
                return u

            u_f32 = u.astype(jnp.float32)
            m_f32 = m_u.astype(jnp.float32)

            dot = dist_reduce(jnp.sum(u_f32 * m_f32), axis_name, "sum")
            norm_u_sq = dist_reduce(jnp.sum(u_f32**2), axis_name, "sum")
            norm_mu_sq = dist_reduce(jnp.sum(m_f32**2), axis_name, "sum")

            # Bounded limit for reverse-mode AD and zero-variance protection
            denom = jnp.maximum(
                jnp.sqrt(norm_u_sq + 1e-12) * jnp.sqrt(norm_mu_sq + 1e-12), 1e-9
            )
            cossim = dot / denom

            s_tilde = jax.nn.sigmoid(cossim / magma_tau)
            return gamma * s_old + (1.0 - gamma) * s_tilde

        new_magma_s = jax.tree.map(
            _compute_magma_state,
            updates,
            mu_f32,
            state.magma_s,
            is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
        )

        is_leaf_fn = lambda x: isinstance(x, _masking.MaskedNode) or x is None
        leaves_delta, treedef = jax.tree.flatten(updates, is_leaf=is_leaf_fn)
        subkeys = jax.random.split(magma_key, len(leaves_delta))
        keys_tree = jax.tree.unflatten(treedef, list(subkeys))

        def _apply_mask(delta, s_new, k_leaf):
            if delta is None or isinstance(delta, _masking.MaskedNode):
                return delta

            m_mask = jax.random.bernoulli(k_leaf, 0.5, shape=delta.shape).astype(
                delta.dtype
            )
            return delta * s_new.astype(delta.dtype) * m_mask

        masked_updates = jax.tree.map(
            _apply_mask,
            updates,
            new_magma_s,
            keys_tree,
            is_leaf=is_leaf_fn,
        )

        return masked_updates, MagmaState(
            mu=mu_f32,
            magma_s=new_magma_s,
            key=next_state_key,
        )

    return base.GradientTransformation(init_fn, update_fn)
