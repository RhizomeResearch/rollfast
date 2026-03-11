import jax
import jax.numpy as jnp
from optax.transforms import _masking

from rollfast.utils import dist_reduce


def apply_magma_internal(
    raw_gradients,
    first_moments,
    base_updates,
    magma_s_prev,
    key,
    tau=2.0,
    axis_name=None,
):
    """
    Computes Momentum-aligned gradient masking (Magma) internally.
    Enforces strict structural preservation by processing flattened leaves.
    """
    is_leaf_fn = lambda x: isinstance(x, _masking.MaskedNode) or x is None

    leaves_g, treedef = jax.tree.flatten(raw_gradients, is_leaf=is_leaf_fn)
    leaves_mu, _ = jax.tree.flatten(first_moments, is_leaf=is_leaf_fn)
    leaves_delta, _ = jax.tree.flatten(base_updates, is_leaf=is_leaf_fn)
    leaves_s, _ = jax.tree.flatten(magma_s_prev, is_leaf=is_leaf_fn)

    subkeys = jax.random.split(key, len(leaves_g))

    new_delta_leaves = []
    new_s_leaves = []

    for g, mu, delta, s_prev, block_key in zip(
        leaves_g, leaves_mu, leaves_delta, leaves_s, subkeys
    ):
        if g is None or isinstance(g, _masking.MaskedNode):
            new_delta_leaves.append(delta)
            new_s_leaves.append(s_prev)
            continue

        g_f32 = g.astype(jnp.float32)
        mu_f32 = mu.astype(jnp.float32)

        # Block-wise inner product and norms
        dot = jnp.sum(g_f32 * mu_f32)
        norm_g_sq = jnp.sum(g_f32**2)
        norm_mu_sq = jnp.sum(mu_f32**2)

        if axis_name is not None:
            dot = dist_reduce(dot, axis_name, "sum")
            norm_g_sq = dist_reduce(norm_g_sq, axis_name, "sum")
            norm_mu_sq = dist_reduce(norm_mu_sq, axis_name, "sum")

        denom = jnp.maximum(
            jnp.sqrt(norm_g_sq + 1e-12) * jnp.sqrt(norm_mu_sq + 1e-12), 1e-9
        )
        cossim = dot / denom

        # Score EMA
        s_tilde = jax.nn.sigmoid(cossim / tau)
        s_new = 0.9 * s_prev + 0.1 * s_tilde

        # Singular block-wise Bernoulli scalar
        m_mask = jax.random.bernoulli(block_key, 0.5).astype(delta.dtype)

        delta_magma = s_new.astype(delta.dtype) * m_mask * delta

        new_delta_leaves.append(delta_magma)
        new_s_leaves.append(s_new)

    magma_updates = jax.tree.unflatten(treedef, new_delta_leaves)
    new_magma_s = jax.tree.unflatten(treedef, new_s_leaves)

    return magma_updates, new_magma_s
