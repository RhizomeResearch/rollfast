import jax
import jax.numpy as jnp
from optax.transforms import _masking

from rollfast.utils import dist_reduce


def validate_magma_args(p: float, tau: float) -> None:
    """Validate Magma constructor arguments before they enter JAX control flow."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"magma_p must be in [0, 1], got {p!r}.")
    if tau <= 0.0:
        raise ValueError(f"magma_tau must be positive, got {tau!r}.")


def apply_magma_internal(
    raw_gradients,
    first_moments,
    base_updates,
    magma_s_prev,
    key,
    p=0.5,
    tau=2.0,
    axis_name=None,
):
    """
    Computes Momentum-aligned gradient masking (Magma) internally.
    Enforces strict structural preservation by processing flattened leaves.

    Args:
        raw_gradients: The raw gradients from the current step.
        first_moments: The first moments (e.g., from Adam/EMA).
        base_updates: The proposed base updates before masking.
        magma_s_prev: The previous Magma scores.
        key: A PRNG key for generating Bernoulli masks.
        tau: The temperature parameter for the sigmoid (default: 2.0).
        axis_name: Optional axis name for pmap/sharding.

    Returns:
        A tuple of (magma_updates, new_magma_s) where magma_updates are the
        masked base updates and new_magma_s are the updated EMA scores.

    Reference:
        Joo, T., Xia, W., Kim, C., Zhang, M., & Ie, E. (2026).
        On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.
        arXiv preprint arXiv:2602.15322.
    """
    is_leaf_fn = lambda x: isinstance(x, _masking.MaskedNode) or x is None

    leaves_g, treedef = jax.tree.flatten(raw_gradients, is_leaf=is_leaf_fn)

    def flatten_matching(tree, name):
        leaves, tree_treedef = jax.tree.flatten(tree, is_leaf=is_leaf_fn)
        if tree_treedef != treedef:
            raise ValueError(
                "Magma input trees must have matching structures; "
                f"`{name}` does not match `raw_gradients`."
            )
        return leaves

    leaves_mu = flatten_matching(first_moments, "first_moments")
    leaves_delta = flatten_matching(base_updates, "base_updates")
    leaves_s = flatten_matching(magma_s_prev, "magma_s_prev")

    if axis_name is not None:
        # Synchronize PRNG key across all devices so Bernoulli masks are identical.
        # pmin over raw key bits is a single O(log D) collective that deterministically
        # selects the same canonical key on every device.
        key_bits = jax.random.key_data(key)
        key_bits = jax.lax.pmin(key_bits, axis_name=axis_name)
        key = jax.random.wrap_key_data(key_bits)

    subkeys = jax.random.split(key, len(leaves_g))

    new_delta_leaves = []
    new_s_leaves = []

    for g, mu, delta, s_prev, block_key in zip(
        leaves_g, leaves_mu, leaves_delta, leaves_s, subkeys
    ):
        if (
            g is None
            or mu is None
            or delta is None
            or isinstance(g, _masking.MaskedNode)
            or isinstance(mu, _masking.MaskedNode)
            or isinstance(delta, _masking.MaskedNode)
        ):
            new_delta_leaves.append(delta)
            new_s_leaves.append(s_prev)
            continue

        compute_dtype = (
            jnp.complex64
            if jnp.issubdtype(g.dtype, jnp.complexfloating)
            or jnp.issubdtype(mu.dtype, jnp.complexfloating)
            else jnp.float32
        )
        g_acc = g.astype(compute_dtype)
        mu_acc = mu.astype(compute_dtype)

        # Block-wise real cosine similarity; vdot conjugates the first argument
        # so complex leaves get the Hermitian inner product.
        dot = jnp.real(jnp.vdot(g_acc, mu_acc)).astype(jnp.float32)
        norm_g_sq = jnp.real(jnp.vdot(g_acc, g_acc)).astype(jnp.float32)
        norm_mu_sq = jnp.real(jnp.vdot(mu_acc, mu_acc)).astype(jnp.float32)

        if axis_name is not None:
            dot = dist_reduce(dot, axis_name, "sum")
            norm_g_sq = dist_reduce(norm_g_sq, axis_name, "sum")
            norm_mu_sq = dist_reduce(norm_mu_sq, axis_name, "sum")

        denom = jnp.maximum(jnp.sqrt(norm_g_sq) * jnp.sqrt(norm_mu_sq), 1e-12)
        cossim = jnp.nan_to_num(dot / denom, nan=0.0)
        cossim = jnp.clip(cossim, -1.0, 1.0)

        # Score EMA
        s_tilde = jax.nn.sigmoid(cossim / tau)
        s_new = 0.9 * s_prev + 0.1 * s_tilde

        # Singular block-wise Bernoulli scalar
        m_mask = jax.random.bernoulli(block_key, p).astype(delta.dtype)

        delta_magma = m_mask * s_new.astype(delta.dtype) * delta

        new_delta_leaves.append(delta_magma)
        new_s_leaves.append(s_new)

    magma_updates = jax.tree.unflatten(treedef, new_delta_leaves)
    new_magma_s = jax.tree.unflatten(treedef, new_s_leaves)

    return magma_updates, new_magma_s
