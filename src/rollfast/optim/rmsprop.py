# This code is coming mostly from Optax itself.
# I just modified it to add support for Magma

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax.tree
from optax._src import base, combine, numerics, transform
from optax.transforms import _masking

from rollfast.utils import dist_reduce

abs_sq = numerics.abs_sq


class ScaleByRmsState(NamedTuple):
    """State for exponential root mean-squared (RMS)-normalized updates."""

    nu: base.Updates
    mu_magma: Optional[base.Updates] = None
    magma_s: Optional[base.Updates] = None
    key: Optional[jax.Array] = None


class ScaleByRmsWithCountState(NamedTuple):
    """State for exponential root mean-squared (RMS)-normalized updates."""

    count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
    nu: base.Updates
    mu_magma: Optional[base.Updates] = None
    magma_s: Optional[base.Updates] = None
    key: Optional[jax.Array] = None


class ScaleByRStdDevState(NamedTuple):
    """State for centered exponential moving average of squares of updates."""

    mu: base.Updates
    nu: base.Updates
    mu_magma: Optional[base.Updates] = None
    magma_s: Optional[base.Updates] = None
    key: Optional[jax.Array] = None


class ScaleByRStdDevWithCountState(NamedTuple):
    """State for centered exponential moving average of squares of updates."""

    count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    mu_magma: Optional[base.Updates] = None
    magma_s: Optional[base.Updates] = None
    key: Optional[jax.Array] = None


def scale_by_rms(
    decay: jax.typing.ArrayLike = 0.9,
    eps: jax.typing.ArrayLike = 1e-8,
    initial_scale: jax.typing.ArrayLike = 0.0,
    eps_in_sqrt: bool = True,
    bias_correction: bool = False,
    use_magma: bool = False,
    magma_tau: float = 2.0,
    magma_b1: float = 0.9,
    axis_name: Optional[str] = None,
    key: int = 42,
) -> base.GradientTransformation:
    r"""Rescale updates by the root of the exp. moving avg of the square.

    See :func:`optax.rmsprop` for more details.

    Args:
        decay: Decay rate for the exponentially weighted average of squared grads.
        eps: Term added to the denominator to improve numerical stability.
        initial_scale: Initial value for second moment.
        eps_in_sqrt: Whether to add ``eps`` in the square root of the denominator or
            outside the square root.
        bias_correction: Whether to apply bias correction to the exponentially
            weighted average of squared grads.
        use_magma: If True, applies Momentum-aligned gradient masking (Magma).
            WARNING: Magma introduces intentional update bias (damping). At an
            equilibrium tau=2.0, non-masked steps scale updates by ~0.5, and
            50% of steps are masked. This yields an expected magnitude attenuation
            of ~0.25x. You may need to scale the global learning rate by ~4x to
            maintain the original update volume.
        magma_tau: Temperature parameter for the alignment sigmoid. Default is 2.0.
        axis_name: Axis name for distributed (SPMD) global norm reduction.
        key: Initial PRNG seed for Magma's Bernoulli sampling.

    Returns:
        A :class:`optax.GradientTransformation` object.

    .. note::
        Using `scale_by_rms(decay=b2, eps_in_sqrt=False, bias_correction=True)`
        will match the behavior of `scale_by_adam(b1=0, b2=b2)`, while sparing the
        memory cost of storing the first moment.
    """

    def init_fn(params):
        nu = optax.tree.full_like(params, initial_scale)
        mu_magma = optax.tree.zeros_like(params) if use_magma else None

        def _init_s(x):
            if x is None:
                return None
            if isinstance(x, _masking.MaskedNode):
                return _masking.MaskedNode()
            return jnp.zeros([], jnp.float32)

        magma_s = (
            jax.tree.map(
                _init_s,
                params,
                is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
            )
            if use_magma
            else None
        )

        _key = jax.random.PRNGKey(key) if use_magma else None

        if bias_correction:
            return ScaleByRmsWithCountState(
                count=jnp.zeros([], jnp.int32),
                nu=nu,
                mu_magma=mu_magma,
                magma_s=magma_s,
                key=_key,
            )
        return ScaleByRmsState(nu=nu, mu_magma=mu_magma, magma_s=magma_s, key=_key)

    def update_fn(updates, state, params=None):
        del params
        raw_gradients = updates

        if use_magma:
            mu_magma = optax.tree.update_moment(
                raw_gradients, state.mu_magma, magma_b1, 1
            )
        else:
            mu_magma = state.mu_magma

        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, decay, 2)
        if bias_correction:
            count_inc = numerics.safe_increment(state.count)
            nu_hat = optax.tree.bias_correction(nu, decay, count_inc)
        else:
            count_inc = jnp.asarray(0)
            nu_hat = nu

        if eps_in_sqrt:
            scaling = jax.tree.map(lambda n: jax.lax.rsqrt(n + eps), nu_hat)
        else:
            scaling = jax.tree.map(lambda n: 1 / (jnp.sqrt(n) + eps), nu_hat)

        updates = jax.tree.map(lambda s, g: s * g, scaling, updates)

        if use_magma:

            def _compute_magma_state(g, m_u, s_old):
                if g is None or isinstance(g, _masking.MaskedNode):
                    return g

                g_f32 = g.astype(jnp.float32)
                m_f32 = m_u.astype(jnp.float32)

                dot = dist_reduce(jnp.sum(g_f32 * m_f32), axis_name, "sum")
                norm_g_sq = dist_reduce(jnp.sum(g_f32**2), axis_name, "sum")
                norm_mu_sq = dist_reduce(jnp.sum(m_f32**2), axis_name, "sum")

                cossim = dot / jnp.maximum(
                    jnp.sqrt(norm_g_sq) * jnp.sqrt(norm_mu_sq), 1e-9
                )
                s_tilde = jax.nn.sigmoid(cossim / magma_tau)
                return 0.9 * s_old + 0.1 * s_tilde

            new_magma_s = jax.tree.map(
                _compute_magma_state,
                raw_gradients,
                mu_magma,
                state.magma_s,
                is_leaf=lambda x: isinstance(x, _masking.MaskedNode),
            )

            leaves_delta, treedef = jax.tree.flatten(updates)
            magma_key, key_next = jax.random.split(state.key)
            subkeys = jax.random.split(magma_key, len(leaves_delta))
            keys_tree = jax.tree.unflatten(treedef, subkeys)

            def _apply_mask(delta, s_new, k_leaf):
                if delta is None or isinstance(delta, _masking.MaskedNode):
                    return delta
                m_mask = jax.random.bernoulli(k_leaf, 0.5).astype(jnp.float32)
                return delta * jnp.array(s_new * m_mask, dtype=delta.dtype)

            updates = jax.tree.map(
                _apply_mask,
                updates,
                new_magma_s,
                keys_tree,
                is_leaf=lambda x: isinstance(x, _masking.MaskedNode),
            )
        else:
            new_magma_s = state.magma_s
            key_next = state.key

        if bias_correction:
            new_state = ScaleByRmsWithCountState(
                count=count_inc,
                nu=nu,
                mu_magma=mu_magma,
                magma_s=new_magma_s,
                key=key_next,
            )
        else:
            new_state = ScaleByRmsState(
                nu=nu, mu_magma=mu_magma, magma_s=new_magma_s, key=key_next
            )
        return updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_stddev(
    decay: jax.typing.ArrayLike = 0.9,
    eps: jax.typing.ArrayLike = 1e-8,
    initial_scale: jax.typing.ArrayLike = 0.0,
    eps_in_sqrt: bool = True,
    bias_correction: bool = False,
    use_magma: bool = False,
    magma_tau: float = 2.0,
    magma_b1: float = 0.9,
    axis_name: Optional[str] = None,
    key: int = 42,
) -> base.GradientTransformation:
    """Rescale updates by the root of the centered exp. moving average of squares.

    See :func:`optax.rmsprop` for more details.

    Args:
        decay: Decay rate for the exponentially weighted average of squared grads.
        eps: Term added to the denominator to improve numerical stability.
        initial_scale: Initial value for second moment.
        eps_in_sqrt: Whether to add ``eps`` in the square root of the denominator or
            outside the square root.
        bias_correction: Whether to apply bias correction to the first and second
            moment.
        use_magma: If True, applies Momentum-aligned gradient masking (Magma).
            WARNING: Magma introduces intentional update bias (damping). At an
            equilibrium tau=2.0, non-masked steps scale updates by ~0.5, and
            50% of steps are masked. This yields an expected magnitude attenuation
            of ~0.25x. You may need to scale the global learning rate by ~4x to
            maintain the original update volume.
        magma_tau: Temperature parameter for the alignment sigmoid. Default is 2.0.
        axis_name: Axis name for distributed (SPMD) global norm reduction.
        key: Initial PRNG seed for Magma's Bernoulli sampling.

    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    def init_fn(params):
        mu = optax.tree.zeros_like(params)
        nu = optax.tree.full_like(params, initial_scale)
        mu_magma = optax.tree.zeros_like(params) if use_magma else None

        def _init_s(x):
            if x is None:
                return None
            if isinstance(x, _masking.MaskedNode):
                return _masking.MaskedNode()
            return jnp.zeros([], jnp.float32)

        magma_s = (
            jax.tree.map(
                _init_s,
                params,
                is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
            )
            if use_magma
            else None
        )

        _key = jax.random.PRNGKey(key) if use_magma else None

        if bias_correction:
            return ScaleByRStdDevWithCountState(
                count=jnp.zeros([], jnp.int32),
                mu=mu,
                nu=nu,
                mu_magma=mu_magma,
                magma_s=magma_s,
                key=_key,
            )
        return ScaleByRStdDevState(
            mu=mu, nu=nu, mu_magma=mu_magma, magma_s=magma_s, key=_key
        )

    def update_fn(updates, state, params=None):
        del params
        raw_gradients = updates

        if use_magma:
            mu_magma = optax.tree.update_moment(
                raw_gradients, state.mu_magma, magma_b1, 1
            )
        else:
            mu_magma = state.mu_magma

        mu = optax.tree.update_moment(updates, state.mu, decay, 1)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, decay, 2)
        if bias_correction:
            count_inc = numerics.safe_increment(state.count)
            mu_hat = optax.tree.bias_correction(mu, decay, count_inc)
            nu_hat = optax.tree.bias_correction(nu, decay, count_inc)
        else:
            count_inc = jnp.asarray(0)
            mu_hat = mu
            nu_hat = nu

        if eps_in_sqrt:
            scaling = jax.tree.map(
                lambda m, n: jax.lax.rsqrt(n - abs_sq(m) + eps), mu_hat, nu_hat
            )
        else:
            scaling = jax.tree.map(
                lambda m, n: 1 / (jnp.sqrt(n - abs_sq(m)) + eps), mu_hat, nu_hat
            )

        updates = jax.tree.map(lambda s, g: s * g, scaling, updates)

        if use_magma:

            def _compute_magma_state(g, m_u, s_old):
                if g is None or isinstance(g, _masking.MaskedNode):
                    return g
                g_f32, m_f32 = g.astype(jnp.float32), m_u.astype(jnp.float32)

                dot = dist_reduce(jnp.sum(g_f32 * m_f32), axis_name, "sum")
                norm_g_sq = dist_reduce(jnp.sum(g_f32**2), axis_name, "sum")
                norm_mu_sq = dist_reduce(jnp.sum(m_f32**2), axis_name, "sum")

                cossim = dot / jnp.maximum(
                    jnp.sqrt(norm_g_sq) * jnp.sqrt(norm_mu_sq), 1e-9
                )
                return 0.9 * s_old + 0.1 * jax.nn.sigmoid(cossim / magma_tau)

            new_magma_s = jax.tree.map(
                _compute_magma_state,
                raw_gradients,
                mu_magma,
                state.magma_s,
                is_leaf=lambda x: isinstance(x, _masking.MaskedNode),
            )

            leaves_delta, treedef = jax.tree.flatten(updates)
            magma_key, key_next = jax.random.split(state.key)
            subkeys = jax.random.split(magma_key, len(leaves_delta))
            keys_tree = jax.tree.unflatten(treedef, subkeys)

            def _apply_mask(delta, s_new, k_leaf):
                if delta is None or isinstance(delta, _masking.MaskedNode):
                    return delta
                m_mask = jax.random.bernoulli(k_leaf, 0.5).astype(jnp.float32)
                return delta * jnp.array(s_new * m_mask, dtype=delta.dtype)

            updates = jax.tree.map(
                _apply_mask,
                updates,
                new_magma_s,
                keys_tree,
                is_leaf=lambda x: isinstance(x, _masking.MaskedNode),
            )
        else:
            new_magma_s = state.magma_s
            key_next = state.key

        if bias_correction:
            new_state = ScaleByRStdDevWithCountState(
                count=count_inc,
                mu=mu,
                nu=nu,
                mu_magma=mu_magma,
                magma_s=new_magma_s,
                key=key_next,
            )
        else:
            new_state = ScaleByRStdDevState(
                mu=mu, nu=nu, mu_magma=mu_magma, magma_s=new_magma_s, key=key_next
            )
        return updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


def rmsprop(
    learning_rate: base.ScalarOrSchedule,
    decay: jax.typing.ArrayLike = 0.9,
    eps: jax.typing.ArrayLike = 1e-8,
    initial_scale: jax.typing.ArrayLike = 0.0,
    eps_in_sqrt: bool = True,
    centered: bool = False,
    momentum: Optional[jax.typing.ArrayLike] = None,
    nesterov: bool = False,
    bias_correction: bool = False,
    use_magma: bool = False,
    magma_tau: float = 2.0,
    magma_b1: float = 0.9,
    axis_name: Optional[str] = None,
    key: int = 42,
) -> base.GradientTransformationExtraArgs:
    r"""A flexible RMSProp optimizer.

    RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
    used for each weight is scaled by a suitable estimate of the magnitude of the
    gradients on previous steps. Several variants of RMSProp can be found
    in the literature. This alias provides an easy to configure RMSProp
    optimizer that can be used to switch between several of these variants.

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along
            iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
        decay: Decay used to track the magnitude of previous gradients.
        eps: A small numerical constant to avoid dividing by zero when rescaling.
        initial_scale: Initial value of accumulators tracking the magnitude of
            previous updates. PyTorch uses `0`, TF1 uses `1`. When reproducing results
            from a paper, verify the value used by the authors.
        eps_in_sqrt: Whether to add ``eps`` in the square root of the denominator or
            outside the square root.
        centered: Whether the second moment or the variance of the past gradients is
            used to rescale the latest gradients.
        momentum: Decay rate used by the momentum term, when it is set to `None`,
            then momentum is not used at all.
        nesterov: Whether Nesterov momentum is used.
        bias_correction: Whether to apply bias correction to the estimates of the
            second moments (and first moment if ``centered=True``).
        use_magma: If True, applies Momentum-aligned gradient masking (Magma).
            WARNING: Magma introduces intentional update bias (damping). At an
            equilibrium tau=2.0, non-masked steps scale updates by ~0.5, and
            50% of steps are masked. This yields an expected magnitude attenuation
            of ~0.25x. You may need to scale the global learning rate by ~4x to
            maintain the original update volume.
        magma_tau: Temperature parameter for the alignment sigmoid. Default is 2.0.
        axis_name: Axis name for distributed (SPMD) global norm reduction.
        key: Initial PRNG seed for Magma's Bernoulli sampling.

    Returns:
        The corresponding :class:`optax.GradientTransformationExtraArgs`.

    Examples:
        >>> import optax
        >>> import jax
        >>> import jax.numpy as jnp
        >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
        >>> solver = optax.rmsprop(learning_rate=0.003)
        >>> params = jnp.array([1., 2., 3.])
        >>> print('Objective function: ', f(params))
        Objective function:  14.0
        >>> opt_state = solver.init(params)
        >>> for _ in range(5):
        ...  grad = jax.grad(f)(params)
        ...  updates, opt_state = solver.update(grad, opt_state, params)
        ...  params = optax.apply_updates(params, updates)
        ...  print('Objective function: {:.2E}'.format(f(params)))
        Objective function: 1.39E+01
        Objective function: 1.38E+01
        Objective function: 1.37E+01
        Objective function: 1.37E+01
        Objective function: 1.36E+01

    References:
        Hinton, `Overview of mini-batch gradient descent`
        <www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_, 2012

        Graves, `Generating Sequences With Recurrent Neural Networks
        <https://arxiv.org/pdf/1308.0850v5>`_, 2014

        Ziyin, `LaProp: Separating Momentum and Adaptivity in Adam
        <https://arxiv.org/pdf/2002.04839>`_, 2021

    .. warning::
        Default behavior of optax's RMSprop (``eps_in_sqrt=True``) differs from
        Pytorch's implementation and could impact performance.
        If ``eps_in_sqrt=True``, in the denominator, optax uses
        :math:`\sqrt{v + \epsilon}` in the denominator whereas PyTorch uses
        :math:`\sqrt{v} + \epsilon`.
        Using ``eps_in_sqrt=False`` in optax will match PyTorch's behavior.
        See
        https://github.com/google-deepmind/optax/issues/532 for more detail.
    """
    if centered:
        return combine.chain(
            scale_by_stddev(
                decay=decay,
                eps=eps,
                initial_scale=initial_scale,
                eps_in_sqrt=eps_in_sqrt,
                bias_correction=bias_correction,
                use_magma=use_magma,
                magma_tau=magma_tau,
                magma_b1=magma_b1,
                axis_name=axis_name,
                key=key,
            ),
            transform.scale_by_learning_rate(learning_rate),
            (
                transform.trace(decay=momentum, nesterov=nesterov)
                if momentum is not None
                else base.identity()
            ),
        )
    return combine.chain(
        scale_by_rms(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
            eps_in_sqrt=eps_in_sqrt,
            bias_correction=bias_correction,
            use_magma=use_magma,
            magma_tau=magma_tau,
            magma_b1=magma_b1,
            axis_name=axis_name,
            key=key,
        ),
        transform.scale_by_learning_rate(learning_rate),
        (
            transform.trace(decay=momentum, nesterov=nesterov)
            if momentum is not None
            else base.identity()
        ),
    )
