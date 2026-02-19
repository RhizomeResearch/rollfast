# This code is coming mostly from Optax itself.
# I just modified it to add support for Magma

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import optax
import optax.tree
from optax._src import base, combine, numerics, transform, utils
from optax.transforms import _masking

from rollfast.utils import dist_reduce


class ScaleByAdamState(NamedTuple):
    """State for the Adam algorithm."""

    count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    magma_s: Optional[base.Updates]  # Tracks alignment EMA per parameter block
    key: Optional[jax.Array]  # Stateful PRNG for Bernoulli masking


def scale_by_adam(
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = False,
    use_magma: bool = False,
    magma_tau: float = 2.0,
    axis_name: Optional[str] = None,
    key: int = 42,
) -> base.GradientTransformation:
    r"""Rescale updates according to the Adam algorithm.

    See :func:`optax.adam` for more details.

    Args:
        b1: Decay rate for the exponentially weighted average of grads.
        b2: Decay rate for the exponentially weighted average of squared grads.
        eps: Term added to the denominator to improve numerical stability.
        eps_root: Term added to the denominator inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
            `None` then the `dtype` is inferred from `params` and `updates`.
        nesterov: Whether to use Nesterov momentum. The variant of Adam with
            Nesterov momentum is described in [Dozat 2016]
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

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
        nu = optax.tree.zeros_like(params)  # Second moment

        def _init_s(x):
            if x is None:
                return None
            # Enforce topological isomorphism for XLA compiler partitioning
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

        return ScaleByAdamState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            nu=nu,
            magma_s=magma_s,
            key=jax.random.PRNGKey(key) if use_magma else None,
        )

    def update_fn(updates, state, params=None):
        del params
        raw_gradients = updates

        mu = optax.tree.update_moment(updates, state.mu, b1, 1)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree.map(
                lambda m, g: b1 * m + (1 - b1) * g,
                optax.tree.bias_correction(mu, b1, numerics.safe_increment(count_inc)),
                optax.tree.bias_correction(updates, b1, count_inc),
            )
        else:
            mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
        # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
        # unclear why. Other Nadam implementations also omit the extra b2 factor.
        nu_hat = optax.tree.bias_correction(nu, b2, count_inc)
        updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = optax.tree.cast(mu, mu_dtype)
        nu = optax.tree.cast_like(nu, state.nu)

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
                mu,
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

        return updates, ScaleByAdamState(
            count=count_inc, mu=mu, nu=nu, magma_s=new_magma_s, key=key_next
        )

    return base.GradientTransformation(init_fn, update_fn)


def adamw(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: base.ScalarOrSchedule = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
    use_magma: bool = False,
    magma_tau: float = 2.0,
    axis_name: Optional[str] = None,
    key: int = 42,
) -> base.GradientTransformationExtraArgs:
    r"""Adam with weight decay regularization.

    AdamW uses weight decay to regularize learning towards small weights, as
    this leads to better generalization. In SGD you can also use L2 regularization
    to implement this as an additive loss term, however L2 regularization
    does not behave as intended for adaptive gradient algorithms such as Adam,
    see [Loshchilov et al, 2019].

    Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
    :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
    ``b1``, ``b2``, ``eps`` and ``eps_root`` respectively. The learning rate is
    indexed by :math:`t` since the learning rate may also be provided by a
    schedule function. Let :math:`\lambda` be the weight decay and
    :math:`\theta_t` the parameter vector at time :math:`t`.

    The ``init`` function of this optimizer initializes an internal state
    :math:`S_0 := (m_0, v_0) = (0, 0)`, representing initial estimates for the
    first and second moments. In practice these values are stored as pytrees
    containing all zeros, with the same shape as the model updates.
    At step :math:`t`, the ``update`` function of this optimizer takes as
    arguments the incoming gradients :math:`g_t`, the optimizer state :math:`S_t`
    and the parameters :math:`\theta_t` and computes updates :math:`u_t` and
    new state :math:`S_{t+1}`. Thus, for :math:`t > 0`, we have,

    .. math::

      \begin{align*}
        m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
        v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
        \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
        \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
        u_t &\leftarrow -\alpha_t \cdot \left( \hat{m}_t / \left({\sqrt{\hat{v}_t
        + \bar{\varepsilon}} + \varepsilon} \right) + \lambda \theta_{t} \right)\\
        S_t &\leftarrow (m_t, v_t).
      \end{align*}

    This implementation can incorporate a momentum a la Nesterov introduced by
    [Dozat 2016]. The resulting optimizer is then often referred as NAdamW.
    With the keyword argument `nesterov=True`, the optimizer uses Nesterov
    momentum, replacing the above :math:`\hat{m}_t` with

    .. math::
        \hat{m}_t \leftarrow
          \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}.

    Args:
        learning_rate: A global scaling factor, either fixed or evolving along
            iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: A small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as
            in RMSProp), to avoid dividing by zero when rescaling. This is needed for
            instance when computing (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
            `None` then the `dtype` is inferred from `params` and `updates`.
        weight_decay: Strength of the weight decay regularization. Note that this
            weight decay is multiplied with the learning rate. This is consistent
            with other frameworks such as PyTorch, but different from
            (Loshchilov et al, 2019) where the weight decay is only multiplied with
            the "schedule multiplier", but not the base learning rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the weight decay to, and `False` for those you want to skip. Note
            that the Adam gradient transformations are applied to all parameters.
        nesterov: Whether to use Nesterov momentum. The solver with
            nesterov=True is equivalent to the :func:`optax.nadamw` optimizer. This
            modification is described in [Dozat 2016].
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
        >>> solver = optax.adamw(learning_rate=0.003)
        >>> params = jnp.array([1., 2., 3.])
        >>> print('Objective function: ', f(params))
        Objective function:  14.0
        >>> opt_state = solver.init(params)
        >>> for _ in range(5):
        ...  grad = jax.grad(f)(params)
        ...  updates, opt_state = solver.update(grad, opt_state, params)
        ...  params = optax.apply_updates(params, updates)
        ...  print('Objective function: {:.2E}'.format(f(params)))
        Objective function: 1.40E+01
        Objective function: 1.39E+01
        Objective function: 1.39E+01
        Objective function: 1.39E+01
        Objective function: 1.38E+01

    References:
        Loshchilov et al, `Decoupled Weight Decay
        Regularization <https://arxiv.org/abs/1711.05101>`_, 2019

        Dozat, `Incorporating Nesterov Momentum into Adam
        <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_, 2016

    .. seealso::
        See the related functions :func:`optax.adam`, :func:`optax.nadamw`, as well
        as the example :doc:`../_collections/examples/nanolm` for a use case.
  """
    return combine.chain(
        scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
            use_magma=use_magma,
            magma_tau=magma_tau,
            axis_name=axis_name,
            key=key,
        ),
        transform.add_decayed_weights(weight_decay, mask),
        transform.scale_by_learning_rate(learning_rate),
    )
