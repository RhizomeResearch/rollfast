# This code is coming mostly from Optax itself.
# I just modified it to add support for Magma
# and stochastic rounding
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import optax
import optax.tree
from optax._src import base, combine, numerics, transform, utils
from optax.transforms import _masking

from rollfast.optim.magma import apply_magma_internal
from rollfast.utils import (
    _safe_bias_correction,
    _tree_stochastic_cast,
    _tree_update_moment_f32,
    _tree_update_moment_sq_f32,
)


class ScaleByAdamState(NamedTuple):
    """State for the Adam algorithm."""

    count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    magma_s: Any
    key: Optional[jax.Array]


def scale_by_adam(
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    nesterov: bool = False,
    use_magma: bool = False,
    magma_tau: float = 2.0,
    axis_name: Optional[str] = None,
    key: jax.Array = jax.random.PRNGKey(42),
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
        weight_decay: Strength of the weight decay regularization (only used if use_magma=True).
        weight_decay_mask: A tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the weight decay to, and `False` for those you want to skip.
        nesterov: Whether to use Nesterov momentum. The variant of Adam with
            Nesterov momentum is described in [Dozat 2016]
        use_magma: If True, applies Momentum-aligned gradient masking (Magma).
        magma_tau: Temperature parameter for the alignment sigmoid. Default is 2.0.
        axis_name: Axis name for distributed (SPMD) global norm reduction.
        key: Initial PRNG seed for Magma's Bernoulli sampling.

    Returns:
        A :class:`optax.GradientTransformation` object.

    Reference:
        Joo, T., Xia, W., Kim, C., Zhang, M., & Ie, E. (2026).
        On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.
        arXiv preprint arXiv:2602.15322.
    """

    if mu_dtype is None:
        mu_dtype = jnp.float32
    else:
        mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
        nu = optax.tree.zeros_like(params, dtype=mu_dtype)  # Second moment

        if use_magma:

            def _init_s(x):
                if x is None:
                    return None
                if isinstance(x, _masking.MaskedNode):
                    return _masking.MaskedNode()
                return jnp.array(0.5, dtype=jnp.float32)

            magma_s = jax.tree.map(
                _init_s,
                params,
                is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
            )
        else:
            magma_s = ()

        return ScaleByAdamState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            nu=nu,
            magma_s=magma_s,
            key=key,
        )

    def update_fn(updates, state, params=None):
        # Preserve raw stochastic gradients (g_t) for Magma alignment
        raw_gradients = updates

        if use_magma:
            next_state_key, sr_key, magma_key = jax.random.split(state.key, 3)
        else:
            next_state_key, sr_key = jax.random.split(state.key, 2)
            magma_key = None

        mu_f32 = _tree_update_moment_f32(updates, state.mu, b1)
        nu_f32 = _tree_update_moment_sq_f32(updates, state.nu, b2)
        count_inc = numerics.safe_increment(state.count)

        mu_bc_factor = 1.0 - b1**count_inc
        nu_bc_factor = 1.0 - b2**count_inc

        if nesterov:
            mu_bc_factor_next = 1.0 - b1 ** numerics.safe_increment(count_inc)
            mu_bc = _safe_bias_correction(mu_f32, mu_bc_factor_next)

            # Explicitly bypass MaskedNodes. Attempting .astype() on a MaskedNode
            # triggers an AttributeError during partitioned gradient routing.
            updates_f32 = jax.tree.map(
                lambda x: (
                    x
                    if isinstance(x, _masking.MaskedNode)
                    else (x.astype(jnp.float32) if x is not None else None)
                ),
                updates,
                is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
            )
            g_bc = _safe_bias_correction(updates_f32, mu_bc_factor)

            # MaskedNodes cannot be mathematically multiplied or added.
            mu_hat = jax.tree.map(
                lambda m, g: (
                    m
                    if isinstance(m, _masking.MaskedNode)
                    else (b1 * m + (1.0 - b1) * g if m is not None else None)
                ),
                mu_bc,
                g_bc,
                is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
            )
        else:
            mu_hat = _safe_bias_correction(mu_f32, mu_bc_factor)

        # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
        # unclear why. Other Nadam implementations also omit the extra b2 factor.
        nu_hat = _safe_bias_correction(nu_f32, nu_bc_factor)

        adam_updates = jax.tree.map(
            lambda m, v: (
                m
                if isinstance(m, _masking.MaskedNode)
                else (None if m is None else m / (jnp.sqrt(v + eps_root) + eps))
            ),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
        )

        wd_step = weight_decay(state.count) if callable(weight_decay) else weight_decay

        if params is not None:
            # Resolve the mask tree natively
            resolved_mask = (
                weight_decay_mask(params)
                if callable(weight_decay_mask)
                else weight_decay_mask
            )
            if resolved_mask is None:
                resolved_mask = jax.tree.map(
                    lambda _: True,
                    params,
                    is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
                )

            adam_updates = jax.tree.map(
                lambda u, p, m_leaf: (
                    u
                    if isinstance(u, _masking.MaskedNode) or u is None
                    else (u + wd_step * p.astype(jnp.float32) if m_leaf else u)
                ),
                adam_updates,
                params,
                resolved_mask,
                is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
            )

        # Intercept and mathematically project Delta_t through Magma logic
        if use_magma:
            final_updates, new_magma_s = apply_magma_internal(
                raw_gradients=raw_gradients,
                first_moments=mu_f32,
                base_updates=adam_updates,
                magma_s_prev=state.magma_s,
                key=magma_key,
                tau=magma_tau,
                axis_name=axis_name,
            )
        else:
            final_updates = adam_updates
            new_magma_s = state.magma_s

        if mu_dtype == jnp.bfloat16:
            k1, k2 = jax.random.split(sr_key)
            mu_stored = _tree_stochastic_cast(mu_f32, jnp.bfloat16, k1)
            nu_stored = _tree_stochastic_cast(nu_f32, jnp.bfloat16, k2)
        else:
            mu_stored = mu_f32
            nu_stored = nu_f32

        return final_updates, ScaleByAdamState(
            count=count_inc,
            mu=mu_stored,
            nu=nu_stored,
            magma_s=new_magma_s,
            key=next_state_key,
        )

    return base.GradientTransformation(init_fn, update_fn)


def adamw(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    weight_decay: base.ScalarOrSchedule = 1e-4,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
    use_magma: bool = False,
    magma_tau: float = 2.0,
    axis_name: Optional[str] = None,
    key: jax.Array = jax.random.PRNGKey(42),
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
        weight_decay_mask: A tree with same structure as (or a prefix of) the params PyTree,
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

        Joo, T., Xia, W., Kim, C., Zhang, M., & Ie, E. (2026).
        On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.
        arXiv preprint arXiv:2602.15322.

    .. seealso::
        See the related functions :func:`optax.adam`, :func:`optax.nadamw`, as well
        as the example :doc:`../_collections/examples/nanolm` for a use case.
    """

    components = [
        scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            weight_decay=weight_decay if use_magma else 0.0,
            weight_decay_mask=weight_decay_mask if use_magma else None,
            nesterov=nesterov,
            use_magma=use_magma,
            magma_tau=magma_tau,
            axis_name=axis_name,
            key=key,
        )
    ]

    _wd_is_nonzero = (
        weight_decay > 0.0 if isinstance(weight_decay, (int, float)) else True
    )
    if _wd_is_nonzero and not use_magma:
        components.append(
            transform.add_decayed_weights(weight_decay, weight_decay_mask)
        )

    components.append(transform.scale_by_learning_rate(learning_rate))
    return combine.chain(*components)
