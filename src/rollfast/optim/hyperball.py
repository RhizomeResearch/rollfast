"""Hyperball terminal transforms and optimizer wrappers.

Hyperball is a weight-decay alternative that constrains selected parameters to
remain on the L2 sphere defined by their initialization norm.  Given a positive,
unscaled optimizer direction ``u_t`` and parameter ``theta_t``, the selected leaf
update is

    d_t       = normalize(u_t + lambda * theta_t)
    theta'   = theta_t - lr * ||theta_0||_2 * d_t
    theta_{t+1} = ||theta_0||_2 * theta' / max(||theta'||_2, eps)

and the Optax update returned by this transform is ``theta_{t+1} - theta_t``.
The transform is therefore terminal: do not append ``scale_by_learning_rate``
after it.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
import optax.contrib._muon as optax_muon
from optax._src import base, combine, numerics
from optax.transforms import _masking

from rollfast.optim.adam import scale_by_adam
from rollfast.optim.aurora import (
    AuroraWeightDimNumOrFn,
    scale_by_aurora,
    scale_by_riemannian_aurora,
)
from rollfast.optim.dimension_numbers import (
    WeightDimNumOrFn,
    _get_dimension_numbers,
    _is_dimension_numbers_leaf,
    _mask_dimension_numbers,
)
from rollfast.optim.orthogonalization import MUON_NS_COEFFS
from rollfast.optim.prism import (
    scale_by_prism,
)
from rollfast.optim.psgd import (
    GradClipMode,
    PreconditionerMode,
    precond_update_prob_schedule,
    scale_by_kron,
)
from rollfast.optim.rmnp import scale_by_rmnp, scale_by_rmnp_shape
from rollfast.utils import dist_reduce

MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]


class HyperballState(NamedTuple):
    """State for the terminal Hyperball transform."""

    count: jax.Array
    init_norm: base.Params


def _is_aux_leaf(x: Any) -> bool:
    return x is None or isinstance(x, _masking.MaskedNode)


def _is_bool_leaf(x: Any) -> bool:
    return _is_aux_leaf(x) or isinstance(x, (bool, int)) or hasattr(x, "dtype")


def _is_array_like(x: Any) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _resolve_scalar(
    value: base.ScalarOrSchedule,
    count: jax.Array,
) -> jax.typing.ArrayLike:
    if callable(value):
        return cast(Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike], value)(
            count
        )
    return value


def _leaf_l2_norm(
    x: jax.Array,
    *,
    axis_name: Optional[str] = None,
) -> jax.Array:
    sq = jnp.sum(numerics.abs_sq(x.astype(jnp.float32)))
    sq = dist_reduce(sq, axis_name, "sum")
    return jnp.sqrt(sq)


def _safe_norm(x: jax.Array, *, eps: float, axis_name: Optional[str]) -> jax.Array:
    return jnp.maximum(
        _leaf_l2_norm(x, axis_name=axis_name),
        jnp.asarray(eps, dtype=jnp.float32),
    )


def _init_norm_tree(params: base.Params, axis_name: Optional[str]) -> base.Params:
    return jax.tree.map(
        lambda p: p if _is_aux_leaf(p) else _leaf_l2_norm(p, axis_name=axis_name),
        params,
        is_leaf=_is_aux_leaf,
    )


def _all_true_mask(params: base.Params) -> base.Params:
    return jax.tree.map(
        lambda p: False if _is_aux_leaf(p) else True,
        params,
        is_leaf=_is_aux_leaf,
    )


def _default_rank2_hyperball_mask(params: base.Params) -> base.Params:
    """Default HeavyBall-style routing: Hyperball for rank >= 2 leaves."""
    return jax.tree.map(
        lambda p: (
            False
            if _is_aux_leaf(p) or not _is_array_like(p)
            else bool(getattr(p, "ndim", 0) >= 2)
        ),
        params,
        is_leaf=_is_aux_leaf,
    )


def _is_mask_callable(mask: Any) -> bool:
    callable_leaves = jax.tree.leaves(jax.tree.map(callable, mask))
    return callable(mask) and len(callable_leaves) > 0 and all(callable_leaves)


def _resolve_mask(
    mask: MaskOrFn,
    params: base.Params,
    default_fn: Callable[[base.Params], base.Params],
) -> base.Params:
    if mask is None:
        return default_fn(params)
    if _is_mask_callable(mask):
        return cast(Callable[[base.Params], Any], mask)(params)
    return cast(base.Params, mask)


def _mask_leaf_value(mask_leaf: Any) -> jax.Array:
    if _is_aux_leaf(mask_leaf):
        return jnp.asarray(False, dtype=jnp.bool_)
    return jnp.asarray(mask_leaf, dtype=jnp.bool_)


def _caution_update(
    grad: Any,
    update: jax.Array,
) -> jax.Array:
    """Cautious-update masking with mean-preserving rescaling.

    Elements whose update direction disagrees with ``grad`` are zeroed; retained
    elements are rescaled by ``numel / max(num_retained, 1)``.  The sign test
    mirrors HeavyBall's ``signbit(grad) == signbit(update)`` rule.
    """
    if _is_aux_leaf(grad):
        return update

    g = grad.astype(jnp.float32)
    u = update.astype(jnp.float32)
    keep = jnp.logical_not(jnp.signbit(g) ^ jnp.signbit(u))
    keep_f = keep.astype(jnp.float32)
    count = jnp.sum(keep_f)
    scale = jnp.asarray(update.size, dtype=jnp.float32) / jnp.maximum(count, 1.0)
    return jnp.where(keep, u * scale, jnp.zeros_like(u)).astype(jnp.float32)


def _add_decay_to_direction(
    direction: jax.Array,
    param: jax.Array,
    weight_decay: jax.Array,
    apply_decay: jax.Array,
    *,
    cautious_weight_decay: bool,
) -> jax.Array:
    p32 = param.astype(jnp.float32)
    d32 = direction.astype(jnp.float32)
    wd32 = jnp.asarray(weight_decay, dtype=jnp.float32)

    if cautious_weight_decay:
        same_sign = jnp.logical_not(jnp.signbit(p32) ^ jnp.signbit(d32))
        decay_term = p32 * wd32 * same_sign.astype(jnp.float32)
    else:
        decay_term = p32 * wd32

    return jnp.where(apply_decay, d32 + decay_term, d32)


def _hyperball_leaf_update(
    update: Any,
    param: Any,
    init_norm: Any,
    raw_grad: Any,
    hyperball_leaf: Any,
    decay_leaf: Any,
    *,
    learning_rate: jax.typing.ArrayLike,
    fallback_learning_rate: jax.typing.ArrayLike,
    weight_decay: jax.typing.ArrayLike,
    caution: bool,
    cautious_weight_decay: bool,
    fallback_weight_decay: bool,
    eps: float,
    axis_name: Optional[str],
) -> Any:
    if _is_aux_leaf(update) or _is_aux_leaf(param) or _is_aux_leaf(init_norm):
        return update

    u32 = update.astype(jnp.float32)
    p32 = param.astype(jnp.float32)
    n32 = init_norm.astype(jnp.float32)

    hb = _mask_leaf_value(hyperball_leaf)
    decay_on = _mask_leaf_value(decay_leaf)
    lr32 = jnp.asarray(learning_rate, dtype=jnp.float32)
    fallback_lr32 = jnp.asarray(fallback_learning_rate, dtype=jnp.float32)
    wd32 = jnp.asarray(weight_decay, dtype=jnp.float32)

    hyperball_direction = _add_decay_to_direction(
        u32,
        p32,
        wd32,
        decay_on,
        cautious_weight_decay=cautious_weight_decay,
    )
    if caution:
        caution_grad = update if _is_aux_leaf(raw_grad) else raw_grad
        hyperball_direction = _caution_update(caution_grad, hyperball_direction)

    hyperball_unit = hyperball_direction / _safe_norm(
        hyperball_direction, eps=eps, axis_name=axis_name
    )
    hyperball_candidate = p32 - lr32 * hyperball_unit * n32
    hyperball_projected = hyperball_candidate * (
        n32 / _safe_norm(hyperball_candidate, eps=eps, axis_name=axis_name)
    )
    hyperball_delta = hyperball_projected - p32

    fallback_direction = u32
    if fallback_weight_decay:
        fallback_direction = _add_decay_to_direction(
            fallback_direction,
            p32,
            wd32,
            decay_on,
            cautious_weight_decay=cautious_weight_decay,
        )
    fallback_delta = -fallback_lr32 * fallback_direction

    return jnp.where(hb, hyperball_delta, fallback_delta).astype(param.dtype)


def apply_hyperball(
    learning_rate: base.ScalarOrSchedule,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: MaskOrFn = None,
    *,
    hyperball_mask: MaskOrFn = None,
    fallback_learning_rate: Optional[base.ScalarOrSchedule] = None,
    fallback_weight_decay: bool = False,
    caution: bool = False,
    cautious_weight_decay: bool = False,
    eps: float = 1e-12,
    axis_name: Optional[str] = None,
) -> base.GradientTransformationExtraArgs:
    """Terminal Optax transform implementing Hyperball projection.

    This transform consumes positive, unscaled directions from a preceding
    optimizer transform and returns ordinary Optax parameter deltas.  It should be
    the final transform in a chain.

    Args:
        learning_rate: Learning rate for Hyperball-selected leaves. May be a scalar
            or Optax schedule.
        weight_decay: Hyperball decay coefficient, corresponding to HeavyBall's
            ``group["weight_decay"]`` argument. May be scalar or schedule.
        weight_decay_mask: Boolean tree/callable selecting leaves or elements that
            receive the decay term. Defaults to all non-auxiliary leaves.
        hyperball_mask: Boolean tree/callable selecting leaves optimized by
            Hyperball. Defaults to rank >= 2 leaves, matching HeavyBall's matrix
            routing in ``HyperBallAdamW``.
        fallback_learning_rate: Learning rate for non-Hyperball leaves. Defaults to
            ``learning_rate``.
        fallback_weight_decay: If True, non-Hyperball leaves receive ordinary
            decoupled AdamW-style decay inside the returned parameter delta.
            Defaults to False so `weight_decay` is fully swapped into Hyperball
            rather than retained as ordinary decay on fallback leaves.
        caution: If True, applies cautious update masking before Hyperball
            normalization. When the caller passes ``grad=...`` through Optax extra
            args, that raw gradient is used for the sign test; otherwise the current
            transformed direction is used.
        cautious_weight_decay: If True, applies decay elementwise only where
            ``sign(param) == sign(direction)``.
        eps: Minimum norm divisor.
        axis_name: Optional pmap axis name for distributed L2-norm reductions.

    Returns:
        A terminal ``GradientTransformationExtraArgs``.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    def init_fn(params: base.Params) -> HyperballState:
        return HyperballState(
            count=jnp.zeros([], dtype=jnp.int32),
            init_norm=_init_norm_tree(params, axis_name),
        )

    def update_fn(
        updates: base.Updates,
        state: base.OptState,
        params: Optional[base.Params] = None,
        **extra_args: Any,
    ) -> tuple[base.Updates, base.OptState]:
        if params is None:
            raise ValueError("`params` must be provided to `apply_hyperball`.")

        hyperball_state = cast(HyperballState, state)

        lr_step = _resolve_scalar(learning_rate, hyperball_state.count)
        fallback_lr_step = (
            lr_step
            if fallback_learning_rate is None
            else _resolve_scalar(fallback_learning_rate, hyperball_state.count)
        )
        wd_step = _resolve_scalar(weight_decay, hyperball_state.count)

        resolved_hyperball_mask = _resolve_mask(
            hyperball_mask, params, _default_rank2_hyperball_mask
        )
        resolved_weight_decay_mask = _resolve_mask(
            weight_decay_mask, params, _all_true_mask
        )
        raw_grad = extra_args.get("grad", extra_args.get("raw_gradients", updates))

        new_updates = jax.tree.map(
            lambda u, p, n, g, hbm, wdm: _hyperball_leaf_update(
                u,
                p,
                n,
                g,
                hbm,
                wdm,
                learning_rate=lr_step,
                fallback_learning_rate=fallback_lr_step,
                weight_decay=wd_step,
                caution=caution,
                cautious_weight_decay=cautious_weight_decay,
                fallback_weight_decay=fallback_weight_decay,
                eps=eps,
                axis_name=axis_name,
            ),
            updates,
            params,
            hyperball_state.init_norm,
            raw_grad,
            resolved_hyperball_mask,
            resolved_weight_decay_mask,
            is_leaf=_is_aux_leaf,
        )
        count_inc = cast(jax.Array, numerics.safe_increment(hyperball_state.count))
        return new_updates, HyperballState(
            count=count_inc, init_norm=hyperball_state.init_norm
        )

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


# Semantic alias: the transform consumes an unscaled direction and applies the
# Hyperball geometry.  It is still terminal and must not be followed by a learning
# rate scale transform.
scale_by_hyperball = apply_hyperball


def adamw_hyperball(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: jax.typing.DTypeLike | None = None,
    weight_decay: base.ScalarOrSchedule = 1e-4,
    weight_decay_mask: MaskOrFn = None,
    *,
    hyperball_mask: MaskOrFn = None,
    fallback_weight_decay: bool = False,
    caution: bool = False,
    cautious_weight_decay: bool = False,
    hyperball_eps: float = 1e-12,
    nesterov: bool = False,
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    axis_name: Optional[str] = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """Adam with Hyperball replacing decoupled weight decay on selected leaves."""
    return combine.chain(
        scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            weight_decay=0.0,
            weight_decay_mask=None,
            nesterov=nesterov,
            use_magma=use_magma,
            magma_p=magma_p,
            magma_tau=magma_tau,
            axis_name=axis_name,
            key=key,
        ),
        apply_hyperball(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            hyperball_mask=hyperball_mask,
            fallback_weight_decay=fallback_weight_decay,
            caution=caution,
            cautious_weight_decay=cautious_weight_decay,
            eps=hyperball_eps,
            axis_name=axis_name,
        ),
    )


def muon_hyperball(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Any = MUON_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: MaskOrFn = None,
    mu_dtype: jax.typing.DTypeLike | None = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    muon_weight_dimension_numbers: optax_muon.WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    hyperball_mask: MaskOrFn = None,
    fallback_weight_decay: bool = False,
    caution: bool = False,
    cautious_weight_decay: bool = False,
    hyperball_eps: float = 1e-12,
    axis_name: str | None = None,
) -> base.GradientTransformationExtraArgs:
    """Muon/Adam partition with Hyperball replacing decoupled weight decay.

    By default, Hyperball is applied to the same leaves routed to Muon. Adam
    fallback leaves receive ordinary Adam-style parameter deltas unless
    `fallback_weight_decay=True`.
    """
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    if muon_weight_dimension_numbers is None:
        use_default_muon_specs = True

        def param_labels(params: base.Params) -> base.Params:
            return jax.tree.map(
                lambda p: (
                    None
                    if _is_aux_leaf(p)
                    else ("muon" if getattr(p, "ndim", 0) == 2 else "adam")
                ),
                params,
                is_leaf=_is_aux_leaf,
            )

        def get_resolved_dim_nums(params: base.Params) -> base.Params:
            return jax.tree.map(
                lambda p: (
                    optax_muon.MuonDimensionNumbers()
                    if not _is_aux_leaf(p) and getattr(p, "ndim", 0) == 2
                    else _masking.MaskedNode()
                ),
                params,
                is_leaf=lambda x: _is_aux_leaf(x),
            )

    else:
        use_default_muon_specs = False

        def get_resolved_dim_nums(params: base.Params) -> Any:
            if callable(muon_weight_dimension_numbers):
                dim_num_fn = cast(
                    Callable[[base.Params], Any], muon_weight_dimension_numbers
                )
                return dim_num_fn(params)
            return muon_weight_dimension_numbers

        def param_labels(params: base.Params) -> base.Params:
            dim_nums = get_resolved_dim_nums(params)

            def populate_subtree(dim_num, subtree):
                return jax.tree.map(
                    lambda p: (
                        None
                        if _is_aux_leaf(p)
                        else ("muon" if dim_num is not None else "adam")
                    ),
                    subtree,
                    is_leaf=_is_aux_leaf,
                )

            return jax.tree.map(
                populate_subtree,
                dim_nums,
                params,
                is_leaf=lambda x: (
                    x is None or isinstance(x, optax_muon.MuonDimensionNumbers)
                ),
            )

        dim_nums_arg = muon_weight_dimension_numbers

    def muon_weight_dim_nums_fn(params: base.Params) -> base.Params:
        if use_default_muon_specs:
            return get_resolved_dim_nums(params)

        if callable(dim_nums_arg):
            dim_num_fn = cast(Callable[[base.Params], Any], dim_nums_arg)
            dim_nums = dim_num_fn(params)
        else:
            dim_nums = dim_nums_arg
        mask = jax.tree.map(lambda label: label == "muon", param_labels(params))

        def populate_subtree(dim_num, submask):
            return jax.tree.map(
                lambda m: dim_num if m else _masking.MaskedNode(),
                submask,
            )

        return jax.tree.map(
            populate_subtree,
            dim_nums,
            mask,
            is_leaf=lambda x: (
                x is None
                or isinstance(x, optax_muon.MuonDimensionNumbers)
                or isinstance(x, _masking.MaskedNode)
            ),
        )

    default_hyperball_mask = _spec_mask_from_resolver(
        get_resolved_dim_nums,
        is_leaf=lambda x: (
            x is None
            or isinstance(x, optax_muon.MuonDimensionNumbers)
            or isinstance(x, _masking.MaskedNode)
        ),
    )
    resolved_hyperball_mask = (
        hyperball_mask if hyperball_mask is not None else default_hyperball_mask
    )

    partitioned_updates = combine.partition(
        transforms={
            "muon": combine.chain(
                optax_muon.scale_by_muon(
                    ns_coeffs=ns_coeffs,
                    ns_steps=ns_steps,
                    beta=beta,
                    eps=eps,
                    mu_dtype=mu_dtype,
                    nesterov=nesterov,
                    adaptive=adaptive,
                    weight_dimension_numbers=muon_weight_dim_nums_fn,
                ),
                optax_muon.scale_by_shape(
                    weight_dimension_numbers=muon_weight_dim_nums_fn,
                    consistent_rms=consistent_rms,
                ),
            ),
            "adam": scale_by_adam(
                b1=adam_b1,
                b2=adam_b2,
                eps=eps,
                eps_root=adam_eps_root,
                mu_dtype=mu_dtype,
                weight_decay=0.0,
                weight_decay_mask=None,
                nesterov=nesterov,
            ),
        },
        param_labels=param_labels,
    )

    return combine.chain(
        partitioned_updates,
        apply_hyperball(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            hyperball_mask=resolved_hyperball_mask,
            fallback_learning_rate=adam_learning_rate,
            fallback_weight_decay=fallback_weight_decay,
            caution=caution,
            cautious_weight_decay=cautious_weight_decay,
            eps=hyperball_eps,
            axis_name=axis_name,
        ),
    )


def rmnp_hyperball(
    learning_rate: base.ScalarOrSchedule,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: MaskOrFn = None,
    mu_dtype: jax.typing.DTypeLike | None = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    rmnp_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    hyperball_mask: MaskOrFn = None,
    fallback_weight_decay: bool = False,
    caution: bool = False,
    cautious_weight_decay: bool = False,
    hyperball_eps: float = 1e-12,
    axis_name: str | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """RMNP/Adam partition with Hyperball replacing decoupled weight decay.

    By default, Hyperball is applied to the same leaves routed to RMNP. Adam
    fallback leaves receive ordinary Adam-style parameter deltas unless
    `fallback_weight_decay=True`.
    """
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    key_rmnp, key_adam = jax.random.split(key, 2)

    def get_resolved_dim_nums(params: base.Params) -> base.Params:
        return _get_dimension_numbers(rmnp_weight_dimension_numbers, params)

    def param_labels(params: base.Params) -> base.Params:
        dim_nums = get_resolved_dim_nums(params)
        return jax.tree.map(
            lambda d, p: None if p is None else ("rmnp" if d is not None else "adam"),
            dim_nums,
            params,
            is_leaf=_is_dimension_numbers_leaf,
        )

    def rmnp_weight_dim_nums_fn(params: base.Params) -> base.Params:
        return _mask_dimension_numbers(get_resolved_dim_nums(params))

    default_hyperball_mask = _spec_mask_from_resolver(
        get_resolved_dim_nums,
        is_leaf=_is_dimension_numbers_leaf,
    )
    resolved_hyperball_mask = (
        hyperball_mask if hyperball_mask is not None else default_hyperball_mask
    )

    partitioned_updates = combine.partition(
        transforms={
            "rmnp": combine.chain(
                scale_by_rmnp(
                    beta=beta,
                    eps=eps,
                    mu_dtype=mu_dtype,
                    nesterov=nesterov,
                    adaptive=adaptive,
                    weight_dimension_numbers=rmnp_weight_dim_nums_fn,
                    key=key_rmnp,
                ),
                scale_by_rmnp_shape(
                    weight_dimension_numbers=rmnp_weight_dim_nums_fn,
                    consistent_rms=consistent_rms,
                ),
            ),
            "adam": scale_by_adam(
                b1=adam_b1,
                b2=adam_b2,
                eps=eps,
                eps_root=adam_eps_root,
                mu_dtype=mu_dtype,
                weight_decay=0.0,
                weight_decay_mask=None,
                nesterov=nesterov,
                key=key_adam,
            ),
        },
        param_labels=param_labels,
    )

    return combine.chain(
        partitioned_updates,
        apply_hyperball(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            hyperball_mask=resolved_hyperball_mask,
            fallback_learning_rate=adam_learning_rate,
            fallback_weight_decay=fallback_weight_decay,
            caution=caution,
            cautious_weight_decay=cautious_weight_decay,
            eps=hyperball_eps,
            axis_name=axis_name,
        ),
    )


def kron_hyperball(
    learning_rate: base.ScalarOrSchedule = 0.001,
    b1: float = 0.9,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: MaskOrFn = None,
    preconditioner_update_probability: base.ScalarOrSchedule = (
        precond_update_prob_schedule()
    ),
    max_size_triangular: int = 8192,
    max_skew_triangular: float = 1.0,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    whiten_grad: bool = True,
    update_preconditioner_first: bool = True,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: Optional[float] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    preconditioner_mode: Union[str, PreconditionerMode] = PreconditionerMode.Q0P5EQ1P5,
    beta_lipschitz: float = 0.9,
    track_lipschitz: bool = True,
    damping: float = 1e-9,
    grad_clip_max_amps: float | Tuple[float, float] = (2.0, 10.0),
    grad_clip_mode: Union[str, GradClipMode] = GradClipMode.PER_TENSOR_RMS,
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    newton_schulz_iters: int = 5,
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    axis_name: Optional[str] = None,
    key: jax.Array = jax.random.PRNGKey(42),
    *,
    hyperball_mask: MaskOrFn = None,
    fallback_weight_decay: bool = False,
    caution: bool = False,
    cautious_weight_decay: bool = False,
    hyperball_eps: float = 1e-12,
) -> base.GradientTransformationExtraArgs:
    """PSGD Kron with Hyperball replacing decoupled weight decay."""
    kron_transform = scale_by_kron(
        b1=b1,
        preconditioner_update_probability=preconditioner_update_probability,
        max_size_triangular=max_size_triangular,
        max_skew_triangular=max_skew_triangular,
        min_ndim_triangular=min_ndim_triangular,
        memory_save_mode=memory_save_mode,
        whiten_grad=whiten_grad,
        update_preconditioner_first=update_preconditioner_first,
        preconditioner_lr=preconditioner_lr,
        preconditioner_init_scale=preconditioner_init_scale,
        mu_dtype=mu_dtype,
        precond_dtype=precond_dtype,
        precond_update_precision=precond_update_precision,
        precond_grads_precision=precond_grads_precision,
        scanned_layers=scanned_layers,
        lax_map_scanned_layers=lax_map_scanned_layers,
        lax_map_batch_size=lax_map_batch_size,
        preconditioner_mode=preconditioner_mode,
        beta_lipschitz=beta_lipschitz,
        track_lipschitz=track_lipschitz,
        damping=damping,
        grad_clip_max_amps=grad_clip_max_amps,
        grad_clip_mode=grad_clip_mode,
        raw_global_grad_clip=raw_global_grad_clip,
        permissive_spike_protection=permissive_spike_protection,
        newton_schulz_iters=newton_schulz_iters,
        use_magma=use_magma,
        magma_p=magma_p,
        magma_tau=magma_tau,
        weight_decay=0.0,
        weight_decay_mask=None,
        axis_name=axis_name,
        key=key,
    )
    return combine.chain(
        base.GradientTransformation(kron_transform.init, kron_transform.update),
        apply_hyperball(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            hyperball_mask=hyperball_mask,
            fallback_weight_decay=fallback_weight_decay,
            caution=caution,
            cautious_weight_decay=cautious_weight_decay,
            eps=hyperball_eps,
            axis_name=axis_name,
        ),
    )


def _spec_mask_from_resolver(
    resolver: Callable[[base.Params], base.Params],
    *,
    is_leaf: Callable[[Any], bool],
) -> Callable[[base.Params], base.Params]:
    def _mask(params: base.Params) -> base.Params:
        specs = resolver(params)
        return jax.tree.map(
            lambda spec, p: (
                False
                if _is_aux_leaf(p)
                else (spec is not None and not isinstance(spec, _masking.MaskedNode))
            ),
            specs,
            params,
            is_leaf=is_leaf,
        )

    return _mask


def prism_hyperball(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.95,
    gamma: float = 1.0,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: MaskOrFn = None,
    ns_iters: int = 5,
    mode: str = "original",
    inv_steps: int = 6,
    inv_eps: float = 1e-5,
    inv_scale: float = 1.001,
    eps_gram: float = 1e-6,
    gamma_l: Optional[float] = None,
    gamma_r: Optional[float] = None,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    axis_name: Optional[str] = None,
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    key: jax.Array = jax.random.PRNGKey(42),
    adam_learning_rate: Optional[base.ScalarOrSchedule] = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    prism_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    *,
    hyperball_mask: MaskOrFn = None,
    caution: bool = False,
    cautious_weight_decay: bool = False,
    hyperball_eps: float = 1e-12,
    fallback_weight_decay: bool = False,
) -> base.GradientTransformationExtraArgs:
    """PRISM/Adam partition with Hyperball replacing decoupled weight decay.

    By default, the Hyperball projection is applied to the same leaves routed to
    PRISM; Adam fallback leaves receive ordinary Adam-style parameter deltas.
    """
    key_prism, key_adam = jax.random.split(key, 2)
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    def get_resolved_dim_nums(params: base.Params) -> base.Params:
        return _get_dimension_numbers(prism_weight_dimension_numbers, params)

    def param_labels(params: base.Params) -> base.Params:
        dim_nums = get_resolved_dim_nums(params)
        return jax.tree.map(
            lambda d, p: None if p is None else ("prism" if d is not None else "adam"),
            dim_nums,
            params,
            is_leaf=_is_dimension_numbers_leaf,
        )

    def prism_weight_dim_nums_fn(params: base.Params) -> base.Params:
        return _mask_dimension_numbers(get_resolved_dim_nums(params))

    default_hyperball_mask = _spec_mask_from_resolver(
        get_resolved_dim_nums,
        is_leaf=_is_dimension_numbers_leaf,
    )
    resolved_hyperball_mask = (
        hyperball_mask if hyperball_mask is not None else default_hyperball_mask
    )

    partitioned_updates = combine.partition(
        transforms={
            "prism": scale_by_prism(
                b1=b1,
                gamma=gamma,
                ns_iters=ns_iters,
                mode=mode,
                inv_steps=inv_steps,
                inv_eps=inv_eps,
                inv_scale=inv_scale,
                eps_gram=eps_gram,
                gamma_l=gamma_l,
                gamma_r=gamma_r,
                precision=precision,
                nesterov=nesterov,
                shape_nesterov=shape_nesterov,
                bias_correction=bias_correction,
                mu_dtype=mu_dtype,
                raw_global_grad_clip=raw_global_grad_clip,
                permissive_spike_protection=permissive_spike_protection,
                grad_clip_max_amps=grad_clip_max_amps,
                weight_dimension_numbers=prism_weight_dim_nums_fn,
                use_magma=use_magma,
                magma_p=magma_p,
                magma_tau=magma_tau,
                weight_decay=0.0,
                weight_decay_mask=None,
                axis_name=axis_name,
                key=key_prism,
            ),
            "adam": scale_by_adam(
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
                mu_dtype=mu_dtype,
                weight_decay=0.0,
                weight_decay_mask=None,
                use_magma=use_magma,
                magma_p=magma_p,
                magma_tau=magma_tau,
                axis_name=axis_name,
                key=key_adam,
            ),
        },
        param_labels=param_labels,
    )

    return combine.chain(
        partitioned_updates,
        apply_hyperball(
            learning_rate=learning_rate,
            fallback_learning_rate=adam_learning_rate,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            hyperball_mask=resolved_hyperball_mask,
            fallback_weight_decay=fallback_weight_decay,
            caution=caution,
            cautious_weight_decay=cautious_weight_decay,
            eps=hyperball_eps,
            axis_name=axis_name,
        ),
    )


def _partitioned_aurora_hyperball(
    *,
    riemannian: bool,
    learning_rate: base.ScalarOrSchedule,
    b1: float,
    weight_decay: base.ScalarOrSchedule,
    weight_decay_mask: MaskOrFn,
    pp_iterations: int,
    pp_beta: float,
    outer_steps: int,
    cg_steps: int,
    riemannian_eta: float,
    retraction_steps: int,
    polar_ns_iters: int,
    polar_compute_dtype: jax.typing.DTypeLike,
    polar_output_dtype: jax.typing.DTypeLike,
    precision: jax.lax.PrecisionLike,
    eps: float,
    nesterov: bool,
    shape_nesterov: bool,
    bias_correction: bool,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]],
    raw_global_grad_clip: Optional[float],
    permissive_spike_protection: bool,
    mu_dtype: Optional[jax.typing.DTypeLike],
    axis_name: Optional[str],
    use_magma: bool,
    magma_p: float,
    magma_tau: float,
    guard_nonfinite: bool,
    key: jax.Array,
    adam_learning_rate: Optional[base.ScalarOrSchedule],
    adam_b1: float,
    adam_b2: float,
    adam_eps: float,
    aurora_weight_dimension_numbers: AuroraWeightDimNumOrFn | None,
    hyperball_mask: MaskOrFn,
    caution: bool,
    cautious_weight_decay: bool,
    hyperball_eps: float,
    fallback_weight_decay: bool,
) -> base.GradientTransformationExtraArgs:
    key_aurora, key_adam = jax.random.split(key, 2)
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    def get_resolved_dim_nums(params: base.Params) -> base.Params:
        return _get_dimension_numbers(aurora_weight_dimension_numbers, params)

    def param_labels(params: base.Params) -> base.Params:
        dim_nums = get_resolved_dim_nums(params)
        return jax.tree.map(
            lambda d, p: None if p is None else ("aurora" if d is not None else "adam"),
            dim_nums,
            params,
            is_leaf=_is_dimension_numbers_leaf,
        )

    def aurora_weight_dim_nums_fn(params: base.Params) -> base.Params:
        return _mask_dimension_numbers(get_resolved_dim_nums(params))

    if riemannian:
        aurora_transform = scale_by_riemannian_aurora(
            b1=b1,
            outer_steps=outer_steps,
            cg_steps=cg_steps,
            riemannian_eta=riemannian_eta,
            retraction_steps=retraction_steps,
            polar_ns_iters=polar_ns_iters,
            polar_compute_dtype=polar_compute_dtype,
            polar_output_dtype=polar_output_dtype,
            precision=precision,
            eps=eps,
            nesterov=nesterov,
            shape_nesterov=shape_nesterov,
            bias_correction=bias_correction,
            mu_dtype=mu_dtype,
            raw_global_grad_clip=raw_global_grad_clip,
            permissive_spike_protection=permissive_spike_protection,
            grad_clip_max_amps=grad_clip_max_amps,
            weight_dimension_numbers=aurora_weight_dim_nums_fn,
            use_magma=use_magma,
            magma_p=magma_p,
            magma_tau=magma_tau,
            weight_decay=0.0,
            weight_decay_mask=None,
            axis_name=axis_name,
            guard_nonfinite=guard_nonfinite,
            key=key_aurora,
        )
    else:
        aurora_transform = scale_by_aurora(
            b1=b1,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
            polar_ns_iters=polar_ns_iters,
            polar_compute_dtype=polar_compute_dtype,
            polar_output_dtype=polar_output_dtype,
            precision=precision,
            eps=eps,
            nesterov=nesterov,
            shape_nesterov=shape_nesterov,
            bias_correction=bias_correction,
            mu_dtype=mu_dtype,
            raw_global_grad_clip=raw_global_grad_clip,
            permissive_spike_protection=permissive_spike_protection,
            grad_clip_max_amps=grad_clip_max_amps,
            weight_dimension_numbers=aurora_weight_dim_nums_fn,
            use_magma=use_magma,
            magma_p=magma_p,
            magma_tau=magma_tau,
            weight_decay=0.0,
            weight_decay_mask=None,
            axis_name=axis_name,
            guard_nonfinite=guard_nonfinite,
            key=key_aurora,
        )

    default_hyperball_mask = _spec_mask_from_resolver(
        get_resolved_dim_nums,
        is_leaf=_is_dimension_numbers_leaf,
    )
    resolved_hyperball_mask = (
        hyperball_mask if hyperball_mask is not None else default_hyperball_mask
    )

    partitioned_updates = combine.partition(
        transforms={
            "aurora": aurora_transform,
            "adam": scale_by_adam(
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
                mu_dtype=mu_dtype,
                weight_decay=0.0,
                weight_decay_mask=None,
                use_magma=use_magma,
                magma_p=magma_p,
                magma_tau=magma_tau,
                axis_name=axis_name,
                key=key_adam,
            ),
        },
        param_labels=param_labels,
    )

    return combine.chain(
        partitioned_updates,
        apply_hyperball(
            learning_rate=learning_rate,
            fallback_learning_rate=adam_learning_rate,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            hyperball_mask=resolved_hyperball_mask,
            fallback_weight_decay=fallback_weight_decay,
            caution=caution,
            cautious_weight_decay=cautious_weight_decay,
            eps=hyperball_eps,
            axis_name=axis_name,
        ),
    )


def aurora_hyperball(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.95,
    weight_decay: base.ScalarOrSchedule = 0.025,
    weight_decay_mask: MaskOrFn = None,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    polar_ns_iters: int = 12,
    polar_compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    polar_output_dtype: jax.typing.DTypeLike = jnp.float32,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    eps: float = 1e-7,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    axis_name: Optional[str] = None,
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    guard_nonfinite: bool = True,
    key: jax.Array = jax.random.PRNGKey(42),
    adam_learning_rate: Optional[base.ScalarOrSchedule] = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    aurora_weight_dimension_numbers: AuroraWeightDimNumOrFn | None = None,
    *,
    hyperball_mask: MaskOrFn = None,
    caution: bool = False,
    cautious_weight_decay: bool = False,
    hyperball_eps: float = 1e-12,
    fallback_weight_decay: bool = False,
) -> base.GradientTransformationExtraArgs:
    """Aurora/Adam partition with Hyperball replacing decoupled weight decay."""
    return _partitioned_aurora_hyperball(
        riemannian=False,
        learning_rate=learning_rate,
        b1=b1,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        pp_iterations=pp_iterations,
        pp_beta=pp_beta,
        outer_steps=3,
        cg_steps=20,
        riemannian_eta=0.1,
        retraction_steps=2,
        polar_ns_iters=polar_ns_iters,
        polar_compute_dtype=polar_compute_dtype,
        polar_output_dtype=polar_output_dtype,
        precision=precision,
        eps=eps,
        nesterov=nesterov,
        shape_nesterov=shape_nesterov,
        bias_correction=bias_correction,
        grad_clip_max_amps=grad_clip_max_amps,
        raw_global_grad_clip=raw_global_grad_clip,
        permissive_spike_protection=permissive_spike_protection,
        mu_dtype=mu_dtype,
        axis_name=axis_name,
        use_magma=use_magma,
        magma_p=magma_p,
        magma_tau=magma_tau,
        guard_nonfinite=guard_nonfinite,
        key=key,
        adam_learning_rate=adam_learning_rate,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps=adam_eps,
        aurora_weight_dimension_numbers=aurora_weight_dimension_numbers,
        hyperball_mask=hyperball_mask,
        caution=caution,
        cautious_weight_decay=cautious_weight_decay,
        hyperball_eps=hyperball_eps,
        fallback_weight_decay=fallback_weight_decay,
    )


def riemannian_aurora_hyperball(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.95,
    weight_decay: base.ScalarOrSchedule = 0.025,
    weight_decay_mask: MaskOrFn = None,
    outer_steps: int = 3,
    cg_steps: int = 20,
    riemannian_eta: float = 0.1,
    retraction_steps: int = 2,
    polar_ns_iters: int = 12,
    polar_compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    polar_output_dtype: jax.typing.DTypeLike = jnp.float32,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    eps: float = 1e-7,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    axis_name: Optional[str] = None,
    use_magma: bool = False,
    magma_p: float = 0.5,
    magma_tau: float = 2.0,
    guard_nonfinite: bool = True,
    key: jax.Array = jax.random.PRNGKey(42),
    adam_learning_rate: Optional[base.ScalarOrSchedule] = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    aurora_weight_dimension_numbers: AuroraWeightDimNumOrFn | None = None,
    *,
    hyperball_mask: MaskOrFn = None,
    caution: bool = False,
    cautious_weight_decay: bool = False,
    hyperball_eps: float = 1e-12,
    fallback_weight_decay: bool = False,
) -> base.GradientTransformationExtraArgs:
    """Riemannian-Aurora/Adam partition with Hyperball projection."""
    return _partitioned_aurora_hyperball(
        riemannian=True,
        learning_rate=learning_rate,
        b1=b1,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        pp_iterations=2,
        pp_beta=0.5,
        outer_steps=outer_steps,
        cg_steps=cg_steps,
        riemannian_eta=riemannian_eta,
        retraction_steps=retraction_steps,
        polar_ns_iters=polar_ns_iters,
        polar_compute_dtype=polar_compute_dtype,
        polar_output_dtype=polar_output_dtype,
        precision=precision,
        eps=eps,
        nesterov=nesterov,
        shape_nesterov=shape_nesterov,
        bias_correction=bias_correction,
        grad_clip_max_amps=grad_clip_max_amps,
        raw_global_grad_clip=raw_global_grad_clip,
        permissive_spike_protection=permissive_spike_protection,
        mu_dtype=mu_dtype,
        axis_name=axis_name,
        use_magma=use_magma,
        magma_p=magma_p,
        magma_tau=magma_tau,
        guard_nonfinite=guard_nonfinite,
        key=key,
        adam_learning_rate=adam_learning_rate,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps=adam_eps,
        aurora_weight_dimension_numbers=aurora_weight_dimension_numbers,
        hyperball_mask=hyperball_mask,
        caution=caution,
        cautious_weight_decay=cautious_weight_decay,
        hyperball_eps=hyperball_eps,
        fallback_weight_decay=fallback_weight_decay,
    )


hyperball_riemannian_aurora = riemannian_aurora_hyperball


__all__ = [
    "HyperballState",
    "apply_hyperball",
    "scale_by_hyperball",
    "adamw_hyperball",
    "kron_hyperball",
    "muon_hyperball",
    "rmnp_hyperball",
    "prism_hyperball",
    "aurora_hyperball",
    "riemannian_aurora_hyperball",
    "hyperball_riemannian_aurora",
]
