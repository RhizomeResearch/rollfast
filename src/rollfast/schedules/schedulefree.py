import inspect
from collections.abc import Callable
from enum import Enum
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from optax._src import base, combine, numerics, transform

from rollfast.optim.adam import adamw
from rollfast.optim.aurora import (
    AuroraWeightDimNumOrFn,
    scale_by_aurora,
    scale_by_riemannian_aurora,
)
from rollfast.optim.dimension_numbers import (
    WeightDimNumOrFn,
    _make_matrix_partition_fns,
)
from rollfast.optim.orthogonalization import (
    MUON_NS_COEFFS,
    MuonNsCoeffs,
    MuonPreconditioning,
)
from rollfast.optim.prism import (
    scale_by_prism,
)
from rollfast.optim.psgd import (
    GradClipMode,
    PreconditionerMode,
    precond_update_prob_schedule,
    scale_by_kron,
)
from rollfast.schedules.wsd import _make_wsd_schedule_pair, wsd_schedule
from rollfast.utils import (
    _fresh_prng_key,
    _stochastic_round_bf16,
    _validate_nonnegative_static_scalar,
)

ScheduleFreeLearningRate = (
    base.ScalarOrSchedule | Callable[[jax.typing.ArrayLike, base.Params | None], Any]
)


class WeightingMode(str, Enum):
    """
    Determines how the iterate averaging parameter c_t is computed.

    Ref: 2511.07767v1 "Schedulers for Schedule-Free"
    """

    THEORETICAL = "theoretical"  # c_t = 1/t (w_t = 1)
    PRACTICAL = "practical"  # c_t = gamma_t^2 / sum(gamma^2) (w_t = gamma_t^2)
    SCHEDULET = "schedulet"  # c_t = gamma_t / sum(gamma) (w_t = gamma_t)
    POWER = "power"  # SF+: c_t from w_t = t^r * gamma_max^p


class ScheduleFreeState(NamedTuple):
    """State for the Schedule-Free wrapper.

    ``b1`` stores the Schedule-Free interpolation coefficient used to produce
    the current ``params``/``y`` leaves from ``x`` and ``z``.  It is stateful so
    beta annealing can reconstruct the implicit ``x`` sequence exactly on the
    next update.  The optional fields are populated by the SF+ path and default
    to ``None`` to make older checkpoints easier to migrate.
    """

    b1: jax.Array
    weight_sum: base.Params
    step_count: jax.Array
    base_state: base.OptState
    z: base.Params
    key: jax.Array
    lr_max: base.Params | None = None
    scheduled_lr: base.Params | None = None
    grad_l1_ema: jax.Array | None = None
    grad_l1_ema_corr: jax.Array | None = None
    polyak_lr: jax.Array | None = None
    polyak_value: jax.Array | None = None
    ip_term: jax.Array | None = None


def _call_learning_rate(
    learning_rate: Callable[..., Any],
    count: jax.Array,
    params: base.Params | None,
) -> Any:
    try:
        signature = inspect.signature(learning_rate)
    except (TypeError, ValueError):
        return learning_rate(count)

    parameters = tuple(signature.parameters.values())
    accepts_two_positional = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    ) or (
        sum(
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for parameter in parameters
        )
        >= 2
    )

    if accepts_two_positional:
        return learning_rate(count, params)
    if "params" in signature.parameters:
        return learning_rate(count, params=params)
    return learning_rate(count)


def _make_dual_schedule_fn(
    partition: Any,
    matrix_label: str,
    matrix_schedule: base.Schedule,
    adam_schedule: base.Schedule,
) -> Callable[[jax.typing.ArrayLike, base.Params | None], Any]:
    def dual_schedule_fn(count, params):
        if params is None:
            raise ValueError("dual schedule functions require `params`.")
        labels = partition.labels(params)
        matrix_lr = matrix_schedule(count)
        adam_lr = adam_schedule(count)
        return jax.tree.map(
            lambda label: (
                None
                if label is None
                else (matrix_lr if label == matrix_label else adam_lr)
            ),
            labels,
            is_leaf=lambda x: x is None,
        )

    return dual_schedule_fn


def _append_decayed_weights_and_lr(
    components: list[Any],
    *,
    weight_decay: float,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None,
    learning_rate: base.ScalarOrSchedule,
) -> list[Any]:
    _validate_nonnegative_static_scalar("weight_decay", weight_decay)
    _wd_is_nonzero = (
        weight_decay > 0.0 if isinstance(weight_decay, (int, float)) else True
    )
    if _wd_is_nonzero:
        components.append(
            transform.add_decayed_weights(weight_decay, weight_decay_mask)
        )
    components.append(transform.scale_by_learning_rate(learning_rate))
    return components


def schedule_free(
    base_optimizer: base.GradientTransformation,
    learning_rate: ScheduleFreeLearningRate,
    b1: float = 0.9,
    weighting_mode: str | WeightingMode = WeightingMode.SCHEDULET,
    state_dtype: jax.typing.DTypeLike | None = None,
    key: jax.Array | None = None,
    *,
    r: float = 0.0,
    c_warmup: int = 0,
    weight_lr_power: float = 2.0,
    use_lr_max: bool = False,
    b1_anneal_steps: int = 0,
    b1_max: float | None = 0.965,
    polyak: bool = False,
    polyak_beta: float = 0.0,
    polyak_f_star: float = 0.0,
    polyak_eps: float = 1e-30,
    polyak_axis_name: str | tuple[str, ...] | None = None,
    lr_max_init: float = 1e-8,
    adamc_weight_decay: float = 0.0,
    adamc_weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free wrapper with optional ScheduleFree+ components.

    The default arguments preserve the original wrapper behavior: the inner
    optimizer returns a fully scaled update for ``z``, and the wrapper maintains
    the averaged ``x`` sequence implicitly from the current ``params``/``y`` and
    the stored ``z`` sequence.

    Setting ``polyak=True`` enables the ScheduleFree+ Polyak scalar.  The caller
    must pass the current objective value to ``tx.update`` as ``function_value``,
    ``loss``, or ``value``.  The scalar multiplies the inner optimizer's update,
    computes the Schedule-Free averaging weights from the effective learning
    rate, and can be paired with ``adamc_weight_decay`` to apply the AdamC
    fully-decoupled weight decay update ``-lr_t**2 * weight_decay * y``.
    """
    if isinstance(weighting_mode, str):
        weighting_mode = WeightingMode(weighting_mode)

    if weighting_mode == WeightingMode.POWER:
        use_lr_max = True
    if b1_max is None:
        b1_max = b1
    if c_warmup < 0:
        raise ValueError("c_warmup must be non-negative.")
    if b1_anneal_steps < 0:
        raise ValueError("b1_anneal_steps must be non-negative.")
    if not 0.0 <= polyak_beta < 1.0:
        raise ValueError("polyak_beta must satisfy 0 <= polyak_beta < 1.")
    if polyak and b1 <= 0.0:
        raise ValueError("polyak=True requires b1 > 0 to reconstruct the x sequence.")

    wrapper_extra_arg_names = frozenset(("function_value", "loss", "value"))

    def _tree_scalar_like_params(value, params):
        return jax.tree.map(
            lambda x: jnp.asarray(value, dtype=jnp.float32) if x is not None else None,
            params,
            is_leaf=lambda x: x is None,
        )

    def _broadcast_to_params(tree_or_scalar, params):
        params_structure = jax.tree.structure(params, is_leaf=lambda x: x is None)
        try:
            tree_structure = jax.tree.structure(
                tree_or_scalar, is_leaf=lambda x: x is None
            )
        except TypeError:
            tree_structure = None
        if tree_structure != params_structure:
            tree_or_scalar = jax.tree.map(
                lambda _: tree_or_scalar, params, is_leaf=lambda x: x is None
            )
        return tree_or_scalar

    def _as_float32_tree(tree):
        return jax.tree.map(
            lambda x: jnp.asarray(x, dtype=jnp.float32) if x is not None else None,
            tree,
            is_leaf=lambda x: x is None,
        )

    def _resolve_learning_rate(count, params):
        if callable(learning_rate):
            lr_fn = cast(Any, learning_rate)
            lr_tree = _call_learning_rate(lr_fn, count, params)
        else:
            lr_tree = learning_rate
        lr_tree = _broadcast_to_params(lr_tree, params)
        return _as_float32_tree(lr_tree)

    def _sum_abs_tree(tree):
        leaves = [
            x
            for x in jax.tree.leaves(tree, is_leaf=lambda x: x is None)
            if x is not None
        ]
        total = jnp.asarray(0.0, dtype=jnp.float32)
        for leaf in leaves:
            total = total + jnp.sum(jnp.abs(leaf.astype(jnp.float32)))
        return total

    def _inner_product_correction(updates, params, z, beta_ratio):
        # Since y_t = beta_prev * x_t + (1 - beta_prev) * z_t,
        # z_t - y_t = beta_prev * (z_t - x_t). Therefore
        # beta_next * <g_t, z_t - x_t> = (beta_next / beta_prev) * <g_t, z_t - y_t>.
        leaves = jax.tree.leaves(
            jax.tree.map(
                lambda g, y, zi: (
                    jnp.sum(
                        g.astype(jnp.float32)
                        * (zi.astype(jnp.float32) - y.astype(jnp.float32))
                    )
                    if (g is not None and y is not None and zi is not None)
                    else None
                ),
                updates,
                params,
                z,
                is_leaf=lambda x: x is None,
            ),
            is_leaf=lambda x: x is None,
        )
        total = sum(
            (leaf for leaf in leaves if leaf is not None),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return beta_ratio * total

    def _annealed_b1(count):
        if b1_anneal_steps > 0:
            progress = jnp.minimum(
                count.astype(jnp.float32) / float(b1_anneal_steps), 1.0
            )
            one_minus_start = jnp.maximum(1.0 - float(b1), 1e-12)
            one_minus_end = jnp.maximum(1.0 - float(b1_max), 1e-12)
            return 1.0 - jnp.exp(
                (1.0 - progress) * jnp.log(one_minus_start)
                + progress * jnp.log(one_minus_end)
            )
        return jnp.asarray(b1, dtype=jnp.float32)

    def _weight_decay_mask_tree(params):
        if adamc_weight_decay_mask is None:
            return jax.tree.map(
                lambda x: True if x is not None else None,
                params,
                is_leaf=lambda x: x is None,
            )
        mask = (
            adamc_weight_decay_mask(params)
            if callable(adamc_weight_decay_mask)
            else adamc_weight_decay_mask
        )
        return _broadcast_to_params(mask, params)

    base_optimizer = base.with_extra_args_support(base_optimizer)

    def init_fn(params):
        state_key = _fresh_prng_key(key)
        dtype = (
            state_dtype
            if state_dtype is not None
            else optax.tree.dtype(params, "lowest")
        )
        z = jax.tree.map(
            lambda t: jnp.array(t, dtype=dtype, copy=True) if t is not None else None,
            params,
            is_leaf=lambda x: x is None,
        )

        return ScheduleFreeState(
            b1=jnp.asarray(b1, dtype=jnp.float32),
            weight_sum=_tree_scalar_like_params(0.0, params),
            step_count=jnp.zeros([], dtype=jnp.int32),
            base_state=base_optimizer.init(params),
            z=z,
            key=state_key,
            lr_max=_tree_scalar_like_params(lr_max_init, params),
            scheduled_lr=_tree_scalar_like_params(0.0, params),
            grad_l1_ema=jnp.asarray(0.0, dtype=jnp.float32),
            grad_l1_ema_corr=jnp.asarray(0.0, dtype=jnp.float32),
            polyak_lr=jnp.asarray(1.0, dtype=jnp.float32),
            polyak_value=jnp.asarray(0.0, dtype=jnp.float32),
            ip_term=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def update_fn(updates, state, params=None, **extra_args):
        if params is None:
            raise ValueError(
                "`params` must be provided to `schedule_free.update`; "
                "Schedule-Free needs the current parameters to interpolate "
                "between the averaged and z sequences."
            )

        next_state_key, sr_key = jax.random.split(state.key, 2)
        lr_tree = _resolve_learning_rate(state.step_count, params)
        lr_leaves = jax.tree.leaves(lr_tree, is_leaf=lambda x: x is None)
        if any(x is not None and x.shape != () for x in lr_leaves):
            raise ValueError(
                "schedule_free learning-rate leaves must be scalar values. "
                "Return one scalar or a PyTree of scalar leaves from callable "
                "learning-rate schedules."
            )

        function_value = extra_args.get("function_value", None)
        if function_value is None:
            function_value = extra_args.get("loss", None)
        if function_value is None:
            function_value = extra_args.get("value", None)
        if polyak and function_value is None:
            raise ValueError(
                "polyak=True requires update(..., function_value=...), "
                "update(..., loss=...), or update(..., value=...)."
            )

        base_extra_args = {
            k: v for k, v in extra_args.items() if k not in wrapper_extra_arg_names
        }
        if base_extra_args:
            base_updates, new_base_state = base_optimizer.update(
                updates, state.base_state, params, **base_extra_args
            )
        else:
            base_updates, new_base_state = base_optimizer.update(
                updates, state.base_state, params
            )

        b1_prev_safe = jnp.maximum(state.b1, 1e-8)
        b1_next = _annealed_b1(state.step_count)
        beta_ratio = b1_next / b1_prev_safe

        state_grad_l1_ema = (
            state.grad_l1_ema
            if state.grad_l1_ema is not None
            else jnp.asarray(0.0, dtype=jnp.float32)
        )
        state_grad_l1_ema_corr = (
            state.grad_l1_ema_corr
            if state.grad_l1_ema_corr is not None
            else jnp.asarray(0.0, dtype=jnp.float32)
        )

        if polyak:
            grad_l1 = _sum_abs_tree(updates)
            ip_term = _inner_product_correction(updates, params, state.z, beta_ratio)
            f_value = jnp.asarray(function_value, dtype=jnp.float32)
            if polyak_axis_name is not None:
                grad_l1 = jax.lax.psum(grad_l1, axis_name=polyak_axis_name)
                ip_term = jax.lax.psum(ip_term, axis_name=polyak_axis_name)
                f_value = jax.lax.pmean(f_value, axis_name=polyak_axis_name)

            grad_l1_ema = float(polyak_beta) * state_grad_l1_ema + (
                1.0 - float(polyak_beta)
            ) * grad_l1 * jnp.sqrt(jnp.asarray(jnp.pi / 2.0, dtype=jnp.float32))
            bias_correction = 1.0 - jnp.asarray(polyak_beta, dtype=jnp.float32) ** (
                state.step_count.astype(jnp.float32) + 1.0
            )
            grad_l1_ema_corr = grad_l1_ema / jnp.maximum(bias_correction, 1e-30)
            polyak_value = f_value - float(polyak_f_star) + ip_term
            polyak_lr = jnp.maximum(0.0, polyak_value) / jnp.maximum(
                grad_l1_ema_corr, float(polyak_eps)
            )
        else:
            grad_l1_ema = state_grad_l1_ema
            grad_l1_ema_corr = state_grad_l1_ema_corr
            ip_term = jnp.asarray(0.0, dtype=jnp.float32)
            polyak_value = jnp.asarray(0.0, dtype=jnp.float32)
            polyak_lr = jnp.asarray(1.0, dtype=jnp.float32)

        effective_lr_tree = jax.tree.map(
            lambda lr: lr * polyak_lr if lr is not None else None,
            lr_tree,
            is_leaf=lambda x: x is None,
        )
        current_lr_max = (
            state.lr_max
            if state.lr_max is not None
            else _tree_scalar_like_params(lr_max_init, params)
        )
        lr_max = jax.tree.map(
            lambda old, lr: (
                jnp.maximum(old, lr) if old is not None and lr is not None else old
            ),
            current_lr_max,
            effective_lr_tree,
            is_leaf=lambda x: x is None,
        )

        lr_for_weight_tree = lr_max if use_lr_max else effective_lr_tree
        if weighting_mode == WeightingMode.SCHEDULET:
            base_weight_tree = lr_for_weight_tree
        elif weighting_mode in (WeightingMode.PRACTICAL, WeightingMode.POWER):
            base_weight_tree = jax.tree.map(
                lambda x: (
                    jnp.power(x, float(weight_lr_power)) if x is not None else None
                ),
                lr_for_weight_tree,
                is_leaf=lambda x: x is None,
            )
        else:  # THEORETICAL
            base_weight_tree = jax.tree.map(
                lambda x: jnp.ones_like(x) if x is not None else None,
                lr_for_weight_tree,
                is_leaf=lambda x: x is None,
            )

        step_weight = jnp.power(state.step_count.astype(jnp.float32) + 1.0, float(r))
        weight_tree = jax.tree.map(
            lambda w: w * step_weight if w is not None else None,
            base_weight_tree,
            is_leaf=lambda x: x is None,
        )

        in_c_warmup = state.step_count < int(c_warmup)
        new_weight_sum = jax.tree.map(
            lambda acc, w: (
                jnp.where(in_c_warmup, acc, acc + w)
                if (acc is not None and w is not None)
                else acc
            ),
            state.weight_sum,
            weight_tree,
            is_leaf=lambda x: x is None,
        )
        ck_tree = jax.tree.map(
            lambda w, sum_w: (
                jnp.where(
                    in_c_warmup,
                    1.0,
                    jnp.where(sum_w > 0.0, w / jnp.maximum(sum_w, 1e-30), 0.0),
                )
                if (w is not None and sum_w is not None)
                else None
            ),
            weight_tree,
            new_weight_sum,
            is_leaf=lambda x: x is None,
        )

        scaled_base_updates = jax.tree.map(
            lambda u: u.astype(jnp.float32) * polyak_lr if u is not None else None,
            base_updates,
            is_leaf=lambda x: x is None,
        )

        if adamc_weight_decay != 0.0:
            mask_tree = _weight_decay_mask_tree(params)

            def _adamc_decay_update(y, lr, mask):
                if y is None or lr is None:
                    return None
                decay = (
                    -jnp.square(lr) * float(adamc_weight_decay) * y.astype(jnp.float32)
                )
                if mask is None:
                    return jnp.zeros_like(decay)
                if isinstance(mask, bool):
                    return decay if mask else jnp.zeros_like(decay)
                return jnp.where(jnp.asarray(mask), decay, jnp.zeros_like(decay))

            adamc_decay_updates = jax.tree.map(
                _adamc_decay_update,
                params,
                effective_lr_tree,
                mask_tree,
                is_leaf=lambda x: x is None,
            )
        else:
            adamc_decay_updates = jax.tree.map(
                lambda x: None if x is None else jnp.zeros_like(x, dtype=jnp.float32),
                params,
                is_leaf=lambda x: x is None,
            )

        z_updates = jax.tree.map(
            lambda u, d: (
                None
                if u is None and d is None
                else ((0.0 if u is None else u) + (0.0 if d is None else d))
            ),
            scaled_base_updates,
            adamc_decay_updates,
            is_leaf=lambda x: x is None,
        )

        leaves, treedef = jax.tree.flatten(state.z, is_leaf=lambda x: x is None)
        subkeys = jax.random.split(sr_key, len(leaves))
        keys_tree = jax.tree.unflatten(treedef, list(subkeys))

        def _safe_z_update(z_old, u, k):
            if z_old is None or u is None:
                return z_old
            sum_f32 = z_old.astype(jnp.float32) + u.astype(jnp.float32)
            if getattr(z_old, "dtype", None) == jnp.bfloat16:
                return _stochastic_round_bf16(sum_f32, k)
            return sum_f32.astype(z_old.dtype)

        z_next = jax.tree.map(
            _safe_z_update,
            state.z,
            z_updates,
            keys_tree,
            is_leaf=lambda x: x is None,
        )

        def _sf_interpolate_y(y, z_old, z_new, ck):
            if y is None or z_old is None or z_new is None:
                return None
            y_f32 = y.astype(jnp.float32)
            z_old_f32 = z_old.astype(jnp.float32)
            z_new_f32 = z_new.astype(jnp.float32)
            if ck is None:
                return y_f32

            # y_t = b_prev*x_t + (1-b_prev)*z_t, but y_{t+1} may use an
            # annealed b_next.  beta_ratio = b_next / b_prev reduces the
            # update to the original formula when the two beta values match.
            term1 = beta_ratio * (1.0 - ck) * y_f32
            term2 = beta_ratio * (1.0 - ck) * (1.0 - b1_prev_safe) * z_old_f32
            term3 = (b1_next * ck + 1.0 - b1_next) * z_new_f32
            return term1 - term2 + term3

        y_next = jax.tree.map(
            _sf_interpolate_y,
            params,
            state.z,
            z_next,
            ck_tree,
            is_leaf=lambda x: x is None,
        )

        final_updates = jax.tree.map(
            lambda y_n, p: (
                (y_n - p.astype(jnp.float32)).astype(p.dtype)
                if y_n is not None and hasattr(p, "astype")
                else None
            ),
            y_next,
            params,
            is_leaf=lambda x: x is None,
        )

        new_state = ScheduleFreeState(
            b1=b1_next,
            weight_sum=new_weight_sum,
            step_count=cast(jax.Array, numerics.safe_increment(state.step_count)),
            base_state=new_base_state,
            z=z_next,
            key=next_state_key,
            lr_max=lr_max,
            scheduled_lr=effective_lr_tree,
            grad_l1_ema=grad_l1_ema,
            grad_l1_ema_corr=grad_l1_ema_corr,
            polyak_lr=polyak_lr,
            polyak_value=polyak_value,
            ip_term=ip_term,
        )

        return final_updates, new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def _resolve_sf_plus_bool(schedule_free_plus: bool, override: bool | None) -> bool:
    return schedule_free_plus if override is None else override


def schedule_free_prism(
    learning_rate: float,
    total_steps: int,
    # Schedule Config
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    weighting_mode: str | WeightingMode = WeightingMode.PRACTICAL,
    # Schedule-Free Config
    sf_b1: float = 0.90,
    state_dtype: jax.typing.DTypeLike | None = None,
    # PRISM Config
    prism_b1: float = 0.0,
    gamma: float = 1.0,
    ns_iters: int = 5,
    ns_coeffs: MuonNsCoeffs = MUON_NS_COEFFS,
    mode: str = "original",
    preconditioning: MuonPreconditioning = "frobenius",
    inv_steps: int = 6,
    inv_eps: float = 1e-5,
    inv_scale: float = 1.001,
    eps_gram: float = 1e-6,
    gamma_l: float | None = None,
    gamma_r: float | None = None,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    shape_nesterov: bool = True,
    weight_decay: float = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    grad_clip_max_amps: float | tuple[float, float] | None = (2.0, 10.0),
    raw_global_grad_clip: float | None = None,
    permissive_spike_protection: bool = True,
    mu_dtype: jax.typing.DTypeLike | None = None,
    axis_name: str | None = None,
    key: jax.Array | None = None,
    # Partitioning Arguments
    adam_learning_rate: float | None = None,
    adam_b1: float | None = None,
    adam_b2: float | None = None,
    adam_eps: float = 1e-8,
    prism_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    *,
    schedule_free_plus: bool = False,
    sf_r: float = 0.0,
    sf_c_warmup: int = 0,
    sf_weight_lr_power: float = 2.0,
    sf_use_lr_max: bool | None = None,
    sf_b1_anneal_steps: int = 0,
    sf_b1_max: float | None = 0.965,
    polyak: bool | None = None,
    use_adamc: bool | None = None,
    polyak_beta: float = 0.0,
    polyak_f_star: float = 0.0,
    polyak_axis_name: str | tuple[str, ...] | None = None,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free PRISM Optimizer with Partitioning, optionally using SF+."""
    polyak_enabled = _resolve_sf_plus_bool(schedule_free_plus, polyak)
    use_adamc_enabled = _resolve_sf_plus_bool(schedule_free_plus, use_adamc)
    use_lr_max_enabled = _resolve_sf_plus_bool(schedule_free_plus, sf_use_lr_max)
    adam_b1_value = (0.9 if schedule_free_plus else 0.0) if adam_b1 is None else adam_b1
    adam_b2_value = (
        (0.95 if schedule_free_plus else 0.999) if adam_b2 is None else adam_b2
    )
    inner_weight_decay = 0.0 if use_adamc_enabled else weight_decay
    inner_weight_decay_mask = None if use_adamc_enabled else weight_decay_mask

    prism_schedule, adam_schedule = _make_wsd_schedule_pair(
        learning_rate=learning_rate,
        adam_learning_rate=adam_learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )

    partition = _make_matrix_partition_fns(prism_weight_dimension_numbers, "prism")

    root_key = _fresh_prng_key(key)
    key_prism, key_adam = jax.random.split(root_key, 2)

    prism_components = [
        scale_by_prism(
            b1=prism_b1,
            gamma=gamma,
            ns_iters=ns_iters,
            ns_coeffs=ns_coeffs,
            mode=mode,
            preconditioning=preconditioning,
            inv_steps=inv_steps,
            inv_eps=inv_eps,
            inv_scale=inv_scale,
            eps_gram=eps_gram,
            gamma_l=gamma_l,
            gamma_r=gamma_r,
            precision=precision,
            nesterov=False,
            shape_nesterov=shape_nesterov,
            mu_dtype=mu_dtype,
            raw_global_grad_clip=raw_global_grad_clip,
            permissive_spike_protection=permissive_spike_protection,
            grad_clip_max_amps=grad_clip_max_amps,
            axis_name=axis_name,
            weight_dimension_numbers=partition.masked_specs,
            use_magma=False,
            weight_decay=0.0,
            weight_decay_mask=None,
            key=key_prism,
        ),
    ]

    _append_decayed_weights_and_lr(
        prism_components,
        weight_decay=inner_weight_decay,
        weight_decay_mask=inner_weight_decay_mask,
        learning_rate=prism_schedule,
    )

    base_opt = combine.partition(
        transforms={
            "prism": combine.chain(*prism_components),
            "adam": adamw(
                learning_rate=adam_schedule,
                b1=adam_b1_value,
                b2=adam_b2_value,
                eps=adam_eps,
                weight_decay=inner_weight_decay,
                weight_decay_mask=inner_weight_decay_mask,
                mu_dtype=mu_dtype,
                use_magma=False,
                axis_name=axis_name,
                key=key_adam,
            ),
        },
        param_labels=partition.labels,
    )

    return schedule_free(
        base_optimizer=base_opt,
        learning_rate=_make_dual_schedule_fn(
            partition,
            "prism",
            prism_schedule,
            adam_schedule,
        ),
        b1=sf_b1,
        weighting_mode=weighting_mode,
        state_dtype=state_dtype,
        key=root_key,
        r=sf_r,
        c_warmup=sf_c_warmup,
        weight_lr_power=sf_weight_lr_power,
        use_lr_max=use_lr_max_enabled,
        b1_anneal_steps=sf_b1_anneal_steps,
        b1_max=sf_b1_max,
        polyak=polyak_enabled,
        polyak_beta=polyak_beta,
        polyak_f_star=polyak_f_star,
        polyak_axis_name=polyak_axis_name,
        adamc_weight_decay=weight_decay if use_adamc_enabled else 0.0,
        adamc_weight_decay_mask=weight_decay_mask,
    )


def schedule_free_kron(
    learning_rate: float,
    total_steps: int,
    # Schedule Config
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    weighting_mode: str | WeightingMode = WeightingMode.PRACTICAL,
    # Schedule-Free Config
    sf_b1: float = 0.9,
    state_dtype: jax.typing.DTypeLike | None = None,
    # Standard Optimizer Args
    weight_decay: float = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    # PSGD Kron parameters
    preconditioner_update_probability: base.ScalarOrSchedule = (
        precond_update_prob_schedule()
    ),
    max_size_triangular: int = 8192,
    max_skew_triangular: float = 1.0,
    min_ndim_triangular: int = 2,
    memory_save_mode: str | None = None,
    update_preconditioner_first: bool = True,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: float | None = None,
    precond_dtype: str | jnp.dtype | None = None,
    precond_update_precision: str | None = "tensorfloat32",
    precond_grads_precision: str | None = None,
    scanned_layers: base.Params | None = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    preconditioner_mode: str | PreconditionerMode = PreconditionerMode.Q0P5EQ1P5,
    beta_lipschitz: float = 0.9,
    track_lipschitz: bool = True,
    damping: float = 1e-9,
    grad_clip_max_amps: float | tuple[float, float] = (2.0, 10.0),
    grad_clip_mode: str | GradClipMode = GradClipMode.PER_TENSOR_RMS,
    raw_global_grad_clip: float | None = None,
    permissive_spike_protection: bool = True,
    newton_schulz_iters: int = 5,
    axis_name: str | None = None,
    key: jax.Array | None = None,
    *,
    schedule_free_plus: bool = False,
    sf_r: float = 0.0,
    sf_c_warmup: int = 0,
    sf_weight_lr_power: float = 2.0,
    sf_use_lr_max: bool | None = None,
    sf_b1_anneal_steps: int = 0,
    sf_b1_max: float | None = 0.965,
    polyak: bool | None = None,
    use_adamc: bool | None = None,
    polyak_beta: float = 0.0,
    polyak_f_star: float = 0.0,
    polyak_axis_name: str | tuple[str, ...] | None = None,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free PSGD Kron optimizer, optionally using SF+.

    Uses the shared `rollfast.schedules.schedulefree` wrapper.

    Args:
        learning_rate: Peak learning rate.
        total_steps: Total training steps (required for WSD schedule generation).
        warmup_fraction: Fraction of steps for warmup.
        decay_fraction: Fraction of steps for decay.
        weighting_mode: The weighting strategy (Practical, Theoretical, Schedulet).
        sf_b1: Schedule-free interpolation parameter (replaces momentum).
        state_dtype: Dtype for schedule-free z-sequence.
        weight_decay: Weight decay applied to the optimizer.
        weight_decay_mask: Mask for weight decay.
        preconditioner_update_probability: Probability (or schedule) of updating
            the preconditioner matrix Q at each step.
        max_size_triangular: Max size for a dimension to be considered for
            dense/triangular preconditioning. Larger dims become diagonal.
        max_skew_triangular: Max aspect ratio skew for dense factors.
        min_ndim_triangular: Minimum tensor rank required for dense preconditioning.
        memory_save_mode: Strategy to force diagonal approximations to save RAM.
            Values: [None, 'one_diag', 'all_diag'].
        update_preconditioner_first: Update Q before applying it to the gradient.
        preconditioner_lr: Learning rate for the preconditioner matrix Q.
        preconditioner_init_scale: Initial scale for Q. If None, computed on-the-fly.
        precond_dtype: Dtype for preconditioner storage (e.g. float32, bfloat16).
        precond_update_precision: JAX precision for Q update matmuls.
        precond_grads_precision: JAX precision for gradient application matmuls.
        scanned_layers: PyTree mask indicating layers that are vmapped/scanned.
        lax_map_scanned_layers: Use lax.map for scanning (saves memory vs vmap).
        lax_map_batch_size: Batch size for lax.map.
        preconditioner_mode: Update rule for Q. See PreconditionerMode enum.
        beta_lipschitz: EMA factor for Lipschitz constant estimation.
        track_lipschitz: Enable adaptive step size for Q based on Lipschitz.
        damping: Numerical damping for stability.
        grad_clip_max_amps: (max_rms, max_val) for gradient clipping.
        grad_clip_mode: Strategy for clipping ('per_tensor_rms' or 'global_rms').
        raw_global_grad_clip: Threshold for global gradient norm clipping (spike protection).
        permissive_spike_protection: If True, allows updates during spikes if prob=1.0.
        newton_schulz_iters: Iterations for NS mode (default 5).
        axis_name: Axis name for distributed (SPMD) reduction.
        key: PRNG key for stochastic elements.

    Returns:
        A Schedule-Free gradient transformation.

    References:
        Defazio, A., Yang, X. A., Mehta, H., Mishchenko, K., Khaled, A., & Cutkosky, A. (2024).
        The Road Less Scheduled.
        arXiv preprint arXiv:2405.15682.

        Pun, Y.-M., Buchholz, M., & Gower, R. M. (2025).
        Schedulers for Schedule-free: Theoretically inspired hyperparameters.
        arXiv preprint arXiv:2511.07767.
    """
    polyak_enabled = _resolve_sf_plus_bool(schedule_free_plus, polyak)
    use_adamc_enabled = _resolve_sf_plus_bool(schedule_free_plus, use_adamc)
    use_lr_max_enabled = _resolve_sf_plus_bool(schedule_free_plus, sf_use_lr_max)
    inner_weight_decay = 0.0 if use_adamc_enabled else weight_decay
    inner_weight_decay_mask = None if use_adamc_enabled else weight_decay_mask

    lr_schedule = wsd_schedule(
        peak_lr=learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )

    base_opt_components: list[Any] = [
        scale_by_kron(
            b1=0.0,
            whiten_grad=True,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            max_skew_triangular=max_skew_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            update_preconditioner_first=update_preconditioner_first,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            mu_dtype=None,
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
            use_magma=False,
            weight_decay=0.0,
            weight_decay_mask=None,
            axis_name=axis_name,
            key=key,
        )
    ]

    _append_decayed_weights_and_lr(
        base_opt_components,
        weight_decay=inner_weight_decay,
        weight_decay_mask=inner_weight_decay_mask,
        learning_rate=lr_schedule,
    )
    base_optimizer = combine.chain(*base_opt_components)

    return schedule_free(
        base_optimizer=base_optimizer,
        learning_rate=lr_schedule,
        b1=sf_b1,
        weighting_mode=weighting_mode,
        state_dtype=state_dtype,
        key=key,
        r=sf_r,
        c_warmup=sf_c_warmup,
        weight_lr_power=sf_weight_lr_power,
        use_lr_max=use_lr_max_enabled,
        b1_anneal_steps=sf_b1_anneal_steps,
        b1_max=sf_b1_max,
        polyak=polyak_enabled,
        polyak_beta=polyak_beta,
        polyak_f_star=polyak_f_star,
        polyak_axis_name=polyak_axis_name,
        adamc_weight_decay=weight_decay if use_adamc_enabled else 0.0,
        adamc_weight_decay_mask=weight_decay_mask,
    )


def schedule_free_adam(
    learning_rate: float,
    total_steps: int,
    # Schedule Config
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    weighting_mode: str | WeightingMode = WeightingMode.PRACTICAL,
    # Schedule-Free Config
    sf_b1: float = 0.9,
    state_dtype: jax.typing.DTypeLike | None = None,
    # Adam Config
    b1: float | None = None,
    b2: float | None = None,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    mu_dtype: jax.typing.DTypeLike | None = None,
    axis_name: str | None = None,
    key: jax.Array | None = None,
    *,
    schedule_free_plus: bool = False,
    sf_r: float = 0.0,
    sf_c_warmup: int = 0,
    sf_weight_lr_power: float = 2.0,
    sf_use_lr_max: bool | None = None,
    sf_b1_anneal_steps: int = 0,
    sf_b1_max: float | None = 0.965,
    polyak: bool | None = None,
    use_adamc: bool | None = None,
    polyak_beta: float = 0.0,
    polyak_f_star: float = 0.0,
    polyak_axis_name: str | tuple[str, ...] | None = None,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free Adam optimizer, optionally using ScheduleFree+.

    ``schedule_free_plus=True`` keeps this as the existing Adam wrapper but
    enables the SF+ components from the paper/reference implementation:
    inner Adam momentum by default, Polyak step-size scaling, beta annealing,
    optional c-warmup/r-weighting, running-lr-max weighting, and AdamC
    fully-decoupled weight decay.  With Polyak enabled, pass the current scalar
    objective value to ``update`` as ``function_value=...``, ``loss=...``, or
    ``value=...``.

    Legacy defaults are preserved when ``schedule_free_plus=False``: inner Adam
    ``b1`` defaults to ``0.0`` and ``b2`` defaults to ``0.999``.  With
    ``schedule_free_plus=True``, omitted ``b1``/``b2`` default to ``0.9`` and
    ``0.95`` respectively, matching the PyTorch SF+ reference defaults.
    """
    polyak_enabled = _resolve_sf_plus_bool(schedule_free_plus, polyak)
    use_adamc_enabled = _resolve_sf_plus_bool(schedule_free_plus, use_adamc)
    use_lr_max_enabled = _resolve_sf_plus_bool(schedule_free_plus, sf_use_lr_max)
    adam_b1 = (0.9 if schedule_free_plus else 0.0) if b1 is None else b1
    adam_b2 = (0.95 if schedule_free_plus else 0.999) if b2 is None else b2
    inner_weight_decay = 0.0 if use_adamc_enabled else weight_decay
    inner_weight_decay_mask = None if use_adamc_enabled else weight_decay_mask

    lr_schedule = wsd_schedule(
        peak_lr=learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )

    base_optimizer = adamw(
        learning_rate=lr_schedule,
        b1=adam_b1,
        b2=adam_b2,
        eps=eps,
        weight_decay=inner_weight_decay,
        weight_decay_mask=inner_weight_decay_mask,
        mu_dtype=mu_dtype,
        use_magma=False,
        axis_name=axis_name,
        key=key,
    )

    return schedule_free(
        base_optimizer=base_optimizer,
        learning_rate=lr_schedule,
        b1=sf_b1,
        weighting_mode=weighting_mode,
        state_dtype=state_dtype,
        key=key,
        r=sf_r,
        c_warmup=sf_c_warmup,
        weight_lr_power=sf_weight_lr_power,
        use_lr_max=use_lr_max_enabled,
        b1_anneal_steps=sf_b1_anneal_steps,
        b1_max=sf_b1_max,
        polyak=polyak_enabled,
        polyak_beta=polyak_beta,
        polyak_f_star=polyak_f_star,
        polyak_axis_name=polyak_axis_name,
        lr_max_init=eps,
        adamc_weight_decay=weight_decay if use_adamc_enabled else 0.0,
        adamc_weight_decay_mask=weight_decay_mask,
    )


def schedule_free_aurora(
    learning_rate: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    weighting_mode: str | Any = "practical",
    sf_b1: float = 0.90,
    state_dtype: jax.typing.DTypeLike | None = None,
    aurora_b1: float = 0.0,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    riemannian: bool = False,
    outer_steps: int = 3,
    cg_steps: int = 20,
    riemannian_eta: float = 0.1,
    retraction_steps: int = 2,
    polar_ns_iters: int = 12,
    polar_compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    polar_output_dtype: jax.typing.DTypeLike = jnp.float32,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    eps: float = 1e-7,
    shape_nesterov: bool = True,
    weight_decay: float = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    grad_clip_max_amps: float | tuple[float, float] | None = (2.0, 10.0),
    raw_global_grad_clip: float | None = None,
    permissive_spike_protection: bool = True,
    mu_dtype: jax.typing.DTypeLike | None = None,
    axis_name: str | None = None,
    guard_nonfinite: bool = True,
    key: jax.Array | None = None,
    adam_learning_rate: float | None = None,
    adam_b1: float | None = None,
    adam_b2: float | None = None,
    adam_eps: float = 1e-8,
    aurora_weight_dimension_numbers: AuroraWeightDimNumOrFn | None = None,
    *,
    schedule_free_plus: bool = False,
    sf_r: float = 0.0,
    sf_c_warmup: int = 0,
    sf_weight_lr_power: float = 2.0,
    sf_use_lr_max: bool | None = None,
    sf_b1_anneal_steps: int = 0,
    sf_b1_max: float | None = 0.965,
    polyak: bool | None = None,
    use_adamc: bool | None = None,
    polyak_beta: float = 0.0,
    polyak_f_star: float = 0.0,
    polyak_axis_name: str | tuple[str, ...] | None = None,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free Aurora with Adam fallback for non-matrix leaves, optionally using SF+."""
    polyak_enabled = _resolve_sf_plus_bool(schedule_free_plus, polyak)
    use_adamc_enabled = _resolve_sf_plus_bool(schedule_free_plus, use_adamc)
    use_lr_max_enabled = _resolve_sf_plus_bool(schedule_free_plus, sf_use_lr_max)
    adam_b1_value = (0.9 if schedule_free_plus else 0.0) if adam_b1 is None else adam_b1
    adam_b2_value = (
        (0.95 if schedule_free_plus else 0.999) if adam_b2 is None else adam_b2
    )
    inner_weight_decay = 0.0 if use_adamc_enabled else weight_decay
    inner_weight_decay_mask = None if use_adamc_enabled else weight_decay_mask

    aurora_schedule, adam_schedule = _make_wsd_schedule_pair(
        learning_rate=learning_rate,
        adam_learning_rate=adam_learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )

    root_key = _fresh_prng_key(key)
    key_aurora, key_adam = jax.random.split(root_key, 2)

    partition = _make_matrix_partition_fns(
        aurora_weight_dimension_numbers,
        "aurora",
    )

    if riemannian:
        scale = scale_by_riemannian_aurora(
            b1=aurora_b1,
            outer_steps=outer_steps,
            cg_steps=cg_steps,
            riemannian_eta=riemannian_eta,
            retraction_steps=retraction_steps,
            polar_ns_iters=polar_ns_iters,
            polar_compute_dtype=polar_compute_dtype,
            polar_output_dtype=polar_output_dtype,
            precision=precision,
            eps=eps,
            nesterov=False,
            shape_nesterov=shape_nesterov,
            mu_dtype=mu_dtype,
            raw_global_grad_clip=raw_global_grad_clip,
            permissive_spike_protection=permissive_spike_protection,
            grad_clip_max_amps=grad_clip_max_amps,
            weight_dimension_numbers=partition.masked_specs,
            use_magma=False,
            weight_decay=0.0,
            weight_decay_mask=None,
            axis_name=axis_name,
            guard_nonfinite=guard_nonfinite,
            key=key_aurora,
        )
    else:
        scale = scale_by_aurora(
            b1=aurora_b1,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
            polar_ns_iters=polar_ns_iters,
            polar_compute_dtype=polar_compute_dtype,
            polar_output_dtype=polar_output_dtype,
            precision=precision,
            eps=eps,
            nesterov=False,
            shape_nesterov=shape_nesterov,
            mu_dtype=mu_dtype,
            raw_global_grad_clip=raw_global_grad_clip,
            permissive_spike_protection=permissive_spike_protection,
            grad_clip_max_amps=grad_clip_max_amps,
            weight_dimension_numbers=partition.masked_specs,
            use_magma=False,
            weight_decay=0.0,
            weight_decay_mask=None,
            axis_name=axis_name,
            guard_nonfinite=guard_nonfinite,
            key=key_aurora,
        )

    aurora_components = [scale]
    _append_decayed_weights_and_lr(
        aurora_components,
        weight_decay=inner_weight_decay,
        weight_decay_mask=inner_weight_decay_mask,
        learning_rate=aurora_schedule,
    )

    base_opt = combine.partition(
        transforms={
            "aurora": combine.chain(*aurora_components),
            "adam": adamw(
                learning_rate=adam_schedule,
                b1=adam_b1_value,
                b2=adam_b2_value,
                eps=adam_eps,
                weight_decay=inner_weight_decay,
                weight_decay_mask=inner_weight_decay_mask,
                mu_dtype=mu_dtype,
                use_magma=False,
                axis_name=axis_name,
                key=key_adam,
            ),
        },
        param_labels=partition.labels,
    )

    return schedule_free(
        base_optimizer=base_opt,
        learning_rate=_make_dual_schedule_fn(
            partition,
            "aurora",
            aurora_schedule,
            adam_schedule,
        ),
        b1=sf_b1,
        weighting_mode=weighting_mode,
        state_dtype=state_dtype,
        key=root_key,
        r=sf_r,
        c_warmup=sf_c_warmup,
        weight_lr_power=sf_weight_lr_power,
        use_lr_max=use_lr_max_enabled,
        b1_anneal_steps=sf_b1_anneal_steps,
        b1_max=sf_b1_max,
        polyak=polyak_enabled,
        polyak_beta=polyak_beta,
        polyak_f_star=polyak_f_star,
        polyak_axis_name=polyak_axis_name,
        adamc_weight_decay=weight_decay if use_adamc_enabled else 0.0,
        adamc_weight_decay_mask=weight_decay_mask,
    )


def schedule_free_eval_params(state: base.OptState, params: base.Params):
    """Params for evaluation from Rollfast's Schedule-Free state.

    Args:
        state: The optimizer state (must be a ScheduleFreeState).
        params: The current parameters (the 'y' sequence).

    Returns:
        The parameters to use for evaluation (the 'x' sequence).
    """
    b1 = getattr(state, "b1", None)
    z = getattr(state, "z", None)
    if b1 is None or z is None:
        raise ValueError(
            "schedule_free_eval_params requires a ScheduleFreeState as input."
        )
    b1_safe = jnp.maximum(b1, 1e-8)
    return jax.tree.map(
        lambda yi, zi: (yi - (1.0 - b1_safe) * zi) / b1_safe if zi is not None else yi,
        params,
        z,
        is_leaf=lambda x: x is None,
    )


__all__ = [
    "ScheduleFreeLearningRate",
    "ScheduleFreeState",
    "WeightingMode",
    "schedule_free",
    "schedule_free_adam",
    "schedule_free_aurora",
    "schedule_free_eval_params",
    "schedule_free_kron",
    "schedule_free_prism",
]
