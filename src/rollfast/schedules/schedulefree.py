from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from optax._src import alias, base, combine, numerics, transform

from rollfast.optim.prism import (
    WeightDimNumOrFn,
    _get_dimension_numbers,
    _is_prism_leaf,
    _mask_dimension_numbers,
    scale_by_prism,
)
from rollfast.optim.psgd import (
    GradClipMode,
    PreconditionerMode,
    precond_update_prob_schedule,
    scale_by_kron,
)
from rollfast.schedules.wsd import wsd_schedule


class WeightingMode(str, Enum):
    """
    Determines how the iterate averaging parameter c_t is computed.

    Ref: 2511.07767v1 "Schedulers for Schedule-Free"
    """

    THEORETICAL = "theoretical"  # c_t = 1/t (w_t = 1)
    PRACTICAL = "practical"  # c_t = gamma_t^2 / sum(gamma^2) (w_t = gamma_t^2)
    SCHEDULET = "schedulet"  # c_t = gamma_t / sum(gamma) (w_t = gamma_t)


class ScheduleFreeState(NamedTuple):
    """State for the Schedule-Free wrapper."""

    b1: jax.Array
    weight_sum: base.Params
    step_count: jax.Array
    base_state: base.OptState
    z: base.Params


def schedule_free(
    base_optimizer: base.GradientTransformation,
    learning_rate: Union[base.ScalarOrSchedule, Callable[[int], base.Params]],
    b1: float = 0.9,
    weighting_mode: Union[str, WeightingMode] = WeightingMode.SCHEDULET,
    state_dtype: Optional[jax.typing.DTypeLike] = None,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free Wrapper supporting Schedulet, Practical, and Theoretical modes.

    Implements the Schedule-Free optimization wrapper. It maintains a primary
    sequence `z` (updated by the base optimizer) and an averaged sequence `x`
    (the parameters used for evaluation).

    Args:
        base_optimizer: The inner optimizer (e.g., PRISM). Must return the full
            update step (including LR scaling).
        learning_rate: The learning rate schedule function or scalar. Used to
            compute the weighting `c_t`.
        b1: The interpolation parameter beta (distinct from optimizer momentum).
            Controls the interpolation between `x` and `z`.
        weighting_mode: Strategy for averaging weights.
        state_dtype: Dtype for the z-sequence storage.

    Returns:
        A `GradientTransformationExtraArgs` wrapper.
    """
    if isinstance(weighting_mode, str):
        weighting_mode = WeightingMode(weighting_mode)

    def init_fn(params):
        dtype = (
            state_dtype
            if state_dtype is not None
            else optax.tree.dtype(params, "lowest")
        )
        z = optax.tree.cast(params, dtype=dtype)
        # Deep copy z to ensure distinct buffer
        z = jax.tree.map(lambda t: jnp.array(t, copy=True), z)

        # Initialize weight_sum as a PyTree of zeros matching params
        # This is O(N) memory but required for exact mathematical correctness
        # with partitioned schedules.
        # weight_sum = jax.tree.map(lambda x: jnp.zeros([], dtype=jnp.float32), params)
        weight_sum = jax.tree.map(
            lambda x: jnp.zeros([], dtype=jnp.float32) if x is not None else None,
            params,
        )

        return ScheduleFreeState(
            b1=jnp.array(b1, dtype=jnp.float32),
            weight_sum=weight_sum,
            # weight_sum=jnp.zeros([], dtype=jnp.float32),
            step_count=jnp.zeros([], dtype=jnp.int32),
            base_state=base_optimizer.init(params),
            z=z,
        )

    def update_fn(updates, state, params=None, **extra_args):
        if callable(learning_rate):
            try:
                lr_tree = learning_rate(state.step_count, params)
            except TypeError:
                # Fallback: The schedule does not accept params (standard optax schedule)
                lr_tree = learning_rate(state.step_count)
        else:
            lr_tree = learning_rate

        lr_tree = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), lr_tree)

        # Compute Base Optimizer Update (Preconditioned Gradient)
        # Note: base_optimizer should output the update step d = -lr * P * g
        # We must handle the LR scaling carefully. Standard SF assumes base_opt
        # returns (D^-1 g).
        base_updates, new_base_state = base_optimizer.update(
            updates, state.base_state, params, **extra_args
        )

        if weighting_mode == WeightingMode.SCHEDULET:
            weight_tree = lr_tree
        elif weighting_mode == WeightingMode.PRACTICAL:
            weight_tree = jax.tree.map(
                lambda x: jnp.square(x) if x is not None else None, lr_tree
            )
        else:  # THEORETICAL
            weight_tree = jax.tree.map(
                lambda x: jnp.ones_like(x) if x is not None else None, lr_tree
            )

        # Accumulate weights: W_t = W_{t-1} + w_t
        # new_weight_sum = jax.tree.map(
        #     lambda acc, w: acc + w, state.weight_sum, weight_tree
        # )
        new_weight_sum = jax.tree.map(
            lambda acc, w: (acc + w) if (acc is not None and w is not None) else None,
            state.weight_sum,
            weight_tree,
        )

        # c_t = w_t / W_t
        # Safety: avoid division by zero
        ck_tree = jax.tree.map(
            lambda w, sum_w: (
                jnp.where(sum_w > 0, w / sum_w, 0.0)
                if (w is not None and sum_w is not None)
                else None
            ),
            weight_tree,
            new_weight_sum,
        )

        # Protect against b1 -> 0
        b1_safe = jnp.maximum(state.b1, 1e-8)

        # Schedule-Free Update Dynamics
        # y_t = params (input)
        # z_t = z_{t-1} - gamma * (Base Update)
        # Note: 'base_updates' from chain usually includes LR scaling.
        # If base_optimizer includes scale_by_learning_rate, base_updates is actual step.
        z_next = jax.tree.map(lambda z, u: z + u, state.z, base_updates)

        # We recover x_t from y_t: x_t = (y_t - (1-b1)z_{t-1}) / b1
        x_curr = jax.tree.map(
            lambda y, z: (y - (1.0 - b1_safe) * z) / b1_safe, params, state.z
        )

        # x_{t+1} = (1 - c_{t+1}) x_t + c_{t+1} z_{t+1}
        x_next = jax.tree.map(
            lambda x, z, ck: (
                (1.0 - ck) * x + ck * z if ck is not None else x
            ),  # Fallback to x (param) if ck is None
            x_curr,
            z_next,
            ck_tree,
        )

        # y_{t+1} = (1-b1) z_{t+1} + b1 x_{t+1}
        y_next = jax.tree.map(
            lambda x, z: b1_safe * x + (1.0 - b1_safe) * z, x_next, z_next
        )

        # Final Update diff
        final_updates = jax.tree.map(lambda n, o: n - o, y_next, params)

        new_state = ScheduleFreeState(
            b1=state.b1,
            weight_sum=new_weight_sum,
            step_count=numerics.safe_increment(state.step_count),
            base_state=new_base_state,
            z=z_next,
        )

        return final_updates, new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def schedule_free_prism(
    learning_rate: float,
    total_steps: int,
    # Schedule Config
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    weighting_mode: Union[str, WeightingMode] = WeightingMode.PRACTICAL,
    # Schedule-Free Config
    sf_b1: float = 0.90,
    state_dtype: Optional[jax.typing.DTypeLike] = None,
    # PRISM Config
    prism_b1: float = 0.0,
    gamma: float = 1.0,
    ns_iters: int = 5,
    shape_nesterov: bool = True,
    weight_decay: float = 0.0,
    grad_clip_max_amps: Optional[Union[float, Tuple[float, float]]] = (2.0, 10.0),
    raw_global_grad_clip: Optional[float] = None,
    permissive_spike_protection: bool = True,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    axis_name: Optional[str] = None,
    # Partitioning Arguments
    adam_learning_rate: Optional[float] = None,
    adam_b1: float = 0.0,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    prism_weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free PRISM Optimizer with Partitioning.

    Combines PRISM's spectral shaping and partitioning (matrices vs vectors)
    with the Schedule-Free optimization wrapper.

    Args:
        learning_rate: Peak learning rate for the PRISM branch.
        total_steps: Total training steps (required for WSD schedule).
        warmup_fraction: Fraction of steps for warmup.
        decay_fraction: Fraction of steps for decay.
        weighting_mode: Schedule-free weighting mode.
        sf_b1: Schedule-free interpolation parameter (distinct from momentum).
        state_dtype: Dtype for schedule-free z-sequence.
        prism_b1: Momentum coefficient for PRISM.
        gamma: Innovation damping coefficient for PRISM.
        ns_iters: Newton-Schulz iterations.
        shape_nesterov: Whether to shape Nesterov momentum.
        weight_decay: Weight decay.
        grad_clip_max_amps: Post-shaping clipping.
        raw_global_grad_clip: Pre-shaping global clipping.
        permissive_spike_protection: Clip vs Skip on spikes.
        mu_dtype: Momentum dtype.
        axis_name: Distributed axis name.
        adam_learning_rate: Peak learning rate for Adam branch. Defaults to `learning_rate`.
        adam_b1: Adam Beta1.
        adam_b2: Adam Beta2.
        adam_eps: Adam Epsilon.
        prism_weight_dimension_numbers: Spec for PRISM parameters.

    Returns:
        A Schedule-Free gradient transformation with partitioned inner optimizer.
    """
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    # We need separate schedules if the peak LRs differ, as the schedule outputs
    # the exact value to apply.
    prism_schedule = wsd_schedule(
        peak_lr=learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )

    if adam_learning_rate == learning_rate:
        adam_schedule = prism_schedule
    else:
        adam_schedule = wsd_schedule(
            peak_lr=adam_learning_rate,
            total_steps=total_steps,
            warmup_fraction=warmup_fraction,
            decay_fraction=decay_fraction,
        )

    def get_resolved_dim_nums(params):
        return _get_dimension_numbers(prism_weight_dimension_numbers, params)

    def param_labels(params):
        dim_nums = get_resolved_dim_nums(params)
        return jax.tree.map(
            lambda d, p: None if p is None else ("prism" if d is not None else "adam"),
            dim_nums,
            params,
            is_leaf=_is_prism_leaf,
        )
        # return _make_param_labels(dim_nums)

    def prism_weight_dim_nums_fn(params):
        dim_nums = get_resolved_dim_nums(params)
        return _mask_dimension_numbers(dim_nums)

    # Note: We must apply the schedule INSIDE the base optimizer branches
    # so that the updates passed to `schedule_free` are fully scaled.
    base_opt = combine.partition(
        transforms={
            "prism": combine.chain(
                scale_by_prism(
                    b1=prism_b1,
                    gamma=gamma,
                    ns_iters=ns_iters,
                    nesterov=False,
                    shape_nesterov=shape_nesterov,
                    # bias_correction=False,  # SF handles bias correction implicitly
                    mu_dtype=mu_dtype,
                    raw_global_grad_clip=raw_global_grad_clip,
                    permissive_spike_protection=permissive_spike_protection,
                    grad_clip_max_amps=grad_clip_max_amps,
                    axis_name=axis_name,
                    weight_dimension_numbers=prism_weight_dim_nums_fn,
                ),
                transform.add_decayed_weights(weight_decay),
                transform.scale_by_learning_rate(prism_schedule),
            ),
            "adam": alias.adamw(
                learning_rate=adam_schedule,
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
                weight_decay=weight_decay,
                mu_dtype=mu_dtype,
            ),
        },
        param_labels=param_labels,
    )

    # This function introspects params to decide which LR schedule to apply
    # for calculation of the schedule-free weighting c_t.
    def dual_schedule_fn(count, params):
        labels = param_labels(params)
        p_lr = prism_schedule(count)
        a_lr = adam_schedule(count)

        # return jax.tree.map(lambda l: p_lr if l == "prism" else a_lr, labels)
        return jax.tree.map(
            lambda l: None if l is None else (p_lr if l == "prism" else a_lr), labels
        )

    # We pass the prism_schedule to the wrapper solely for calculating `c_t`.
    # Since c_t depends on the *relative* progress of the schedule (and weighting mode),
    # using the prism schedule is sufficient even if Adam LR differs, provided the
    # shape (warmup/decay ratios) is the same.
    return schedule_free(
        base_optimizer=base_opt,
        learning_rate=dual_schedule_fn,
        b1=sf_b1,
        weighting_mode=weighting_mode,
        state_dtype=state_dtype,
    )


def schedule_free_kron(
    learning_rate: float,
    total_steps: int,
    # Schedule Config
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    weighting_mode: Union[str, WeightingMode] = WeightingMode.PRACTICAL,
    # Schedule-Free Config
    sf_b1: float = 0.9,
    state_dtype: Optional[jax.typing.DTypeLike] = None,
    # Standard Optimizer Args
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    # PSGD Kron parameters
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    max_skew_triangular: float = 1.0,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    update_preconditioner_first: bool = True,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: Optional[float] = None,
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
    axis_name: Optional[str] = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free PSGD Kron optimizer.

    Refactored to use the authoritative `rollfast.optim.schedulefree` wrapper.

    Args:
        learning_rate: Peak learning rate.
        total_steps: Total training steps (required for WSD schedule generation).
        weighting_mode: The weighting strategy (Practical, Theoretical, Schedulet).
        sf_b1: Schedule-free interpolation parameter (replaces momentum).
        preconditioner_update_probability: Probability (or schedule) of updating
            the preconditioner matrix Q at each step.
        max_size_triangular: Max size for a dimension to be considered for
            dense/triangular preconditioning. Larger dims become diagonal.
        max_skew_triangular: Max aspect ratio skew for dense factors.
        min_ndim_triangular: Minimum tensor rank required for dense preconditioning.
        memory_save_mode: Strategy to force diagonal approximations to save RAM.
            Values: [None, 'one_diag', 'all_diag'].
        preconditioner_lr: Learning rate for the preconditioner matrix Q.
        preconditioner_init_scale: Initial scale for Q. If None, computed on-the-fly.
        update_preconditioner_first: Update Q before applying it to the gradient.
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
    """
    lr_schedule = wsd_schedule(
        peak_lr=learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )

    # CRITICAL: b1 must be 0.0 because Schedule-Free handles momentum via the z-sequence.
    # CRITICAL: whiten_grad must be True because there is no internal momentum buffer to whiten.
    base_opt_components = [
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
            mu_dtype=None,  # No momentum buffer needed
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
            axis_name=axis_name,
            key=key,
        )
    ]

    if weight_decay > 0.0:
        base_opt_components.append(
            transform.add_decayed_weights(weight_decay, weight_decay_mask)
        )

    base_opt_components.append(transform.scale_by_learning_rate(lr_schedule))
    base_optimizer = combine.chain(*base_opt_components)

    return schedule_free(
        base_optimizer=base_optimizer,
        learning_rate=lr_schedule,
        b1=sf_b1,
        weighting_mode=weighting_mode,
        state_dtype=state_dtype,
    )


def schedule_free_eval_params(state: base.OptState, params: base.Params):
    """Params for evaluation of :func:`optax.contrib.schedule_free`.

    Args:
        state: The optimizer state (must be a ScheduleFreeState).
        params: The current parameters (the 'y' sequence).

    Returns:
        The parameters to use for evaluation (the 'x' sequence).
    """
    # Using ScheduleFreeState as a type hint above results in pytype errors in tests.
    b1 = getattr(state, "b1")
    z = getattr(state, "z")
    if b1 is None or z is None:
        raise ValueError(
            "schedule_free_eval_params requires a ScheduleFreeState as input."
        )
    b1_safe = jnp.maximum(b1, 1e-8)
    return jax.tree.map(lambda yi, zi: (yi - (1.0 - b1_safe) * zi) / b1_safe, params, z)
