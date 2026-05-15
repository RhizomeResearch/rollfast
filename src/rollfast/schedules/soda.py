from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax._src import base

from rollfast.optim.adam import adamw
from rollfast.optim.orthogonalization import MUON_NS_COEFFS
from rollfast.optim.prism import WeightDimNumOrFn, prism
from rollfast.optim.psgd import (
    GradClipMode,
    PreconditionerMode,
    precond_update_prob_schedule,
    kron,
)
from rollfast.optim.rmnp import rmnp
from rollfast.schedules.wsd import wsd_schedule

"""SODA usage notes.

SODA is a wrapper for base optimizers that already return full additive Optax
parameter deltas, including learning-rate scaling. It adds the practical SODA
anchor term ``(z0 - params) / (k + 2)``, where ``z0`` is the initialization. This
anchor is toward initialization, not toward zero, and it is not multiplied by
the learning rate.

The convenience wrappers disable ordinary base optimizer weight decay because
SODA is intended to replace tuned weight decay by a parameter-free anchor term.
Stacking SODA with ordinary weight decay is possible through the generic
``soda`` wrapper, but it changes the intended algorithm and should be treated as
an experiment.

SODA stores one full copy of the initialization. This is similar in scale to an
EMA or schedule-free auxiliary parameter sequence. The update requires current
``params`` so the anchor can be computed. Be deliberate when composing SODA with
other terminal geometry transforms such as Hyperball, and avoid wrapping
non-additive optimizers whose invariants would be broken by an additive anchor
term, such as Pion's spectrum-preserving step.
"""


class SodaState(NamedTuple):
    """State for the practical SODA wrapper."""

    count: jax.Array
    base_state: base.OptState
    z0: base.Params


def _call_base_update(base_optimizer, updates, state, params, extra_args):
    try:
        return base_optimizer.update(updates, state, params, **extra_args)
    except TypeError:
        return base_optimizer.update(updates, state, params)


def soda(
    base_optimizer: base.GradientTransformation,
    state_dtype: jax.typing.DTypeLike | None = None,
) -> base.GradientTransformationExtraArgs:
    r"""Practical SODA wrapper for an existing base optimizer.

    This implements Algorithm 1 from "Optimistic Dual Averaging Unifies Modern
    Optimizers" in its expanded Optax form:

    ``x_{k+1} - x_k = BaseUpdate(g_k) + (z0 - x_k) / (k + 2)``.

    The base optimizer should include its learning-rate schedule and should not
    include weight decay. SODA replaces tuned weight decay with a parameter-free
    initialization-centered anchor term.
    """

    def init_fn(params):
        dtype = (
            state_dtype
            if state_dtype is not None
            else optax.tree.dtype(params, "lowest")
        )
        z0 = jax.tree.map(
            lambda x: jnp.array(x, dtype=dtype, copy=True) if x is not None else None,
            params,
            is_leaf=lambda x: x is None,
        )
        return SodaState(
            count=jnp.zeros([], dtype=jnp.int32),
            base_state=base_optimizer.init(params),
            z0=z0,
        )

    def update_fn(updates, state, params=None, **extra_args):
        if params is None:
            raise ValueError("`params` must be provided to `soda`.")

        base_updates, new_base_state = _call_base_update(
            base_optimizer, updates, state.base_state, params, extra_args
        )
        denom = state.count.astype(jnp.float32) + 2.0

        anchor_updates = jax.tree.map(
            lambda p0, p: (
                (p0.astype(jnp.float32) - p.astype(jnp.float32)) / denom
                if p0 is not None and p is not None
                else None
            ),
            state.z0,
            params,
            is_leaf=lambda x: x is None,
        )

        final_updates = jax.tree.map(
            lambda u, a: (
                u if a is None else (a if u is None else u + a.astype(u.dtype))
            ),
            base_updates,
            anchor_updates,
            is_leaf=lambda x: x is None,
        )

        return final_updates, SodaState(
            count=state.count + jnp.array(1, dtype=jnp.int32),
            base_state=new_base_state,
            z0=state.z0,
        )

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def soda_adam(
    learning_rate: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    state_dtype: jax.typing.DTypeLike | None = None,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: jax.typing.DTypeLike | None = None,
    nesterov: bool = False,
    axis_name: str | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """Adam base optimizer wrapped with SODA."""
    lr_schedule = wsd_schedule(
        peak_lr=learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )
    base_optimizer = adamw(
        learning_rate=lr_schedule,
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        mu_dtype=mu_dtype,
        weight_decay=0.0,
        weight_decay_mask=None,
        nesterov=nesterov,
        use_magma=False,
        axis_name=axis_name,
        key=key,
    )
    return soda(base_optimizer, state_dtype=state_dtype)


def soda_prism(
    learning_rate: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    state_dtype: jax.typing.DTypeLike | None = None,
    b1: float = 0.95,
    gamma: float = 1.0,
    ns_iters: int = 5,
    mode: str = "original",
    inv_steps: int = 6,
    inv_eps: float = 1e-5,
    inv_scale: float = 1.001,
    eps_gram: float = 1e-6,
    gamma_l: float | None = None,
    gamma_r: float | None = None,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    nesterov: bool = True,
    shape_nesterov: bool = True,
    bias_correction: bool = False,
    grad_clip_max_amps: float | tuple[float, float] | None = (2.0, 10.0),
    raw_global_grad_clip: float | None = None,
    permissive_spike_protection: bool = True,
    mu_dtype: jax.typing.DTypeLike | None = None,
    axis_name: str | None = None,
    adam_learning_rate: float | None = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    prism_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """PRISM base optimizer wrapped with SODA."""
    prism_schedule = wsd_schedule(
        peak_lr=learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )
    if adam_learning_rate is None:
        adam_schedule = prism_schedule
    else:
        adam_schedule = wsd_schedule(
            peak_lr=adam_learning_rate,
            total_steps=total_steps,
            warmup_fraction=warmup_fraction,
            decay_fraction=decay_fraction,
        )

    base_optimizer = prism(
        learning_rate=prism_schedule,
        b1=b1,
        gamma=gamma,
        weight_decay=0.0,
        weight_decay_mask=None,
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
        grad_clip_max_amps=grad_clip_max_amps,
        raw_global_grad_clip=raw_global_grad_clip,
        permissive_spike_protection=permissive_spike_protection,
        mu_dtype=mu_dtype,
        axis_name=axis_name,
        use_magma=False,
        adam_learning_rate=adam_schedule,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps=adam_eps,
        prism_weight_dimension_numbers=prism_weight_dimension_numbers,
        key=key,
    )
    return soda(base_optimizer, state_dtype=state_dtype)


def soda_kron(
    learning_rate: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    state_dtype: jax.typing.DTypeLike | None = None,
    b1: float = 0.9,
    preconditioner_update_probability: base.ScalarOrSchedule = (
        precond_update_prob_schedule()
    ),
    max_size_triangular: int = 8192,
    max_skew_triangular: float = 1.0,
    min_ndim_triangular: int = 2,
    memory_save_mode: str | None = None,
    whiten_grad: bool = True,
    update_preconditioner_first: bool = True,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: float | None = None,
    mu_dtype: str | jnp.dtype | None = None,
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
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """PSGD Kron base optimizer wrapped with SODA."""
    lr_schedule = wsd_schedule(
        peak_lr=learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )
    base_optimizer = kron(
        learning_rate=lr_schedule,
        b1=b1,
        weight_decay=0.0,
        weight_decay_mask=None,
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
        use_magma=False,
        axis_name=axis_name,
        key=key,
    )
    return soda(base_optimizer, state_dtype=state_dtype)


def soda_muon(
    learning_rate: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    state_dtype: jax.typing.DTypeLike | None = None,
    ns_coeffs: Any = MUON_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype: jax.typing.DTypeLike | None = None,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_learning_rate: float | None = None,
    muon_weight_dimension_numbers: Any | Callable[[base.Params], Any] | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformationExtraArgs:
    """Optax contrib Muon base optimizer wrapped with SODA."""
    muon_schedule = wsd_schedule(
        peak_lr=learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )
    if adam_learning_rate is None:
        adam_schedule = None
    else:
        adam_schedule = wsd_schedule(
            peak_lr=adam_learning_rate,
            total_steps=total_steps,
            warmup_fraction=warmup_fraction,
            decay_fraction=decay_fraction,
        )

    base_optimizer = optax.contrib.muon(
        learning_rate=muon_schedule,
        ns_coeffs=ns_coeffs,
        ns_steps=ns_steps,
        beta=beta,
        eps=eps,
        weight_decay=0.0,
        weight_decay_mask=None,
        mu_dtype=mu_dtype,
        nesterov=nesterov,
        adaptive=adaptive,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps_root=adam_eps_root,
        adam_weight_decay=0.0,
        adam_learning_rate=adam_schedule,
        muon_weight_dimension_numbers=muon_weight_dimension_numbers,
        consistent_rms=consistent_rms,
    )
    return soda(base_optimizer, state_dtype=state_dtype)


def soda_rmnp(
    learning_rate: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.1,
    state_dtype: jax.typing.DTypeLike | None = None,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype: jax.typing.DTypeLike | None = None,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_learning_rate: float | None = None,
    rmnp_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """RMNP base optimizer wrapped with SODA."""
    rmnp_schedule = wsd_schedule(
        peak_lr=learning_rate,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        decay_fraction=decay_fraction,
    )
    if adam_learning_rate is None:
        adam_schedule = rmnp_schedule
    else:
        adam_schedule = wsd_schedule(
            peak_lr=adam_learning_rate,
            total_steps=total_steps,
            warmup_fraction=warmup_fraction,
            decay_fraction=decay_fraction,
        )

    base_optimizer = rmnp(
        learning_rate=rmnp_schedule,
        beta=beta,
        eps=eps,
        weight_decay=0.0,
        weight_decay_mask=None,
        mu_dtype=mu_dtype,
        nesterov=nesterov,
        adaptive=adaptive,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps_root=adam_eps_root,
        adam_weight_decay=0.0,
        adam_learning_rate=adam_schedule,
        rmnp_weight_dimension_numbers=rmnp_weight_dimension_numbers,
        consistent_rms=consistent_rms,
        key=key,
    )
    return soda(base_optimizer, state_dtype=state_dtype)
