"""Small functional update helpers for plan-aware optimizers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax

from rollfast.optim.sam import add_perturbation, global_l2_norm, sam_perturbation

from ._protocols import FineTunePlanProtocol
from .config import OptimizerBundle, SAMConfig


class SAMStepInfo(NamedTuple):
    """Diagnostics returned by a two-pass SAM/ASAM update."""

    loss: Any
    perturbed_loss: Any
    aux: Any | None
    perturbed_aux: Any | None
    grad_norm: Any
    perturbation_norm: Any


def make_update_step(
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return a one-update function over a trainable PyTree."""

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def step(trainable, opt_state, *args, **kwargs):
        value, grads = value_and_grad(trainable, *args, **kwargs)
        updates, opt_state = optimizer.update(grads, opt_state, trainable)
        trainable = optax.apply_updates(trainable, updates)
        return trainable, opt_state, value

    return step


def make_plan_update_step(
    plan: FineTunePlanProtocol,
    loss_fn: Callable[..., Any],
    optimizer: OptimizerBundle,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return a one-update function whose loss receives ``plan.combine`` output."""

    def trainable_loss(trainable, *args, **kwargs):
        return loss_fn(plan.combine(trainable), *args, **kwargs)

    return make_update_step(trainable_loss, optimizer, has_aux=has_aux)


def make_sam_step(
    *,
    plan: FineTunePlanProtocol,
    base_optimizer: OptimizerBundle,
    config: SAMConfig | None = None,
    loss_fn: Callable[..., Any],
    has_aux: bool = False,
    microbatch_axis: int | None = None,
    microbatch_count: int | None = None,
    microbatch_reduction: Literal["mean", "sum"] = "mean",
) -> Callable[..., Any]:
    """Return a functional two-pass SAM/ASAM update step.

    The base optimizer is applied to gradients from perturbed parameters, but
    the returned trainable tree descends from the original unperturbed params.
    """

    config = SAMConfig(enabled=True) if config is None else config
    if base_optimizer.accumulation_config.steps != 1:
        raise ValueError(
            "SAM/ASAM steps require a base optimizer with accumulation_steps=1; "
            "use microbatch_axis on make_sam_step for exact SAM accumulation."
        )
    if microbatch_axis is not None and microbatch_axis < 0:
        raise ValueError("microbatch_axis must be non-negative when provided.")
    if microbatch_count is not None and microbatch_count <= 0:
        raise ValueError("microbatch_count must be positive when provided.")
    if microbatch_reduction not in {"mean", "sum"}:
        raise ValueError("microbatch_reduction must be 'mean' or 'sum'.")
    mask = _sam_perturb_mask(plan, config)
    value_and_grad = jax.value_and_grad(
        lambda trainable, *args, **kwargs: loss_fn(
            plan.combine(trainable),
            *args,
            **kwargs,
        ),
        has_aux=has_aux,
    )

    def step(trainable, opt_state, *args, **kwargs):
        value, grads = _evaluate_sam_value_and_grad(
            value_and_grad,
            trainable,
            args,
            kwargs,
            microbatch_axis=microbatch_axis,
            microbatch_count=microbatch_count,
            microbatch_reduction=microbatch_reduction,
        )
        loss, aux = _split_value_aux(value, has_aux=has_aux)
        perturbation, perturbation_norm = sam_perturbation(
            grads,
            params=trainable,
            rho=config.rho,
            adaptive=config.adaptive,
            eta=config.eta,
            eps=config.eps,
            mask=mask,
            axis_name=config.axis_name,
        )
        perturbed = add_perturbation(trainable, perturbation)
        perturbed_value, perturbed_grads = _evaluate_sam_value_and_grad(
            value_and_grad,
            perturbed,
            args,
            kwargs,
            microbatch_axis=microbatch_axis,
            microbatch_count=microbatch_count,
            microbatch_reduction=microbatch_reduction,
        )
        perturbed_loss, perturbed_aux = _split_value_aux(
            perturbed_value,
            has_aux=has_aux,
        )
        updates, opt_state = base_optimizer.update(
            perturbed_grads,
            opt_state,
            trainable,
        )
        trainable = optax.apply_updates(trainable, updates)
        info = SAMStepInfo(
            loss=loss,
            perturbed_loss=perturbed_loss,
            aux=aux,
            perturbed_aux=perturbed_aux,
            grad_norm=global_l2_norm(grads, axis_name=config.axis_name),
            perturbation_norm=perturbation_norm,
        )
        return trainable, opt_state, info

    return step


def _evaluate_sam_value_and_grad(
    value_and_grad: Callable[..., Any],
    trainable: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    microbatch_axis: int | None,
    microbatch_count: int | None,
    microbatch_reduction: Literal["mean", "sum"],
) -> tuple[Any, Any]:
    if microbatch_axis is None:
        return value_and_grad(trainable, *args, **kwargs)

    count = _resolve_microbatch_count(
        args,
        kwargs,
        axis=microbatch_axis,
        count=microbatch_count,
    )
    total_value = None
    total_grads = None
    for index in range(count):
        micro_args = _slice_microbatch_tree(
            args,
            index=index,
            axis=microbatch_axis,
            count=count,
        )
        micro_kwargs = _slice_microbatch_tree(
            kwargs,
            index=index,
            axis=microbatch_axis,
            count=count,
        )
        value, grads = value_and_grad(trainable, *micro_args, **micro_kwargs)
        total_value = value if total_value is None else _tree_add(total_value, value)
        total_grads = grads if total_grads is None else _tree_add(total_grads, grads)

    if microbatch_reduction == "mean":
        scale = 1.0 / count
        total_value = _tree_scale(total_value, scale)
        total_grads = _tree_scale(total_grads, scale)
    return total_value, total_grads


def _resolve_microbatch_count(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    axis: int,
    count: int | None,
) -> int:
    if count is not None:
        return count
    sizes = {
        int(leaf.shape[axis])
        for leaf in jax.tree.leaves((args, kwargs), is_leaf=lambda x: x is None)
        if leaf is not None and hasattr(leaf, "shape") and len(leaf.shape) > axis
    }
    if not sizes:
        raise ValueError(
            "microbatch_axis was provided but no array argument has that axis; "
            "pass microbatch_count for broadcast-only arguments."
        )
    if len(sizes) != 1:
        raise ValueError(
            "microbatch_axis inferred inconsistent leading sizes; pass "
            "microbatch_count to broadcast non-microbatched arrays explicitly."
        )
    return sizes.pop()


def _slice_microbatch_tree(
    tree: Any,
    *,
    index: int,
    axis: int,
    count: int,
) -> Any:
    return jax.tree.map(
        lambda leaf: _slice_microbatch_leaf(
            leaf,
            index=index,
            axis=axis,
            count=count,
        ),
        tree,
        is_leaf=lambda x: x is None,
    )


def _slice_microbatch_leaf(
    leaf: Any,
    *,
    index: int,
    axis: int,
    count: int,
) -> Any:
    if leaf is None or not hasattr(leaf, "shape") or len(leaf.shape) <= axis:
        return leaf
    if int(leaf.shape[axis]) != count:
        return leaf
    return jnp.take(leaf, index, axis=axis)


def _tree_add(left: Any, right: Any) -> Any:
    return jax.tree.map(
        lambda lhs, rhs: None if lhs is None else lhs + rhs,
        left,
        right,
        is_leaf=lambda x: x is None,
    )


def _tree_scale(tree: Any, scale: float) -> Any:
    return jax.tree.map(
        lambda leaf: None if leaf is None else leaf * scale,
        tree,
        is_leaf=lambda x: x is None,
    )


def sam_cost_report(
    plan: FineTunePlanProtocol,
    config: SAMConfig | None = None,
) -> dict[str, Any]:
    """Return static cost metadata for a SAM/ASAM fine-tuning step."""

    config = SAMConfig(enabled=True) if config is None else config
    mask = _sam_perturb_mask(plan, config)
    perturbation_bytes = 0
    for param, include in zip(
        jax.tree.leaves(plan.trainable, is_leaf=lambda x: x is None),
        jax.tree.leaves(mask, is_leaf=lambda x: x is None),
        strict=True,
    ):
        if param is not None and include:
            perturbation_bytes += int(param.size * param.dtype.itemsize)
    return {
        "method": "ASAM" if config.adaptive else "SAM",
        "forward_backward_evaluations": 2,
        "perturbation_bytes": perturbation_bytes,
        "rho": config.rho,
        "adaptive": config.adaptive,
        "eta": config.eta,
        "norm": config.norm,
    }


def _split_value_aux(value: Any, *, has_aux: bool) -> tuple[Any, Any | None]:
    if not has_aux:
        return value, None
    loss, aux = value
    return loss, aux


def _sam_perturb_mask(
    plan: FineTunePlanProtocol,
    config: SAMConfig,
) -> Any:
    group_specs = getattr(plan, "group_specs", {})
    return jax.tree.map(
        lambda label: (
            None
            if label is None
            else _perturb_label(str(label), group_specs.get(str(label)), config)
        ),
        plan.labels,
        is_leaf=lambda x: x is None,
    )


def _perturb_label(label: str, group: Any, config: SAMConfig) -> bool:
    terms = {label.lower()}
    if group is not None:
        terms.add(str(getattr(group, "role", "")).lower())
        terms.update(str(tag).lower() for tag in getattr(group, "tags", ()))
    if not config.perturb_bias and any("bias" in term for term in terms):
        return False
    if not config.perturb_norm and any("norm" in term for term in terms):
        return False
    return True


__all__ = (
    "SAMStepInfo",
    "make_plan_update_step",
    "make_sam_step",
    "make_update_step",
    "sam_cost_report",
)
