"""EMA/SWA wrappers and evaluation views for fine-tuning optimizers."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from rollfast.utils import astype_preserving_sharding, zeros_like_preserving_sharding

from .config import AveragingSourceView, EMAConfig, SWAConfig


class AveragingState(NamedTuple):
    """Optimizer state augmented with optional EMA/SWA parameter averages."""

    inner_state: optax.OptState
    step: jax.Array
    ema_params: Any
    ema_mask: Any
    ema_count: jax.Array
    swa_params: Any
    swa_count: jax.Array


EvalFn = Callable[[Any, optax.OptState | None, str], Any]


def averaging_enabled(ema: EMAConfig, swa: SWAConfig) -> bool:
    """Return whether any parameter averaging wrapper is enabled."""

    return ema.enabled or swa.enabled


def wrap_with_averaging(
    tx: optax.GradientTransformation,
    *,
    ema: EMAConfig,
    swa: SWAConfig,
    total_steps: int | None,
    labels: Any | None = None,
    groups: tuple[Any, ...] = (),
    source_eval_fn: EvalFn | None = None,
) -> optax.GradientTransformationExtraArgs:
    """Wrap an optimizer and maintain EMA/SWA params after applied updates."""

    if not averaging_enabled(ema, swa):
        return optax.GradientTransformationExtraArgs(tx.init, tx.update)
    if (
        ema.source_view == "schedule_free_eval"
        or swa.source_view == "schedule_free_eval"
    ) and source_eval_fn is None:
        raise ValueError(
            "schedule_free_eval averaging requires a schedule-free source eval fn."
        )
    ema_mask = _ema_mask_from_labels(labels, groups, ema)

    swa_start_step = swa.resolved_start_step(total_steps)

    def init_fn(params):
        ema_params = _init_ema_params(params, ema)
        swa_params = _cast_tree(params, swa.state_dtype) if swa.enabled else None
        return AveragingState(
            inner_state=tx.init(params),
            step=jnp.zeros([], dtype=jnp.int32),
            ema_params=ema_params,
            ema_mask=ema_mask,
            ema_count=jnp.zeros([], dtype=jnp.int32),
            swa_params=swa_params,
            swa_count=jnp.zeros([], dtype=jnp.int32),
        )

    def update_fn(updates, state, params=None, **extra_args):
        if params is None:
            raise ValueError("EMA/SWA averaging requires params in optimizer.update.")
        if extra_args:
            base_updates, inner_state = tx.update(
                updates,
                state.inner_state,
                params,
                **extra_args,
            )
        else:
            base_updates, inner_state = tx.update(updates, state.inner_state, params)

        next_params = optax.apply_updates(params, base_updates)
        applied = _update_applied(state.inner_state, inner_state)
        next_step = state.step + applied.astype(jnp.int32)
        ema_source_params = _averaging_source_params(
            ema.source_view,
            next_params,
            inner_state,
            source_eval_fn,
        )
        swa_source_params = _averaging_source_params(
            swa.source_view,
            next_params,
            inner_state,
            source_eval_fn,
        )

        ema_params, ema_count = _update_ema(
            state.ema_params,
            state.ema_count,
            ema_source_params,
            ema,
            next_step,
            applied,
            mask=state.ema_mask,
        )
        swa_params, swa_count = _update_swa(
            state.swa_params,
            state.swa_count,
            swa_source_params,
            swa,
            next_step,
            applied,
            swa_start_step=swa_start_step,
        )
        return base_updates, AveragingState(
            inner_state=inner_state,
            step=next_step,
            ema_params=ema_params,
            ema_mask=state.ema_mask,
            ema_count=ema_count,
            swa_params=swa_params,
            swa_count=swa_count,
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _averaging_source_params(
    source_view: AveragingSourceView | None,
    params: Any,
    inner_state: optax.OptState,
    source_eval_fn: EvalFn | None,
) -> Any:
    if source_view in {None, "optimizer"}:
        return params
    if source_view == "schedule_free_eval":
        if source_eval_fn is None:
            raise ValueError(
                "schedule_free_eval averaging requires a schedule-free source eval fn."
            )
        return source_eval_fn(params, inner_state, "schedule_free")
    raise ValueError(f"unknown averaging source_view: {source_view!r}.")


def make_averaging_eval_fn(
    *,
    ema: EMAConfig,
    swa: SWAConfig,
    inner_eval_fn: EvalFn | None = None,
) -> EvalFn | None:
    """Build an eval-parameter resolver for optimizer, EMA, and SWA views."""

    if not averaging_enabled(ema, swa) and inner_eval_fn is None:
        return None

    def eval_fn(params: Any, state: optax.OptState | None, view: str) -> Any:
        if view == "optimizer":
            return params
        averaging_state = state if isinstance(state, AveragingState) else None
        inner_state = (
            averaging_state.inner_state if averaging_state is not None else state
        )
        if view == "ema":
            if not ema.enabled or averaging_state is None:
                raise ValueError("EMA eval params are not enabled for this bundle.")
            return _eval_ema(params, averaging_state, ema)
        if view == "swa":
            if not swa.enabled or averaging_state is None:
                raise ValueError("SWA eval params are not enabled for this bundle.")
            return _eval_swa(params, averaging_state)
        if inner_eval_fn is not None:
            return inner_eval_fn(params, inner_state, view)
        raise ValueError(f"unknown eval params view: {view!r}.")

    return eval_fn


def eval_views(
    *,
    ema: EMAConfig,
    swa: SWAConfig,
    inner_views: tuple[str, ...] = ("optimizer",),
) -> tuple[str, ...]:
    """Return the named evaluation views exposed by averaging configuration."""

    views = list(inner_views)
    if ema.enabled:
        views.append("ema")
    if swa.enabled:
        views.append("swa")
    return tuple(dict.fromkeys(views))


def default_eval_view(
    *,
    ema: EMAConfig,
    swa: SWAConfig,
    inner_default: str = "optimizer",
) -> str:
    """Return the preferred evaluation view for a compiled optimizer bundle."""

    if ema.enabled:
        return "ema"
    if swa.enabled:
        return "swa"
    return inner_default


def _init_ema_params(params: Any, ema: EMAConfig) -> Any:
    if not ema.enabled:
        return None
    if ema.debias:
        return jax.tree.map(
            lambda x: zeros_like_preserving_sharding(x, ema.state_dtype),
            params,
            is_leaf=lambda x: x is None,
        )
    return _cast_tree(params, ema.state_dtype)


def _update_ema(
    ema_params: Any,
    ema_count: jax.Array,
    params: Any,
    config: EMAConfig,
    step: jax.Array,
    applied: jax.Array,
    *,
    mask: Any,
) -> tuple[Any, jax.Array]:
    if not config.enabled:
        return ema_params, ema_count

    cadence = (step - config.start_step) % config.update_every == 0
    should_update = applied & (step >= config.start_step) & cadence
    cast_params = _cast_tree(params, config.state_dtype)
    next_ema = jax.tree.map(
        lambda old, new: (
            None if old is None else config.decay * old + (1.0 - config.decay) * new
        ),
        ema_params,
        cast_params,
        is_leaf=lambda x: x is None,
    )
    next_ema = _where_mask_tree(mask, next_ema, cast_params)
    return (
        _where_tree(should_update, next_ema, ema_params),
        ema_count + should_update.astype(jnp.int32),
    )


def _update_swa(
    swa_params: Any,
    swa_count: jax.Array,
    params: Any,
    config: SWAConfig,
    step: jax.Array,
    applied: jax.Array,
    *,
    swa_start_step: int,
) -> tuple[Any, jax.Array]:
    if not config.enabled:
        return swa_params, swa_count

    cadence = (step - swa_start_step) % config.frequency == 0
    should_update = applied & (step >= swa_start_step) & cadence
    cast_params = _cast_tree(params, config.state_dtype)
    next_count = swa_count + jnp.asarray(1, dtype=jnp.int32)
    next_swa = jax.tree.map(
        lambda old, new: (
            None if old is None else old + (new - old) / next_count.astype(new.dtype)
        ),
        swa_params,
        cast_params,
        is_leaf=lambda x: x is None,
    )
    return (
        _where_tree(should_update, next_swa, swa_params),
        swa_count + should_update.astype(jnp.int32),
    )


def _eval_ema(params: Any, state: AveragingState, config: EMAConfig) -> Any:
    if not config.debias:
        return _where_mask_tree(state.ema_mask, state.ema_params, params)
    correction = 1.0 - jnp.power(
        jnp.asarray(config.decay, dtype=jnp.float32),
        state.ema_count.astype(jnp.float32),
    )
    corrected = jax.tree.map(
        lambda x: None if x is None else x / jnp.maximum(correction, 1e-8),
        state.ema_params,
        is_leaf=lambda x: x is None,
    )
    debiased = _where_tree(state.ema_count > 0, corrected, params)
    return _where_mask_tree(state.ema_mask, debiased, params)


def _eval_swa(params: Any, state: AveragingState) -> Any:
    return _where_tree(state.swa_count > 0, state.swa_params, params)


def _cast_tree(tree: Any, dtype: Any) -> Any:
    return jax.tree.map(
        lambda x: (
            astype_preserving_sharding(x, dtype)
            if x is not None and hasattr(x, "astype")
            else x
        ),
        tree,
        is_leaf=lambda x: x is None,
    )


def _where_tree(condition: jax.Array, true_tree: Any, false_tree: Any) -> Any:
    return jax.tree.map(
        lambda true, false: None if true is None else jnp.where(condition, true, false),
        true_tree,
        false_tree,
        is_leaf=lambda x: x is None,
    )


def _where_mask_tree(mask: Any, true_tree: Any, false_tree: Any) -> Any:
    if mask is None:
        return true_tree
    return jax.tree.map(
        lambda include, true, false: (
            None if false is None else jnp.where(include, true, false)
        ),
        mask,
        true_tree,
        false_tree,
        is_leaf=lambda x: x is None,
    )


def _ema_mask_from_labels(
    labels: Any | None,
    groups: tuple[Any, ...],
    ema: EMAConfig,
) -> Any | None:
    if not ema.enabled or (not ema.include_tags and not ema.exclude_tags):
        return None
    if labels is None:
        raise ValueError("EMA include_tags/exclude_tags require plan labels.")
    groups_by_label = {group.source_label: group for group in groups}
    include_tags = {tag.lower() for tag in ema.include_tags}
    exclude_tags = {tag.lower() for tag in ema.exclude_tags}
    return jax.tree.map(
        lambda label: (
            None
            if label is None
            else _ema_include_label(
                str(label),
                groups_by_label.get(str(label)),
                include_tags=include_tags,
                exclude_tags=exclude_tags,
            )
        ),
        labels,
        is_leaf=lambda x: x is None,
    )


def _ema_include_label(
    label: str,
    group: Any,
    *,
    include_tags: set[str],
    exclude_tags: set[str],
) -> bool:
    terms = {label.lower()}
    if group is not None:
        terms.add(str(getattr(group, "role", "")).lower())
        terms.update(str(tag).lower() for tag in getattr(group, "tags", ()))
    included = not include_tags or bool(terms & include_tags)
    excluded = bool(terms & exclude_tags)
    return included and not excluded


def _update_applied(old_state: Any, new_state: Any) -> jax.Array:
    if hasattr(old_state, "gradient_step") and hasattr(new_state, "gradient_step"):
        return new_state.gradient_step > old_state.gradient_step
    if hasattr(new_state, "last_finite"):
        return jnp.asarray(new_state.last_finite, dtype=jnp.bool_)
    if hasattr(old_state, "inner_opt_state") and hasattr(new_state, "inner_opt_state"):
        return _update_applied(old_state.inner_opt_state, new_state.inner_opt_state)
    if hasattr(old_state, "inner_state") and hasattr(new_state, "inner_state"):
        return _update_applied(old_state.inner_state, new_state.inner_state)
    return jnp.asarray(True, dtype=jnp.bool_)


__all__ = (
    "AveragingState",
    "averaging_enabled",
    "default_eval_view",
    "eval_views",
    "make_averaging_eval_fn",
    "wrap_with_averaging",
)
