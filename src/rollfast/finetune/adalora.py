"""Optimizer-side AdaLoRA budget and rank-mask controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp

from .config import AdaLoRAControllerConfig, SCHEMA_VERSION


class AdaLoRAState(NamedTuple):
    """JAX PyTree state for fixed-shape AdaLoRA rank allocation."""

    step: jax.Array
    sensitivity_ema: jax.Array
    uncertainty_ema: jax.Array
    rank_mask: jax.Array
    ranks: jax.Array
    current_budget: jax.Array


@dataclass(frozen=True)
class AdaLoRAController:
    """Fixed-group AdaLoRA controller.

    ``group_names`` are static Python metadata. The mutable state is a PyTree of
    fixed-shape JAX arrays so it can be carried through JIT-compiled training
    loops. The controller emits masks; Equimo or user code owns applying them to
    rank-masked adapter modules.
    """

    group_names: tuple[str, ...]
    max_ranks: tuple[int, ...]
    total_steps: int
    config: AdaLoRAControllerConfig = AdaLoRAControllerConfig(enabled=True)

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if not self.group_names:
            raise ValueError("AdaLoRAController requires at least one rank group.")
        if len(self.group_names) != len(self.max_ranks):
            raise ValueError("group_names and max_ranks must have equal length.")
        if any(rank < 1 for rank in self.max_ranks):
            raise ValueError("max_ranks must all be positive.")
        if self.config.min_rank > min(self.max_ranks):
            raise ValueError("AdaLoRA min_rank cannot exceed any max_rank.")

    @property
    def group_count(self) -> int:
        return len(self.group_names)

    @property
    def max_rank(self) -> int:
        return max(self.max_ranks)

    @property
    def t_init(self) -> int:
        return int(round(self.config.initial_warmup_fraction * self.total_steps))

    @property
    def t_final(self) -> int:
        return int(round(self.config.final_tuning_fraction * self.total_steps))

    @property
    def allocation_interval(self) -> int:
        return self.config.resolved_interval(self.total_steps)

    def init(self) -> AdaLoRAState:
        ranks = jnp.asarray(
            [
                min(max_rank, self.config.init_rank)
                for max_rank in self.max_ranks
            ],
            dtype=jnp.int32,
        )
        rank_mask = _mask_from_ranks(ranks, self.max_rank)
        budget = jnp.asarray(int(jnp.sum(ranks)), dtype=jnp.int32)
        zeros = jnp.zeros((self.group_count, self.max_rank), dtype=jnp.float32)
        return AdaLoRAState(
            step=jnp.zeros([], jnp.int32),
            sensitivity_ema=zeros,
            uncertainty_ema=zeros,
            rank_mask=rank_mask,
            ranks=ranks,
            current_budget=budget,
        )

    def budget_at(self, step: int | jax.Array) -> jax.Array:
        """Return the monotonic global rank budget at an optimizer step."""

        step = jnp.asarray(step, dtype=jnp.int32)
        init_budget = _clipped_budget(
            self.group_count * self.config.init_rank,
            self.max_ranks,
            self.config.min_rank,
        )
        target_budget = _clipped_budget(
            self.group_count * self.config.resolved_target_rank(),
            self.max_ranks,
            self.config.min_rank,
        )
        decay_start = jnp.asarray(self.t_init, dtype=jnp.int32)
        decay_end = jnp.asarray(
            max(self.t_init + 1, self.total_steps - self.t_final),
            dtype=jnp.int32,
        )
        span = jnp.maximum(decay_end - decay_start, 1)
        progress = jnp.clip(step - decay_start, 0, span)
        delta = init_budget - target_budget
        decayed = init_budget - jnp.floor(delta * progress / span).astype(jnp.int32)
        return jnp.clip(decayed, target_budget, init_budget).astype(jnp.int32)

    def update(
        self,
        state: AdaLoRAState,
        importance: jax.Array,
        *,
        applied: bool | jax.Array = True,
    ) -> AdaLoRAState:
        """Advance EMAs and rank masks after one applied optimizer step."""

        applied = jnp.asarray(applied, dtype=jnp.bool_)
        importance = self._padded_scores(importance)
        sensitivity = self.config.beta1 * state.sensitivity_ema + (
            1.0 - self.config.beta1
        ) * jnp.abs(importance)
        uncertainty = self.config.beta2 * state.uncertainty_ema + (
            1.0 - self.config.beta2
        ) * jnp.abs(jnp.abs(importance) - state.sensitivity_ema)
        next_step = state.step + applied.astype(jnp.int32)
        budget = self.budget_at(next_step)
        should_allocate = jnp.logical_and(
            applied,
            jnp.logical_and(
                next_step >= self.t_init,
                (next_step - self.t_init) % self.allocation_interval == 0,
            ),
        )
        scores = sensitivity * (uncertainty + self.config.score_eps)
        new_mask = allocate_rank_mask(
            scores,
            max_ranks=self.max_ranks,
            budget=budget,
            min_rank=self.config.min_rank,
        )
        rank_mask = jnp.where(should_allocate, new_mask, state.rank_mask)
        ranks = jnp.sum(rank_mask.astype(jnp.int32), axis=1)
        current_budget = jnp.where(should_allocate, budget, state.current_budget)
        return AdaLoRAState(
            step=next_step,
            sensitivity_ema=jnp.where(applied, sensitivity, state.sensitivity_ema),
            uncertainty_ema=jnp.where(applied, uncertainty, state.uncertainty_ema),
            rank_mask=rank_mask,
            ranks=ranks,
            current_budget=current_budget,
        )

    def rank_pattern(self, state: AdaLoRAState) -> dict[str, jax.Array]:
        """Return a mapping from static group name to fixed-shape rank mask."""

        return {
            name: state.rank_mask[index, : self.max_ranks[index]]
            for index, name in enumerate(self.group_names)
        }

    def rank_report(self, state: AdaLoRAState) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "group": name,
                "rank": int(state.ranks[index]),
                "max_rank": self.max_ranks[index],
            }
            for index, name in enumerate(self.group_names)
        )

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "group_names": self.group_names,
            "max_ranks": self.max_ranks,
            "total_steps": self.total_steps,
            "t_init": self.t_init,
            "t_final": self.t_final,
            "allocation_interval": self.allocation_interval,
            "orthogonal_reg_weight": self.config.orthogonal_reg_weight,
            "config": self.config.to_dict(),
        }

    def _padded_scores(self, scores: jax.Array) -> jax.Array:
        scores = jnp.asarray(scores, dtype=jnp.float32)
        if scores.shape != (self.group_count, self.max_rank):
            raise ValueError(
                "AdaLoRA importance scores must have shape "
                f"{(self.group_count, self.max_rank)}, got {scores.shape}."
            )
        valid = _valid_mask(self.max_ranks, self.max_rank)
        return jnp.where(valid, scores, -jnp.inf)


def make_adalora_controller(
    rank_groups: Mapping[str, int],
    *,
    total_steps: int,
    config: AdaLoRAControllerConfig | None = None,
) -> AdaLoRAController:
    """Build a deterministic fixed-group AdaLoRA controller."""

    if not rank_groups:
        raise ValueError("rank_groups must not be empty.")
    group_names = tuple(sorted(rank_groups))
    max_ranks = tuple(int(rank_groups[name]) for name in group_names)
    return AdaLoRAController(
        group_names=group_names,
        max_ranks=max_ranks,
        total_steps=total_steps,
        config=AdaLoRAControllerConfig(enabled=True) if config is None else config,
    )


def allocate_rank_mask(
    scores: jax.Array,
    *,
    max_ranks: tuple[int, ...],
    budget: int | jax.Array,
    min_rank: int = 1,
) -> jax.Array:
    """Allocate a fixed-shape rank mask from per-slot scores."""

    scores = jnp.asarray(scores, dtype=jnp.float32)
    group_count, max_rank = scores.shape
    valid = _valid_mask(max_ranks, max_rank)
    mandatory = jnp.arange(max_rank)[None, :] < min_rank
    mandatory = jnp.logical_and(mandatory, valid)
    min_budget = int(sum(min(max_rank_, min_rank) for max_rank_ in max_ranks))
    max_budget = int(sum(max_ranks))
    budget = jnp.clip(jnp.asarray(budget, dtype=jnp.int32), min_budget, max_budget)
    remaining = budget - jnp.sum(mandatory.astype(jnp.int32))
    candidate = jnp.logical_and(valid, jnp.logical_not(mandatory))
    flat_scores = jnp.reshape(scores, (-1,))
    flat_candidate = jnp.reshape(candidate, (-1,))
    tie_break = jnp.arange(flat_scores.size, dtype=jnp.float32) * 1e-12
    adjusted = jnp.where(flat_candidate, flat_scores - tie_break, -jnp.inf)
    order = jnp.argsort(-adjusted)
    selected_flat = jnp.arange(flat_scores.size) < remaining
    selected = jnp.zeros(flat_scores.shape, dtype=jnp.bool_).at[order].set(selected_flat)
    selected = jnp.reshape(selected, (group_count, max_rank))
    return jnp.logical_or(mandatory, jnp.logical_and(selected, candidate))


def _valid_mask(max_ranks: tuple[int, ...], max_rank: int) -> jax.Array:
    ranks = jnp.asarray(max_ranks, dtype=jnp.int32)
    return jnp.arange(max_rank)[None, :] < ranks[:, None]


def _mask_from_ranks(ranks: jax.Array, max_rank: int) -> jax.Array:
    return jnp.arange(max_rank)[None, :] < ranks[:, None]


def _clipped_budget(
    requested_budget: int,
    max_ranks: tuple[int, ...],
    min_rank: int,
) -> jax.Array:
    min_budget = sum(min(max_rank, min_rank) for max_rank in max_ranks)
    max_budget = sum(max_ranks)
    return jnp.asarray(
        max(min_budget, min(max_budget, requested_budget)),
        dtype=jnp.int32,
    )


__all__ = (
    "AdaLoRAController",
    "AdaLoRAState",
    "allocate_rank_mask",
    "make_adalora_controller",
)
