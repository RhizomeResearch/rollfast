# `rollfast.schedules`

## Overview

Learning-rate schedules, interval-stepped schedule wrappers, and Schedule-Free iterate-averaging optimizer wrappers
built around Optax schedules and `GradientTransformationExtraArgs`.

## Interface

- Learning-Rate Schedules (`[__init__.py:__all__]`):
  - `cosine_annealing_schedule(peak_lr, t_max, *, end_lr_ratio=0.0, warmup_steps=0, after_end="clamp") -> optax.Schedule`
    (`[cosine_annealing.py:cosine_annealing_schedule]`), with `AfterEnd = Literal["clamp", "cycle"]`
    (`[cosine_annealing.py:AfterEnd]`).
  - `interval_schedule(schedule, steps_per_interval) -> optax.Schedule` (`[interval.py:interval_schedule]`): Indexes
    `schedule` by integer-divided update intervals (`count // steps_per_interval`).
  - `wsd_schedule(peak_lr, total_steps, warmup_fraction=0.1, decay_fraction=0.1, warmup_shape="linear", decay_shape="linear", final_lr_ratio=0.0, *, warmup_steps=None, decay_steps=None, end_lr_ratio=None, warmup_power=1.0, decay_power=1.0, exponential_rate=5.0) -> optax.Schedule`
    (`[wsd.py:wsd_schedule]`), with `ScheduleShape = Literal["linear", "cosine", "power", "exponential"]`
    (`[wsd.py:ScheduleShape]`).
  - `power_decay_schedule(peak_lr, total_steps, power, warmup_fraction=0.0, warmup_shape="linear", final_lr_ratio=0.0, warmup_power=1.0, exponential_rate=5.0) -> optax.Schedule`
    (`[wsd.py:power_decay_schedule]`).
- Schedule-Free Transformations & Helpers (`[schedulefree.py:__all__]`, `[__init__.py:__all__]`):
  - `WeightingMode` (`[schedulefree.py:WeightingMode]`): `str` `Enum` with modes `THEORETICAL` (`"theoretical"`),
    `PRACTICAL` (`"practical"`), `SCHEDULET` (`"schedulet"`), and `POWER` (`"power"`).
  - `ScheduleFreeLearningRate` (`[schedulefree.py:ScheduleFreeLearningRate]`): Schedule/scalar or
    `(count, params) -> tree` learning-rate type alias.
  - `ScheduleFreeState` (`[schedulefree.py:ScheduleFreeState]`): `NamedTuple` carrying
    `(b1, weight_sum, step_count, base_state, z, key, lr_max, scheduled_lr, grad_l1_ema, grad_l1_ema_corr, polyak_lr, polyak_value, ip_term)`.
  - `schedule_free(base_optimizer, learning_rate, b1=0.9, weighting_mode=WeightingMode.SCHEDULET, state_dtype=None, key=None, *, r=0.0, c_warmup=0, weight_lr_power=2.0, use_lr_max=False, b1_anneal_steps=0, b1_max=0.965, polyak=False, polyak_beta=0.0, polyak_f_star=0.0, polyak_eps=1e-30, polyak_axis_name=None, lr_max_init=1e-8, adamc_weight_decay=0.0, adamc_weight_decay_mask=None) -> optax.GradientTransformationExtraArgs`
    (`[schedulefree.py:schedule_free]`).
  - `schedule_free_adam`, `schedule_free_aurora`, `schedule_free_kron`, `schedule_free_prism`
    (`[schedulefree.py:schedule_free_adam]`, `[schedulefree.py:schedule_free_aurora]`,
    `[schedulefree.py:schedule_free_kron]`, `[schedulefree.py:schedule_free_prism]`): Pre-configured WSD-scheduled
    Schedule-Free (and Schedule-Free+) optimizer wrappers.
  - `schedule_free_eval_params(state, params) -> optax.Params` (`[schedulefree.py:schedule_free_eval_params]`):
    Reconstructs the averaged `x` evaluation sequence from training iterate `y` (`params`) and `ScheduleFreeState.z`.
- Compatibility Re-exports:
  - `GradClipMode`, `PreconditionerMode`, `precond_update_prob_schedule` (`[__init__.py:__all__]`): Re-exported from
    `rollfast.optim.psgd`.
  - `soda`, `soda_adam`, `soda_kron`, `soda_muon`, `soda_prism`, `soda_rmnp` (`[soda.py:__all__]`): Backward-compatible
    submodule re-exports from `rollfast.optim.soda`.

## Invariants

- Cosine annealing bounds (`[cosine_annealing.py:cosine_annealing_schedule]`): Requires `t_max > 0`,
  `end_lr_ratio >= 0.0`, `warmup_steps >= 0`, and `after_end in ("clamp", "cycle")` (`ValueError`); casts `count` to
  `float32` and evaluates `sin(0.5 * pi * remaining)^2` to prevent `float32` cancellation near zero.
- Interval stepping (`[interval.py:interval_schedule]`): Requires `steps_per_interval` to be a non-`bool` `int >= 1`
  (`ValueError`); returns `schedule` unchanged when `steps_per_interval == 1`.
- WSD and power-decay validation (`[wsd.py:wsd_schedule]`, `[wsd.py:power_decay_schedule]`, `[wsd.py:_validate_shape]`):
  Requires `total_steps > 0`, valid `ScheduleShape`, fractions and `final_lr_ratio` in `[0.0, 1.0]`, matching
  `final_lr_ratio` and `end_lr_ratio` when both are non-default/provided, positive power exponents,
  `exponential_rate != 0.0`, `resolved_warmup_steps >= 0`, `resolved_decay_steps >= 0`, and
  `resolved_warmup_steps + resolved_decay_steps <= total_steps` (`ValueError`); clamps outputs via
  `jnp.maximum(0.0, val)`.
- Schedule-Free constructor, parameter, and dtype contracts (`[schedulefree.py:schedule_free]`): Validates
  `c_warmup >= 0`, `b1_anneal_steps >= 0`, `0.0 <= polyak_beta < 1.0`, and `b1 > 0.0` when `polyak=True` (`ValueError`);
  rejects complex-valued PyTrees in `init` and `update` via `_reject_complex_tree`; requires `params is not None` on
  `update`; allows `None` learning-rate leaves while requiring every non-`None` learning-rate leaf to be scalar
  (`shape == ()`); accumulates `z` updates in `float32` with stochastic `bfloat16` rounding when `z` leaves are
  `bfloat16`; strips `{"function_value", "loss", "value"}` from `extra_args` before calling `base_optimizer.update`
  (requiring one of them when `polyak=True`).
- Schedule-Free warmup and Schedule-Free+ defaults (`[schedulefree.py:schedule_free]`,
  `[schedulefree.py:schedule_free_adam]`, `[schedulefree.py:schedule_free_aurora]`,
  `[schedulefree.py:schedule_free_kron]`, `[schedulefree.py:schedule_free_prism]`): `WeightingMode.POWER` forces
  `use_lr_max=True`; while `step_count < c_warmup`, `weight_sum` is held constant and `ck = 1.0`;
  `schedule_free_plus=True` defaults `polyak`, `use_adamc`, and `sf_use_lr_max` to `True`, moves weight decay to outer
  `adamc_weight_decay`, and shifts omitted Adam `(b1, b2)` defaults to `(0.9, 0.95)`.
- Evaluation reconstruction (`[schedulefree.py:schedule_free_eval_params]`): Requires `state` to carry non-`None` `b1`
  and `z` attributes (`ValueError`) and clamps `b1` to `>= 1e-8`.

## Side Effects

- Import-time schedule construction (`[schedulefree.py:_DEFAULT_SCHEDULE_FREE_KRON_SCHEDULE]`): Evaluates
  `precond_update_prob_schedule()` at module import time.
- SPMD collectives (`[schedulefree.py:schedule_free]`): Emits `jax.lax.psum` (on `grad_l1` and `ip_term`) and
  `jax.lax.pmean` (on `f_value`) across `polyak_axis_name` during `update` when `polyak=True` and `polyak_axis_name` is
  set.
- PRNG initialization (`[schedulefree.py:schedule_free]`, `[schedulefree.py:schedule_free_prism]`,
  `[schedulefree.py:schedule_free_aurora]`): Generates a fresh JAX PRNG key via `_fresh_prng_key(key)` when
  `key is None`.

## Verification

- Run: `uv run pytest tests/test_schedulefree.py tests/test_wsd.py`
