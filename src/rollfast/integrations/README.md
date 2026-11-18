# `rollfast.integrations`

## Overview

Optional third-party integration adapters connecting external fine-tuning plan objects to `rollfast.finetune` optimizer
compilation and step-building primitives without adding hard third-party dependencies to Rollfast's base import path.

## Interface

- Submodules (`[__init__.py:__all__]`):
  - `equimo` (`[__init__.py:equimo]`): Structural adapter helpers for Equimo-compatible fine-tuning plans.
- Equimo Plan Adapters (`[equimo.py:__all__]`):
  - `from_equimo_plan(plan: Any, **kwargs: Any) -> OptimizerBundle` (`[equimo.py:from_equimo_plan]`): Validates
    structural plan attributes and compiles a `rollfast.finetune.OptimizerBundle` via `compile_optimizer`.
  - `adamw_from_equimo_plan(plan: Any, **kwargs: Any) -> OptimizerBundle` (`[equimo.py:adamw_from_equimo_plan]`):
    Validates structural plan attributes and compiles a grouped AdamW `OptimizerBundle` via `adamw_from_plan`.
  - `make_equimo_update_step(plan: Any, loss_fn: Any, optimizer: OptimizerBundle, **kwargs: Any)`
    (`[equimo.py:make_equimo_update_step]`): Validates structural plan attributes plus `combine` support and delegates
    to `make_plan_update_step`.
  - `transition_lpft_stage(plan: Any, **kwargs: Any) -> OptimizerBundle` (`[equimo.py:transition_lpft_stage]`): Compiles
    a fresh `OptimizerBundle` for a prepared linear-probe-to-fine-tuning (LP-FT) stage via `from_equimo_plan`.

## Invariants

- Structural plan validation (`[equimo.py:_validate_equimo_like_plan]`): `plan` passed to `from_equimo_plan`,
  `adamw_from_equimo_plan`, `make_equimo_update_step`, or `transition_lpft_stage` must expose `trainable`, `frozen`,
  `labels`, `group_specs`, and `identities` attributes; missing attributes raise `TypeError` listing their names.
- Combinable plan requirement (`[equimo.py:_require_combine]`): `plan` passed to `make_equimo_update_step` must
  additionally expose a `combine` attribute (`hasattr(plan, "combine")`), raising `TypeError` if absent.
- Optional-dependency isolation (`[equimo.py:__all__]`): `equimo.py` uses structural duck-typing (`typing.Any`) and
  imports exclusively from `rollfast.finetune` (`OptimizerBundle`, `adamw_from_plan`, `compile_optimizer`,
  `make_plan_update_step`), never importing `equimo` or `equinox` at module import time.

## Side Effects

- None (`[equimo.py:__all__]`).

## Verification

- Run: `uv run pytest tests/integrations tests/test_equimo_integration.py`
