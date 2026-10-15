# Rollfast Usage Guide

This guide covers the non-fine-tuning surface of Rollfast: Optax-compatible optimizers, schedule wrappers, parameter
PyTrees, and low-precision update helpers. Plan-aware fine-tuning utilities are documented separately under
[`docs/finetuning`](./finetuning).

## Optimizer Shape

Rollfast optimizers are Optax gradient transformations. They consume a parameter PyTree and return update PyTrees with
the same structure. Apply those updates with `rollfast.apply_updates`, explicitly choosing `stochastic=False` for
deterministic rounding.

```python
import jax
import jax.numpy as jnp
from rollfast import adamw, apply_updates

params = {"w": jnp.ones((4, 3)), "b": jnp.zeros((4,))}
grads = jax.tree.map(lambda x: jnp.full_like(x, 0.1), params)

optimizer = adamw(learning_rate=1e-2, weight_decay=0.01)
state = optimizer.init(params)
updates, state = optimizer.update(grads, state, params)
params = apply_updates(params, updates, stochastic=False)
```

## Structured Matrix Optimizers

Structured optimizers such as PRISM, Aurora, PSGD/Kron, Pion, and RMNP route matrix-like leaves to the structured branch
and scalar/vector leaves to a fallback Adam-style branch.

```python
from rollfast import aurora, prism

prism_tx = prism(learning_rate=1e-3, mode="bidirectional")
aurora_tx = aurora(learning_rate=3e-4, polar_ns_iters=12)
```

For Equinox modules or convolution kernels, use the exported dimension-number helpers when a high-rank tensor should be
flattened as a matrix:

```python
from rollfast import get_equinox_aurora_spec

optimizer = aurora(
    learning_rate=3e-4,
    aurora_weight_dimension_numbers=get_equinox_aurora_spec,
)
```

## Schedule-Free Evaluation

Schedule-Free optimizers maintain training parameters and an averaged evaluation view. Apply updates to the training
parameters, then call `schedule_free_eval_params` for validation or checkpointing.

```python
from rollfast import apply_updates, schedule_free_adam, schedule_free_eval_params

optimizer = schedule_free_adam(learning_rate=1e-2, total_steps=20)
state = optimizer.init(params)

updates, state = optimizer.update(grads, state, params)
params = apply_updates(params, updates, stochastic=False)
eval_params = schedule_free_eval_params(state, params)
```

## Low-Precision Updates

`apply_updates` and `apply_updates_prefix` support stochastic rounding for pure BF16 parameter updates. They default to
`stochastic=True`, which requires a PRNG key. Use `stochastic=False` for deterministic updates without a key, as in the
examples above. For stochastic rounding, split a fresh key on each invocation and retain its stream in training
checkpoints.

```python
import jax.random as jr
from rollfast import apply_updates

rounding_key = jr.PRNGKey(0)  # Initialize once, outside the training loop.
rounding_key, step_key = jr.split(rounding_key)
params = apply_updates(params, updates, key=step_key, stochastic=True)
```

Use `apply_updates_prefix` when updates are a prefix of an Equinox model PyTree.

Stochastic rounding rounds a finite value away from zero with probability equal to its distance from the truncated BF16
value, measured in BF16 spacings. Exactly representable values, signed zeros, infinities and NaNs keep their class;
values beyond the largest finite BF16 number may round to infinity. For plan-aware fine-tuning steps that own the
rounding stream, master parameters, and loss scaling, see [low-precision fine-tuning](finetuning/precision.md).

## Runnable Examples

- [`examples/adamw_quickstart.py`](../examples/adamw_quickstart.py)
- [`examples/schedule_free_eval.py`](../examples/schedule_free_eval.py)
- [`examples/finetuning/`](../examples/finetuning)
- [`examples/finetuning/low_precision.py`](../examples/finetuning/low_precision.py)
