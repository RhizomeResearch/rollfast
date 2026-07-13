# Migrating from Rollfast 0.x to 1.0

Rollfast 1.0 establishes the first compatibility contract. Review the items
below before resuming an existing training job.

## Runtime Requirements

- Use Python 3.12 or newer.
- Use JAX `>=0.6.2` and Optax `>=0.2.8,<0.3.0`.
- Install `rollfast[equinox]` only when using the optional Equinox helpers.
  Equimo remains a development/integration dependency rather than a core
  import requirement.

## Imports

Existing top-level optimizer imports remain available. New fine-tuning APIs are
under `rollfast.finetune`, optional adapters are under
`rollfast.integrations`, and implementation modules beginning with `_` are
private. The exact v1 public boundary is defined in
[`stability.md`](./stability.md).

## Optimizer State and Checkpoints

Do not assume that a raw optimizer-state PyTree created by 0.x can be restored
into a 1.0 optimizer. Reinitialize optimizer state when restarting is
acceptable, or explicitly migrate and validate application checkpoints before
resuming training.

For plan-aware fine-tuning, use `make_state_checkpoint` and
`restore_state_checkpoint`. The restore path validates schema version, plan
fingerprint, optimizer method, state structure, and sharding metadata. Pickle
helpers require `trusted=True` and must not be used with untrusted files.

## Schedule-Free Evaluation

Schedule-Free optimizers maintain training and averaged parameter sequences.
Continue applying updates to the training parameters and use
`schedule_free_eval_params`, or an `OptimizerBundle` evaluation view, for
validation and evaluation checkpoints.

## Structured Optimizers

Muon, PRISM, Aurora, PSGD/Kron, Pion, RMNP, and related wrappers route leaves by
shape or explicit dimension-number specifications. Verify routing reports when
moving a convolutional or otherwise high-rank model from 0.x; do not assume all
rank-greater-than-two tensors enter the structured branch automatically.

## Validation Before Resuming Training

1. Initialize the optimizer and inspect its report or state structure.
2. Run one eager and one JIT-compiled update on representative data.
3. Check that updates and state are finite in the intended dtype.
4. On sharded workloads, verify placement and collectives on the target
   hardware.
5. Compare a short resumed run against a fresh 1.0 baseline before committing a
   long training job.
