# Fine-Tuning Compatibility and Migration

Rollfast accepts a structural fine-tuning plan: `trainable`, `frozen`, `labels`, `group_specs`, and `identities`. The
trees must align, and trainable leaves need stable logical IDs. Compilation does not import Equimo; plan-aware update
helpers additionally require `combine(trainable)`. Equimo is an optional integration dependency. The declared
development range in `pyproject.toml` is an API range, not a claim that every release in it has been tested. Record the
exact Python, Equimo, Equinox, JAX, jaxlib, Optax, and Rollfast versions used for each training run.

## Moving to the current fine-tuning APIs

Use `rollfast.finetune.adamw_from_plan` or `compile_optimizer` for grouped plans. `GroupRule(required=True)` rejects a
missing required selector; inspect `plan.report.parameters` and `bundle.report.policy_table()` to confirm the selected
logical IDs and effective learning rates and decay. The fine-tuning AdamW default is `eps=1e-6`. Schedule previews use
the first applied update at index `0` and the final planned update at `total_steps - 1`.

Choose an evaluation view explicitly. Ordinary training parameters use `view="optimizer"`; `ema`, `swa`, and
`schedule_free` require their corresponding state. `bind_evaluation_view` associates the view with the model checkpoint,
optimizer checkpoint, model-state snapshot, and applied-update count. Changing the view changes the evaluated parameter
set. For accumulation, `bundle.update_with_status` distinguishes accumulating microsteps, applied updates, and discarded
windows. A skipped batch requires the caller to omit the optimizer update. See [configuration](configuration.md) and
[averaging](averaging.md) for the supported policies.

## Choosing a checkpoint reader

The `rollfast.finetune` logical checkpoint uses schema version 1. `make_state_checkpoint` packages the optimizer state
and manifest; `restore_state_checkpoint` checks the schema, plan fingerprint, effective group table, optimizer settings,
model identity, and declared state components. Recent checkpoints include a `state_schema` description. When restoring
an older logical checkpoint without that description, supply a freshly built `template` to check its tree. The
trusted-local pickle helper requires `trusted=True` to load; it is not a reader for untrusted input.

`save_array_state_checkpoint` and `load_array_state_checkpoint` use a separate version 1 array format. Loading requires
a template rebuilt from the same compiled plan, explicit allocation limits, and the matching model identity. The reader
checks the array layout and checksums and requires the original process count and equivalent sharding. It does not read
pickle files or arbitrary Equinox model checkpoints. See [state and checkpoints](state_and_checkpoints.md) for the
complete save and restore example.

An older training application may have stored `(model, optimizer_state, random_key)` directly with Equinox rather than
using a Rollfast fine-tuning checkpoint. Its own reader, model template, configuration, and dependency lock define that
format. Retain that pinned reader for historical resume or export. Do not pass its raw optimizer state to
`restore_state_checkpoint` or treat a freshly initialized optimizer as a resumed run. New training runs can use the
current fine-tuning checkpoint APIs without changing historical artifacts. If conversion is needed later, qualify it
against actual saved artifacts and compare the next update and random state before relying on it.
