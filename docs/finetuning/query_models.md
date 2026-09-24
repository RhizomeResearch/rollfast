# Query Models and Stateful Updates

Query-based segmentation plans can select learned queries, lateral layers, decoder blocks, prediction heads, and encoder
blocks at the same time. Compile the model library's plan directly. Inspect `plan.report.parameters` to see each
selected logical ID, physical path, label, and recorded depth; compare those rows with `bundle.report.policy_table()`
and `bundle.report.group_table()` before training. In a frozen-encoder plan, confirm that no encoder leaf is trainable
and that the optimizer state contains no encoder moments. When the recipe requests layer-wise rates, use the plan's
depth metadata and inspect the resolved effective rate for every encoder depth. Generic model buffers and mutable
normalization state belong to the model, not the optimizer.

For attention-mask annealing, derive the controller position from `step_state.counters.successful_update` **before** the
next update. The first update has position zero. Pass that position to the loss function, where the model library builds
its mask state. A rejected update leaves the position unchanged, so the next attempt uses the same annealing position.
Split the loss function's `forward_key` for point sampling and mask removal in a fixed order; save the resulting
`RNGStreams` with the optimizer state to reproduce the next stochastic forward. If these operations use separate caller
owned keys, checkpoint those keys too.

Start with a physical batch and `accumulation_steps=1`. Declare whether each loss term is normalized by valid examples,
points, pixels, or another explicit denominator. Remove entirely ignored and padded examples before any batch-dependent
normalization; setting their loss weight to zero still changes BatchNorm statistics. If the whole batch has no valid
target, do not call the update step. This leaves optimizer state, schedule position, model state, and RNG unchanged. If
accumulation is needed later, qualify the loss denominator and normalization-state evolution at the chosen physical and
effective batch sizes.

For mutable normalization, pass the state supplied by the model library to `init_finetune_step_state` and use
`state_policy="microbatch_sequential"` with `make_finetune_update_step`. A successful step commits the proposed forward
state; a nonfinite step discards it with the parameter update and forward-key advance. Save the complete step state:
optimizer state, any master parameters or loss scale, accumulation state, model state, counters, and RNG. The optimizer
checkpoint must identify the matching model-parameter artifact. On restore, rebuild the same plan and bundle, load the
model artifact and matching checkpoint, and resume from the saved successful-update count. The array reader's trusted
restore template must have the same optional component structure as the saved step, including master parameters if the
step helper has materialized them. For evaluation, call `bind_evaluation_view` with an explicit `optimizer` or `ema`
view and the matching model-state snapshot; the selected parameter view does not create a different normalization-state
snapshot.

See [state and checkpoints](state_and_checkpoints.md) for save and restore examples and [averaging](averaging.md) for
evaluation views.
