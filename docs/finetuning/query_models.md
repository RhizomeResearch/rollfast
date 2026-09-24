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
points, pixels, or another explicit denominator. Query classification, mask BCE, and Dice can have different
denominators: sum each component's numerators and denominators across the physical batch before dividing and applying
its coefficient. Averaging per-image scalar losses gives different weighting. A weighted sum of these ratios cannot
generally use one `LossBundle.normalizer` for exact accumulation. With Equimo's query-training utilities, use
`reduce_query_losses` on the vmapped results. Its named-axis reduction combines loss statistics; the caller still owns
parameter-gradient aggregation.

Exclude entirely ignored and padded examples from batch-dependent normalization. Remove them before the forward or use
the model library's explicit normalization-validity argument; setting their loss weight to zero is insufficient. For
PMT's validity-aware BatchNorm route, pass `example_valid=targets.supervised` per image when wholly unlabelled images
should not participate, and supply finite padded image inputs. A supervised example with an empty target set still
supplies no-object classification loss. If the whole batch has no supervision, do not call the update step, even when
the loss library returns zero. This leaves optimizer state, schedule position, model state, and RNG unchanged. If
accumulation is needed later, qualify each loss denominator and normalization-state evolution at the chosen physical and
effective batch sizes.

Prepare assignments and sampled-point context from the same forward whose predictions enter the differentiated loss.
Equimo's `prepare_query_loss` and `query_segmentation_loss` can run together inside that loss callback; preparation
stops differentiation through matching and point selection and does not require a second model forward. Return the
proposed normalization state with the loss and commit it once through the update helper. Keep images, targets, and
incoming model state as explicit compiled-step arguments. If a forward is intentionally recomputed, replay its keys,
mask state, sampling context, and incoming normalization state. An invalid active target or prediction produces an
invalid result and nonfinite total; return that total so the update helper rejects the attempt instead of turning it
into a zero-loss update.

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
