"""Migrate optimizer state from a linear probe to full fine-tuning."""

import jax

import equimo.finetune as eqft
import equimo.vision.models as em
import rollfast.finetune as rfft


key = jax.random.PRNGKey(0)
model = em.vit_tiny_patch16_224(num_classes=10, key=key)

stage1 = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(mode="head"),
)
stage1_optim = rfft.adamw_from_plan(
    stage1,
    total_steps=1_000,
    base_lr=1e-3,
    schedule="constant",
    weight_decay=0.0,
    clip_global_norm=None,
)
stage1_state = stage1_optim.init(stage1.trainable)

stage2 = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(mode="full"),
    labels=eqft.LLRDConfig(decay=0.75),
)
stage2_optim = rfft.adamw_from_plan(
    stage2,
    total_steps=20_000,
    base_lr=5e-4,
    schedule="warmup_cosine",
    weight_decay=0.05,
    clip_global_norm=1.0,
)

_, stage2_state, migration = rfft.reconfigure_optimizer(
    old_plan=stage1,
    old_bundle=stage1_optim,
    old_state=stage1_state,
    new_plan=stage2,
    new_bundle=stage2_optim,
    state_policy="preserve_shared",
    counter_policy="restart_schedule",
)
del stage2_state

print(
    {
        "preserved_state_leaves": len(migration.preserved_state_leaves),
        "initialized_state_leaves": len(migration.initialized_state_leaves),
        "dropped_state_leaves": len(migration.dropped_state_leaves),
        "preserved_param_leaves": len(migration.preserved_param_leaves),
        "initialized_param_leaves": len(migration.initialized_param_leaves),
        "old_state_bytes": migration.old_state_bytes,
        "new_state_bytes": migration.new_state_bytes,
        "counter_policy": migration.counter_policy,
    }
)
