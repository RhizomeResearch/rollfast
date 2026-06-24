"""Compile Rollfast Aurora/Adam for an Equimo fine-tuning plan."""

import jax

import equimo.finetune as eqft
import equimo.vision.models as em
import rollfast.finetune as rfft


key = jax.random.PRNGKey(0)
model = em.vit_tiny_patch16_224(num_classes=10, key=key)
plan = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(mode="head"),
)

optim = rfft.hybrid_aurora_adam_from_plan(
    plan,
    total_steps=2_000,
    base_lr=1e-3,
    weight_decay=0.0,
    polar_ns_iters=6,
)
opt_state = optim.init(plan.trainable)
del opt_state

print(optim.report.group_table())
