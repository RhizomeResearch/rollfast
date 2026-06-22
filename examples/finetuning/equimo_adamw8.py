"""Compile Rollfast AdamW with blockwise 8-bit state for an Equimo plan."""

import jax

import equimo.finetune as eqft
import equimo.vision.models as em
import rollfast.finetune as rfft
from rollfast.optim.adam8 import tree_state_nbytes


key = jax.random.PRNGKey(0)
model = em.vit_tiny_patch16_224(num_classes=10, key=key)
plan = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(mode="full"),
    labels=eqft.LLRDConfig(decay=0.75),
)

optim = rfft.adamw8_from_plan(
    plan,
    total_steps=20_000,
    base_lr=5e-4,
    weight_decay=0.05,
    state_quantization=rfft.StateQuantizationConfig(
        enabled=True,
        block_size=2048,
        min_size=4096,
    ),
)
opt_state = optim.init(plan.trainable)

print(optim.report.group_table())
print({"optimizer_state_bytes": tree_state_nbytes(opt_state)})
