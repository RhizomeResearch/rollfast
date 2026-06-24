"""Print measured optimizer-state memory diagnostics for a fine-tuning bundle."""

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

optim = rfft.hybrid_kron_adam_from_plan(
    plan,
    total_steps=2_000,
    schedule="constant",
    weight_decay=0.0,
    clip_global_norm=None,
)
state = optim.init(plan.trainable)
summary = rfft.optimizer_state_memory_summary(optim, state)

print(
    {
        "total_bytes": summary.total_bytes,
        "estimated_state_bytes": summary.estimated_state_bytes,
        "by_category": summary.by_category,
        "preconditioner_factors": [
            factor.to_dict() for factor in summary.preconditioner_factors[:4]
        ],
    }
)
