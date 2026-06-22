"""Build a Rollfast AdaLoRA rank controller for Equimo rank-masked LoRA."""

import equinox as eqx
import jax
import jax.numpy as jnp

import equimo.finetune as eqft
import rollfast.finetune as rfft


key = jax.random.PRNGKey(0)
model = eqx.nn.MLP(4, 2, 8, 2, key=key)
lora_model = eqft.apply_lora(
    model,
    eqft.RankMaskedLoRAConfig(
        target=eqft.TargetSpec(predicate=eqft.is_linear),
        rank=4,
        initial_rank=4,
        target_rank=2,
        min_rank=1,
        max_rank=4,
    ),
    key=jax.random.PRNGKey(1),
)

rank_groups = eqft.lora_rank_groups(lora_model)
controller = rfft.make_adalora_controller(
    rank_groups,
    total_steps=1_000,
    config=rfft.AdaLoRAControllerConfig(
        enabled=True,
        init_rank=4,
        target_rank=2,
    ),
)
state = controller.init()

# User code supplies rank-importance scores from adapter statistics.
scores = jnp.ones((controller.group_count, controller.max_rank), dtype=jnp.float32)
state = controller.update(state, scores)
rank_pattern = controller.rank_pattern(state)
lora_model = eqft.apply_lora_rank_pattern(lora_model, rank_pattern)

print(controller.report())
print(controller.rank_report(state))
print(eqft.lora_rank_groups(lora_model))
