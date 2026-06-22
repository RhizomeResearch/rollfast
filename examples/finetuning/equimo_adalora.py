"""Build a Rollfast AdaLoRA rank controller for Equimo AdaLoRA modules."""

import equinox as eqx
import jax
import jax.numpy as jnp

import equimo.finetune as eqft
import rollfast.finetune as rfft


key = jax.random.PRNGKey(0)
model = eqx.nn.MLP(4, 2, 8, 2, key=key)
adalora_model = eqft.apply_adalora(
    model,
    eqft.AdaLoRAConfig(
        target=eqft.TargetSpec(predicate=eqft.is_linear),
        rank=4,
    ),
    key=jax.random.PRNGKey(1),
)

rank_groups = eqft.lora_rank_groups(adalora_model)
controller = rfft.make_adalora_controller(
    rank_groups,
    total_steps=1_000,
    config=rfft.AdaLoRAControllerConfig(
        initial_budget=4,
        target_budget=2,
        min_rank=1,
    ),
)
state = controller.init()

# User code supplies rank-importance scores from adapter statistics.
scores = jnp.ones((controller.group_count, controller.max_rank), dtype=jnp.float32)
state = controller.update(state, scores)
rank_pattern = controller.rank_pattern(state)
adalora_model = eqft.apply_lora_rank_pattern(adalora_model, rank_pattern)

print(controller.report())
print(controller.rank_report(state))
print(eqft.lora_rank_groups(adalora_model))
