"""Configure Rollfast ASAM from an Equimo fine-tuning plan."""

import equinox as eqx
import jax

import equimo.finetune as eqft
import rollfast.finetune as rfft


key = jax.random.PRNGKey(0)
model = eqx.nn.MLP(4, 2, 8, 2, key=key)
plan = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(mode="full"),
)

base = rfft.adamw_from_plan(
    plan,
    total_steps=1_000,
    base_lr=1e-3,
    schedule="constant",
    weight_decay=0.01,
    clip_global_norm=None,
)

asam = rfft.SAMConfig(
    rho=0.5,
    adaptive=True,
    eta=0.01,
)

print(rfft.sam_cost_report(plan, asam))
print(base.report.group_table())
