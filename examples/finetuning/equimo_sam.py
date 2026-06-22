"""Run one Rollfast SAM step from an Equimo fine-tuning plan."""

import equinox as eqx
import jax
import jax.numpy as jnp

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
opt_state = base.init(plan.trainable)


def loss_fn(model, x, y):
    pred = jax.vmap(model)(x)
    return jnp.mean((pred - y) ** 2)


step = rfft.make_sam_step(
    plan=plan,
    base_optimizer=base,
    config=rfft.SAMConfig(rho=0.05),
    loss_fn=loss_fn,
)

x = jnp.ones((3, 4), dtype=jnp.float32)
y = jnp.zeros((3, 2), dtype=jnp.float32)
trainable, opt_state, info = step(plan.trainable, opt_state, x, y)
del trainable, opt_state

print(
    {
        "loss": float(info.loss),
        "perturbed_loss": float(info.perturbed_loss),
        "cost": rfft.sam_cost_report(plan, rfft.SAMConfig(rho=0.05)),
    }
)
