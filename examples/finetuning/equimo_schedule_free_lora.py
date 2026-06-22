"""Build grouped Schedule-Free Adam for an Equimo LoRA plan."""

import jax
import jax.numpy as jnp
import optax

import equimo.finetune as eqft
import equimo.vision.models as em
import rollfast.finetune as rfft


key = jax.random.PRNGKey(0)
model = em.vit_tiny_patch16_224(num_classes=10, key=key)
lora_model = eqft.apply_lora(
    model,
    eqft.LoRAConfig(rank=8, alpha=16.0),
    key=jax.random.PRNGKey(1),
)
plan = eqft.prepare_finetune(
    lora_model,
    trainable=eqft.TrainableSpec(mode="peft", method_name="lora"),
)

optim = rfft.schedule_free_adam_from_plan(
    plan,
    total_steps=10_000,
    base_lr=2e-4,
    weight_decay=0.0,
    lora_b_lr_ratio=16.0,
)
opt_state = optim.init(plan.trainable)


def loss_fn(trainable, x, y):
    model = plan.combine(trainable)
    logits = jax.vmap(lambda image: model(image, key=key, inference=False))(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


x = jnp.ones((2, 3, 224, 224), dtype=jnp.float32)
y = jnp.array([0, 1], dtype=jnp.int32)
loss, grads = jax.value_and_grad(loss_fn)(plan.trainable, x, y)
updates, opt_state = optim.update(grads, opt_state, plan.trainable)
trainable = optax.apply_updates(plan.trainable, updates)

# Use eval params for validation/checkpointing with Schedule-Free optimizers.
eval_model = plan.combine(optim.eval_params(trainable, opt_state))
del eval_model

print(loss, optim.report.group_table())
