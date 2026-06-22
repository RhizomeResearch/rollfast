"""Build grouped AdamW with EMA evaluation params for an Equimo plan."""

import jax
import jax.numpy as jnp
import optax

import equimo.finetune as eqft
import equimo.vision.models as em
import rollfast.finetune as rfft


key = jax.random.PRNGKey(0)
model = em.vit_tiny_patch16_224(num_classes=10, key=key)
plan = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(mode="head"),
)

optim = rfft.adamw_from_plan(
    plan,
    total_steps=2_000,
    base_lr=1e-3,
    weight_decay=0.0,
    ema=rfft.EMAConfig(enabled=True, decay=0.999),
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
eval_model = plan.combine(optim.eval_params(trainable, opt_state, view="ema"))
del eval_model

print(loss, optim.eval_views)
