from types import SimpleNamespace

import equimo.finetune as eqft
import equimo.finetune.vision as eqv
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

import rollfast.finetune as rfft
from rollfast.integrations import equimo
from tests.finetune.helpers import tiny_plan


@pytest.mark.integration
def test_real_equimo_plan_compiles_and_updates_finitely():
    key = jax.random.PRNGKey(0)
    model = eqx.nn.MLP(4, 2, 8, 2, key=key)
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(mode="full"),
    )
    optimizer = equimo.adamw_from_equimo_plan(
        plan,
        total_steps=10,
        base_lr=1e-3,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
    )
    opt_state = optimizer.init(plan.trainable)
    x = jnp.ones((4,), dtype=jnp.float32)

    def loss_fn(trainable):
        prediction = plan.combine(trainable)(x)
        return jnp.mean(jnp.square(prediction))

    loss, grads = jax.value_and_grad(loss_fn)(plan.trainable)
    updates, _ = optimizer.update(grads, opt_state, plan.trainable)
    updated = optax.apply_updates(plan.trainable, updates)

    assert jnp.isfinite(loss)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(updated))


def test_equimo_integration_accepts_structural_plan_without_importing_equimo():
    bundle = equimo.adamw_from_equimo_plan(
        tiny_plan(),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )

    assert bundle.report.trainable_params == 14


def test_equimo_compiler_integration_does_not_require_combine():
    source = tiny_plan()
    plan = SimpleNamespace(
        trainable=source.trainable,
        frozen=source.frozen,
        labels=source.labels,
        group_specs=source.group_specs,
        identities=source.identities,
    )

    bundle = equimo.adamw_from_equimo_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )

    assert bundle.report.trainable_params == 14


def test_equimo_update_integration_requires_combine():
    source = tiny_plan()
    plan = SimpleNamespace(
        trainable=source.trainable,
        frozen=source.frozen,
        labels=source.labels,
        group_specs=source.group_specs,
        identities=source.identities,
    )

    optimizer = equimo.adamw_from_equimo_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )

    with pytest.raises(TypeError, match="combine"):
        equimo.make_equimo_update_step(plan, lambda _: 0.0, optimizer)


class _TinyVision(eqx.Module):
    weight: jax.Array

    def features(self, image):
        return image @ self.weight

    def feature_metadata(self, image, *, endpoint, endpoint_options):
        del image, endpoint, endpoint_options
        return {
            "grid_size": (2, 2),
            "tokens_include_prefix": False,
            "patch_size": (1, 1),
        }


@pytest.mark.integration
@pytest.mark.parametrize("task", ("classification", "dense"))
@pytest.mark.parametrize("trainability", ("head", "full"))
def test_safe_checkpoint_resumes_equimo_probe_plans(tmp_path, task, trainability):
    backbone = _TinyVision(jnp.ones((2, 3), dtype=jnp.float32))
    key = jax.random.key(0)
    spec = eqft.FeatureSpec(
        endpoint="features",
        output_layout="BNC",
        token_selection="patches",
        pooling="mean_patch" if task == "classification" else "none",
        return_metadata=True,
    )
    if task == "classification":
        model = eqft.make_linear_probe(
            backbone, in_features=3, out_features=2, key=key, feature_spec=spec
        )
    else:
        model = eqv.make_dense_probe(
            backbone, in_features=3, out_features=3, key=key, feature_spec=spec
        )
    plan = eqft.prepare_finetune(model, trainable=eqft.TrainableSpec(mode=trainability))
    bundle = rfft.adamw_from_plan(
        plan, total_steps=4, schedule="constant", clip_global_norm=None
    )
    image = jnp.ones((4, 2), dtype=jnp.float32)

    def loss_fn(trainable):
        output = plan.combine(trainable)(image)
        return jnp.mean(jnp.square(output))

    grads = jax.grad(loss_fn)(plan.trainable)
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(grads, state, plan.trainable)
    trainable = optax.apply_updates(plan.trainable, updates)
    assert plan.combine(trainable)(image).shape == (
        (2,) if task == "classification" else (3, 2, 2)
    )

    checkpoint = rfft.make_state_checkpoint(
        bundle,
        state,
        model_checkpoint_id="training-model-1",
        counters={"successful_update": 1},
    )
    template = rfft.make_state_checkpoint(
        bundle,
        bundle.init(plan.trainable),
        model_checkpoint_id="training-model-1",
        counters={"successful_update": 0},
    )
    limits = rfft.ArrayCheckpointLimits(
        2_000_000, 10_000, 20_000, 10_000_000, 100_000_000
    )
    path = tmp_path / "optimizer.rfarr"
    rfft.save_array_state_checkpoint(path, checkpoint, limits=limits)
    restored = rfft.load_array_state_checkpoint(
        path,
        bundle,
        template=template,
        model_checkpoint_id="training-model-1",
        applied_updates=1,
        limits=limits,
    )

    next_grads = jax.grad(loss_fn)(trainable)
    expected, _ = jax.jit(bundle.update)(next_grads, state, trainable)
    actual, _ = jax.jit(bundle.update)(next_grads, restored.state, trainable)
    for left, right in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        assert jnp.allclose(left, right, rtol=1e-6, atol=1e-7)
