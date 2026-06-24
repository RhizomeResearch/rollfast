import jax
import jax.numpy as jnp
import optax
import pytest

import rollfast.finetune as rfft
from rollfast.optim.sam import add_perturbation, sam_perturbation

from .helpers import tiny_plan


def _tree_allclose(left, right):
    pairs = zip(
        jax.tree.leaves(left, is_leaf=lambda x: x is None),
        jax.tree.leaves(right, is_leaf=lambda x: x is None),
        strict=True,
    )
    for lhs, rhs in pairs:
        if lhs is None:
            assert rhs is None
        else:
            assert jnp.allclose(lhs, rhs, rtol=1e-6, atol=1e-6)


def _loss(model):
    leaves = [
        leaf
        for leaf in jax.tree.leaves(model, is_leaf=lambda x: x is None)
        if leaf is not None
    ]
    return sum(jnp.sum((leaf - 0.25) ** 2) for leaf in leaves)


def _scaled_loss(model, batch):
    leaves = [
        leaf
        for leaf in jax.tree.leaves(model, is_leaf=lambda x: x is None)
        if leaf is not None
    ]
    target = batch["target"]
    return sum(jnp.sum((leaf - target) ** 2) for leaf in leaves)


def test_make_sam_step_matches_manual_two_pass_update():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        base_lr=1e-2,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
    )
    config = rfft.SAMConfig(rho=0.05)
    state = bundle.init(plan.trainable)
    value_and_grad = jax.value_and_grad(
        lambda trainable: _loss(plan.combine(trainable))
    )
    first_loss, grads = value_and_grad(plan.trainable)
    perturbation, perturbation_norm = sam_perturbation(
        grads,
        params=plan.trainable,
        rho=config.rho,
        adaptive=config.adaptive,
        eta=config.eta,
        eps=config.eps,
    )
    perturbed = add_perturbation(plan.trainable, perturbation)
    second_loss, second_grads = value_and_grad(perturbed)
    updates, manual_state = bundle.update(second_grads, state, plan.trainable)
    manual_trainable = optax.apply_updates(plan.trainable, updates)

    step = rfft.make_sam_step(
        plan=plan,
        base_optimizer=bundle,
        config=config,
        loss_fn=_loss,
    )
    stepped_trainable, stepped_state, info = step(plan.trainable, state)

    _tree_allclose(stepped_trainable, manual_trainable)
    assert stepped_state is not None
    assert manual_state is not None
    assert jnp.allclose(info.loss, first_loss)
    assert jnp.allclose(info.perturbed_loss, second_loss)
    assert jnp.allclose(info.perturbation_norm, perturbation_norm)
    assert not jnp.allclose(
        stepped_trainable["blocks"][0]["w"],
        perturbed["blocks"][0]["w"],
    )


def test_make_sam_step_accumulates_microbatches_exactly():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        base_lr=1e-2,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
    )
    config = rfft.SAMConfig(rho=0.05)
    state = bundle.init(plan.trainable)
    batch = {"target": jnp.array([0.1, 0.4, -0.2], dtype=jnp.float32)}

    def accumulated_loss(trainable):
        losses = jax.vmap(
            lambda target: _scaled_loss(
                plan.combine(trainable),
                {"target": target},
            )
        )(batch["target"])
        return jnp.mean(losses)

    value_and_grad = jax.value_and_grad(accumulated_loss)
    _, grads = value_and_grad(plan.trainable)
    perturbation, _ = sam_perturbation(
        grads,
        params=plan.trainable,
        rho=config.rho,
        adaptive=config.adaptive,
        eta=config.eta,
        eps=config.eps,
    )
    perturbed = add_perturbation(plan.trainable, perturbation)
    _, second_grads = value_and_grad(perturbed)
    updates, manual_state = bundle.update(second_grads, state, plan.trainable)
    manual_trainable = optax.apply_updates(plan.trainable, updates)

    step = rfft.make_sam_step(
        plan=plan,
        base_optimizer=bundle,
        config=config,
        loss_fn=_scaled_loss,
        microbatch_axis=0,
    )
    stepped_trainable, stepped_state, info = step(plan.trainable, state, batch)

    _tree_allclose(stepped_trainable, manual_trainable)
    assert stepped_state is not None
    assert manual_state is not None
    assert jnp.allclose(info.loss, accumulated_loss(plan.trainable))
    assert jnp.allclose(info.perturbed_loss, accumulated_loss(perturbed))


def test_make_sam_step_is_jittable():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
    )
    state = bundle.init(plan.trainable)
    step = jax.jit(
        rfft.make_sam_step(
            plan=plan,
            base_optimizer=bundle,
            config=rfft.SAMConfig(rho=0.05),
            loss_fn=_loss,
        )
    )

    trainable, state, info = step(plan.trainable, state)

    assert state is not None
    assert info.perturbed_loss > info.loss
    assert trainable["embed"] is None


def test_make_sam_step_rejects_ambiguous_accumulation():
    plan = tiny_plan()
    bundle = rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        accumulation_steps=2,
    )

    with pytest.raises(ValueError, match="accumulation_steps=1"):
        rfft.make_sam_step(
            plan=plan,
            base_optimizer=bundle,
            config=rfft.SAMConfig(rho=0.05),
            loss_fn=_loss,
        )


def test_sam_cost_report_states_two_pass_cost_and_storage():
    plan = tiny_plan()
    report = rfft.sam_cost_report(plan, rfft.SAMConfig(rho=0.05))

    assert report["method"] == "SAM"
    assert report["forward_backward_evaluations"] == 2
    assert report["perturbation_bytes"] == 14 * 4


def test_sam_cost_report_honors_bias_exclusion():
    plan = tiny_plan()
    report = rfft.sam_cost_report(
        plan,
        rfft.SAMConfig(rho=0.05, perturb_bias=False),
    )

    assert report["perturbation_bytes"] == 10 * 4
