import jax
import jax.numpy as jnp
import pytest

import rollfast.finetune as rfft


def test_adalora_budget_is_monotonic_and_reaches_target():
    controller = rfft.make_adalora_controller(
        {"adapter_a": 4, "adapter_b": 4},
        total_steps=10,
        config=rfft.AdaLoRAControllerConfig(
            initial_budget=8,
            target_budget=4,
            t_init=2,
            t_final=2,
            allocation_interval=1,
        ),
    )
    budgets = [int(controller.budget_at(step)) for step in range(11)]

    assert budgets[:3] == [8, 8, 8]
    assert budgets[-2:] == [4, 4]
    assert budgets == sorted(budgets, reverse=True)


def test_adalora_allocation_prefers_high_scores_with_static_masks():
    scores = jnp.array(
        [
            [0.1, 0.2, 0.3, 10.0],
            [0.1, 9.0, 8.0, 7.0],
        ],
        dtype=jnp.float32,
    )
    mask = rfft.allocate_rank_mask(
        scores,
        max_ranks=(4, 4),
        budget=4,
        min_rank=1,
    )

    assert mask.shape == (2, 4)
    assert jnp.array_equal(mask[0], jnp.array([True, False, False, True]))
    assert jnp.array_equal(mask[1], jnp.array([True, True, False, False]))


def test_adalora_triplet_importance_matches_reference_equation():
    p = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    lambda_ = jnp.asarray([2.0, 4.0], dtype=jnp.float32)
    q = jnp.asarray([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=jnp.float32)
    grad_p = jnp.full_like(p, 0.5)
    grad_lambda = jnp.asarray([0.1, 0.2], dtype=jnp.float32)
    grad_q = jnp.full_like(q, 0.1)

    scores = rfft.adalora_triplet_importance(
        p,
        lambda_,
        q,
        grad_p,
        grad_lambda,
        grad_q,
    )

    assert scores[0] == pytest.approx(1.5)
    assert scores[1] == pytest.approx(2.7)


def test_adalora_triplet_importance_rejects_rank_mismatch():
    with pytest.raises(ValueError, match="rank dimension"):
        rfft.adalora_triplet_importance(
            jnp.ones((2, 2)),
            jnp.ones((3,)),
            jnp.ones((2, 2)),
            jnp.ones((2, 2)),
            jnp.ones((3,)),
            jnp.ones((2, 2)),
        )


def test_adalora_skipped_update_does_not_advance_controller():
    controller = rfft.make_adalora_controller(
        {"adapter": 4},
        total_steps=2,
        config=rfft.AdaLoRAControllerConfig(
            initial_budget=4,
            target_budget=2,
            t_init=0,
            t_final=0,
            allocation_interval=1,
        ),
    )
    state = controller.init()
    scores = jnp.ones((1, 4), dtype=jnp.float32)
    skipped = controller.update(state, scores, applied=False)
    applied = controller.update(state, scores, applied=True)

    assert int(skipped.last_allocation_step) == 0
    assert int(skipped.current_budget) == int(state.current_budget)
    assert int(applied.last_allocation_step) == 1
    assert int(applied.current_budget) < int(state.current_budget)


def test_adalora_uncertainty_uses_updated_sensitivity_ema():
    controller = rfft.make_adalora_controller(
        {"adapter": 1},
        total_steps=4,
        config=rfft.AdaLoRAControllerConfig(
            initial_budget=1,
            target_budget=1,
            t_init=0,
            t_final=0,
            allocation_interval=1,
            beta_sensitivity=0.5,
            beta_uncertainty=0.5,
        ),
    )
    state = controller.init()

    state = controller.update(state, jnp.asarray([[2.0]], dtype=jnp.float32))

    assert state.sensitivity_ema[0, 0] == pytest.approx(1.0)
    assert state.uncertainty_ema[0, 0] == pytest.approx(0.5)


def test_adalora_final_support_freezes_after_entering_final_phase():
    controller = rfft.make_adalora_controller(
        {"adapter": 2},
        total_steps=4,
        config=rfft.AdaLoRAControllerConfig(
            initial_budget=2,
            target_budget=1,
            t_init=0,
            t_final=2,
            allocation_interval=1,
            min_rank=0,
            beta_sensitivity=0.0,
            beta_uncertainty=0.0,
            final_support_frozen=True,
        ),
    )
    state = controller.init()
    state = controller.update(state, jnp.asarray([[10.0, 1.0]], dtype=jnp.float32))
    state = controller.update(state, jnp.asarray([[1.0, 10.0]], dtype=jnp.float32))
    frozen_support = state.current_support
    state = controller.update(state, jnp.asarray([[10.0, 1.0]], dtype=jnp.float32))

    assert int(state.current_budget) == 1
    assert jnp.array_equal(frozen_support, jnp.asarray([[False, True]]))
    assert jnp.array_equal(state.current_support, frozen_support)


def test_adalora_update_is_jittable_and_mask_shape_is_static():
    controller = rfft.make_adalora_controller(
        {"adapter_a": 4, "adapter_b": 3},
        total_steps=10,
        config=rfft.AdaLoRAControllerConfig(
            initial_budget=7,
            target_budget=4,
            t_init=0,
            t_final=0,
            allocation_interval=1,
        ),
    )
    state = controller.init()
    scores = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    update = jax.jit(lambda state, scores: controller.update(state, scores))

    state = update(state, scores)
    pattern = controller.rank_pattern(state)

    assert state.current_support.shape == (2, 4)
    assert pattern["adapter_b"].shape == (3,)
    assert sum(int(rank) for rank in state.ranks) == int(state.current_budget)


def test_adalora_controller_config_round_trip_and_validation():
    config = rfft.AdaLoRAControllerConfig(
        initial_budget=12,
        target_budget=8,
        t_init=2,
        t_final=3,
        allocation_interval=4,
    )

    assert rfft.AdaLoRAControllerConfig.from_dict(config.to_dict()) == config
    assert config.t_init == 2
    assert config.t_final == 3
    assert config.allocation_interval == 4
    with pytest.raises(ValueError, match="target_budget"):
        rfft.AdaLoRAControllerConfig(target_budget=0)
