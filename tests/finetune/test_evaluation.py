import pickle
from dataclasses import replace

import jax
import jax.numpy as jnp
import optax
import pytest

import rollfast.finetune as rfft

from .helpers import ones_like_trainable, tiny_plan


class CombiningPlan:
    def __init__(self, plan):
        self.trainable = plan.trainable
        self.frozen = plan.frozen
        self.labels = plan.labels
        self.group_specs = plan.group_specs
        self.identities = plan.identities
        self.model_state = plan.model_state

    def combine(self, trainable):
        return jax.tree.map(
            lambda selected, frozen: frozen if selected is None else selected,
            trainable,
            self.frozen,
            is_leaf=lambda x: x is None,
        )


def _bundle(plan, *, averaging=False):
    return rfft.adamw_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        ema=rfft.EMAConfig(enabled=averaging, decay=0.5),
        swa=rfft.SWAConfig(enabled=averaging, start_step=1),
    )


def test_raw_view_binds_checkpoint_without_averaging():
    base = tiny_plan()
    plan = CombiningPlan(
        replace(base, frozen={**base.frozen, "embed": jnp.array(42.0)})
    )
    bundle = _bundle(plan)
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(
        ones_like_trainable(plan.trainable), state, plan.trainable
    )
    params = optax.apply_updates(plan.trainable, updates)
    checkpoint = rfft.make_state_checkpoint(
        bundle,
        state,
        model_checkpoint_id="model-1",
        counters={"successful_update": 1},
    )

    result = rfft.bind_evaluation_view(
        bundle,
        plan,
        params,
        checkpoint,
        view="optimizer",
        model_checkpoint_id="model-1",
        optimizer_checkpoint_id="optimizer-1",
        applied_updates=1,
    )

    assert result.view == "optimizer"
    assert result.model_checkpoint_id == "model-1"
    assert result.optimizer_checkpoint_id == "optimizer-1"
    assert result.optimizer_fingerprint == bundle.report.fingerprint
    assert result.applied_updates == 1
    assert result.average_updates is None
    assert result.trainable is params
    assert jnp.array_equal(result.model["head"]["w"], params["head"]["w"])
    assert jnp.array_equal(result.model["embed"], plan.frozen["embed"])


def test_divergent_views_survive_checkpoint_round_trip_with_model_state():
    base = replace(tiny_plan(), model_state={"norm": jnp.array(7.0)})
    plan = CombiningPlan(base)
    bundle = _bundle(plan, averaging=True)
    params = plan.trainable
    state = bundle.init(params)
    for _ in range(2):
        updates, state = bundle.update(ones_like_trainable(params), state, params)
        params = optax.apply_updates(params, updates)
    state = state._replace(
        ema_params=jax.tree.map(
            lambda x: None if x is None else jnp.full_like(x, 2.0),
            params,
            is_leaf=lambda x: x is None,
        ),
        swa_params=jax.tree.map(
            lambda x: None if x is None else jnp.full_like(x, 3.0),
            params,
            is_leaf=lambda x: x is None,
        ),
    )
    checkpoint = rfft.make_state_checkpoint(
        bundle,
        state,
        model_checkpoint_id="model-2",
        model_state=plan.model_state,
        counters={"successful_update": 2},
    )
    loaded = pickle.loads(pickle.dumps(checkpoint))

    results = {
        view: rfft.bind_evaluation_view(
            bundle,
            plan,
            params,
            loaded,
            view=view,
            model_checkpoint_id="model-2",
            optimizer_checkpoint_id="optimizer-2",
            applied_updates=2,
        )
        for view in ("optimizer", "ema", "swa")
    }

    assert jnp.array_equal(
        results["optimizer"].model["head"]["w"],
        params["head"]["w"],
    )
    assert jnp.array_equal(
        results["ema"].model["head"]["w"],
        jnp.full_like(params["head"]["w"], 2.0),
    )
    assert jnp.array_equal(
        results["swa"].model["head"]["w"],
        jnp.full_like(params["head"]["w"], 3.0),
    )
    assert results["ema"].average_updates == 2
    assert results["swa"].average_updates == 2
    assert all(result.model_state is loaded.model_state for result in results.values())
    assert len({result.view for result in results.values()}) == 3


@pytest.mark.parametrize("view", ["ema", "swa"])
def test_uninitialized_averaged_view_is_rejected(view):
    plan = tiny_plan()
    bundle = _bundle(plan, averaging=True)
    state = bundle.init(plan.trainable)

    with pytest.raises(ValueError, match="no recorded averages"):
        bundle.eval_params(plan.trainable, state, view=view)


def test_view_binding_rejects_missing_or_mismatched_state():
    plan = CombiningPlan(tiny_plan())
    bundle = _bundle(plan, averaging=True)
    checkpoint = rfft.make_state_checkpoint(
        bundle,
        bundle.init(plan.trainable),
        model_checkpoint_id="model-0",
        counters={"successful_update": 0},
    )
    kwargs = {
        "view": "optimizer",
        "model_checkpoint_id": "model-0",
        "optimizer_checkpoint_id": "optimizer-0",
        "applied_updates": 0,
    }

    with pytest.raises(ValueError, match="unsupported evaluation view"):
        rfft.bind_evaluation_view(
            bundle, plan, plan.trainable, checkpoint, **{**kwargs, "view": "other"}
        )
    with pytest.raises(ValueError, match="successful_update counter"):
        rfft.bind_evaluation_view(
            bundle,
            plan,
            plan.trainable,
            checkpoint,
            **{**kwargs, "applied_updates": 1},
        )
    checkpoint_without_counters = rfft.make_state_checkpoint(
        bundle,
        checkpoint.state,
        model_checkpoint_id="model-0",
    )
    with pytest.raises(ValueError, match="averaging step"):
        rfft.bind_evaluation_view(
            bundle,
            plan,
            plan.trainable,
            checkpoint_without_counters,
            **{**kwargs, "applied_updates": 1},
        )
    with pytest.raises(rfft.OptimizerStateRestoreError, match="model checkpoint"):
        rfft.bind_evaluation_view(
            bundle,
            plan,
            plan.trainable,
            checkpoint,
            **{**kwargs, "model_checkpoint_id": "wrong"},
        )
    with pytest.raises(ValueError, match="no recorded averages"):
        rfft.bind_evaluation_view(
            bundle,
            plan,
            plan.trainable,
            checkpoint,
            **{**kwargs, "view": "ema"},
        )


def test_view_binding_rejects_missing_model_state_snapshot():
    plan = CombiningPlan(replace(tiny_plan(), model_state={"norm": jnp.array(1.0)}))
    bundle = _bundle(plan)
    checkpoint = rfft.make_state_checkpoint(
        bundle, bundle.init(plan.trainable), model_checkpoint_id="model-0"
    )

    with pytest.raises(ValueError, match="model-state snapshot"):
        rfft.bind_evaluation_view(
            bundle,
            plan,
            plan.trainable,
            checkpoint,
            view="optimizer",
            model_checkpoint_id="model-0",
            optimizer_checkpoint_id="optimizer-0",
            applied_updates=0,
        )
