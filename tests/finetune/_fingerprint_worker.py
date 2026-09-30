"""Fresh-interpreter fingerprint and non-executable checkpoint regression fixture."""

from __future__ import annotations

import functools
import hashlib
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import rollfast.finetune as rfft
from rollfast import apply_updates

from .helpers import ARRAY_LIMITS, TinyGroup, TinyPlan


class Toy(eqx.Module):
    weight: jax.Array
    activation: object = eqx.field(static=True)

    def __call__(self):
        return self.activation(self.weight)


def _toy(weight=None):
    return Toy(
        jnp.asarray([0.5, -1.0], dtype=jnp.float32) if weight is None else weight,
        functools.partial(jax.nn.gelu, approximate=True),
    )


def checkpoint(mode: str, path: Path) -> dict:
    tree = _toy()
    plan = TinyPlan(
        trainable=tree,
        labels=jax.tree.map(lambda _: "head", tree),
        group_specs={"head": TinyGroup("head", "head", None, 1.0, True)},
    )
    bundle = rfft.adamw_from_plan(
        plan, total_steps=4, schedule="constant", clip_global_norm=None
    )
    state = bundle.init(tree)
    gradient = jax.grad(lambda model: jnp.sum(model() ** 2))
    update = jax.jit(bundle.update)
    model_path = path.with_suffix(".model.npy")
    reference_path = path.with_suffix(".reference.npz")
    if mode == "write":
        updates, state = update(gradient(tree), state, tree)
        tree = apply_updates(tree, updates, stochastic=False)
        np.save(model_path, np.asarray(tree.weight), allow_pickle=False)
        saved = rfft.make_state_checkpoint(
            bundle,
            state,
            model_checkpoint_id="toy-step-1",
            base_model_value_hash=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            counters={"successful_update": 1},
        )
        rfft.save_array_state_checkpoint(path, saved, limits=ARRAY_LIMITS)
    else:
        tree = eqx.tree_at(
            lambda model: model.weight,
            tree,
            jnp.asarray(np.load(model_path, allow_pickle=False)),
        )
        template = rfft.make_state_checkpoint(
            bundle,
            state,
            model_checkpoint_id="toy-step-1",
            counters={"successful_update": 0},
        )
        restored = rfft.load_array_state_checkpoint(
            path,
            bundle,
            template=template,
            model_checkpoint_id="toy-step-1",
            base_model_value_hash=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            applied_updates=1,
            limits=ARRAY_LIMITS,
        )
        state = restored.state
    evaluated = tree()
    updates, next_state = update(gradient(tree), state, tree)
    next_tree = apply_updates(tree, updates, stochastic=False)
    results = {
        "evaluation": np.asarray(evaluated),
        "next_parameters": np.asarray(next_tree.weight),
        **{
            f"state_{index}": np.asarray(leaf)
            for index, leaf in enumerate(jax.tree.leaves(next_state))
        },
    }
    if mode == "write":
        np.savez(reference_path, **results)
    else:
        with np.load(reference_path, allow_pickle=False) as expected:
            assert set(expected.files) == results.keys()
            for name, actual in results.items():
                np.testing.assert_allclose(actual, expected[name], rtol=1e-6, atol=1e-7)
    return {"fingerprint": bundle.report.fingerprint, "mode": mode}


def equimo_fingerprints() -> dict:
    import equimo.finetune as eqft
    import equimo.finetune.vision as eqv
    from equimo.vision.models.eomt import EoMT
    from equimo.vision.models.pmd import PMTConfig
    from equimo.vision.models.pmt import PMT
    from equimo.vision.models.vit import VisionTransformer

    key = jax.random.key(0)
    backbone = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        depths=[1],
        key=key,
        dynamic_img_size=True,
        reg_tokens=0,
        use_global_pos_embed=False,
        use_local_pos_embed=True,
        num_classes=0,
        act_layer=functools.partial(jax.nn.gelu, approximate=True),
    )
    fingerprints = {}
    for name, factory, pooling in (
        ("pooled", eqft.make_linear_probe, "mean_patch"),
        ("pointwise", eqv.make_dense_probe, "none"),
    ):
        model = factory(
            backbone,
            in_features=8,
            out_features=2,
            key=key,
            feature_spec=eqft.FeatureSpec(
                endpoint="features",
                output_layout="BNC",
                token_selection="patches",
                pooling=pooling,
                return_metadata=True,
            ),
        )
        fingerprints[name] = rfft.validate_plan(
            eqft.prepare_finetune(model, trainable=eqft.TrainableSpec(mode="head"))
        ).fingerprint
    fingerprints["eomt"] = rfft.validate_plan(
        eqv.eomt_full_finetune(
            EoMT(backbone, num_classes=2, num_queries=2, num_blocks=1, key=key)
        )
    ).fingerprint
    fingerprints["pmt"] = rfft.validate_plan(
        eqv.pmt_head_finetune(
            PMT(
                backbone,
                num_classes=2,
                key=key,
                config=PMTConfig(
                    dim=8,
                    num_heads=2,
                    hidden_dim=8,
                    taps=(0,),
                    num_queries=2,
                    num_blocks=1,
                ),
            )
        )
    ).fingerprint
    return fingerprints


if __name__ == "__main__":
    print(
        json.dumps(
            equimo_fingerprints()
            if sys.argv[1] == "equimo"
            else checkpoint(sys.argv[1], Path(sys.argv[2])),
            sort_keys=True,
        )
    )
