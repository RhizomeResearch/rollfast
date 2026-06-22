import jax
import jax.numpy as jnp
import optax
import pytest

import rollfast.finetune as rfft

from .helpers import tiny_plan


def _ones_like_trainable(tree):
    return jax.tree.map(
        lambda x: jnp.ones_like(x) if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
    )


def _zeros_like_trainable(tree):
    return jax.tree.map(
        lambda x: jnp.zeros_like(x) if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
    )


@pytest.mark.parametrize(
    ("builder", "optimizer_name"),
    (
        (rfft.hybrid_aurora_adam_from_plan, "aurora_adam"),
        (rfft.hybrid_prism_adam_from_plan, "prism_adam"),
        (rfft.hybrid_kron_adam_from_plan, "kron_adam"),
    ),
)
def test_hybrid_from_plan_updates_and_reports_groups(builder, optimizer_name):
    plan = tiny_plan()
    kwargs = {"total_steps": 10, "schedule": "constant", "clip_global_norm": None}
    if optimizer_name == "kron_adam":
        kwargs["preconditioner_update_probability"] = 1.0
    bundle = builder(plan, **kwargs)
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(
        _ones_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )
    params = optax.apply_updates(plan.trainable, updates)
    groups = {group.source_label: group for group in bundle.report.groups}

    assert state is not None
    assert params["embed"] is None
    assert not jnp.allclose(params["blocks"][0]["w"], plan.trainable["blocks"][0]["w"])
    assert groups["block_00_decay"].optimizer == optimizer_name
    assert groups["block_00_decay"].effective_lr == pytest.approx(2.5e-4)
    assert groups["block_00_no_decay"].weight_decay_value == 0.0


def test_hybrid_no_decay_leaf_unchanged_under_zero_gradients():
    plan = tiny_plan()
    bundle = rfft.hybrid_aurora_adam_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        weight_decay=0.1,
        clip_global_norm=None,
        polar_ns_iters=2,
    )
    state = bundle.init(plan.trainable)
    updates, _ = bundle.update(
        _zeros_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )

    assert jnp.allclose(updates["blocks"][0]["b"], 0.0)
    assert jnp.all(updates["blocks"][0]["w"] < 0.0)


def test_hybrid_supports_ema_eval_view_and_manifest():
    plan = tiny_plan()
    bundle = rfft.hybrid_prism_adam_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        ns_iters=2,
        ema=rfft.EMAConfig(enabled=True, decay=0.5),
    )
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(
        _ones_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )
    params = optax.apply_updates(plan.trainable, updates)

    assert bundle.eval_params(params, state, view="ema")["head"]["w"].shape == (
        2,
        1,
    )
    assert bundle.manifest()["groups"][0]["optimizer"] == "prism_adam"


@pytest.mark.parametrize(
    ("builder", "kwargs", "method", "implemented_profile", "reference_algorithm"),
    (
        (
            rfft.hybrid_aurora_adam_from_plan,
            {"polar_ns_iters": 2},
            "aurora",
            "rollfast_aurora_balanced_polar",
            "tilde_aurora_balanced_polar",
        ),
        (
            rfft.hybrid_prism_adam_from_plan,
            {"ns_iters": 2},
            "prism",
            "rollfast_prism_original",
            "prism_innovation_augmented_spectral_shaping",
        ),
    ),
)
def test_aurora_prism_manifests_mark_experimental_profiles(
    builder,
    kwargs,
    method,
    implemented_profile,
    reference_algorithm,
):
    plan = tiny_plan()
    bundle = builder(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        **kwargs,
    )
    method_config = bundle.manifest()["method_config"]

    assert method_config["method"] == method
    assert method_config["implemented_profile"] == implemented_profile
    assert method_config["reference_algorithm"] == reference_algorithm
    assert method_config["matrix_optimizer"] == f"rollfast.optim.{method}"
    assert method_config["fallback_optimizer"] == "adamw"
    assert method_config["paper_profile"] is False
    assert method_config["known_deviations"] == (
        "reference_parity_fixture_pending",
        "pretrained_finetuning_benchmark_gate_not_satisfied",
    )
    assert method_config["reference_validated"] is False
    assert any("experimental" in warning.lower() for warning in bundle.report.warnings)


def test_kron_manifest_marks_psgd_profile_and_no_internal_adam_fallback():
    plan = tiny_plan()
    bundle = rfft.hybrid_kron_adam_from_plan(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        preconditioner_update_probability=1.0,
    )
    method_config = bundle.manifest()["method_config"]

    assert method_config["method"] == "kron"
    assert method_config["implemented_profile"] == "rollfast_psgd_kron"
    assert (
        method_config["reference_algorithm"]
        == "psgd_kronecker_factored_preconditioner"
    )
    assert method_config["matrix_optimizer"] == "rollfast.optim.psgd.kron"
    assert method_config["fallback_optimizer"] is None
    assert method_config["matrix_eligibility"] == "caller_routed_group_leaves"
    assert method_config["paper_profile"] is False
    assert method_config["known_deviations"] == (
        "reference_parity_fixture_pending",
        "no_internal_adam_fallback_in_kron_transform",
    )
    assert method_config["reference_validated"] is False
    assert any("psgd" in warning.lower() for warning in bundle.report.warnings)
    assert any(
        "internal adam fallback" in warning.lower()
        for warning in bundle.report.warnings
    )


@pytest.mark.parametrize(
    ("builder", "optimizer_name", "kwargs"),
    (
        (
            rfft.muon_adam_from_plan,
            "muon_adam",
            {"muon": rfft.MuonConfig(ns_steps=2)},
        ),
    ),
)
def test_p2_matrix_builders_update_and_manifest(builder, optimizer_name, kwargs):
    plan = tiny_plan()
    bundle = builder(
        plan,
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
        **kwargs,
    )
    state = bundle.init(plan.trainable)
    updates, state = bundle.update(
        _ones_like_trainable(plan.trainable),
        state,
        plan.trainable,
    )
    params = optax.apply_updates(plan.trainable, updates)
    manifest = bundle.manifest()

    assert state is not None
    assert not jnp.allclose(params["blocks"][0]["w"], plan.trainable["blocks"][0]["w"])
    assert bundle.report.groups[0].optimizer == optimizer_name
    assert manifest["method_config"]["profile_fidelity"] == "experimental"
    assert manifest["method_config"]["fallback_optimizer"] == "adamw"
    assert any("experimental" in warning.lower() for warning in bundle.report.warnings)


def test_muon_config_round_trip():
    muon = rfft.MuonConfig(
        ns_steps=3,
        beta=0.9,
        preconditioning="spectral",
        consistent_rms=0.2,
    )

    assert rfft.MuonConfig.from_dict(muon.to_dict()) == muon


def test_muon_manifest_marks_optax_profile_and_benchmark_gate():
    plan = tiny_plan()
    bundle = rfft.muon_adam_from_plan(
        plan,
        muon=rfft.MuonConfig(ns_steps=3, consistent_rms=0.2),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )
    method_config = bundle.manifest()["method_config"]

    assert method_config["method"] == "muon"
    assert method_config["implemented_profile"] == "optax_contrib_muon"
    assert method_config["reference_algorithm"] == "muon_newton_schulz_matrix_momentum"
    assert method_config["matrix_optimizer"] == "optax.contrib.muon"
    assert method_config["paper_profile"] is False
    assert method_config["pretrained_finetuning_recommendation"] == "benchmark_required"
    assert method_config["known_deviations"] == (
        "uses_optax_library_profile_not_moonlight_distributed_reference",
        "pretrained_finetuning_benchmark_gate_not_satisfied",
    )
    assert method_config["consistent_rms"] == 0.2
