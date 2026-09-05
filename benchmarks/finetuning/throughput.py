"""Emit tiny plan step-time smoke benchmarks for AdamW and SAM."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from _common import benchmark_step, emit, metadata, rfft, tiny_plan, tree_l2_loss


def _scaled_loss(model, batch):
    return tree_l2_loss(model, target=batch["target"])


def _loss_bundle(model):
    return rfft.LossBundle(
        loss_sum=tree_l2_loss(model),
        normalizer=jnp.asarray(1.0, dtype=jnp.float32),
        metrics_sums={},
        metric_normalizers={},
        new_model_state=None,
    )


def main() -> None:
    warmup_steps = 2
    measured_steps = 5
    plan = tiny_plan()

    adamw = rfft.adamw_from_plan(
        plan,
        total_steps=100,
        schedule="constant",
        clip_global_norm=None,
    )
    adamw_step = jax.jit(rfft.make_update_step(tree_l2_loss, adamw))
    adamw_params, _adamw_state, adamw_loss, adamw_seconds = benchmark_step(
        adamw_step,
        adamw.init(plan.trainable),
        plan.trainable,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )

    sam_base = rfft.adamw_from_plan(
        plan,
        total_steps=100,
        schedule="constant",
        clip_global_norm=None,
    )
    sam_step_raw = rfft.make_sam_step(
        plan=plan,
        base_optimizer=sam_base,
        config=rfft.SAMConfig(rho=0.05),
        loss_fn=_scaled_loss,
        microbatch_axis=0,
    )
    batch = {"target": jnp.array([0.0, 0.25, 0.5, 0.75], dtype=jnp.float32)}
    sam_step = jax.jit(lambda params, state: sam_step_raw(params, state, batch))
    _, _, sam_info, sam_seconds = benchmark_step(
        sam_step,
        sam_base.init(plan.trainable),
        plan.trainable,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )

    accumulation_factor = 4
    accumulation = rfft.AccumulationConfig(steps=accumulation_factor)
    accumulating_step = jax.jit(
        rfft.make_accumulating_loss_bundle_update_step(
            _loss_bundle,
            adamw,
            accumulation=accumulation,
        )
    )
    accumulation_params = plan.trainable
    accumulation_optimizer_state = adamw.init(accumulation_params)
    accumulation_state = rfft.init_accumulation_state(
        accumulation_params,
        accumulation,
    )
    for _ in range(warmup_steps * accumulation_factor):
        (
            accumulation_params,
            accumulation_optimizer_state,
            accumulation_state,
            accumulation_info,
        ) = accumulating_step(
            accumulation_params,
            accumulation_optimizer_state,
            accumulation_state,
        )
    jax.block_until_ready(accumulation_params)

    started = time.perf_counter()
    for _ in range(measured_steps * accumulation_factor):
        (
            accumulation_params,
            accumulation_optimizer_state,
            accumulation_state,
            accumulation_info,
        ) = accumulating_step(
            accumulation_params,
            accumulation_optimizer_state,
            accumulation_state,
        )
    jax.block_until_ready(accumulation_params)
    accumulation_seconds = (time.perf_counter() - started) / measured_steps

    emit(
        {
            "metadata": metadata(
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
            ),
            "scenarios": {
                "adamw_step_seconds": adamw_seconds,
                "adamw_final_loss": float(adamw_loss),
                "sam_step_seconds": sam_seconds,
                "sam_final_loss": float(sam_info.loss),
                "sam_perturbed_loss": float(sam_info.perturbed_loss),
                "sam_microbatches": int(batch["target"].shape[0]),
                "accumulation_factor": accumulation_factor,
                "accumulation_seconds_per_applied_update": accumulation_seconds,
                "accumulation_final_loss": float(
                    accumulation_info.loss_bundle.loss_sum
                ),
            },
            "notes": [
                "Tiny CPU/GPU smoke timing; use task hardware for publishable claims.",
                f"AdamW final head norm sample: {float(jnp.linalg.norm(adamw_params['head']['w']))}",
            ],
        }
    )


if __name__ == "__main__":
    main()
