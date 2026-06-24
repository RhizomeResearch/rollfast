"""Emit tiny plan step-time smoke benchmarks for AdamW and SAM."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from _common import benchmark_step, emit, metadata, rfft, tiny_plan, tree_l2_loss


def _scaled_loss(model, batch):
    return tree_l2_loss(model, target=batch["target"])


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
    adamw_params, adamw_state, adamw_loss, adamw_seconds = benchmark_step(
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
            },
            "notes": [
                "Tiny CPU/GPU smoke timing; use task hardware for publishable claims.",
                f"AdamW final head norm sample: {float(jnp.linalg.norm(adamw_params['head']['w']))}",
            ],
        }
    )


if __name__ == "__main__":
    main()
