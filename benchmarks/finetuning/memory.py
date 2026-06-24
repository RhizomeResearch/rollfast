"""Emit state-memory smoke benchmarks for plan-aware fine-tuning optimizers."""

from __future__ import annotations

from contextlib import redirect_stdout
import io

from _common import emit, large_plan, metadata, rfft, tiny_plan


def _compact_summary(summary):
    return {
        "optimizer": summary.optimizer,
        "total_bytes": summary.total_bytes,
        "estimated_state_bytes": summary.estimated_state_bytes,
        "by_category": dict(summary.by_category),
        "by_group": dict(summary.by_group),
        "preconditioner_bytes": summary.preconditioner_bytes,
        "preconditioner_factors": [
            {
                "shape": factor.shape,
                "dtype": factor.dtype,
                "bytes": factor.bytes,
                "storage": factor.storage,
                "group": factor.group,
            }
            for factor in summary.preconditioner_factors
        ],
    }


def _compact_estimate(estimate):
    return {
        "optimizer": estimate.optimizer,
        "total_bytes": estimate.total_bytes,
        "moment_bytes": estimate.moment_bytes,
        "preconditioner_bytes": estimate.preconditioner_bytes,
        "preconditioner_aux_bytes": estimate.preconditioner_aux_bytes,
        "by_category": dict(estimate.by_category),
        "by_group": dict(estimate.by_group),
        "preconditioner_factors": [
            {
                "shape": factor.shape,
                "dtype": factor.dtype,
                "bytes": factor.bytes,
                "storage": factor.storage,
                "group": factor.group,
            }
            for factor in estimate.preconditioner_factors
        ],
        "warnings": estimate.warnings,
    }


def _measured_summary(bundle, trainable):
    with redirect_stdout(io.StringIO()):
        state = bundle.init(trainable)
    return rfft.optimizer_state_memory_summary(bundle, state)


def main() -> None:
    tiny = tiny_plan()
    large = large_plan()

    adamw = rfft.adamw_from_plan(
        tiny,
        total_steps=100,
        schedule="constant",
        clip_global_norm=None,
    )
    kron = rfft.hybrid_kron_adam_from_plan(
        tiny,
        total_steps=100,
        schedule="constant",
        clip_global_norm=None,
        preconditioner_update_probability=1.0,
    )
    adamw8 = rfft.adamw8_from_plan(
        large,
        total_steps=100,
        schedule="constant",
        clip_global_norm=None,
        state_quantization=rfft.StateQuantizationConfig(
            enabled=True,
            block_size=2048,
            min_size=4096,
            stochastic_rounding=False,
        ),
    )
    fp32_large = rfft.adamw_from_plan(
        large,
        total_steps=100,
        schedule="constant",
        clip_global_norm=None,
    )

    kron_estimate = rfft.estimate_optimizer_state_memory(tiny, kron)
    payload = {
        "metadata": metadata(warmup_steps=0, measured_steps=1),
        "scenarios": {
            "tiny_adamw_measured": _compact_summary(
                _measured_summary(adamw, tiny.trainable)
            ),
            "tiny_kron_static_estimate": _compact_estimate(kron_estimate),
            "tiny_kron_measured": _compact_summary(
                _measured_summary(kron, tiny.trainable)
            ),
            "large_adamw_fp32_measured": _compact_summary(
                _measured_summary(fp32_large, large.trainable)
            ),
            "large_adamw8_measured": _compact_summary(
                _measured_summary(adamw8, large.trainable)
            ),
        },
    }
    emit(payload)


if __name__ == "__main__":
    main()
