"""Save and restore a small AdamW optimizer with non-executable storage."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp

import rollfast.finetune as rfft


@dataclass(frozen=True)
class Identity:
    """Stable logical identity for one trainable parameter."""

    logical_id: str


@dataclass(frozen=True)
class Group:
    """Minimal effective optimizer group."""

    label: str
    role: str = "head"
    depth: int | None = None
    lr_multiplier: float = 1.0
    weight_decay: bool = True
    tags: tuple[str, ...] = ()


class Plan:
    """Minimal structural fine-tuning plan for this storage example."""

    def __init__(self) -> None:
        self.trainable = {"head": {"weight": jnp.ones((2,), dtype=jnp.float32)}}
        self.frozen = {"head": {"weight": None}}
        self.labels = {"head": {"weight": "head"}}
        self.identities = {"head": {"weight": Identity("head.weight")}}
        self.group_specs = {"head": Group("head")}


plan = Plan()
optimizer = rfft.adamw_from_plan(plan, total_steps=4, schedule="constant")
limits = rfft.ArrayCheckpointLimits(
    max_manifest_bytes=1_000_000,
    max_arrays=1_000,
    max_shards=2_000,
    max_array_bytes=1_000_000,
    max_total_bytes=10_000_000,
)

with TemporaryDirectory() as directory:
    path = Path(directory) / "optimizer.rfarr"
    state = optimizer.init(plan.trainable)
    saved = rfft.make_state_checkpoint(
        optimizer,
        state,
        model_checkpoint_id="training-model-0",
        counters={"successful_update": 0},
    )
    rfft.save_array_state_checkpoint(path, saved, limits=limits)

    # Build the trusted template from the same compiled plan and expected parts.
    template = rfft.make_state_checkpoint(
        optimizer,
        optimizer.init(plan.trainable),
        model_checkpoint_id="training-model-0",
        counters={"successful_update": 0},
    )
    restored = rfft.load_array_state_checkpoint(
        path,
        optimizer,
        template=template,
        model_checkpoint_id="training-model-0",
        applied_updates=0,
        limits=limits,
    )
    print("Restored state leaves:", len(jax.tree.leaves(restored.state)))
