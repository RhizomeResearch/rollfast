"""Compare BF16 fine-tuning precision modes and resume one from a checkpoint."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp

import rollfast.finetune as rfft

STEPS = 300


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
    weight_decay: bool = False
    tags: tuple[str, ...] = ()


class Plan:
    """Minimal structural fine-tuning plan with one BF16 linear head."""

    def __init__(self, weight: jax.Array) -> None:
        self.trainable = {"head": {"weight": weight}}
        self.frozen = {"head": {"weight": None}}
        self.labels = {"head": {"weight": "head"}}
        self.identities = {"head": {"weight": Identity("head.weight")}}
        self.group_specs = {"head": Group("head")}

    def combine(self, trainable):
        return trainable


weight_key, input_key = jax.random.split(jax.random.key(0))
true_weight = jax.random.normal(weight_key, (64,))
inputs = jax.random.normal(input_key, (256, 64))
targets = inputs @ true_weight
# Start near the solution, where AdamW steps of about the learning rate are
# smaller than half the BF16 spacing of most weights.
plan = Plan((true_weight + 0.05).astype(jnp.bfloat16))


def loss_fn(model, model_state, key):
    del key
    weight = model["head"]["weight"].astype(jnp.float32)
    return jnp.mean((inputs @ weight - targets) ** 2), model_state


def train(precision: rfft.PrecisionConfig):
    optimizer = rfft.adamw_from_plan(
        plan,
        total_steps=STEPS,
        base_lr=1e-3,
        schedule="constant",
        weight_decay=0.0,
        clip_global_norm=None,
        precision=precision,
    )
    state = rfft.init_finetune_step_state(optimizer, plan.trainable, rng_seed=0)
    step = rfft.make_finetune_update_step(plan, loss_fn, optimizer)

    @jax.jit
    def run(params, state):
        def body(carry, _):
            params, state, info = step(*carry)
            return (params, state), info.value

        return jax.lax.scan(body, (params, state), length=STEPS)

    (params, state), losses = run(plan.trainable, state)
    return optimizer, params, state, losses


adam_sized_update = jax.tree.map(lambda p: jnp.full(p.shape, 1e-3), plan.trainable)
metrics = rfft.low_precision_update_metrics(plan.trainable, adam_sized_update)
print(
    "Fraction of 1e-3 updates that round-to-nearest BF16 storage discards: "
    f"{float(metrics['below_half_spacing_fraction']):.2f}"
)

modes = {
    "nearest, no masters": rfft.PrecisionConfig(master_params="never"),
    "float32 masters": rfft.bf16_mixed_precision(),
    "stochastic, no masters": rfft.bf16_master_free_precision(),
}
for name, precision in modes.items():
    optimizer, params, state, losses = train(precision)
    print(f"{name:>22}: loss {float(losses[0]):.2e} -> {float(losses[-1]):.2e}")
    print(f"{'':>22}  {optimizer.report.precision}")

# The stochastic run's checkpoint carries its rounding stream and counters, so
# the resumed run continues exactly where it stopped.
limits = rfft.ArrayCheckpointLimits(
    max_manifest_bytes=1_000_000,
    max_arrays=1_000,
    max_shards=2_000,
    max_array_bytes=1_000_000,
    max_total_bytes=10_000_000,
)
with TemporaryDirectory() as directory:
    path = Path(directory) / "optimizer.rfarr"
    saved = rfft.make_step_state_checkpoint(
        optimizer, state, model_checkpoint_id="bf16-run"
    )
    rfft.save_array_state_checkpoint(path, saved, limits=limits)
    template = rfft.make_step_state_checkpoint(
        optimizer,
        rfft.init_finetune_step_state(optimizer, plan.trainable),
        model_checkpoint_id="bf16-run",
    )
    loaded = rfft.load_array_state_checkpoint(
        path,
        optimizer,
        template=template,
        model_checkpoint_id="bf16-run",
        applied_updates=int(state.counters.successful_update),
        limits=limits,
    )
    resumed = rfft.step_state_from_checkpoint(loaded)
    step = jax.jit(rfft.make_finetune_update_step(plan, loss_fn, optimizer))
    expected, _, _ = step(params, state)
    actual, _, _ = step(params, resumed)
    print(
        "Resumed step matches the uninterrupted run:",
        bool(jnp.array_equal(actual["head"]["weight"], expected["head"]["weight"])),
    )
