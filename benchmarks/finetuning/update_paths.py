"""Time jitted update steps for every Rollfast optimizer and fine-tuning path.

Each case compiles one donated training step (forward, backward, optimizer
update, parameter update) and reports its compile time and steady-state time.
Run the script twice, once with default XLA flags and once with
``XLA_FLAGS=--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN``,
to see which update paths stall in CUDA-graph control flow on a given GPU.

Run from the Rollfast repo root:

    PYTHONPATH=src uv run python benchmarks/finetuning/update_paths.py \
        --size small --suites optim,finetune,multi --output update_paths.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import traceback
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from _common import matrix_plan, metadata, rfft
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import rollfast
from rollfast import apply_updates

SIZES = {"tiny": (2, 64), "small": (8, 2048), "large": (12, 4096)}
BATCH = 256
LR = 1e-3
TOTAL_STEPS = 10_000


def _mlp_loss(params: Any, batch: dict[str, jax.Array]) -> jax.Array:
    hidden = batch["x"]
    for layer in params.values():
        hidden = jnp.tanh(hidden @ layer["w"] + layer["b"])
    return jnp.mean((hidden - batch["y"]) ** 2)


def _batch(width: int, sharding: Any | None = None) -> dict[str, jax.Array]:
    batch = {
        "x": jax.random.normal(jax.random.PRNGKey(1), (BATCH, width)),
        "y": jnp.tanh(jax.random.normal(jax.random.PRNGKey(2), (BATCH, width))),
    }
    return batch if sharding is None else jax.device_put(batch, sharding)


def _optimizers() -> dict[str, Callable[[], Any]]:
    """Complete optimizers exported by ``rollfast``, with branch-selecting variants."""

    r = rollfast
    sf = {"learning_rate": LR, "total_steps": TOTAL_STEPS}
    return {
        "adamw": lambda: r.adamw(LR),
        "adamw8": lambda: r.adamw8(LR),
        "adamw_hyperball": lambda: r.adamw_hyperball(LR),
        "aurora": lambda: r.aurora(LR),
        "aurora_hyperball": lambda: r.aurora_hyperball(LR),
        "riemannian_aurora": lambda: r.riemannian_aurora(LR),
        "riemannian_aurora_hyperball": lambda: r.riemannian_aurora_hyperball(LR),
        "hyperball_riemannian_aurora": lambda: r.hyperball_riemannian_aurora(LR),
        "contramuon": lambda: r.contramuon(LR),
        "contranormuon": lambda: r.contranormuon(LR),
        "kron": lambda: r.kron(learning_rate=LR),
        "kron_p1": lambda: r.kron(
            learning_rate=LR, preconditioner_update_probability=1.0
        ),
        "kron_p0.1": lambda: r.kron(
            learning_rate=LR, preconditioner_update_probability=0.1
        ),
        "kron_hyperball": lambda: r.kron_hyperball(learning_rate=LR),
        "muon": lambda: r.muon(LR),
        "muon_aol": lambda: r.muon(LR, preconditioning="aol"),
        "muon_schatten": lambda: r.muon(LR, preconditioning="schatten"),
        "muon_spectral": lambda: r.muon(LR, preconditioning="spectral"),
        "muon_hyperball": lambda: r.muon_hyperball(LR),
        "normuon": lambda: r.normuon(LR),
        "pion": lambda: r.pion(LR),
        "pion_bilateral": lambda: r.pion(LR, alternating=False),
        "prism": lambda: r.prism(LR),
        "prism_hyperball": lambda: r.prism_hyperball(LR),
        "rmnp": lambda: r.rmnp(LR),
        "rmnp_hyperball": lambda: r.rmnp_hyperball(LR),
        "trasmuon": lambda: r.trasmuon(LR),
        "schedule_free_adam": lambda: r.schedule_free_adam(**sf),
        "schedule_free_aurora": lambda: r.schedule_free_aurora(**sf),
        "schedule_free_kron": lambda: r.schedule_free_kron(**sf),
        "schedule_free_prism": lambda: r.schedule_free_prism(**sf),
        "soda_adam": lambda: r.soda_adam(**sf),
        "soda_kron": lambda: r.soda_kron(**sf),
        "soda_muon": lambda: r.soda_muon(**sf),
        "soda_prism": lambda: r.soda_prism(**sf),
        "soda_rmnp": lambda: r.soda_rmnp(**sf),
    }


def _optim_case(name: str, plan: Any, sharding: Any | None):
    tx = _optimizers()[name]()
    params = plan.trainable
    if sharding is not None:
        params = jax.device_put(params, sharding["params"])
    value_and_grad = jax.value_and_grad(_mlp_loss)

    def step(carry, batch):
        params, state = carry
        _, grads = value_and_grad(params, batch)
        updates, state = tx.update(grads, state, params)
        return apply_updates(params, updates, stochastic=False), state

    return step, (params, tx.init(params)), 1


def _finetune_loss(model, model_state, key, batch):
    del key
    return _mlp_loss(model, batch), model_state


def _bundle_loss(model, batch):
    hidden = batch["x"]
    for layer in model.values():
        hidden = jnp.tanh(hidden @ layer["w"] + layer["b"])
    return rfft.LossBundle(
        loss_sum=jnp.sum((hidden - batch["y"]) ** 2),
        normalizer=jnp.asarray(hidden.size, jnp.float32),
        metrics_sums={},
        metric_normalizers={},
        new_model_state=None,
    )


def _bundles(plan: Any) -> dict[str, Callable[[], Any]]:
    common = {"total_steps": TOTAL_STEPS, "base_lr": LR, "schedule": "wsd"}
    averaging = {
        "ema": rfft.EMAConfig(enabled=True, decay=0.999),
        "swa": rfft.SWAConfig(enabled=True, start_step=0, frequency=1),
    }
    return {
        "adamw": lambda: rfft.adamw_from_plan(plan, **common),
        "adamw_magma": lambda: rfft.adamw_from_plan(plan, use_magma=True, **common),
        "adamw8": lambda: rfft.adamw8_from_plan(plan, **common),
        "adamw_ema_swa": lambda: rfft.adamw_from_plan(plan, **averaging, **common),
        "adamw_ema_swa_every4": lambda: rfft.adamw_from_plan(
            plan,
            ema=rfft.EMAConfig(enabled=True, decay=0.999, update_every=4),
            swa=rfft.SWAConfig(enabled=True, start_step=0, frequency=4),
            **common,
        ),
        "adamw_accum4": lambda: rfft.adamw_from_plan(
            plan, accumulation_steps=4, **common
        ),
        "apollo_adamw": lambda: rfft.apollo_adamw_from_plan(
            plan,
            apollo=rfft.APOLLOConfig(rank=64, projection_refresh_interval=10),
            **common,
        ),
        "galore_adamw": lambda: rfft.galore_adamw_from_plan(
            plan,
            galore=rfft.GaLoreConfig(rank=64, update_interval=10),
            **common,
        ),
        "schedule_free_adam": lambda: rfft.schedule_free_adam_from_plan(plan, **common),
        "muon_adam": lambda: rfft.muon_adam_from_plan(plan, **common),
        "muon_adam_aol": lambda: rfft.muon_adam_from_plan(
            plan, muon=rfft.MuonConfig(preconditioning="aol"), **common
        ),
        "muon_adam_schatten": lambda: rfft.muon_adam_from_plan(
            plan, muon=rfft.MuonConfig(preconditioning="schatten"), **common
        ),
        "aurora_adam": lambda: rfft.hybrid_aurora_adam_from_plan(plan, **common),
        "prism_adam": lambda: rfft.hybrid_prism_adam_from_plan(plan, **common),
        "kron_adam": lambda: rfft.hybrid_kron_adam_from_plan(plan, **common),
        "kron_adam_p0.1": lambda: rfft.hybrid_kron_adam_from_plan(
            plan, preconditioner_update_probability=0.1, **common
        ),
        "kron_adam_accum4": lambda: rfft.hybrid_kron_adam_from_plan(
            plan, accumulation_steps=4, **common
        ),
        "kron_adam_ema_swa": lambda: rfft.hybrid_kron_adam_from_plan(
            plan, **averaging, **common
        ),
    }


def _finetune_case(name: str, plan: Any, sharding: Any | None):
    bundle = _bundles(plan)[name]()
    params = plan.trainable
    if sharding is not None:
        params = jax.device_put(params, sharding["params"])
    update = rfft.make_finetune_update_step(plan, _finetune_loss, bundle)

    def step(carry, batch):
        params, step_state = carry
        params, step_state, _ = update(params, step_state, batch)
        return params, step_state

    return (
        step,
        (params, rfft.init_finetune_step_state(bundle, params)),
        bundle.accumulation_config.steps,
    )


def _loss_bundle_case(name: str, plan: Any, sharding: Any | None):
    family = name.removeprefix("loss_bundle_accum4_")
    bundle = _bundles(plan)[family]()
    accumulation = rfft.AccumulationConfig(steps=4)
    params = plan.trainable
    if sharding is not None:
        params = jax.device_put(params, sharding["params"])
    update = rfft.make_accumulating_loss_bundle_update_step(
        _bundle_loss, bundle, accumulation=accumulation
    )

    def step(carry, batch):
        params, opt_state, window = carry
        params, opt_state, window, _ = update(params, opt_state, window, batch)
        return params, opt_state, window

    window = rfft.init_accumulation_state(params, accumulation)
    return step, (params, bundle.init(params), window), 4


def _adalora_case(name: str, plan: Any, sharding: Any | None):
    del name, plan, sharding
    controller = rfft.make_adalora_controller(
        {f"group_{index:02d}": 16 for index in range(32)},
        total_steps=TOTAL_STEPS,
        config=rfft.AdaLoRAControllerConfig(
            initial_budget=384, target_budget=256, allocation_interval=10
        ),
    )
    importance = jax.random.uniform(
        jax.random.PRNGKey(3), (controller.group_count, controller.max_rank)
    )

    def step(carry, batch):
        del batch
        return controller.update(carry, importance)

    return step, controller.init(), 1


def _cases(suite: str) -> list[tuple[str, Callable[..., Any]]]:
    if suite == "optim":
        return [(name, _optim_case) for name in _optimizers()]
    if suite == "finetune":
        return [
            *((name, _finetune_case) for name in _bundles(None)),
            ("loss_bundle_accum4_adamw", _loss_bundle_case),
            ("loss_bundle_accum4_kron_adam", _loss_bundle_case),
            ("adalora_controller", _adalora_case),
        ]
    if suite == "multi":
        multi_optim = ("adamw", "muon", "kron_p1", "aurora", "schedule_free_adam")
        multi_finetune = ("adamw", "adamw_accum4", "muon_adam", "kron_adam")
        return [
            *((name, _optim_case) for name in multi_optim),
            *((name, _finetune_case) for name in multi_finetune),
            ("loss_bundle_accum4_adamw", _loss_bundle_case),
        ]
    raise ValueError(f"unknown suite {suite!r}")


def _shardings(plan: Any) -> dict[str, Any]:
    """Shard matrices over their first axis (FSDP) and the batch over examples."""

    mesh = Mesh(jax.devices(), ("data",))

    def leaf_sharding(leaf):
        spec = P("data", None) if leaf.ndim == 2 else P()
        return NamedSharding(mesh, spec)

    return {
        "params": jax.tree.map(leaf_sharding, plan.trainable),
        "batch": NamedSharding(mesh, P("data", None)),
    }


def _time_case(step, carry, batch, steps_per_update: int) -> dict[str, Any]:
    # Donation consumes the carry; copy it so cases never share buffers.
    carry = jax.tree.map(jnp.copy, carry)
    batch_sharding = jax.tree.leaves(batch)[0].sharding
    if isinstance(batch_sharding, NamedSharding):
        # Scalars and keys created on one device join the mesh replicated.
        replicated = NamedSharding(batch_sharding.mesh, P())
        carry = jax.tree.map(
            lambda leaf: (
                leaf
                if leaf.sharding.device_set == batch_sharding.device_set
                else jax.device_put(leaf, replicated)
            ),
            carry,
        )
    # Pin the carry's shardings, as a training loop does, so each call
    # receives the layout the step was compiled for.
    carry_shardings = jax.tree.map(lambda leaf: leaf.sharding, carry)
    jitted = jax.jit(
        step,
        in_shardings=(carry_shardings, jax.tree.map(lambda x: x.sharding, batch)),
        out_shardings=carry_shardings,
        donate_argnums=(0,),
    )
    started = time.perf_counter()
    compiled = jitted.lower(carry, batch).compile()
    compile_seconds = time.perf_counter() - started
    memory = compiled.memory_analysis()

    for _ in range(3 * steps_per_update):
        carry = compiled(carry, batch)
    jax.block_until_ready(carry)

    synced = []
    budget_end = time.perf_counter() + 1.0
    while len(synced) < 4 * steps_per_update or (
        time.perf_counter() < budget_end and len(synced) < 100
    ):
        tick = time.perf_counter()
        carry = compiled(carry, batch)
        jax.block_until_ready(carry)
        synced.append((time.perf_counter() - tick) * 1e3)

    calls = max(10, 4 * steps_per_update, min(200, int(2e3 / max(min(synced), 1e-3))))
    calls -= calls % steps_per_update
    started = time.perf_counter()
    for _ in range(calls):
        carry = compiled(carry, batch)
    jax.block_until_ready(carry)
    pipelined = (time.perf_counter() - started) * 1e3 / calls

    leaves = jax.tree.leaves(carry)
    finite = all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in leaves
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    )
    return {
        "compile_s": round(compile_seconds, 2),
        "temp_mib": round(memory.temp_size_in_bytes / 2**20, 1) if memory else None,
        "ms_per_call": round(pipelined, 3),
        "ms_per_call_synced_median": round(statistics.median(synced), 3),
        "ms_per_call_synced_max": round(max(synced), 3),
        "calls_per_update": steps_per_update,
        "finite": finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=tuple(SIZES), default="small")
    parser.add_argument("--suites", default="optim,finetune")
    parser.add_argument("--only", default="", help="comma-separated case names")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    layers, width = SIZES[args.size]
    plan = matrix_plan(layers, width)
    only = set(filter(None, args.only.split(",")))
    header = {
        **metadata(warmup_steps=3, measured_steps=0),
        "device_kind": jax.devices()[0].device_kind,
        "device_count": jax.device_count(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "size": args.size,
        "layers": layers,
        "width": width,
        "batch": BATCH,
    }
    with open(args.output, "a") as output:
        output.write(json.dumps({"header": header}) + "\n")
        for suite in args.suites.split(","):
            if suite == "multi" and jax.device_count() == 1:
                continue
            sharding = _shardings(plan) if suite == "multi" else None
            batch = _batch(width, None if sharding is None else sharding["batch"])
            for name, build in _cases(suite):
                if only and name not in only:
                    continue
                record: dict[str, Any] = {"suite": suite, "case": name}
                try:
                    step, carry, steps_per_update = build(name, plan, sharding)
                    record.update(_time_case(step, carry, batch, steps_per_update))
                except Exception as error:  # noqa: BLE001 - a failing case must not end the sweep
                    record["error"] = "".join(
                        traceback.format_exception_only(error)
                    ).strip()[:500]
                record.update(
                    device_kind=header["device_kind"],
                    device_count=header["device_count"],
                    jax=header["jax"],
                    xla_flags=header["xla_flags"],
                    size=args.size,
                )
                print(json.dumps(record), flush=True)
                output.write(json.dumps(record) + "\n")
                output.flush()


if __name__ == "__main__":
    main()
