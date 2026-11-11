# Fine-Tuning Benchmark Smoke Scripts

These scripts emit JSON with environment metadata and small local measurements. They are meant to prevent undocumented
performance claims in the Rollfast fine-tuning integration; they are not substitutes for task-specific reports on target
hardware.

Run from the Rollfast repo root:

```bash
PYTHONPATH=src uv run python benchmarks/finetuning/memory.py
PYTHONPATH=src uv run python benchmarks/finetuning/throughput.py
PYTHONPATH=src uv run python benchmarks/finetuning/convergence.py
```

`update_paths.py` times one jitted, donated training step for every exported optimizer and the main fine-tuning update
paths, and appends one JSON line per case to `--output`. The `multi` suite shards matrices and the batch over all
visible devices. To find update paths that stall in CUDA-graph control flow, run it once with default flags and once
with `XLA_FLAGS=--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN`, then compare `ms_per_call`:

```bash
PYTHONPATH=src uv run python benchmarks/finetuning/update_paths.py --size small --suites optim,finetune,multi \
    --output update_paths.jsonl
```

Each payload records Python/JAX/Rollfast versions, devices, warmup iterations, measured iterations, and whether metrics
are model-only or end-to-end. For publishable comparisons, rerun these patterns with the target Equimo model, real batch
shape, dtype, sharding, and enough warmup/measured iterations for the hardware.
