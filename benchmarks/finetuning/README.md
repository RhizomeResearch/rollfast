# Fine-Tuning Benchmark Smoke Scripts

These scripts emit JSON with environment metadata and small local measurements.
They are meant to prevent undocumented performance claims in the Rollfast
fine-tuning integration; they are not substitutes for task-specific reports on
target hardware.

Run from the Rollfast repo root:

```bash
PYTHONPATH=src uv run python benchmarks/finetuning/memory.py
PYTHONPATH=src uv run python benchmarks/finetuning/throughput.py
PYTHONPATH=src uv run python benchmarks/finetuning/convergence.py
```

Each payload records Python/JAX/Rollfast versions, devices, warmup iterations,
measured iterations, and whether metrics are model-only or end-to-end. For
publishable comparisons, rerun these patterns with the target Equimo model,
real batch shape, dtype, sharding, and enough warmup/measured iterations for the
hardware.
