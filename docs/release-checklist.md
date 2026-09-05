# Stable Release Checklist

Run these gates from a clean checkout of the release commit using the Python
version declared by the project.

Ruff and ty are pinned in `pyproject.toml` so local checks and fresh CI installs
use the same versions. Update these pins deliberately and run all static checks
when doing so; tool upgrades can change default lint rules and formatting.

## Required Gates

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest tests/test_multidevice_cpu.py
uv run python examples/adamw_quickstart.py
uv run python examples/schedule_free_eval.py
uv run python examples/finetuning/state_memory_report.py
uv run python benchmarks/finetuning/memory.py
uv run python benchmarks/finetuning/throughput.py
uv run python benchmarks/finetuning/convergence.py
uv build
```

Install the built wheel into a fresh environment and import Rollfast from a
directory outside the repository:

```bash
uv venv --clear /tmp/rollfast-wheel-venv
uv pip install --python /tmp/rollfast-wheel-venv/bin/python dist/*.whl
(cd /tmp && /tmp/rollfast-wheel-venv/bin/python -c "import rollfast")
```

CI additionally runs the suite against the minimum supported JAX and Optax
versions. The release tag, `pyproject.toml`, and `rollfast.__version__` must be
identical.

## Accelerator Validation

GPU or TPU validation is required before making hardware-specific performance,
memory, or sharding claims, but is not a prerequisite for the CPU-validated API
release. Record the accelerator model, JAX/JAXLIB build, dtype, mesh, partition
specifications, model, batch shape, and benchmark warmup in any published
result.

## Release Hygiene

- Update `CHANGELOG.md` and the versioned release notes.
- Confirm project, documentation, issue, and source links in the built metadata.
- Confirm the release commit is on the default branch and the worktree is clean.
- Run the tag pipeline and publish only its tested artifacts.
