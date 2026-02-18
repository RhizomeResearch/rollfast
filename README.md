# rollfast: Advanced Optimization Primitives in JAX

`rollfast` is a high-performance optimization library for JAX, designed to
implement cutting-edge optimizers that go beyond standard Euclidean gradient
descent. It provides production-ready implementations of optimizers like
**PSGD** (Preconditioned Stochastic Gradient Descent) and **PRISM** (Anisotropic
Spectral Shaping), along with a robust **Schedule-Free** wrapper.

Built on top of the [Optax](https://github.com/google-deepmind/optax) ecosystem,
`rollfast` prioritizes memory efficiency (via scanned layers and Kronecker
factorizations), multi-gpu compatibility, mixed-precision trainings and
scalability for large models.

## Algorithms

### 1. PRISM (Anisotropic Spectral Shaping)

PRISM allows for structured optimization by applying anisotropic spectral
shaping to parameter updates. Unlike standard adaptive methods (Adam) that
operate element-wise, or full-matrix second-order methods (Shampoo/PSGD) that
approximate the Hessian, PRISM optimizes the singular value distribution of
weight matrices directly.

- **Mechanism**: Decomposes updates using Newton-Schulz iterations to
  approximate SVD, applying "innovation" updates to the singular vectors while
  damping singular values.
- **Partitioning**: Automatically partitions parameters. High-rank tensors
  (Linear/Conv weights) are optimized via PRISM; vectors (biases, layernorms) are
  optimized via AdamW.
- **Reference**: *PRISM: Structured Optimization via Anisotropic Spectral
  Shaping* (Yang, 2026).

### 2. PSGD Kron (Lie Group Preconditioning)

PSGD reformulates preconditioner estimation as a strongly convex optimization
problem on Lie groups. It updates the preconditioner $Q$ (where $P = Q^T Q$)
using multiplicative updates that avoid explicit matrix inversion.

- **Mechanism**: Maintains a Kronecker-factored preconditioner updated via the
  triangular or orthogonal group.
- **Reference**: *Stochastic Hessian Fittings with Lie Groups* (Li, 2024).

### 3. Schedule-Free Optimization

A wrapper that eliminates the need for complex learning rate schedules by
maintaining two sequences of parameters: a primary sequence $z$ (stepped via the
base optimizer) and an averaged sequence $x$ (used for evaluation).

- **Features**: Supports "Practical" and "Schedulet" weighting modes for
  theoretically grounded averaging.
- **Reference**: *The Road Less Scheduled* (Defazio et al., 2024).

______________________________________________________________________

## Installation

```bash
pip install rollfast
```

## Usage

### 1. PRISM (Standard)

PRISM automatically handles parameter partitioning. You simply provide the
learning rate and structural hyperparameters.

```python
import jax
import jax.numpy as jnp
from rollfast import prism

# Define parameters
params = {
    'linear': {'w': jnp.zeros((128, 128)), 'b': jnp.zeros((128,))},
}

# Initialize PRISM
# 'w' will be optimized by PRISM (Spectral Shaping)
# 'b' will be optimized by AdamW
optimizer = prism(
    learning_rate=1e-3,
    ns_iters=5,          # Newton-Schulz iterations for orthogonalization
    gamma=1.0,           # Innovation damping
    weight_decay=0.01
)

opt_state = optimizer.init(params)
```

### 2. Schedule-Free PRISM

The `schedule_free_prism` function wraps the PRISM optimizer with the
Schedule-Free logic and the WSD (Warmup-Stable-Decay) scheduler for the internal
step size.

```python
from rollfast.optim import schedule_free_prism

optimizer = schedule_free_prism(
    learning_rate=1.0,   # Peak LR for internal steps
    total_steps=10000,   # Required for WSD schedule generation
    warmup_fraction=0.1,
    weighting_mode="schedulet",
    sf_b1=0.9,           # Schedule-Free interpolation (beta)
    gamma=0.8,           # PRISM specific arg
)

# Note: In Schedule-Free, you must compute gradients at the averaged location 'x'
# but apply updates to the state 'z'.
```

### 3. PSGD Kron

The classic Kronecker-factored PSGD optimizer.

```python
from rollfast.optim import kron

optimizer = kron(
    learning_rate=1e-3,
    b1=0.9,
    preconditioner_lr=0.1,
    preconditioner_mode='Q0.5EQ1.5',  # Procrustes-regularized update
    whiten_grad=True
)
```

### Advanced: Scanned Layers (Memory Efficiency)

For deep architectures (e.g., Transformers) implemented via `jax.lax.scan`,
`rollfast` supports explicit handling of scanned layers to prevent unrolling
computation graphs.

```python
import jax
from rollfast.optim import kron

# Boolean pytree mask where True indicates a scanned parameter
scanned_layers_mask = ... 

optimizer = kron(
    learning_rate=3e-4,
    scanned_layers=scanned_layers_mask,
    lax_map_scanned_layers=True, # Use lax.map for preconditioner updates
    lax_map_batch_size=8
)
```

______________________________________________________________________

## Configuration

### PRISM Hyperparameters

| Parameter        | Default | Description                                                                                           |
| :--------------- | :------ | :---------------------------------------------------------------------------------------------------- |
| `ns_iters`       | `5`     | Number of Newton-Schulz iterations. Higher values provide better orthogonality but cost more compute. |
| `gamma`          | `1.0`   | Damping coefficient for the innovation term. Controls the "anisotropy" of spectral shaping.           |
| `shape_nesterov` | `True`  | If True, applies spectral shaping to Nesterov momentum; otherwise shapes raw momentum.                |

### PSGD Modes

| Mode        | Formula                             | Description                                                                                                                       |
| :---------- | :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| `Q0.5EQ1.5` | $dQ = Q^{0.5} \\mathcal{E} Q^{1.5}$ | **Recommended**. Uses an online orthogonal Procrustes solver to keep $Q$ approximately SPD. Numerically stable for low precision. |
| `EQ`        | $dQ = \\mathcal{E} Q$               | The original triangular update. Requires triangular solves. Only mode compatible with triangular $Q$.                             |
| `QUAD`      | Quadratic Form                      | Ensures $Q$ remains symmetric positive definite via quadratic form updates.                                                       |
| `NS`        | Newton-Schulz                       | Iteratively projects $Q$ onto the SPD manifold using Newton-Schulz iterations. Exact but more expensive.                          |
| `EXP`       | Matrix Exponential                  | Geodesic update on the SPD manifold. Uses matrix exponential.                                                                     |
| `TAYLOR2`   | Taylor Expansion                    | Second-order Taylor approximation of the matrix exponential update.                                                               |
| `HYPER`     | Hyperbolic                          | Multiplicative hyperbolic update.                                                                                                 |

______________________________________________________________________

## Citations

If you use `rollfast` in your research, please cite the relevant papers for the algorithms you utilize.

**PRISM:**

```bibtex
@misc{2602.03096,
  Author = {Yujie Yang},
  Title = {PRISM: Structured Optimization via Anisotropic Spectral Shaping},
  Year = {2026},
  Eprint = {arXiv:2602.03096},
}
```

**Schedule-Free:**

```bibtex
@misc{2405.15682,
  Author = {Aaron Defazio and Xingyu Alice Yang and Harsh Mehta and Konstantin Mishchenko and Ahmed Khaled and Ashok Cutkosky},
  Title = {The Road Less Scheduled},
  Year = {2024},
  Eprint = {arXiv:2405.15682},
}

@misc{2511.07767,
  Author = {Yuen-Man Pun and Matthew Buchholz and Robert M. Gower},
  Title = {Schedulers for Schedule-free: Theoretically inspired hyperparameters},
  Year = {2025},
  Eprint = {arXiv:2511.07767},
}
```

**PSGD:**

```bibtex
@article{li2024stochastic,
  title={Stochastic Hessian Fittings with Lie Groups},
  author={Li, Xi-Lin},
  journal={arXiv preprint arXiv:2402.11858},
  year={2024}
}
```
