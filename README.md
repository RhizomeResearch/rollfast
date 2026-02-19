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

### 4. Magma (Momentum-Aligned Gradient Masking)

While training large language models (LLMs) typically relies almost exclusively
on dense adaptive optimizers, `rollfast` implements a stochastic masking
intervention that proves randomly masking parameter updates can be highly
effective.

- **Mechanism**: Random masking induces a curvature-dependent geometric
  regularization that smooths the optimization trajectory.
- **Alignment**: Momentum-aligned gradient masking (Magma) modulates the masked
  updates using momentum-gradient alignment.
- **Integration**: It acts as a simple drop-in replacement for adaptive
  optimizers with consistent gains and negligible computational overhead.

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

### Stability & Clipping Parameters

These parameters ensure robustness against gradient spikes and numerical
instability, critical for training at scale.

| Parameter                     | Default       | Description                                                                                                                                                |
| :---------------------------- | :------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `raw_global_grad_clip`        | `None`        | If set, computes the global L2 norm of gradients *before* the optimizer step. If the norm exceeds this threshold, the update is either clipped or skipped. |
| `permissive_spike_protection` | `True`        | Controls behavior when `raw_global_grad_clip` is triggered. `True` clips the gradient and proceeds; `False` strictly skips the update (zeroing the step).  |
| `grad_clip_max_amps`          | `(2.0, 10.0)` | Post-processing clipping. Clips individual tensors by RMS (`2.0`) and absolute value (`10.0`) to prevent heavy tails in the update distribution.           |

### Schedule-Free Hyperparameters

When using `schedule_free_*` optimizers, these arguments control the underlying
WSD (Warmup-Stable-Decay) schedule and the iterate averaging.

| Parameter         | Default     | Description                                                                                                       |
| :---------------- | :---------- | :---------------------------------------------------------------------------------------------------------------- |
| `warmup_fraction` | `0.1`       | Fraction of `total_steps` used for linear warmup.                                                                 |
| `decay_fraction`  | `0.1`       | Fraction of `total_steps` used for linear decay (cooldown) at the end of training.                                |
| `weighting_mode`  | `PRACTICAL` | Strategy for $c_t$ calculation: `THEORETICAL` ($1/t$), `PRACTICAL` ($\gamma_t^2$), or `SCHEDULET` ($\gamma_t$). |

### PRISM Specifics

| Parameter            | Default | Description                                                                                 |
| :------------------- | :------ | :------------------------------------------------------------------------------------------ |
| `ns_iters`           | `5`     | Newton-Schulz iterations. Higher values provide better orthogonality but cost more compute. |
| `gamma`              | `1.0`   | Damping coefficient for the innovation term. Controls the "anisotropy" of spectral shaping. |
| `shape_nesterov`     | `True`  | If True, shapes Nesterov momentum; otherwise shapes raw momentum.                           |
| `adam_learning_rate` | `None`  | Optional override for the Adam branch learning rate. Defaults to `learning_rate` if None.   |

### PSGD Specifics

| Parameter                   | Default | Description                                                                                                                                                 |
| :-------------------------- | :------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `track_lipschitz`           | `True`  | Enables adaptive step sizes for the preconditioner $Q$ by tracking the Lipschitz constant of the gradient.                                                  |
| `max_skew_triangular`       | `1.0`   | Threshold for diagonal approximation. If a dimension's aspect ratio squared exceeds this relative to total numel, it is treated as diagonal to save memory. |
| `preconditioner_init_scale` | `None`  | Initial scale for $Q$. If `None`, it is estimated on the first step using gradient statistics.                                                              |

### Magma Specifics

Magma acts as an intervention layer applicable to both PRISM and PSGD optimizers
by passing `use_magma=True`.

**Architectural Warning:** Magma introduces intentional update bias (damping)
that scales down the expected update magnitude. At equilibrium, you may need to
scale your global learning rate by ~4x to maintain the original update volume
and prevent vanishing progress.

| Parameter   | Default | Description                                                                                                                                                                                                                                                            |
| :---------- | :------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_magma` | `False` | Enables Momentum-aligned gradient masking. Operates at the PyTree leaf level to ensure strict cryptographic PRNG independence and JAX topological isomorphism.                                                                                                         |
| `magma_tau` | `2.0`   | Temperature parameter for the alignment sigmoid $\sigma(\text{cossim} / \tau)$. At default `2.0`, non-masked steps scale updates by ~0.5, which combined with 50% Bernoulli masking yields an expected magnitude attenuation of ~0.25x.                             |
| `key`       | `42`    | Stateful PRNG seed initialized for Magma's Bernoulli sampling. `rollfast` dynamically cycles this key across shards and layers to prevent cryptographic correlation and ensure statistical independence from the base optimizer's noise injections (e.g., Procrustes). |

#### Preconditioner Modes

The geometry of the preconditioner update $dQ$ is controlled via
`preconditioner_mode`.

| Mode        | Formula                             | Description                                                                                                                       |
| :---------- | :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| `Q0.5EQ1.5` | $dQ = Q^{0.5} \mathcal{E} Q^{1.5}$ | **Recommended**. Uses an online orthogonal Procrustes solver to keep $Q$ approximately SPD. Numerically stable for low precision. |
| `EQ`        | $dQ = \mathcal{E} Q$               | The original triangular update. Requires triangular solves. Only mode compatible with triangular $Q$.                             |
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

**Magma:**

```bibtex
@misc{2602.15322,
  Author = {Taejong Joo and Wenhan Xia and Cheolmin Kim and Ming Zhang and Eugene Ie},
  Title = {On Surprising Effectiveness of Masking Updates in Adaptive Optimizers},
  Year = {2026},
  Eprint = {arXiv:2602.15322},
}
```
