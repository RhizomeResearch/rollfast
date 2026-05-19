# rollfast: Advanced Optimization Primitives in JAX

`rollfast` is a high-performance optimization library for JAX, designed to
implement cutting-edge optimizers that go beyond standard Euclidean gradient
descent. It provides experimental implementations of optimizers like
**Muon** (orthogonalized momentum), **PSGD** (Preconditioned Stochastic Gradient Descent), **PRISM** (Anisotropic
Spectral Shaping), **Aurora** (leverage-aware rectangular matrix optimization),
and **Pion** (spectrum-preserving orthogonal equivalence updates), plus
**RMNP** (row-momentum normalized preconditioning), **NorMuon**, **ContraMuon**,
**TrasMuon** (trust-region adaptive scaling for Muon), **Hyperball**
norm-preserving weight decay, **SODA**, and a robust
**Schedule-Free** wrapper.

Built on top of the [Optax](https://github.com/google-deepmind/optax) ecosystem,
`rollfast` prioritizes memory efficiency (via scanned layers and Kronecker
factorizations), multi-gpu compatibility, mixed-precision trainings and
scalability for large models.

## Algorithms

### 1. Muon

Muon is a matrix optimizer that applies momentum, orthogonalizes the matrix
direction with quintic Newton-Schulz iterations, and uses AdamW as the fallback
for non-matrix leaves.

- **Mechanism**: Tracks first momentum, optionally forms a Nesterov lookahead,
  applies shared Muon-family Newton-Schulz coefficient schedules, and scales
  matrix updates by shape.
- **Coefficient schedules**: `ns_coeffs` accepts `"standard"`, `"dion"`,
  `"polar_express"`, a single `(a, b, c)` tuple, or an ordered `(n, 3)` schedule.
- **Momentum**: `momentum_accumulator="ema"` matches the default exponential
  moving average; `"heavy_ball"` uses heavy-ball accumulation. Muon-family
  matrix optimizers expose the same choice.
- **Partitioning**: Automatically routes 2D matrices to Muon and routes vectors,
  scalars, and unspecified tensors to AdamW. Use `muon_weight_dimension_numbers`
  for high-rank tensors.

### 2. PRISM (Anisotropic Spectral Shaping)

PRISM allows for structured optimization by applying anisotropic spectral
shaping to parameter updates. Unlike standard adaptive methods (Adam) that
operate element-wise, or full-matrix second-order methods (Shampoo/PSGD) that
approximate the Hessian, PRISM optimizes the singular value distribution of
weight matrices directly.

- **Mechanism**: Decomposes updates using Newton-Schulz iterations to
  approximate SVD, applying "innovation" updates to the singular vectors while
  damping singular values.
- **Modes**: Supports `original` (Newton-Schulz iterations on an augmented matrix) and `bidirectional` (Shampoo-style bilateral shaping of both left and right singular-vector spaces). `ns_coeffs` applies to `original`; `bidirectional` uses separate inverse-root coefficients.
- **Partitioning**: Automatically routes 2D matrix leaves to PRISM and routes
  vectors, scalars, and unspecified tensors to AdamW. Use
  `prism_weight_dimension_numbers` or `get_equinox_prism_spec` to opt
  high-rank/Conv tensors into PRISM.
- **Reference**: *PRISM: Structured Optimization via Anisotropic Spectral
  Shaping* (Yang, 2026) and *Bidirectional-PRISM: Kronecker-Factored Optimization via Anisotropic Spectral Shaping* (Cesista, 2026).

### 3. Aurora (Leverage-Aware Matrix Optimization)

Aurora optimizes matrix-shaped parameters by applying polar-style updates with
leverage-aware balancing for rectangular matrices. This gives rectangular layers
row/column-aware update geometry instead of treating every element
independently.

- **Mechanism**: Uses Newton-Schulz polar iterations, with practical diagonal
  balancing for rectangular matrices. `riemannian_aurora` provides a more
  expensive balanced-Stiefel variant for reference-quality updates.
- **Coefficient schedules**: Aurora intentionally does not accept Muon/PRISM
  `ns_coeffs`; it keeps the fixed simple-quintic polar path from the Aurora
  reference implementation.
- **Partitioning**: Automatically applies Aurora to matrix leaves and AdamW to
  vectors/scalars. Explicit dimension specs can opt convolution kernels or other
  high-rank tensors into Aurora.
- **Reference**: *Aurora: A Leverage-Aware Optimizer for Rectangular Matrices*
  (Dewulf et al., 2026).

### 4. PSGD Kron (Lie Group Preconditioning)

PSGD reformulates preconditioner estimation as a strongly convex optimization
problem on Lie groups. It updates the preconditioner $Q$ (where $P = Q^T Q$)
using multiplicative updates that avoid explicit matrix inversion.

- **Mechanism**: Maintains a Kronecker-factored preconditioner updated via the
  triangular or orthogonal group.
- **Reference**: *Stochastic Hessian Fittings with Lie Groups* (Li, 2024).

### 5. Hyperball Optimization

Hyperball replaces ordinary decoupled weight decay with a terminal projection
that keeps selected parameters on the L2 sphere defined by their initialization
norm. Given an unscaled optimizer direction, Hyperball adds the decay direction,
normalizes the result, takes a step proportional to the initial norm, and
projects the parameter back to that sphere.

- **Mechanism**: `apply_hyperball` is a terminal Optax transform; it should be
  the final transform in a chain and should not be followed by a learning-rate
  scale transform.
- **Wrappers**: `adamw_hyperball`, `kron_hyperball`, `muon_hyperball`,
  `rmnp_hyperball`, `prism_hyperball`, `aurora_hyperball`, and
  `riemannian_aurora_hyperball`
  replace decoupled weight decay in the corresponding optimizer families.
- **Partitioning**: By default, Hyperball applies to rank >= 2 leaves. Muon,
  PRISM, RMNP, and Aurora wrappers apply Hyperball to the same leaves routed to
  the structured optimizer branch, while fallback leaves use Adam-style updates.
- **Reference**: *Fantastic Pretraining Optimizers and Where to Find Them 2.1:
  Hyperball Optimization* (Wen et al., 2025).

### 6. Pion (Spectrum-Preserving Orthogonal Equivalence)

Pion updates matrix parameters by multiplying the current weight by left and
right orthogonal transformations rather than adding an ambient-space update.
This keeps each optimized matrix on its iso-spectral manifold, preserving its
singular values up to the numerical error of the second-order exponential
approximation.

- **Mechanism**: Builds in-side and out-side skew-symmetric Lie-algebra
  gradients, tracks Adam-style moments in those tangent spaces, and applies
  RMS-normalized second-order exponential updates.
- **Partitioning**: Automatically routes 2D matrices to Pion and routes vectors,
  scalars, and unspecified tensors to AdamW.
- **Weight decay**: `pion(..., weight_decay=...)` applies decay only on the
  AdamW fallback leaves. Pion-managed matrices stay on their iso-spectral path.
- **Reference**: *Pion: A Spectrum-Preserving Optimizer via Orthogonal
  Equivalence Transformation* (Shi et al., 2026).

### 7. RMNP (Row-Momentum Normalized Preconditioning)

RMNP is a Muon-style matrix optimizer that replaces Newton-Schulz
orthogonalization with row-wise L2 normalization of the momentum matrix. This
keeps the matrix branch linear in the number of parameters while preserving the
matrix/fallback partitioning pattern used by Muon-like optimizers.

- **Mechanism**: Tracks first momentum, normalizes rows in the configured matrix
  layout, and applies Muon-style shape scaling.
- **Partitioning**: Automatically routes 2D matrices to RMNP and routes vectors,
  scalars, and unspecified tensors to AdamW.
- **Wrappers**: `soda_rmnp` and `rmnp_hyperball` are provided.
- **Reference**: *RMNP: Row-Momentum Normalized Preconditioning for Scalable
  Matrix-Based Optimization* (Deng et al., 2026).

### 8. TrasMuon (Trust-Region Adaptive Scaling Muon)

TrasMuon is a Muon-style matrix optimizer that keeps Newton-Schulz
orthogonalized momentum, then stabilizes update magnitude with global RMS
calibration and feature-wise trust-region damping.

- **Mechanism**: Tracks first momentum, applies Muon Newton-Schulz
  orthogonalization, normalizes row RMS, calibrates each matrix update to a
  global Frobenius target, and damps high-energy columns using relative energy
  ratios.
- **Partitioning**: Automatically routes 2D matrices to TrasMuon and routes
  vectors, scalars, and unspecified tensors to AdamW.
- **Reference**: *TrasMuon: Trust-Region Adaptive Scaling for Orthogonalized
  Momentum Optimizers* (Cheng et al., 2026).

### 9. NorMuon and ContraMuon

`normuon` implements NorMuon's second-moment normalization in Rollfast's
standard matrix/AdamW partitioning style.

- **NorMuon**: Applies Muon Newton-Schulz orthogonalization, then tracks a second
  moment along the configured matrix layout. By default this is row-wise after
  reshaping tensors to `(..., reduction, output)`, matching the reference
  NorMuon implementation's `mean(..., dim=-1, keepdim=True)`. Set
  `normalization_axis="auto"` to use rows for tall matrices and columns for wide
  matrices. The default `normalization_rescale="preserve_update_norm"` preserves
  the Muon update norm after normalization; use `"fixed_rms"` with
  `normalization_rms` for a fixed-RMS direction, or `"none"` to skip
  post-normalization rescaling.
- **ContraMuon**: Subtracts `contra_coeff / 2` times a power-iteration estimate
  of the operator-normalized momentum gradient from the Muon update, then
  restores the pre-Contra Frobenius norm.
- **ContraNorMuon**: Combines the ContraMuon spectral correction with
  NorMuon second-moment normalization.
- **Partitioning**: Automatically routes 2D matrices to the Muon variant and
  routes vectors, scalars, and unspecified tensors to AdamW.

### 10. SODA (Optimistic Dual Averaging Wrapper)

SODA wraps an existing base optimizer and replaces tuned weight decay with a
parameter-free initialization-centered anchor term that decays as `1 / (k + 2)`.

- **Mechanism**: Adds `(z0 - params) / (k + 2)` to the base optimizer update,
  where `z0` is the initialization. The base optimizer should include its
  learning-rate schedule and should not include weight decay.
- **Wrappers**: `soda_adam`, `soda_prism`, `soda_kron`, `soda_muon`, and
  `soda_rmnp` are provided. `soda_muon` uses Rollfast's local Muon as its base
  optimizer.
- **Reference**: *Optimistic Dual Averaging Unifies Modern Optimizers*
  (Pethick et al., 2026).

### 11. Schedule-Free Optimization

A wrapper that eliminates the need for complex learning rate schedules by
maintaining two sequences of parameters: a primary sequence $z$ (stepped via the
base optimizer) and an averaged sequence $x$ (used for evaluation). Available
for PRISM, Aurora, PSGD Kron, and Adam.

- **Features**: Supports "Practical", "Schedulet", "Theoretical", and "Power" (SF+) weighting modes for
  theoretically grounded averaging. Optionally supports the newer **ScheduleFree+** components (e.g., Polyak step sizes).
- **Reference**: *The Road Less Scheduled* (Defazio et al., 2024) and *ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models* (Defazio, 2026).

### 12. Magma (Momentum-Aligned Gradient Masking)

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

Equinox integration helpers such as `get_equinox_prism_spec` and
`get_equinox_aurora_spec` require Equinox, which is optional:

```bash
pip install "rollfast[equinox]"
```

## Usage

### 1. Muon

Muon automatically partitions parameters: 2D leaves use the Muon matrix branch,
while vectors and scalars use AdamW.

```python
import jax.numpy as jnp
from rollfast import MatrixDimensionNumbers, muon

params = {
    "linear": {"w": jnp.zeros((128, 128)), "b": jnp.zeros((128,))},
}

optimizer = muon(
    learning_rate=1e-3,
    ns_steps=5,
    ns_coeffs="polar_express",
    momentum_accumulator="ema",  # or "heavy_ball"
)
```

For convolution kernels or other high-rank tensors, pass explicit dimension
specs with `MatrixDimensionNumbers`:

```python
specs = {
    "conv": {
        "kernel": MatrixDimensionNumbers(
            reduction_axis=(1, 2, 3),
            output_axis=0,
        )
    }
}

optimizer = muon(
    learning_rate=1e-3,
    muon_weight_dimension_numbers=specs,
)
```

### 2. PRISM (Standard)

PRISM automatically handles parameter partitioning. You simply provide the
learning rate and structural hyperparameters.

```python
import jax
import jax.numpy as jnp
from rollfast import prism, get_equinox_prism_spec

# Define parameters
params = {
    'linear': {'w': jnp.zeros((128, 128)), 'b': jnp.zeros((128,))},
}

# Initialize PRISM
# 'w' will be optimized by PRISM (Spectral Shaping)
# 'b' will be optimized by AdamW
optimizer = prism(
    learning_rate=1e-3,
    mode='bidirectional', # or 'original'
    ns_iters=5,           # Newton-Schulz iterations (for 'original' mode)
    ns_coeffs="standard", # only used by 'original' mode
    inv_steps=6,          # Polynomial iterations (for 'bidirectional' mode)
    gamma=1.0,            # Innovation damping
    weight_decay=0.01
)

opt_state = optimizer.init(params)
```

**Equinox Integration:** `get_equinox_prism_spec` automatically constructs the
dimension specification PyTree matching an Equinox model. Install the optional
Equinox dependency with `pip install "rollfast[equinox]"` before using it.

```python
import equinox as eqx

model = ... # Your Equinox model
specs = get_equinox_prism_spec(model)
optimizer = prism(
    learning_rate=1e-3,
    prism_weight_dimension_numbers=specs,
)
```

You may also pass `get_equinox_prism_spec` itself as a callable dimension spec;
Rollfast will call it with the parameter/model PyTree supplied to `init` and
`update`.

### 3. Aurora

Aurora uses the same automatic matrix/AdamW partitioning pattern as PRISM: 2D
leaves are optimized with Aurora, while vectors and scalars fall back to AdamW.

```python
import jax.numpy as jnp
from rollfast import aurora

params = {
    'linear': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros((256,))},
}

# 'w' will be optimized by Aurora
# 'b' will be optimized by AdamW
optimizer = aurora(
    learning_rate=3e-4,
    b1=0.95,
    pp_iterations=2,
    pp_beta=0.5,
    polar_ns_iters=12,
    weight_decay=0.025,
)

opt_state = optimizer.init(params)
```

For convolution kernels or other high-rank tensors, pass an explicit dimension
spec. The Equinox Aurora helper returns compatible specs for Aurora and requires
the optional Equinox dependency.

```python
from rollfast import aurora, get_equinox_aurora_spec

model = ... # Your Equinox model
optimizer = aurora(
    learning_rate=3e-4,
    aurora_weight_dimension_numbers=get_equinox_aurora_spec(model),
)
```

As with PRISM, `aurora_weight_dimension_numbers` also accepts the helper itself
as a callable spec.

Use `riemannian_aurora` when you want the more expensive balanced-Stiefel
variant. Aurora does not use Muon/PRISM `ns_coeffs`; tune
`polar_ns_iters`, `polar_compute_dtype`, and the Aurora-specific balancing
arguments instead.

### 4. Hyperball

Hyperball wrappers use the same base optimizer arguments as their non-Hyperball
counterparts, but interpret `weight_decay` as the Hyperball decay coefficient on
selected leaves.

```python
from rollfast.optim.hyperball import prism_hyperball

optimizer = prism_hyperball(
    learning_rate=1e-3,
    weight_decay=0.01,
    mode='bidirectional',
    inv_steps=6,
    hyperball_mask=None,       # Defaults to the PRISM-routed leaves
    fallback_weight_decay=False
)
```

Muon is available as a first-class Rollfast optimizer, with Hyperball support:

```python
from rollfast import muon, muon_hyperball

optimizer = muon(
    learning_rate=1e-3,
    ns_coeffs="polar_express",
    ns_steps=5,
)

hyperball_optimizer = muon_hyperball(
    learning_rate=1e-3,
    weight_decay=0.01,
    ns_steps=5,
)
```

RMNP has a matching Hyperball wrapper:

```python
from rollfast import rmnp_hyperball

optimizer = rmnp_hyperball(
    learning_rate=1e-3,
    weight_decay=0.01,
)
```

For direct Optax composition, use `apply_hyperball` as the final transform in
the chain.

```python
import optax
from rollfast.optim.hyperball import apply_hyperball

optimizer = optax.chain(
    optax.scale_by_adam(),
    apply_hyperball(learning_rate=1e-3, weight_decay=0.01)
)
```

### 5. Schedule-Free Optimization

The `schedule_free_*` convenience functions wrap base optimizers with the
Schedule-Free logic and generate an internal WSD (Warmup-Stable-Decay) schedule
for the step size. The generic `schedule_free` wrapper takes an already scaled
base optimizer and an explicit learning-rate scalar or schedule used for
averaging weights. Their `update` method needs the current `params`, because
Schedule-Free interpolates between the averaged iterate and its internal
z-sequence.

```python
import jax.numpy as jnp
import optax

from rollfast import schedule_free_eval_params, schedule_free_prism

optimizer = schedule_free_prism(
    learning_rate=1.0,   # Peak LR for internal steps
    total_steps=10000,   # Required for WSD schedule generation
    warmup_fraction=0.1,
    weighting_mode="schedulet",
    sf_b1=0.9,           # Schedule-Free interpolation (beta)
    gamma=0.8,           # PRISM specific arg
)

params = {"linear": {"w": jnp.zeros((128, 128)), "b": jnp.zeros((128,))}}
grads = {"linear": {"w": jnp.ones((128, 128)), "b": jnp.ones((128,))}}
opt_state = optimizer.init(params)

# During training, apply updates to the z-sequence parameters.
updates, opt_state = optimizer.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)

# For evaluation/validation, use the averaged x-sequence parameters:
eval_params = schedule_free_eval_params(opt_state, params)
```

*Note: We also provide `schedule_free_kron`, `schedule_free_aurora`, and
`schedule_free_adam`.*

### 6. Pion

```python
from rollfast import pion

optimizer = pion(
    learning_rate=1e-3,
    b1=0.9,
    b2=0.999,
    rms_constant=1.0,
    alternating=True,
    momentum_accumulator="ema",
)
```

### 7. RMNP

```python
from rollfast import rmnp

optimizer = rmnp(
    learning_rate=1e-3,
    beta=0.95,
    momentum_accumulator="ema",
    consistent_rms=None,
)
```

For Equinox modules or convolution kernels, pass explicit dimension specs with
`MatrixDimensionNumbers` via `rmnp_weight_dimension_numbers` so row
normalization uses the intended flattened matrix layout.

### 8. TrasMuon

```python
from rollfast import trasmuon

optimizer = trasmuon(
    learning_rate=1e-3,
    beta1=0.95,
    beta2=0.95,
    momentum_accumulator="ema",
    ns_iters=5,
    ns_coeffs="polar_express",
    trigger=1.0,
    clip_min=0.1,
)
```

For Equinox modules or convolution kernels, pass explicit dimension specs with
`MatrixDimensionNumbers` via `trasmuon_weight_dimension_numbers`.

### 9. NorMuon / ContraMuon

`normuon` uses NorMuon's second-moment normalization recipe in Rollfast's
`MatrixDimensionNumbers` layout. The default `normalization_axis="row"` keeps
one statistic per reduction row after reshaping to `(..., reduction, output)`,
matching the reference implementation. Use `normalization_axis="auto"` for rows
on tall matrices and columns on wide matrices.

```python
from rollfast import contramuon, contranormuon, normuon

optimizer = normuon(
    learning_rate=1e-3,
    beta1=0.95,
    beta2=0.95,
    momentum_accumulator="ema",
    ns_iters=5,
    ns_coeffs="dion",
    normalization_axis="row",
    normalization_rescale="preserve_update_norm",
    preconditioning="frobenius",
)

contra_optimizer = contramuon(
    learning_rate=1e-3,
    beta1=0.95,
    contra_coeff=0.4,
    ns_iters=5,
)

contra_nor_optimizer = contranormuon(
    learning_rate=1e-3,
    beta1=0.95,
    beta2=0.95,
    contra_coeff=0.4,
    ns_iters=5,
    normalization_rescale="fixed_rms",
    normalization_rms=0.2,
)
```

Use `normuon_weight_dimension_numbers`, `contramuon_weight_dimension_numbers`,
or `contranormuon_weight_dimension_numbers` for high-rank tensors.

### Direct `scale_by_*` transforms

The direct transforms such as `scale_by_rmnp`, `scale_by_normuon`,
`scale_by_trasmuon`, and `scale_by_pion` are low-level matrix-branch
primitives. They do not include the AdamW fallback used by public wrappers such
as `rmnp`, `normuon`, `trasmuon`, and `pion`; direct matrix transforms reject
updated non-auxiliary leaves without matrix dimension specs instead of silently
using them as fallback leaves.

By default, only 2D array leaves receive matrix dimension numbers. For
convolution kernels or other high-rank tensors, either use the public wrapper so
unsupported leaves route to AdamW, or pass an explicit PyTree/callable
`weight_dimension_numbers` spec to the direct transform.

Matrix optimizers operate on real matrix geometry. Complex parameters should be
routed to an Adam fallback branch or represented as real-valued tensors before
using Muon/PRISM/Aurora/RMNP/NorMuon/TrasMuon/Pion matrix branches.

A bare `MatrixDimensionNumbers(...)` is only for a single array leaf. It is not
broadcast across PyTrees; for structured params, pass a matching PyTree of
specs/`None` or a callable such as `get_equinox_prism_spec`.

### 10. SODA

```python
from rollfast import soda_adam, soda_kron, soda_muon, soda_prism, soda_rmnp

optimizer = soda_prism(
    learning_rate=1e-3,
    total_steps=10000,
    mode="bidirectional",
    inv_steps=6,
)

muon_optimizer = soda_muon(
    learning_rate=1e-3,
    total_steps=10000,
    ns_steps=5,
)

rmnp_optimizer = soda_rmnp(
    learning_rate=1e-3,
    total_steps=10000,
)

adam_optimizer = soda_adam(
    learning_rate=1e-3,
    total_steps=10000,
)

kron_optimizer = soda_kron(
    learning_rate=1e-3,
    total_steps=10000,
)
```

SODA convenience wrappers disable base optimizer weight decay and add the
initialization-centered `1 / (k + 2)` anchor term from the paper.

### 11. PSGD Kron

The classic Kronecker-factored PSGD optimizer.

```python
from rollfast import kron

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
from rollfast import kron

# Boolean pytree mask where True indicates a scanned parameter
scanned_layers_mask = ... 

optimizer = kron(
    learning_rate=3e-4,
    scanned_layers=scanned_layers_mask,
    lax_map_scanned_layers=True, # Use lax.map for preconditioner updates
    lax_map_batch_size=8
)
```

### Advanced: Stochastic Rounding (Low Precision)

Training models in **pure BF16** (where parameters, moments, and gradients are all low precision) can lead to the "vanishing update" problem: when the weight update $\Delta \theta$ is smaller than the precision limit of the current weight $\theta$, deterministic rounding (Round-to-Nearest-Even) collapses it to zero.

Stochastic Rounding (SR) solves this by mapping the update's fractional part to a probability of rounding up, ensuring that even small updates contribute to the training trajectory on average.

`rollfast` provides a high-performance SR implementation using an **integer-only bit manipulation pipeline** that is fully fusible by XLA, avoiding HBM spills and slow float conversions.

#### 1. Pure BF16 Training Step

To enable SR for your model parameters, use `apply_updates_prefix` (compatible with Equinox/PyTrees) or `apply_updates` (Optax-style) within your JIT-compiled step.

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from rollfast import apply_updates_prefix

# Initial Model (usually in FP32)
model = ...

# Cast model to BF16 for pure low-precision training
model = jax.tree.map(
    lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, 
    model
)

@eqx.filter_jit
def step(model, opt_state, batch, key):
    fwd_key, sr_key = jax.random.split(key)
    
    # Compute gradients (model is BF16, gradients will be BF16)
    (loss, aux), grads = eqx.filter_value_and_grad(compute_loss)(model, batch, fwd_key)
    
    # Update optimizer
    # 'updates' PyTree structure matches the filtered model.
    filtered_model = eqx.filter(model, eqx.is_inexact_array)
    updates, new_opt_state = optimizer.update(grads, opt_state, filtered_model)
    
    # Apply updates with Stochastic Rounding
    # This is critical when 'model' is BF16 to prevent vanishing updates.
    new_model = apply_updates_prefix(model, updates, sr_key, stochastic=True)
    
    return new_model, new_opt_state, loss
```

#### 2. Memory-Efficient Optimizer Moments

If you are bottlenecked by VRAM, you can also store selected optimizer state in
**BF16 with stochastic rounding** by passing `mu_dtype=jnp.bfloat16`. The exact
state affected depends on the optimizer:

| Optimizer family | State affected by `mu_dtype` |
| :--------------- | :--------------------------- |
| `adamw` | First and second Adam moment trees, `mu` and `nu`. |
| `muon`, `prism`, `aurora`, `rmnp`, `psgd/kron` | First momentum tree, `mu`. PSGD preconditioners use `precond_dtype` separately. |
| `normuon`, `contramuon`, `contranormuon` | First momentum tree, `mu`; NorMuon second-moment statistics stay FP32. |
| `trasmuon` | First momentum tree, `mu`; row, energy, and clip statistics stay FP32. |
| `pion` | All Pion Lie-algebra moment trees: `m_in`, `v_in`, `m_out`, and `v_out`. |

Composition wrappers such as SODA, Hyperball, and Schedule-Free forward
`mu_dtype` to their base optimizer branches when those branches expose momentum
state. `schedule_free_kron` is the exception: it runs Kron with `b1=0.0`, so
there is no momentum buffer for `mu_dtype` to control; use `precond_dtype` for
Kron preconditioner storage.

```python
import jax.numpy as jnp
from rollfast import adamw

optimizer = adamw(
    learning_rate=1e-3,
    mu_dtype=jnp.bfloat16  # Selected optimizer state is stored as BF16 with SR
)
```

______________________________________________________________________

## Configuration

### Stability & Clipping Parameters

These parameters ensure robustness against gradient spikes and numerical
instability, critical for training at scale.

| Parameter                     | Default          | Description                                                                                                                                                |
| :---------------------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `raw_global_grad_clip`        | `None`           | If set, computes the global L2 norm of gradients *before* the optimizer step. If the norm exceeds this threshold, the update is either clipped or skipped. |
| `permissive_spike_protection` | `True`           | Controls behavior when `raw_global_grad_clip` is triggered. `True` clips the gradient and proceeds; `False` strictly skips the update (zeroing the step).  |
| `grad_clip_mode`              | `per_tensor_rms` | Strategy for clipping gradients (`per_tensor_rms` or `global_rms`). Used by PSGD.                                                                          |
| `grad_clip_max_amps`          | `(2.0, 10.0)`    | Post-processing clipping. Clips individual tensors by RMS (`2.0`) and absolute value (`10.0`) to prevent heavy tails in the update distribution.           |

### Schedule-Free Hyperparameters

When using `schedule_free_*` optimizers, these arguments control the underlying
WSD (Warmup-Stable-Decay) schedule and the iterate averaging.

| Parameter         | Default     | Description                                                                                                       |
| :---------------- | :---------- | :---------------------------------------------------------------------------------------------------------------- |
| `warmup_fraction` | `0.1`       | Fraction of `total_steps` used for linear warmup.                                                                 |
| `decay_fraction`  | `0.1`       | Fraction of `total_steps` used for linear decay (cooldown) at the end of training.                                |
| `weighting_mode`  | `PRACTICAL` | Strategy for $c_t$ calculation: `THEORETICAL` ($1/t$), `PRACTICAL` ($\gamma_t^2$), or `SCHEDULET` ($\gamma_t$). |

For the WSD schedule, `warmup_steps = int(total_steps * warmup_fraction)` and
`decay_steps = int(total_steps * decay_fraction)`. Counts `0..warmup_steps`
are warmup, counts after `total_steps - decay_steps` are decay, and the final
count `total_steps - 1` reaches `final_lr_ratio * peak_lr`.

### Muon Specifics

| Parameter                     | Default       | Description                                                                                                                                  |
| :---------------------------- | :------------ | :------------------------------------------------------------------------------------------------------------------------------------------- |
| `ns_steps`                    | `5`           | Newton-Schulz iterations for Muon orthogonalization.                                                                                         |
| `ns_coeffs`                   | Muon default  | Shared Newton-Schulz coefficients; accepts `"standard"`, `"dion"`, `"polar_express"`, a single `(a, b, c)` tuple, or an ordered `(n, 3)` schedule. |
| `momentum_accumulator`        | `"ema"`       | `"ema"` for exponential moving average momentum, or `"heavy_ball"` for heavy-ball accumulation.                                               |
| `preconditioning`             | `"frobenius"` | Matrix normalization before Newton-Schulz: `"frobenius"`, `"spectral"`, `"aol"`, or `"schatten"`.                                           |
| `muon_weight_dimension_numbers` | `None`      | Optional PyTree/callable spec for reshaping high-rank tensors into matrices. Defaults to 2D leaves.                                           |

`momentum_accumulator` is also available on PRISM, Aurora, NorMuon,
ContraNorMuon, ContraMuon, TrasMuon, RMNP, and Pion. Schedule-Free wrappers keep
their own interpolation state and do not expose this option.

### PRISM Specifics

| Parameter            | Default    | Description                                                                                 |
| :------------------- | :--------- | :------------------------------------------------------------------------------------------ |
| `mode`               | `original` | `original` (Newton-Schulz augmented) or `bidirectional` (Shampoo-style bilateral shaping).  |
| `ns_iters`           | `5`        | Newton-Schulz iterations. Higher values provide better orthogonality but cost more compute. |
| `ns_coeffs`          | Muon default | Shared Newton-Schulz coefficients for `mode="original"`; ignored by `mode="bidirectional"`, which uses inverse-root coefficients. |
| `inv_steps`          | `6`        | Polynomial iterations for mode `bidirectional`.                                             |
| `gamma`              | `1.0`      | Damping coefficient for the innovation term. Controls the "anisotropy" of spectral shaping. |
| `shape_nesterov`     | `True`     | If True, shapes Nesterov momentum; otherwise shapes raw momentum.                           |
| `momentum_accumulator` | `"ema"`  | `"ema"` for exponential moving average momentum, or `"heavy_ball"` for heavy-ball accumulation. |
| `adam_learning_rate` | `None`     | Optional override for the Adam branch learning rate. Defaults to `learning_rate` if None.   |

### Aurora Specifics

| Parameter                         | Default        | Description                                                                                         |
| :-------------------------------- | :------------- | :-------------------------------------------------------------------------------------------------- |
| `b1`                              | `0.95`         | Momentum used before Aurora shaping.                                                               |
| `momentum_accumulator`            | `"ema"`        | `"ema"` for exponential moving average momentum, or `"heavy_ball"` for heavy-ball accumulation.    |
| `pp_iterations`                   | `2`            | Practical diagonal-balancing iterations for rectangular matrices.                                   |
| `pp_beta`                         | `0.5`          | Exponent for Aurora's row-balance multiplier updates.                                               |
| `polar_ns_iters`                  | `12`           | Newton-Schulz iterations used to approximate the polar factor.                                      |
| `polar_compute_dtype`             | `jnp.bfloat16` | Compute dtype for the polar iterations.                                                            |
| `aurora_weight_dimension_numbers` | `None`         | Optional PyTree/callable spec for reshaping high-rank tensors into matrices. Defaults to 2D leaves. |

`riemannian_aurora` also exposes `outer_steps`, `cg_steps`, `riemannian_eta`,
and `retraction_steps` for its balanced-Stiefel solve.

### Hyperball Specifics

| Parameter                 | Default | Description                                                                                                            |
| :------------------------ | :------ | :--------------------------------------------------------------------------------------------------------------------- |
| `hyperball_mask`          | `None`  | Boolean PyTree/callable selecting leaves projected by Hyperball. Defaults to rank >= 2 leaves, or the structured branch for Muon/PRISM/RMNP/Aurora wrappers. |
| `fallback_weight_decay`   | `False` | If True, non-Hyperball leaves receive ordinary decoupled AdamW-style decay.                                            |
| `caution`                 | `False` | Applies cautious update masking before Hyperball normalization.                                                        |
| `cautious_weight_decay`   | `False` | Applies the decay term elementwise only where `sign(param) == sign(direction)`.                                        |
| `hyperball_eps` / `eps`   | `1e-12` | Minimum norm divisor used by the projection. Wrapper functions expose this as `hyperball_eps`; `apply_hyperball` uses `eps`. |

### PSGD Specifics

| Parameter                   | Default | Description                                                                                                                                                 |
| :-------------------------- | :------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `track_lipschitz`           | `True`  | Enables adaptive step sizes for the preconditioner $Q$ by tracking the Lipschitz constant of the gradient.                                                  |
| `max_skew_triangular`       | `1.0`   | Threshold for diagonal approximation. If a dimension's aspect ratio squared exceeds this relative to total numel, it is treated as diagonal to save memory. |
| `preconditioner_init_scale` | `None`  | Initial scale for $Q$. If `None`, it is estimated on the first step using gradient statistics.                                                              |

#### Preconditioner Modes

The geometry of the preconditioner update $dQ$ is controlled via
`preconditioner_mode`.

| Mode        | Formula                             | Description                                                                                                                       |
| :---------- | :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| `Q0.5EQ1.5` | $dQ = Q^{0.5} \mathcal{E} Q^{1.5}$ | **Recommended**. Uses an online orthogonal Procrustes solver to keep $Q$ approximately SPD. Numerically stable for low precision. |
| `EQ`        | $dQ = \mathcal{E} Q$               | Triangular preconditioner update. Requires triangular solves. Only mode compatible with triangular $Q$.                           |
| `QUAD`      | Quadratic Form                      | Ensures $Q$ remains symmetric positive definite via quadratic form updates.                                                       |
| `NS`        | Newton-Schulz                       | Iteratively projects $Q$ onto the SPD manifold using Newton-Schulz iterations. Exact but more expensive.                          |
| `EXP`       | Matrix Exponential                  | Geodesic update on the SPD manifold. Uses matrix exponential.                                                                     |
| `TAYLOR2`   | Taylor Expansion                    | Second-order Taylor approximation of the matrix exponential update.                                                               |
| `HYPER`     | Hyperbolic                          | Multiplicative hyperbolic update.                                                                                                 |

### Magma Specifics

Magma acts as an intervention layer for optimizer updates that expose
`use_magma=True`: AdamW, Muon, PRISM, Aurora, PSGD/Kron, and wrappers that
explicitly forward those arguments. Composition wrappers can intentionally keep
Magma out of their public API or apply it only to their base optimizer branch;
check the wrapper signature rather than assuming every wrapper exposes Magma.

When Magma is enabled on AdamW, Muon, PRISM, Aurora, or PSGD/Kron with weight
decay, the decay contribution is part of the base update passed through Magma.
Hyperball and SODA wrappers are different compositions: Hyperball replaces
ordinary decoupled decay with a terminal projection, while SODA disables base
weight decay in its convenience wrappers and adds its initialization anchor
separately.

**Architectural Warning:** Magma introduces intentional update bias (damping)
that scales down the expected update magnitude. At equilibrium, you may need to
scale your global learning rate by ~4x to maintain the undamped update volume
and prevent vanishing progress.

| Parameter   | Default | Description                                                                                                                                                                                                                                                            |
| :---------- | :------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_magma` | `False` | Enables Momentum-aligned gradient masking. Operates at the PyTree leaf level to ensure strict cryptographic PRNG independence and JAX topological isomorphism.                                                                                                         |
| `magma_p`   | `0.5`   | Bernoulli masking probability in `[0, 1]`. A value of `1.0` keeps every leaf active while still applying the alignment score; `0.0` suppresses Magma-managed leaves.                                                                                                  |
| `magma_tau` | `2.0`   | Positive temperature parameter for the alignment sigmoid $\sigma(\text{cossim} / \tau)$. At default `2.0`, non-masked steps scale updates by ~0.5, which combined with 50% Bernoulli masking yields an expected magnitude attenuation of ~0.25x.                    |
| `key`       | `jax.random.PRNGKey(42)` | PRNG key used to initialize Magma's stateful Bernoulli sampling stream. Pass an explicit JAX key when composing multiple Magma-enabled branches and split keys per branch when deterministic independence matters. |

______________________________________________________________________

## Development Validation

Before release, run the same validation surfaces used by CI:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run ty check
uv run pytest -q
```

## Citations

If you use `rollfast` in your research, please cite the relevant papers for the algorithms you utilize.

**PRISM:**

```bibtex
@misc{yang2026prism,
  Author = {Yujie Yang},
  Title = {PRISM: Structured Optimization via Anisotropic Spectral Shaping},
  Year = {2026},
  Eprint = {arXiv:2602.03096},
}

@misc{cesista2026bidirectional,
  Author = {Ferth Louie Cesista},
  Title = {Bidirectional-PRISM: Kronecker-Factored Optimization via Anisotropic Spectral Shaping},
  Year = {2026},
  Url = {https://leloykun.github.io/ponder/shampoo-prism/},
}
```

**Aurora:**

```bibtex
@article{dewulf2026aurora,
  title   = {Aurora: A Leverage-Aware Optimizer for Rectangular Matrices},
  author  = {Dewulf, Alec and Pai, Dhruv and Yang, Li and Zhang, Ashley
             and Keigwin, Ben},
  year    = {2026},
  url     = {https://tilderesearch.com/blog/aurora}
}
```

**Pion:**

```bibtex
@article{pion2026,
  title   = {Pion: A Spectrum-Preserving Optimizer via Orthogonal Equivalence Transformation},
  author  = {Shi, Kexuan and Li, Hanxuan and Qiu, Zeju and Wen, Yandong
             and Buchholz, Simon and Liu, Weiyang},
  journal = {arXiv preprint arXiv:2605.12492},
  year    = {2026}
}
```

**RMNP:**

```bibtex
@article{deng2026rmnp,
  title   = {RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization},
  author  = {Deng, Shenyang and Ouyang, Zhuoli and Pang, Tianyu and Liu, Zihang
             and Jin, Ruochen and Yu, Shuhua and Yang, Yaoqing},
  journal = {arXiv preprint arXiv:2603.20527},
  year    = {2026}
}
```

**Schedule-Free:**

```bibtex
@misc{defazio2024road,
  Author = {Aaron Defazio and Xingyu Alice Yang and Harsh Mehta and Konstantin Mishchenko and Ahmed Khaled and Ashok Cutkosky},
  Title = {The Road Less Scheduled},
  Year = {2024},
  Eprint = {arXiv:2405.15682},
}

@misc{pun2025schedulers,
  Author = {Yuen-Man Pun and Matthew Buchholz and Robert M. Gower},
  Title = {Schedulers for Schedule-free: Theoretically inspired hyperparameters},
  Year = {2025},
  Eprint = {arXiv:2511.07767},
}

@misc{defazio2026schedulefreeplus,
  Author = {A. Defazio},
  Title = {ScheduleFree+: Scaling Learning-Rate-Free \& Schedule-Free Learning to Large Language Models},
  Year = {2026},
  Eprint = {arXiv:2605.19095},
}
```

**SODA:**

```bibtex
@misc{pethick2026optimistic,
  Author = {Thomas Pethick and Wanyun Xie and Roman Machacek and Volkan Cevher},
  Title = {Optimistic Dual Averaging Unifies Modern Optimizers},
  Year = {2026},
  Eprint = {arXiv:2605.11172},
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

**Hyperball:**

```bibtex
@online{wen2025hyperball,
  title   = {Fantastic Pretraining Optimizers and Where to Find Them 2.1: Hyperball Optimization},
  author  = {Wen, Kaiyue and Dang, Xingyu and Lyu, Kaifeng and Ma, Tengyu and Liang, Percy},
  year    = {2025},
  month   = {12},
  day     = {15},
  url     = {https://tinyurl.com/muonh},
  urldate = {2025-12-15}
}
```

**Magma:**

```bibtex
@misc{joo2026surprising,
  Author = {Taejong Joo and Wenhan Xia and Cheolmin Kim and Ming Zhang and Eugene Ie},
  Title = {On Surprising Effectiveness of Masking Updates in Adaptive Optimizers},
  Year = {2026},
  Eprint = {arXiv:2602.15322},
}
```
