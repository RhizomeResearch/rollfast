# rollfast: Advanced Optimization Primitives in JAX

`rollfast` is a high-performance optimization library for JAX, designed to
implement cutting-edge optimizers that go beyond standard Euclidean gradient
descent. It provides production-ready implementations of optimizers like
**PSGD** (Preconditioned Stochastic Gradient Descent), **PRISM** (Anisotropic
Spectral Shaping), **Aurora** (leverage-aware rectangular matrix optimization),
and **Pion** (spectrum-preserving orthogonal equivalence updates), plus
**RMNP** (row-momentum normalized preconditioning), **Hyperball**
norm-preserving weight decay, **SODA**, and a robust
**Schedule-Free** wrapper.

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
- **Modes**: Supports `original` (Newton-Schulz iterations on an augmented matrix) and `bidirectional` (Shampoo-style bilateral shaping of both left and right singular-vector spaces).
- **Partitioning**: Automatically partitions parameters. High-rank tensors
  (Linear/Conv weights) are optimized via PRISM; vectors (biases, layernorms) are
  optimized via AdamW.
- **Reference**: *PRISM: Structured Optimization via Anisotropic Spectral
  Shaping* (Yang, 2026) and *Bidirectional-PRISM: Kronecker-Factored Optimization via Anisotropic Spectral Shaping* (Cesista, 2026).

### 2. Aurora (Leverage-Aware Matrix Optimization)

Aurora optimizes matrix-shaped parameters by applying polar-style updates with
leverage-aware balancing for rectangular matrices. This gives rectangular layers
row/column-aware update geometry instead of treating every element
independently.

- **Mechanism**: Uses Newton-Schulz polar iterations, with practical diagonal
  balancing for rectangular matrices. `riemannian_aurora` provides a more
  expensive balanced-Stiefel variant for reference-quality updates.
- **Partitioning**: Automatically applies Aurora to matrix leaves and AdamW to
  vectors/scalars. Explicit dimension specs can opt convolution kernels or other
  high-rank tensors into Aurora.
- **Reference**: *Aurora: A Leverage-Aware Optimizer for Rectangular Matrices*
  (Dewulf et al., 2026).

### 3. PSGD Kron (Lie Group Preconditioning)

PSGD reformulates preconditioner estimation as a strongly convex optimization
problem on Lie groups. It updates the preconditioner $Q$ (where $P = Q^T Q$)
using multiplicative updates that avoid explicit matrix inversion.

- **Mechanism**: Maintains a Kronecker-factored preconditioner updated via the
  triangular or orthogonal group.
- **Reference**: *Stochastic Hessian Fittings with Lie Groups* (Li, 2024).

### 4. Hyperball Optimization

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
- **Partitioning**: By default, Hyperball applies to rank >= 2 leaves. PRISM and
  Aurora wrappers apply Hyperball to the same leaves routed to the structured
  optimizer branch, while fallback leaves use Adam-style updates.
- **Reference**: *Fantastic Pretraining Optimizers and Where to Find Them 2.1:
  Hyperball Optimization* (Wen et al., 2025).

### 5. Pion (Spectrum-Preserving Orthogonal Equivalence)

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
- **Reference**: *Pion: A Spectrum-Preserving Optimizer via Orthogonal
  Equivalence Transformation* (Shi et al., 2026).

### 6. RMNP (Row-Momentum Normalized Preconditioning)

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

### 7. SODA (Optimistic Dual Averaging Wrapper)

SODA wraps an existing base optimizer and replaces tuned weight decay with a
parameter-free initialization-centered anchor term that decays as `1 / (k + 2)`.

- **Mechanism**: Adds `(z0 - params) / (k + 2)` to the base optimizer update,
  where `z0` is the initialization. The base optimizer should include its
  learning-rate schedule and should not include weight decay.
- **Wrappers**: `soda_adam`, `soda_prism`, `soda_kron`, `soda_muon`, and
  `soda_rmnp` are provided. `soda_muon` uses `optax.contrib.muon` as its base
  optimizer.
- **Reference**: *Optimistic Dual Averaging Unifies Modern Optimizers*
  (Pethick et al., 2026).

### 8. Schedule-Free Optimization

A wrapper that eliminates the need for complex learning rate schedules by
maintaining two sequences of parameters: a primary sequence $z$ (stepped via the
base optimizer) and an averaged sequence $x$ (used for evaluation). Available for PRISM, PSGD Kron, and Adam.

- **Features**: Supports "Practical", "Schedulet", "Theoretical", and "Power" (SF+) weighting modes for
  theoretically grounded averaging. Optionally supports the newer **ScheduleFree+** components (e.g., Polyak step sizes).
- **Reference**: *The Road Less Scheduled* (Defazio et al., 2024) and *ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models* (Defazio, 2026).

### 9. Magma (Momentum-Aligned Gradient Masking)

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

## Fine-Tuning Plans

`rollfast.finetune` compiles model-library fine-tuning plans into grouped Optax
optimizers. It is designed for Equimo's `equimo.finetune.FineTunePlan`, but the
core only requires a structural plan with `trainable`, `labels`, and
`group_specs` fields.

```python
import rollfast.finetune as rfft

optim = rfft.adamw_from_plan(
    plan,
    total_steps=20_000,
    base_lr=5e-4,
    schedule="warmup_cosine",
    weight_decay=0.05,
    clip_global_norm=1.0,
)

print(optim.report)
opt_state = optim.init(plan.trainable)
```

The compiler validates exact PyTree alignment between trainable leaves and
labels, rejects labels for frozen leaves, applies layer-wise multipliers exactly
once, and keeps no-decay groups out of weight decay. Frozen Equimo leaves remain
`None` in `plan.trainable`, so Adam moments are not allocated for them.

PEFT policies are optimizer-side rules. For example, LoRA+ can give `lora_B`
groups a higher learning rate without changing Equimo's LoRA modules:

```python
optim = rfft.adamw_from_plan(
    lora_plan,
    total_steps=10_000,
    base_lr=2e-4,
    weight_decay=0.0,
    lora_b_lr_ratio=16.0,
)
```

Large effective batches and mixed precision are configured in the optimizer
bundle, not in the model plan:

```python
import jax.numpy as jnp

optim = rfft.adamw_from_plan(
    plan,
    total_steps=20_000,
    accumulation_steps=8,
    moment_dtype=jnp.float32,   # use jnp.bfloat16 for lower-memory moments
    axis_name="data",           # distributed global-norm clipping under pmap
)
```

Schedule-Free Adam is also plan-aware. Use `eval_params` for validation or
checkpointing of the averaged sequence:

```python
import optax

optim = rfft.schedule_free_adam_from_plan(
    plan,
    total_steps=20_000,
    base_lr=5e-4,
    weight_decay=0.0,
    accumulation_steps=4,
)

updates, opt_state = optim.update(grads, opt_state, plan.trainable)
trainable = optax.apply_updates(plan.trainable, updates)
eval_trainable = optim.eval_params(trainable, opt_state)
```

EMA/SWA averaging is explicit and exposes named evaluation views:

```python
optim = rfft.adamw_from_plan(
    plan,
    total_steps=20_000,
    ema=rfft.EMAConfig(enabled=True, decay=0.9999),
    swa=rfft.SWAConfig(enabled=True, start_fraction=0.75),
)

eval_ema = optim.eval_params(trainable, opt_state, view="ema")
eval_swa = optim.eval_params(trainable, opt_state, view="swa")
```

SAM and ASAM are exposed as dedicated two-pass steps, not as ordinary one-pass
Optax transforms. The base optimizer still owns LLRD, weight decay, clipping,
precision, and schedules:

```python
base = rfft.adamw_from_plan(
    plan,
    total_steps=20_000,
    accumulation_steps=1,
)

step = rfft.make_sam_step(
    plan=plan,
    base_optimizer=base,
    config=rfft.SAMConfig(rho=0.05),
    loss_fn=loss_fn,
    microbatch_axis=0,
)

trainable, opt_state, info = step(trainable, opt_state, batch)
```

For ASAM, use `SAMConfig(rho=0.5, adaptive=True, eta=0.01)` as a starting
preset and sweep it for the task. Set `microbatch_axis` when `batch` carries a
leading microbatch dimension; Rollfast accumulates both SAM gradient passes
before applying one base optimizer update.

AdaLoRA rank scheduling is optimizer-side metadata. Rollfast provides a
fixed-shape controller that tracks importance EMAs and emits static rank masks;
Equimo applies those masks to rank-masked adapter modules:

```python
rank_groups = eqft.lora_rank_groups(lora_model)
controller = rfft.make_adalora_controller(
    rank_groups,
    total_steps=20_000,
    config=rfft.AdaLoRAControllerConfig(init_rank=12, target_rank=8),
)
state = controller.init()
state = controller.update(state, importance_scores, applied=True)
rank_pattern = controller.rank_pattern(state)
lora_model = eqft.apply_lora_rank_pattern(lora_model, rank_pattern)
```

Staged fine-tuning can preserve compatible optimizer state while changing the
plan. The conservative default initializes new leaves, preserves shared moment
state by path and shape, and restarts stage-local schedule counters:

```python
stage2_bundle, stage2_state, migration = rfft.reconfigure_optimizer(
    old_plan=linear_probe_plan,
    old_bundle=stage1_bundle,
    old_state=stage1_state,
    new_plan=full_ft_plan,
    new_bundle=stage2_bundle,
    state_policy="preserve_shared",
    counter_policy="restart_schedule",
)
```

The migration report lists preserved, initialized, dropped, incompatible, and
group-changed leaves plus state-byte changes. With
`state_policy="preserve_by_path_and_shape"`, compatible Kron/PSGD
preconditioner and Lipschitz leaves are preserved by parameter path and factor
shape; incompatible factors are initialized from the new optimizer.

Static state-memory diagnostics are available before optimizer initialization.
They estimate optimizer-family moment state plus Kron/PSGD preconditioner factors
and Lipschitz auxiliaries without materializing the Optax state:

```python
estimate = rfft.estimate_optimizer_state_memory(
    plan,
    optim,
    preconditioner_dtype=jnp.bfloat16,
)
print(estimate.preconditioner_bytes)
print(estimate.warnings)
```

Measured state-memory diagnostics are available after optimizer initialization:

```python
summary = rfft.optimizer_state_memory_summary(optim, opt_state)
print(summary.total_bytes)
print(summary.by_category)
print([factor.to_dict() for factor in summary.preconditioner_factors])
```

For Kron/PSGD, preconditioner factors report actual initialized shapes, dtypes,
bytes, and storage class such as `matrix_factor` or `diagonal_factor`. This is
measured from the real optimizer state, so it reflects memory-saving diagonal
modes and low-precision preconditioner dtypes after initialization.

Optimizer state manifests include the plan fingerprint, resolved group table,
schedule metadata, precision policy, accumulation policy, and eval views:

```python
checkpoint = rfft.make_state_checkpoint(
    optim,
    opt_state,
    metadata={"step": 10_000},
)
opt_state = rfft.restore_state_checkpoint(optim, checkpoint)
```

Blockwise 8-bit AdamW state is opt-in. It quantizes eligible first and second
moment leaves only; model parameters and gradients are not quantized, and small
or sensitive groups such as bias, norm, embedding, prompt, IA3, and scale-shift
groups stay in fp32 state by default:

```python
optim = rfft.adamw8_from_plan(
    plan,
    total_steps=20_000,
    base_lr=5e-4,
    state_quantization=rfft.StateQuantizationConfig(
        enabled=True,
        block_size=2048,
        min_size=4096,
    ),
)
```

Structured hybrid optimizers are plan-aware too. They preserve the same labels,
per-group LR multipliers, decay policy, clipping, accumulation, and EMA/SWA
views:

```python
optim = rfft.hybrid_aurora_adam_from_plan(
    plan,
    total_steps=20_000,
    base_lr=5e-4,
    schedule="warmup_cosine",
    weight_decay=0.05,
)
```

Use `hybrid_prism_adam_from_plan` for PRISM/Adam groups or
`hybrid_kron_adam_from_plan` for PSGD/Kron groups. Aurora and PRISM keep their
primitive matrix/Adam partitioning inside each fine-tuning group. Lightweight
JSON smoke harnesses live under `benchmarks/finetuning/`; rerun them on target
hardware and real Equimo models before making task-quality performance claims.

## Usage

### 1. PRISM (Standard)

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
    inv_steps=8,          # Polynomial iterations (for 'bidirectional' mode)
    gamma=1.0,            # Innovation damping
    weight_decay=0.01
)

opt_state = optimizer.init(params)
```

**Equinox Integration:** `rollfast` natively supports Equinox. You can use `get_equinox_prism_spec` to automatically construct the exact dimension specification PyTree matching your model.

```python
import equinox as eqx

model = ... # Your Equinox model
optimizer = prism(
    learning_rate=1e-3,
    prism_weight_dimension_numbers=get_equinox_prism_spec
)
```

### 2. Aurora

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
spec. The Equinox Aurora helper returns compatible specs for Aurora.

```python
from rollfast import aurora, get_equinox_aurora_spec

optimizer = aurora(
    learning_rate=3e-4,
    aurora_weight_dimension_numbers=get_equinox_aurora_spec,
)
```

Use `riemannian_aurora` when you want the more expensive balanced-Stiefel
variant.

### 3. Hyperball

Hyperball wrappers use the same base optimizer arguments as their non-Hyperball
counterparts, but interpret `weight_decay` as the Hyperball decay coefficient on
selected leaves.

```python
from rollfast.optim.hyperball import prism_hyperball

optimizer = prism_hyperball(
    learning_rate=1e-3,
    weight_decay=0.01,
    mode='bidirectional',
    inv_steps=8,
    hyperball_mask=None,       # Defaults to the PRISM-routed leaves
    fallback_weight_decay=False
)
```

Muon is available through Optax's contrib implementation:

```python
from rollfast import muon_hyperball

optimizer = muon_hyperball(
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

### 4. Schedule-Free Optimization

The `schedule_free_*` functions wrap base optimizers with the Schedule-Free logic and the WSD (Warmup-Stable-Decay) scheduler for the internal step size.

```python
from rollfast import schedule_free_prism, schedule_free_eval_params

optimizer = schedule_free_prism(
    learning_rate=1.0,   # Peak LR for internal steps
    total_steps=10000,   # Required for WSD schedule generation
    warmup_fraction=0.1,
    weighting_mode="schedulet",
    sf_b1=0.9,           # Schedule-Free interpolation (beta)
    gamma=0.8,           # PRISM specific arg
)

# In Schedule-Free, updates are applied to the z-sequence parameters.
# For evaluation/validation, use the averaged x-sequence parameters:
eval_params = schedule_free_eval_params(opt_state, params)
```

*Note: We also provide `schedule_free_kron` and `schedule_free_adam`.*

### 5. Pion

```python
from rollfast import pion

optimizer = pion(
    learning_rate=1e-3,
    b1=0.9,
    b2=0.999,
    rms_constant=1.0,
    alternating=True,
)
```

### 6. RMNP

```python
from rollfast import rmnp

optimizer = rmnp(
    learning_rate=1e-3,
    beta=0.95,
    consistent_rms=None,
)
```

For Equinox modules or convolution kernels, pass explicit dimension specs with
`MatrixDimensionNumbers` via `rmnp_weight_dimension_numbers` so row
normalization uses the intended flattened matrix layout.

### 7. SODA

```python
from rollfast import soda_prism, soda_muon, soda_rmnp

optimizer = soda_prism(
    learning_rate=1e-3,
    total_steps=10000,
    mode="bidirectional",
    inv_steps=8,
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
```

SODA convenience wrappers disable base optimizer weight decay and add the
initialization-centered `1 / (k + 2)` anchor term from the paper.

### 8. PSGD Kron

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

If you are bottlenecked by VRAM, you can also store the optimizer's first and second moments in **BF16 with stochastic rounding** by passing `mu_dtype=jnp.bfloat16` to `adamw` or `prism`.

```python
from rollfast import adamw

optimizer = adamw(
    learning_rate=1e-3,
    mu_dtype=jnp.bfloat16  # Moments will be stored as BF16 with SR
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

### PRISM Specifics

| Parameter            | Default    | Description                                                                                 |
| :------------------- | :--------- | :------------------------------------------------------------------------------------------ |
| `mode`               | `original` | `original` (Newton-Schulz augmented) or `bidirectional` (Shampoo-style bilateral shaping).  |
| `ns_iters`           | `5`        | Newton-Schulz iterations. Higher values provide better orthogonality but cost more compute. |
| `inv_steps`          | `8`        | Polynomial iterations for mode `bidirectional`.                                             |
| `gamma`              | `1.0`      | Damping coefficient for the innovation term. Controls the "anisotropy" of spectral shaping. |
| `shape_nesterov`     | `True`     | If True, shapes Nesterov momentum; otherwise shapes raw momentum.                           |
| `adam_learning_rate` | `None`     | Optional override for the Adam branch learning rate. Defaults to `learning_rate` if None.   |

### Aurora Specifics

| Parameter                         | Default        | Description                                                                                         |
| :-------------------------------- | :------------- | :-------------------------------------------------------------------------------------------------- |
| `b1`                              | `0.95`         | Momentum used before Aurora shaping.                                                               |
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
| `hyperball_mask`          | `None`  | Boolean PyTree/callable selecting leaves projected by Hyperball. Defaults to rank >= 2 leaves, or the structured branch for PRISM/Aurora wrappers. |
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
