"""NorMuon, ContraMuon, and ContraNorMuon matrix optimizers.

NorMuon applies Muon-style Newton-Schulz orthogonalization, then normalizes
rows or columns with a second-moment accumulator in the matrix layout described
by ``MatrixDimensionNumbers``. The default rescaling preserves the Muon update
norm after normalization, matching the reference implementation.
"""

from numbers import Real
from typing import Any, Callable, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
from optax._src import base, combine, numerics, transform, utils

from rollfast.optim.adam import adamw
from rollfast.optim.dimension_numbers import (
    MatrixDimensionNumbers,
    WeightDimNumOrFn,
    _compute_matrix_reshape,
    _get_dimension_numbers,
    _is_dimension_numbers_leaf,
    _make_matrix_partition_fns,
    _resolve_update_dimension_numbers,
    _validate_matrix_operand,
)
from rollfast.optim.muon import (
    MuonDimensionNumbers,
    scale_by_muon_shape,
)
from rollfast.optim.orthogonalization import MUON_NS_COEFFS
from rollfast.optim.orthogonalization import (
    MuonNsCoeffs,
    MuonPreconditioning,
    orthogonalize_via_newton_schulz,
    resolve_ns_coeffs,
)
from rollfast.utils import (
    MomentumAccumulator,
    _has_nonzero_or_scheduled,
    _is_aux_leaf,
    _prepare_first_moment_runtime,
    _unzip_leaf_tuple_tree,
    _validate_beta_static_scalar,
    _validate_positive_static_scalar,
    _zeros_like_tree,
)

NorMuonNormalizationAxis = Literal["row", "reduction", "column", "output", "auto"]
NorMuonRescale = Literal["preserve_update_norm", "fixed_rms", "none"]


class ScaleByNorMuonState(NamedTuple):
    """State for Muon variants with optional NorMuon second moment."""

    count: jax.Array
    mu: base.Updates
    nu: base.Updates
    key: jax.Array | None


def _zeros_for_nu(
    param: Any,
    dim_nums: MatrixDimensionNumbers | None,
    dtype: jax.typing.DTypeLike,
    normalization_axis: NorMuonNormalizationAxis,
) -> Any:
    if _is_aux_leaf(param) or dim_nums is None:
        return param
    reshape_fn, _ = _compute_matrix_reshape(param, dim_nums)
    matrix = reshape_fn(param)
    axis, _ = _normalize_axis_for_nor(matrix, normalization_axis)
    if axis == -1:
        shape = matrix.shape[:-1] + (1,)
    else:
        shape = matrix.shape[:-2] + (1, matrix.shape[-1])
    return jnp.zeros(shape, dtype=dtype)


def _orthogonalize_muon(
    x: jax.Array,
    ns_iters: int,
    eps: jax.typing.ArrayLike,
    ns_coeffs: jax.Array,
    preconditioning: MuonPreconditioning,
) -> jax.Array:
    return orthogonalize_via_newton_schulz(
        x,
        jnp.asarray(ns_coeffs),
        ns_steps=ns_iters,
        preconditioning=preconditioning,
        eps=eps,
        dimension_numbers=MuonDimensionNumbers(reduction_axis=-2, output_axis=-1),
    )


def _operator_normalize_power_iter(
    x: jax.Array,
    eps: jax.typing.ArrayLike,
    power_iters: int,
) -> jax.Array:
    x = x.astype(jnp.float32)
    eps32 = jnp.asarray(eps, dtype=jnp.float32)
    col_norms = jnp.sum(numerics.abs_sq(x), axis=-2)
    col_idx = jnp.argmax(col_norms, axis=-1)
    v = jax.nn.one_hot(col_idx, x.shape[-1], dtype=x.dtype)[..., None]

    def body_fn(_, v_):
        u = x @ v_
        u = u / jnp.maximum(jnp.linalg.norm(u, axis=(-2, -1), keepdims=True), eps32)
        v_next = jnp.swapaxes(x, -1, -2) @ u
        return v_next / jnp.maximum(
            jnp.linalg.norm(v_next, axis=(-2, -1), keepdims=True), eps32
        )

    v = jax.lax.fori_loop(0, power_iters, body_fn, v, unroll=True)
    op_norm = jnp.maximum(jnp.linalg.norm(x @ v, axis=(-2, -1), keepdims=True), eps32)
    return x / op_norm


def _normalize_axis_for_nor(
    update: jax.Array,
    normalization_axis: NorMuonNormalizationAxis,
) -> tuple[int, bool]:
    if normalization_axis in ("row", "reduction"):
        return -1, True
    if normalization_axis in ("column", "output"):
        return -2, True
    if normalization_axis != "auto":
        raise ValueError(
            "normalization_axis must be one of 'row', 'reduction', 'column', "
            f"'output', or 'auto', got {normalization_axis!r}."
        )
    if update.shape[-2] >= update.shape[-1]:
        return -1, True
    return -2, True


def _normuon_leaf_update(
    direction: Any,
    dim_nums: MatrixDimensionNumbers | None,
    nu: Any,
    *,
    beta2: jax.typing.ArrayLike | None,
    eps: jax.typing.ArrayLike,
    ns_iters: int,
    ns_coeffs: jax.Array,
    preconditioning: MuonPreconditioning,
    contra_coeff: jax.typing.ArrayLike,
    contra_enabled: bool,
    contra_power_iters: int,
    normalization_axis: NorMuonNormalizationAxis,
    normalization_rescale: NorMuonRescale,
    normalization_rms: jax.typing.ArrayLike,
) -> tuple[Any, Any]:
    if _is_aux_leaf(direction) or dim_nums is None:
        return direction, nu
    _validate_matrix_operand(direction, dim_nums, "scale_by_normuon")
    if direction.ndim < 2:
        raise ValueError(
            f"Muon-variant optimized parameters must have rank >= 2, got {direction.ndim=}."
        )

    reshape_fn, inverse_fn = _compute_matrix_reshape(direction, dim_nums)
    matrix = reshape_fn(direction).astype(jnp.float32)
    eps32 = jnp.asarray(eps, dtype=jnp.float32)
    muon_update = _orthogonalize_muon(
        matrix,
        ns_iters,
        eps32,
        ns_coeffs,
        preconditioning,
    )

    contra32 = jnp.asarray(contra_coeff, dtype=jnp.float32)
    if contra_enabled:
        normalized_momentum = _operator_normalize_power_iter(
            matrix, eps32, contra_power_iters
        )
        original_norm = jnp.linalg.norm(muon_update, axis=(-2, -1), keepdims=True)
        muon_update = muon_update - 0.5 * contra32 * normalized_momentum
        new_norm = jnp.linalg.norm(muon_update, axis=(-2, -1), keepdims=True)
        muon_update = muon_update * (original_norm / jnp.maximum(new_norm, eps32))

    if beta2 is None:
        return inverse_fn(muon_update).astype(direction.dtype), nu

    axis, keepdims = _normalize_axis_for_nor(muon_update, normalization_axis)
    nu_new = jnp.asarray(beta2, dtype=jnp.float32) * nu.astype(jnp.float32) + (
        1.0 - jnp.asarray(beta2, dtype=jnp.float32)
    ) * jnp.mean(numerics.abs_sq(muon_update), axis=axis, keepdims=keepdims)
    pre_norm = jnp.linalg.norm(muon_update, axis=(-2, -1), keepdims=True)
    adapted = muon_update / jnp.sqrt(nu_new + eps32)
    if normalization_rescale == "preserve_update_norm":
        post_norm = jnp.linalg.norm(adapted, axis=(-2, -1), keepdims=True)
        adapted = adapted * (pre_norm / (post_norm + eps32))
    elif normalization_rescale == "fixed_rms":
        rms = jnp.sqrt(jnp.mean(numerics.abs_sq(adapted), axis=(-2, -1), keepdims=True))
        adapted = adapted * (
            jnp.asarray(normalization_rms, dtype=jnp.float32) / (rms + eps32)
        )
    elif normalization_rescale != "none":
        raise ValueError(
            "normalization_rescale must be 'preserve_update_norm', "
            f"'fixed_rms', or 'none', got {normalization_rescale!r}."
        )
    return inverse_fn(adapted).astype(direction.dtype), nu_new


def scale_by_normuon(
    beta1: jax.typing.ArrayLike = 0.95,
    beta2: jax.typing.ArrayLike | None = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    ns_iters: int = 5,
    ns_coeffs: MuonNsCoeffs = MUON_NS_COEFFS,
    *,
    contra_coeff: jax.typing.ArrayLike = 0.0,
    contra_power_iters: int = 5,
    normalization_axis: NorMuonNormalizationAxis = "row",
    normalization_rescale: NorMuonRescale = "preserve_update_norm",
    normalization_rms: jax.typing.ArrayLike = 0.2,
    preconditioning: MuonPreconditioning = "frobenius",
    momentum_accumulator: MomentumAccumulator = "ema",
    mu_dtype: jax.typing.DTypeLike | None = None,
    nesterov: bool = True,
    bias_correction: bool = False,
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """Scale updates with NorMuon/ContraMuon matrix directions.

    ``beta2=None`` disables NorMuon's second-moment normalization, giving plain
    Muon when ``contra_coeff=0`` and ContraMuon when ``contra_coeff>0``.
    ``normalization_axis`` is interpreted after reshaping with
    ``MatrixDimensionNumbers`` into ``(..., reduction, output)``. The default
    ``"row"`` matches the reference NorMuon implementation by averaging across
    output columns. ``normalization_rescale="preserve_update_norm"`` preserves
    the pre-normalization Muon update norm; ``"fixed_rms"`` scales to
    ``normalization_rms`` RMS.
    """
    if ns_iters < 1:
        raise ValueError(f"ns_iters must be >= 1, got {ns_iters}")
    if contra_power_iters < 1:
        raise ValueError(f"contra_power_iters must be >= 1, got {contra_power_iters}")
    _validate_beta_static_scalar("beta1", beta1)
    if beta2 is not None:
        _validate_beta_static_scalar("beta2", beta2)
    _validate_positive_static_scalar("eps", eps)
    if (
        normalization_rescale == "fixed_rms"
        and isinstance(normalization_rms, Real)
        and not isinstance(normalization_rms, bool)
        and normalization_rms <= 0
    ):
        raise ValueError(
            f"normalization_rms must be positive, got {normalization_rms!r}"
        )

    canonical_mu_dtype = cast(
        jax.typing.DTypeLike,
        jnp.float32 if mu_dtype is None else utils.canonicalize_dtype(mu_dtype),
    )

    def resolve_dim_nums(params_or_updates):
        return _get_dimension_numbers(weight_dimension_numbers, params_or_updates)

    def init_fn(params):
        dim_nums = resolve_dim_nums(params)
        return ScaleByNorMuonState(
            count=jnp.zeros([], jnp.int32),
            mu=_zeros_like_tree(params, canonical_mu_dtype),
            nu=jax.tree.map(
                lambda p, d: _zeros_for_nu(p, d, jnp.float32, normalization_axis),
                params,
                dim_nums,
                is_leaf=_is_dimension_numbers_leaf,
            ),
            key=key,
        )

    def update_fn(updates, state, params=None):
        dim_nums = _resolve_update_dimension_numbers(
            weight_dimension_numbers,
            params=params,
            updates=updates,
            transform_name="scale_by_normuon",
        )
        jax.tree.map(
            lambda u, d: _validate_matrix_operand(u, d, "scale_by_normuon"),
            updates,
            dim_nums,
            is_leaf=_is_dimension_numbers_leaf,
        )
        runtime = _prepare_first_moment_runtime(
            updates,
            state.mu,
            state.count,
            state.key,
            beta1,
            canonical_mu_dtype,
            nesterov=nesterov,
            bias_correction=bias_correction,
            momentum_accumulator=momentum_accumulator,
        )
        count_inc = runtime.count
        direction = runtime.direction

        resolved_ns_coeffs = resolve_ns_coeffs(ns_coeffs, ns_iters)
        try:
            contra_enabled = bool(jnp.any(jnp.asarray(contra_coeff) != 0.0))
        except TypeError:
            contra_enabled = True

        result = jax.tree.map(
            lambda u, d, v: _normuon_leaf_update(
                u,
                d,
                v,
                beta2=beta2,
                eps=eps,
                ns_iters=ns_iters,
                ns_coeffs=resolved_ns_coeffs,
                preconditioning=preconditioning,
                contra_coeff=contra_coeff,
                contra_enabled=contra_enabled,
                contra_power_iters=contra_power_iters,
                normalization_axis=normalization_axis,
                normalization_rescale=normalization_rescale,
                normalization_rms=normalization_rms,
            ),
            direction,
            dim_nums,
            state.nu,
            is_leaf=_is_dimension_numbers_leaf,
        )
        new_updates, nu = _unzip_leaf_tuple_tree(result, 2)

        return new_updates, ScaleByNorMuonState(
            count=count_inc,
            mu=runtime.mu_stored,
            nu=nu,
            key=runtime.key,
        )

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_normuon_shape(
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformation:
    """Scale Muon-variant matrix directions using Muon-style shape factors."""
    return scale_by_muon_shape(
        weight_dimension_numbers=weight_dimension_numbers,
        consistent_rms=consistent_rms,
    )


def _partitioned_muon_variant(
    *,
    label: str,
    learning_rate: base.ScalarOrSchedule,
    beta1: jax.typing.ArrayLike,
    beta2: jax.typing.ArrayLike | None,
    eps: jax.typing.ArrayLike,
    weight_decay: base.ScalarOrSchedule,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None,
    ns_iters: int,
    ns_coeffs: MuonNsCoeffs,
    contra_coeff: jax.typing.ArrayLike,
    contra_power_iters: int,
    normalization_axis: NorMuonNormalizationAxis,
    normalization_rescale: NorMuonRescale,
    normalization_rms: jax.typing.ArrayLike,
    preconditioning: MuonPreconditioning,
    momentum_accumulator: MomentumAccumulator,
    mu_dtype: jax.typing.DTypeLike | None,
    nesterov: bool,
    bias_correction: bool,
    adam_b1: jax.typing.ArrayLike,
    adam_b2: jax.typing.ArrayLike,
    adam_eps_root: jax.typing.ArrayLike,
    adam_weight_decay: base.ScalarOrSchedule | None,
    adam_learning_rate: base.ScalarOrSchedule | None,
    weight_dimension_numbers: WeightDimNumOrFn | None,
    consistent_rms: jax.typing.ArrayLike | None,
    key: jax.Array,
) -> base.GradientTransformation:
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate
    effective_adam_weight_decay = (
        weight_decay if adam_weight_decay is None else adam_weight_decay
    )

    key_muon, key_adam = jax.random.split(key, 2)

    partition = _make_matrix_partition_fns(weight_dimension_numbers, label)
    matrix_components = [
        scale_by_normuon(
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            ns_iters=ns_iters,
            ns_coeffs=ns_coeffs,
            contra_coeff=contra_coeff,
            contra_power_iters=contra_power_iters,
            normalization_axis=normalization_axis,
            normalization_rescale=normalization_rescale,
            normalization_rms=normalization_rms,
            preconditioning=preconditioning,
            momentum_accumulator=momentum_accumulator,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
            bias_correction=bias_correction,
            weight_dimension_numbers=partition.masked_specs,
            key=key_muon,
        ),
        scale_by_normuon_shape(
            weight_dimension_numbers=partition.masked_specs,
            consistent_rms=consistent_rms,
        ),
    ]
    if _has_nonzero_or_scheduled(weight_decay):
        matrix_components.append(
            transform.add_decayed_weights(weight_decay, weight_decay_mask)
        )
    matrix_components.append(transform.scale_by_learning_rate(learning_rate))

    return combine.partition(
        transforms={
            label: combine.chain(*matrix_components),
            "adam": adamw(
                learning_rate=adam_learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=eps,
                eps_root=adam_eps_root,
                weight_decay=effective_adam_weight_decay,
                weight_decay_mask=weight_decay_mask,
                mu_dtype=mu_dtype,
                nesterov=nesterov,
                key=key_adam,
            ),
        },
        param_labels=partition.labels,
    )


def normuon(
    learning_rate: base.ScalarOrSchedule,
    beta1: jax.typing.ArrayLike = 0.95,
    beta2: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    ns_iters: int = 5,
    ns_coeffs: MuonNsCoeffs = MUON_NS_COEFFS,
    *,
    mu_dtype: jax.typing.DTypeLike | None = None,
    normalization_axis: NorMuonNormalizationAxis = "row",
    normalization_rescale: NorMuonRescale = "preserve_update_norm",
    normalization_rms: jax.typing.ArrayLike = 0.2,
    preconditioning: MuonPreconditioning = "frobenius",
    momentum_accumulator: MomentumAccumulator = "ema",
    nesterov: bool = True,
    bias_correction: bool = False,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: base.ScalarOrSchedule | None = None,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    normuon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """NorMuon optimizer with automatic matrix/AdamW partitioning.

    This tracks one second moment along the configured matrix layout after
    reshaping tensors into ``(..., reduction, output)``. The default ``"row"``
    follows the reference NorMuon implementation by averaging over output
    columns and keeping one statistic per reduction row; set ``"auto"`` to use
    rows for tall matrices and columns for wide matrices.
    """
    return _partitioned_muon_variant(
        label="normuon",
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        ns_iters=ns_iters,
        ns_coeffs=ns_coeffs,
        contra_coeff=0.0,
        contra_power_iters=5,
        normalization_axis=normalization_axis,
        normalization_rescale=normalization_rescale,
        normalization_rms=normalization_rms,
        preconditioning=preconditioning,
        momentum_accumulator=momentum_accumulator,
        mu_dtype=mu_dtype,
        nesterov=nesterov,
        bias_correction=bias_correction,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps_root=adam_eps_root,
        adam_weight_decay=adam_weight_decay,
        adam_learning_rate=adam_learning_rate,
        weight_dimension_numbers=normuon_weight_dimension_numbers,
        consistent_rms=consistent_rms,
        key=key,
    )


def contramuon(
    learning_rate: base.ScalarOrSchedule,
    beta1: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    ns_iters: int = 5,
    ns_coeffs: MuonNsCoeffs = MUON_NS_COEFFS,
    *,
    contra_coeff: jax.typing.ArrayLike = 0.4,
    contra_power_iters: int = 5,
    mu_dtype: jax.typing.DTypeLike | None = None,
    preconditioning: MuonPreconditioning = "frobenius",
    momentum_accumulator: MomentumAccumulator = "ema",
    nesterov: bool = True,
    bias_correction: bool = False,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: base.ScalarOrSchedule | None = None,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    contramuon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """ContraMuon optimizer with automatic matrix/AdamW partitioning.

    Subtracts ``contra_coeff / 2`` times a power-iteration estimate of the
    operator-normalized momentum direction, then restores the Muon update norm.
    """
    return _partitioned_muon_variant(
        label="contramuon",
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=None,
        eps=eps,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        ns_iters=ns_iters,
        ns_coeffs=ns_coeffs,
        contra_coeff=contra_coeff,
        contra_power_iters=contra_power_iters,
        normalization_axis="row",
        normalization_rescale="preserve_update_norm",
        normalization_rms=0.2,
        preconditioning=preconditioning,
        momentum_accumulator=momentum_accumulator,
        mu_dtype=mu_dtype,
        nesterov=nesterov,
        bias_correction=bias_correction,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps_root=adam_eps_root,
        adam_weight_decay=adam_weight_decay,
        adam_learning_rate=adam_learning_rate,
        weight_dimension_numbers=contramuon_weight_dimension_numbers,
        consistent_rms=consistent_rms,
        key=key,
    )


def contranormuon(
    learning_rate: base.ScalarOrSchedule,
    beta1: jax.typing.ArrayLike = 0.95,
    beta2: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    ns_iters: int = 5,
    ns_coeffs: MuonNsCoeffs = MUON_NS_COEFFS,
    *,
    contra_coeff: jax.typing.ArrayLike = 0.4,
    contra_power_iters: int = 5,
    mu_dtype: jax.typing.DTypeLike | None = None,
    normalization_axis: NorMuonNormalizationAxis = "row",
    normalization_rescale: NorMuonRescale = "preserve_update_norm",
    normalization_rms: jax.typing.ArrayLike = 0.2,
    preconditioning: MuonPreconditioning = "frobenius",
    momentum_accumulator: MomentumAccumulator = "ema",
    nesterov: bool = True,
    bias_correction: bool = False,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: base.ScalarOrSchedule | None = None,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    contranormuon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """ContraMuon plus NorMuon normalization with AdamW fallback."""
    return _partitioned_muon_variant(
        label="contranormuon",
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        ns_iters=ns_iters,
        ns_coeffs=ns_coeffs,
        contra_coeff=contra_coeff,
        contra_power_iters=contra_power_iters,
        normalization_axis=normalization_axis,
        normalization_rescale=normalization_rescale,
        normalization_rms=normalization_rms,
        preconditioning=preconditioning,
        momentum_accumulator=momentum_accumulator,
        mu_dtype=mu_dtype,
        nesterov=nesterov,
        bias_correction=bias_correction,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps_root=adam_eps_root,
        adam_weight_decay=adam_weight_decay,
        adam_learning_rate=adam_learning_rate,
        weight_dimension_numbers=contranormuon_weight_dimension_numbers,
        consistent_rms=consistent_rms,
        key=key,
    )


__all__ = [
    "ScaleByNorMuonState",
    "NorMuonNormalizationAxis",
    "NorMuonRescale",
    "contramuon",
    "contranormuon",
    "normuon",
    "scale_by_normuon",
    "scale_by_normuon_shape",
]
