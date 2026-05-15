import math
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from optax._src import base, combine, numerics, utils
from optax.transforms import _masking

from rollfast.optim.adam import adamw
from rollfast.optim.dimension_numbers import (
    MatrixDimensionNumbers,
    WeightDimNumOrFn,
    _get_dimension_numbers,
    _compute_matrix_reshape,
    _is_dimension_numbers_leaf,
    _mask_dimension_numbers,
)
from rollfast.utils import _tree_stochastic_cast

"""Pion usage notes.

Pion preserves each optimized matrix's singular values by construction. Treat the
initial spectrum as part of the model design: avoid zero, rank-deficient, badly
conditioned, or intentionally under-scaled initialization for matrices routed to
Pion, because the optimizer will not repair those properties later.

Do not route every 2D tensor blindly. Large embedding tables and LM heads can be
expensive because Pion stores in-side and out-side square moment matrices; route
them to the AdamW fallback with ``pion_weight_dimension_numbers`` when that cost
is not acceptable. Keep learnable scale paths, such as normalization parameters,
gates, residual scales, and biases, on the AdamW branch when the model needs
magnitude adaptation.

Weight decay is intentionally applied only to the AdamW fallback branch in
``pion``. Applying additive decay to Pion-managed matrices would change the
singular spectrum and work against the optimizer's core invariant. For
higher-rank tensors such as convolution kernels, provide explicit
``MatrixDimensionNumbers`` so the preserved matrix spectrum matches the intended
input/output axes.
"""


class ScaleByPionState(NamedTuple):
    """State for Pion's Lie-algebra moment estimates."""

    count: jax.Array
    m_in: base.Updates
    v_in: base.Updates
    m_out: base.Updates
    v_out: base.Updates
    key: jax.Array | None


def _zeros_for_pion(
    param: Any,
    dim_nums: MatrixDimensionNumbers | None,
    dtype: jax.typing.DTypeLike,
) -> tuple[Any, Any, Any, Any]:
    if isinstance(param, _masking.MaskedNode):
        node = _masking.MaskedNode()
        return node, node, node, node
    if param is None or dim_nums is None:
        return None, None, None, None
    if param.ndim < 2:
        raise ValueError(
            f"Pion optimized parameters must have rank >= 2, got {param.ndim=}."
        )

    reshape_fn, _ = _compute_matrix_reshape(param, dim_nums)
    matrix = reshape_fn(param)
    batch, rows, cols = matrix.shape
    m_in = jnp.zeros((batch, cols, cols), dtype=dtype)
    v_in = jnp.zeros((batch, cols, cols), dtype=dtype)
    m_out = jnp.zeros((batch, rows, rows), dtype=dtype)
    v_out = jnp.zeros((batch, rows, rows), dtype=dtype)
    return m_in, v_in, m_out, v_out


def _cast_state_tree(tree: base.Params, dtype: jax.typing.DTypeLike) -> base.Params:
    return jax.tree.map(
        lambda x: (
            x if isinstance(x, _masking.MaskedNode) or x is None else x.astype(dtype)
        ),
        tree,
        is_leaf=lambda x: isinstance(x, _masking.MaskedNode) or x is None,
    )


def _e2(
    A: jax.Array,
    eta_alpha: jax.Array,
    precision: jax.lax.PrecisionLike,
) -> jax.Array:
    size = A.shape[-1]
    eye = jnp.eye(size, dtype=A.dtype)
    X = eta_alpha[..., None, None] * A
    return eye + X + 0.5 * jnp.matmul(X, X, precision=precision)


def _pion_matrix_update(
    grad: Any,
    param: Any,
    m_in_prev: Any,
    v_in_prev: Any,
    m_out_prev: Any,
    v_out_prev: Any,
    dim_nums: MatrixDimensionNumbers | None,
    learning_rate: base.ScalarOrSchedule,
    b1: float,
    b2: float,
    rms_constant: float,
    eps: float,
    alternating: bool,
    count: jax.Array,
    precision: jax.lax.PrecisionLike,
) -> tuple[Any, Any, Any, Any, Any]:
    if isinstance(param, _masking.MaskedNode) or isinstance(grad, _masking.MaskedNode):
        node = _masking.MaskedNode()
        return node, node, node, node, node
    if grad is None or param is None or dim_nums is None:
        return grad, m_in_prev, v_in_prev, m_out_prev, v_out_prev

    reshape_fn, inverse_fn = _compute_matrix_reshape(param, dim_nums)
    W = reshape_fn(param).astype(jnp.float32)
    G = reshape_fn(grad).astype(jnp.float32)

    G_in = jnp.matmul(jnp.swapaxes(W, -1, -2), G, precision=precision)
    G_in = G_in - jnp.swapaxes(G_in, -1, -2)
    G_out = jnp.matmul(G, jnp.swapaxes(W, -1, -2), precision=precision)
    G_out = G_out - jnp.swapaxes(G_out, -1, -2)

    m_in = b1 * m_in_prev.astype(jnp.float32) + (1.0 - b1) * G_in
    v_in = b2 * v_in_prev.astype(jnp.float32) + (1.0 - b2) * jnp.square(G_in)
    m_out = b1 * m_out_prev.astype(jnp.float32) + (1.0 - b1) * G_out
    v_out = b2 * v_out_prev.astype(jnp.float32) + (1.0 - b2) * jnp.square(G_out)

    A_in = -m_in / (jnp.sqrt(v_in) + eps)
    A_out = -m_out / (jnp.sqrt(v_out) + eps)

    rows, cols = W.shape[-2], W.shape[-1]
    target_rms = rms_constant * math.sqrt(rows * cols)
    lr = learning_rate(count) if callable(learning_rate) else learning_rate
    lr = jnp.asarray(lr, dtype=jnp.float32)

    def out_step():
        direction = jnp.matmul(A_out, W, precision=precision)
        alpha = target_rms / (jnp.linalg.norm(direction, axis=(-2, -1)) + eps)
        E_out = _e2(A_out, lr * alpha, precision)
        return jnp.matmul(E_out, W, precision=precision)

    def in_step():
        direction = jnp.matmul(W, A_in, precision=precision)
        alpha = target_rms / (jnp.linalg.norm(direction, axis=(-2, -1)) + eps)
        E_in = _e2(A_in, lr * alpha, precision)
        return jnp.matmul(W, E_in, precision=precision)

    def bilateral_step():
        left = jnp.matmul(A_out, W, precision=precision)
        right = jnp.matmul(W, A_in, precision=precision)
        alpha = target_rms / (jnp.linalg.norm(left + right, axis=(-2, -1)) + eps)
        E_out = _e2(A_out, lr * alpha, precision)
        E_in = _e2(A_in, lr * alpha, precision)
        return jnp.matmul(
            jnp.matmul(E_out, W, precision=precision), E_in, precision=precision
        )

    if alternating:
        W_new = jax.lax.cond(count % 2 == 0, out_step, in_step)
    else:
        W_new = bilateral_step()

    update = inverse_fn(W_new - W).astype(param.dtype)
    return update, m_in, v_in, m_out, v_out


def scale_by_pion(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    rms_constant: float = 1.0,
    eps: float = 1e-8,
    alternating: bool = True,
    mu_dtype: jax.typing.DTypeLike | None = None,
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    r"""Pion's spectrum-preserving matrix update as an Optax transform.

    For a matrix parameter ``W`` and gradient ``G``, Pion forms skew-symmetric
    in/out Lie-algebra gradients, tracks Adam-style first and second moments in
    those tangent spaces, and applies the paper's second-order exponential
    approximation. The returned Optax update is ``W_next - W`` because Optax
    applies updates by addition.
    """
    if mu_dtype is None:
        mu_dtype = jnp.float32
    else:
        mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        dim_nums = _get_dimension_numbers(weight_dimension_numbers, params)
        states = jax.tree.map(
            lambda p, d: _zeros_for_pion(p, d, mu_dtype),
            params,
            dim_nums,
            is_leaf=_is_dimension_numbers_leaf,
        )
        m_in = jax.tree.map(
            lambda x: x[0], states, is_leaf=lambda x: isinstance(x, tuple)
        )
        v_in = jax.tree.map(
            lambda x: x[1], states, is_leaf=lambda x: isinstance(x, tuple)
        )
        m_out = jax.tree.map(
            lambda x: x[2], states, is_leaf=lambda x: isinstance(x, tuple)
        )
        v_out = jax.tree.map(
            lambda x: x[3], states, is_leaf=lambda x: isinstance(x, tuple)
        )
        return ScaleByPionState(
            count=jnp.zeros([], jnp.int32),
            m_in=m_in,
            v_in=v_in,
            m_out=m_out,
            v_out=v_out,
            key=key,
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("`params` must be provided to `scale_by_pion`.")

        count_inc = numerics.safe_increment(state.count)
        dim_nums = _get_dimension_numbers(weight_dimension_numbers, params)

        out = jax.tree.map(
            lambda g, p, mi, vi, mo, vo, d: _pion_matrix_update(
                g,
                p,
                mi,
                vi,
                mo,
                vo,
                d,
                learning_rate,
                b1,
                b2,
                rms_constant,
                eps,
                alternating,
                state.count,
                precision,
            ),
            updates,
            params,
            state.m_in,
            state.v_in,
            state.m_out,
            state.v_out,
            dim_nums,
            is_leaf=_is_dimension_numbers_leaf,
        )

        new_updates = jax.tree.map(
            lambda x: x[0], out, is_leaf=lambda x: isinstance(x, tuple)
        )
        m_in = jax.tree.map(lambda x: x[1], out, is_leaf=lambda x: isinstance(x, tuple))
        v_in = jax.tree.map(lambda x: x[2], out, is_leaf=lambda x: isinstance(x, tuple))
        m_out = jax.tree.map(
            lambda x: x[3], out, is_leaf=lambda x: isinstance(x, tuple)
        )
        v_out = jax.tree.map(
            lambda x: x[4], out, is_leaf=lambda x: isinstance(x, tuple)
        )

        if mu_dtype == jnp.bfloat16:
            key, sr_in, sr_vin, sr_out, sr_vout = jax.random.split(state.key, 5)
            m_in = _tree_stochastic_cast(m_in, mu_dtype, sr_in)
            v_in = _tree_stochastic_cast(v_in, mu_dtype, sr_vin)
            m_out = _tree_stochastic_cast(m_out, mu_dtype, sr_out)
            v_out = _tree_stochastic_cast(v_out, mu_dtype, sr_vout)
        else:
            key = state.key
            m_in = _cast_state_tree(m_in, mu_dtype)
            v_in = _cast_state_tree(v_in, mu_dtype)
            m_out = _cast_state_tree(m_out, mu_dtype)
            v_out = _cast_state_tree(v_out, mu_dtype)

        return new_updates, ScaleByPionState(
            count=count_inc,
            m_in=m_in,
            v_in=v_in,
            m_out=m_out,
            v_out=v_out,
            key=key,
        )

    return base.GradientTransformation(init_fn, update_fn)


def pion(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    rms_constant: float = 1.0,
    eps: float = 1e-8,
    alternating: bool = True,
    mu_dtype: jax.typing.DTypeLike | None = None,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    pion_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformation:
    """Pion optimizer with AdamW fallback for non-matrix parameters.

    Matrix leaves are optimized by Pion's orthogonal-equivalence update. Leaves
    without a Pion dimension specification, such as biases and normalization
    scales, are optimized by AdamW.
    """
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    key_pion, key_adam = jax.random.split(key, 2)

    def get_resolved_dim_nums(params):
        return _get_dimension_numbers(pion_weight_dimension_numbers, params)

    def param_labels(params):
        dim_nums = get_resolved_dim_nums(params)
        return jax.tree.map(
            lambda d, p: None if p is None else ("pion" if d is not None else "adam"),
            dim_nums,
            params,
            is_leaf=_is_dimension_numbers_leaf,
        )

    def pion_weight_dim_nums_fn(params):
        return _mask_dimension_numbers(get_resolved_dim_nums(params))

    return combine.partition(
        transforms={
            "pion": scale_by_pion(
                learning_rate=learning_rate,
                b1=b1,
                b2=b2,
                rms_constant=rms_constant,
                eps=eps,
                alternating=alternating,
                mu_dtype=mu_dtype,
                weight_dimension_numbers=pion_weight_dim_nums_fn,
                precision=precision,
                key=key_pion,
            ),
            "adam": adamw(
                learning_rate=adam_learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
                weight_decay=weight_decay,
                weight_decay_mask=weight_decay_mask,
                mu_dtype=mu_dtype,
                key=key_adam,
            ),
        },
        param_labels=param_labels,
    )
