"""
Tucker decomposition via gradient descent with L1 loss (noise robust).
L1 loss is more robust to outliers (salt-and-pepper noise) than Frobenius/L2.
"""
from typing import Tuple, List

import jax
import jax.numpy as jnp
import optax

from src.decomposition import TuckerResult
from src.decomposition.tucker_grad import _params_from_tucker, _tucker_from_params
from src.decomposition.hosvd import decompose as hosvd_decompose
from src.utils.tensor_ops import multi_mode_product


def _make_step_l1(optimizer, ndim: int):
    @jax.jit
    def step(params, opt_state, tensor):
        def loss_fn(p):
            X_hat = multi_mode_product(p["core"], [p[f"A{n}"] for n in range(ndim)])
            return jnp.sum(jnp.abs(tensor - X_hat))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    return step


def decompose(
    tensor: jnp.ndarray,
    rank: list,
    learning_rate: float = 1e-2,
    max_iter: int = 1000,
    tol: float = 1e-6,
    **kwargs,
) -> Tuple[TuckerResult, List[float]]:
    """Tucker decomposition via Adam + L1 loss. Returns (TuckerResult, errors)."""
    params = _params_from_tucker(hosvd_decompose(tensor, rank))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    step = _make_step_l1(optimizer, tensor.ndim)

    tensor_norm = float(jnp.linalg.norm(tensor))
    errors: List[float] = []
    prev_error = float("inf")

    for _ in range(max_iter):
        params, opt_state, loss = step(params, opt_state, tensor)
        # Report relative Frobenius error for comparability
        X_hat = multi_mode_product(params["core"], [params[f"A{n}"] for n in range(tensor.ndim)])
        error = float(jnp.linalg.norm(tensor - X_hat) / tensor_norm)
        errors.append(error)
        if abs(prev_error - error) < tol:
            break
        prev_error = error

    return _tucker_from_params(params, tensor.ndim), errors


def reconstruct(result: TuckerResult) -> jnp.ndarray:
    return multi_mode_product(result.core, result.factors)
