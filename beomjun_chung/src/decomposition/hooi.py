"""
Tucker HOOI: Higher-Order Orthogonal Iteration. Kolda & Bader (2009) Fig 4.4.

Algorithm:
    initialize A^(n) using HOSVD
    repeat:
        for n = 1, ..., N:
            Y <- X ×_{k!=n} A^(k)T
            A^(n) <- R_n leading left singular vectors of Y_(n)
    until convergence
    G <- X ×_1 A^(1)T ... ×_N A^(N)T
"""
from functools import partial
from typing import Tuple, List

import jax
import jax.numpy as jnp

from src.decomposition import TuckerResult
from src.decomposition.hosvd import decompose as hosvd_decompose
from src.utils.tensor_ops import unfold, mode_n_product, multi_mode_product


@partial(jax.jit, static_argnums=(2, 3))
def _update_factor(tensor: jnp.ndarray, factors: list, n: int, rank_n: int) -> jnp.ndarray:
    """HOOI inner update for mode n. JIT'd for efficiency."""
    Y = tensor
    for k in range(tensor.ndim):
        if k != n:
            Y = mode_n_product(Y, factors[k].T, k)
    U, _, _ = jnp.linalg.svd(unfold(Y, n), full_matrices=False)
    return U[:, :rank_n]


def decompose(
    tensor: jnp.ndarray,
    rank: list,
    max_iter: int = 500,
    tol: float = 1e-6,
    **kwargs,
) -> Tuple[TuckerResult, List[float]]:
    """HOOI Tucker decomposition. Returns (TuckerResult, per-iteration errors)."""
    factors = list(hosvd_decompose(tensor, rank).factors)
    errors: List[float] = []
    prev_error = float("inf")

    for _ in range(max_iter):
        for n in range(tensor.ndim):
            factors[n] = _update_factor(tensor, factors, n, rank[n])

        core = multi_mode_product(tensor, [A.T for A in factors])
        X_hat = multi_mode_product(core, factors)
        error = float(jnp.linalg.norm(tensor - X_hat) / jnp.linalg.norm(tensor))
        errors.append(error)

        if abs(prev_error - error) < tol:
            break
        prev_error = error

    core = multi_mode_product(tensor, [A.T for A in factors])
    return TuckerResult(core=core, factors=factors), errors


def reconstruct(result: TuckerResult) -> jnp.ndarray:
    return multi_mode_product(result.core, result.factors)
