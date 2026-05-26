"""
CP-ALS: CANDECOMP/PARAFAC via Alternating Least Squares. Kolda & Bader (2009) Fig 3.3.

Algorithm:
    initialize A^(n) for n = 1, ..., N (SVD or random)
    repeat:
        for n = 1, ..., N:
            V = Hadamard product of A^(k)T A^(k) for all k != n
            KR = Khatri-Rao product of A^(k) for all k != n (in order)
            A^(n) <- X_(n) KR V^(-1)
            normalize columns: lambda_r = ||a_r||, a_r /= lambda_r
    until convergence
"""
from functools import partial
from typing import Tuple, List

import jax
import jax.numpy as jnp

from src.decomposition import CPResult
from src.utils.tensor_ops import unfold, khatri_rao

_DEGENERACY_THRESHOLD = 1e6

def _cp_reconstruct(weights: jnp.ndarray, factors: list) -> jnp.ndarray:
    """Reconstruct CP tensor: sum_r lambda_r a_r ⊗ b_r ⊗ c_r (general N-way)."""
    N = len(factors)
    indices = "".join(chr(ord("i") + n) for n in range(N))
    subscripts = "r," + ",".join(f"{chr(ord('i') + n)}r" for n in range(N)) + "->" + indices
    return jnp.einsum(subscripts, weights, *factors)

@partial(jax.jit, static_argnums=(2, 3))
def _update_mode(tensor: jnp.ndarray, factors: list, n: int, rank: int):
    """CP-ALS inner update for mode n. JIT'd for efficiency."""
    N = tensor.ndim
    other = [k for k in range(N) if k != n]

    # Gram matrix product V = Hadamard of A_k^T A_k for k != n
    V = jnp.ones((rank, rank))
    for k in other:
        V = V * (factors[k].T @ factors[k])

    # Khatri-Rao product of all factors except n (in index order)
    KR = factors[other[0]]
    for k in other[1:]:
        KR = khatri_rao(KR, factors[k])

    # Least-squares update: A_n = X_(n) @ KR @ pinv(V)
    A_new = unfold(tensor, n) @ KR @ jnp.linalg.pinv(V)

    # Column normalization
    norms = jnp.linalg.norm(A_new, axis=0)
    norms = jnp.where(norms > 0, norms, jnp.ones_like(norms))
    return A_new / norms[None, :], norms

def _init_factors_svd(tensor: jnp.ndarray, rank: int) -> list:
    """Initialize factors using leading singular vectors of each mode unfolding."""
    factors = []
    for n in range(tensor.ndim):
        X_n = unfold(tensor, n)
        U, _, _ = jnp.linalg.svd(X_n, full_matrices=False)
        r = min(rank, U.shape[1])
        if r < rank:
            pad = jax.random.normal(jax.random.PRNGKey(n), (U.shape[0], rank - r))
            U = jnp.concatenate([U[:, :r], pad], axis=1)
        else:
            U = U[:, :rank]
        factors.append(U)
    return factors

def decompose(
    tensor: jnp.ndarray,
    rank: int,
    max_iter: int = 500,
    tol: float = 1e-6,
    init: str = "svd",
    random_seed: int = 0,
    **kwargs,
) -> Tuple[CPResult, List[float]]:
    """
    CP-ALS decomposition. Returns (CPResult, per-iteration errors).
    rank: scalar integer (CP rank).
    """
    if init == "svd":
        factors = _init_factors_svd(tensor, rank)
    else:
        key = jax.random.PRNGKey(random_seed)
        factors = [
            jax.random.normal(key, (tensor.shape[n], rank)) for n in range(tensor.ndim)
        ]

    weights = jnp.ones(rank)
    errors: List[float] = []
    prev_error = float("inf")

    for it in range(max_iter):
        for n in range(tensor.ndim):
            factors[n], col_norms = _update_mode(tensor, factors, n, rank)
            # Overwrite (not accumulate): K&B Fig 3.3 absorbs norms into λ at each mode step.
            # Accumulating across modes would give weights = λ^N at convergence (wrong).
            weights = col_norms

        # Check for degeneracy
        max_norm = float(jnp.max(jnp.abs(weights)))
        if max_norm > _DEGENERACY_THRESHOLD:
            print(f"[CP-ALS] iter {it}: degeneracy detected (max norm={max_norm:.2e})")

        X_hat = _cp_reconstruct(weights, factors)
        error = float(jnp.linalg.norm(tensor - X_hat) / jnp.linalg.norm(tensor))
        errors.append(error)

        if abs(prev_error - error) < tol:
            break
        prev_error = error

    return CPResult(weights=weights, factors=factors), errors

def reconstruct(result: CPResult) -> jnp.ndarray:
    return _cp_reconstruct(result.weights, result.factors)