"""
Tucker HOSVD: Higher-Order SVD. Kolda & Bader (2009) Fig 4.3.

Algorithm:
    for n = 1, ..., N:
        A^(n) <- R_n leading left singular vectors of X_(n)
    G <- X ×_1 A^(1)T ×_2 A^(2)T ... ×_N A^(N)T
"""
from functools import partial

import jax
import jax.numpy as jnp

from src.decomposition import TuckerResult
from src.utils.tensor_ops import unfold, multi_mode_product


@partial(jax.jit, static_argnums=(1,))
def _hosvd_jit(tensor: jnp.ndarray, rank: tuple) -> TuckerResult:
    factors = []
    for n in range(tensor.ndim):
        U, _, _ = jnp.linalg.svd(unfold(tensor, n), full_matrices=False)
        factors.append(U[:, :rank[n]])
    core = multi_mode_product(tensor, [A.T for A in factors])
    return TuckerResult(core=core, factors=factors)


def decompose(tensor: jnp.ndarray, rank: list, **kwargs) -> TuckerResult:
    """HOSVD Tucker decomposition (non-iterative)."""
    return _hosvd_jit(tensor, tuple(rank))


def reconstruct(result: TuckerResult) -> jnp.ndarray:
    return multi_mode_product(result.core, result.factors)
