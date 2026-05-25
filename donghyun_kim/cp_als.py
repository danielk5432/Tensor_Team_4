"""
CP Decomposition via Alternating Least Squares (ALS)
Frobenius norm objective: minimize ||T - T'||_F

T' = sum_{r=1}^{R} a_r ⊗ b_r ⊗ c_r  (rank-R approximation)

Supports complex-valued tensors in C^n1 x C^n2 x C^n3.
"""

import numpy as np


def khatri_rao(A, B):
    """Column-wise Khatri-Rao product: (A ⊙ B)_r = a_r ⊗ b_r."""
    n_cols = A.shape[1]
    return np.vstack([np.kron(A[:, r], B[:, r]) for r in range(n_cols)]).T


def unfold(T, mode):
    """Mode-n unfolding of tensor T (0-indexed mode)."""
    n = T.ndim
    order = [mode] + [i for i in range(n) if i != mode]
    T_perm = np.transpose(T, order)
    return T_perm.reshape(T.shape[mode], -1)


def reconstruct(factors):
    """Reconstruct tensor from CP factor matrices [A, B, C, ...]."""
    rank = factors[0].shape[1]
    shape = [f.shape[0] for f in factors]
    T = np.zeros(shape, dtype=complex)
    for r in range(rank):
        component = factors[0][:, r]
        for f in factors[1:]:
            component = np.tensordot(component, f[:, r], axes=0)
        T += component
    return T


def frobenius_norm(T):
    return np.sqrt(np.sum(np.abs(T) ** 2))


def cp_als(T, rank, n_iter_max=1000, tol=1e-8, n_restarts=5, random_state=None):
    """
    CP decomposition of T via ALS, minimizing Frobenius norm ||T - T'||_F.

    Returns (factors, reconstruction_error, converged).
    factors: list of factor matrices [A, B, C] each of shape (dim_i, rank)
    """
    rng = np.random.default_rng(random_state)
    shape = T.shape
    ndim = T.ndim
    T_norm = frobenius_norm(T)

    best_factors = None
    best_err = np.inf

    is_complex = np.iscomplexobj(T)

    for restart in range(n_restarts):
        # Initialize factors: real for real tensors, complex for complex tensors
        if is_complex:
            factors = [
                rng.standard_normal((s, rank)) + 1j * rng.standard_normal((s, rank))
                for s in shape
            ]
        else:
            factors = [rng.standard_normal((s, rank)) for s in shape]

        prev_err = np.inf
        for iteration in range(n_iter_max):
            for mode in range(ndim):
                # ALS update for mode-n factor matrix
                # T_(n) = A_n (A_{N} ⊙ ... ⊙ A_{n+1} ⊙ A_{n-1} ⊙ ... ⊙ A_1)^T
                other = [factors[i] for i in range(ndim) if i != mode]
                # Khatri-Rao of all other factors (reverse order for correct unfolding)
                kr = other[-1]
                for f in reversed(other[:-1]):
                    kr = khatri_rao(f, kr)

                T_unfold = unfold(T, mode)
                # Gram matrix: ∏_k A_k^H A_k = KR^H KR
                gram = np.ones((rank, rank), dtype=T.dtype if not is_complex else complex)
                for f in other:
                    gram *= (f.conj().T @ f)

                # ALS normal equations (complex case):
                # A_n = T_(n) @ KR^* @ (KR^T KR^*)^{-1} = T_(n) @ KR^* @ conj(gram)^{-1}
                # For real tensors: reduces to T_(n) @ KR @ gram^{-1}
                reg = gram.conj() + 1e-12 * np.eye(rank, dtype=gram.dtype)
                factors[mode] = T_unfold @ kr.conj() @ np.linalg.pinv(reg)

            T_approx = reconstruct(factors)
            err = frobenius_norm(T - T_approx) / (T_norm + 1e-16)

            if abs(prev_err - err) < tol:
                break
            prev_err = err

        if err < best_err:
            best_err = err
            best_factors = [f.copy() for f in factors]

    T_best = reconstruct(best_factors)
    abs_err = frobenius_norm(T - T_best)
    return best_factors, abs_err, best_err


def rank2_approximation(T, **kwargs):
    """Find best rank-2 approximation of T using ALS."""
    assert T.shape == (3, 3, 3), f"Expected (3,3,3) tensor, got {T.shape}"
    factors, abs_err, rel_err = cp_als(T, rank=2, **kwargs)
    T_prime = reconstruct(factors)
    return T_prime, factors, abs_err, rel_err
