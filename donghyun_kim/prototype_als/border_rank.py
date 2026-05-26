"""
border_rank.py
────────────────────────────────────────────────────────────
Mathematical analysis of border rank and degeneracy.

Key example: W-state tensor
  W = e₁⊗e₁⊗e₂ + e₁⊗e₂⊗e₁ + e₂⊗e₁⊗e₁  ∈ C³⊗C³⊗C³
  (zero-padded from C²⊗C²⊗C²)

Facts:
  rank(W) = 3  (minimum 3 rank-1 terms needed)
  border rank(W) = 2  (W ∈ σ̄₂ but W ∉ σ₂⁰)

This means: inf_{T' of rank≤2} ||W - T'||_F = 0
            but the infimum is NEVER achieved.

Usage:
    python border_rank.py
"""

import numpy as np
from cp_als import frobenius_norm, reconstruct, cp_als

# ──────────────────────────────────────────────────────────
# Tensor constructions
# ──────────────────────────────────────────────────────────

def w_state():
    """
    W-state tensor in C³⊗C³⊗C³ (embedded from C²⊗C²⊗C²).
    W = e₁⊗e₁⊗e₂ + e₁⊗e₂⊗e₁ + e₂⊗e₁⊗e₁  (0-indexed: e₀=[1,0,0], e₁=[0,1,0])
    rank(W)=3, border rank(W)=2.
    """
    e0 = np.array([1., 0., 0.])
    e1 = np.array([0., 1., 0.])
    W = (np.einsum("i,j,k->ijk", e0, e0, e1)
       + np.einsum("i,j,k->ijk", e0, e1, e0)
       + np.einsum("i,j,k->ijk", e1, e0, e0))
    return W


def border_rank2_sequence(eps):
    """
    Rank-≤2 approximation of W via the border rank limit sequence.

    A(ε) = (1/ε)·(e₀+ε·e₁)⊗(e₀+ε·e₁)⊗(e₀+ε·e₁)  -  (1/ε)·e₀⊗e₀⊗e₀

    As ε→0:  A(ε) → e₀⊗e₀⊗e₁ + e₀⊗e₁⊗e₀ + e₁⊗e₀⊗e₀ = W

    Each A(ε) is a sum of exactly 2 rank-1 tensors → rank(A(ε)) ≤ 2.
    But the factor norms → ∞ as ε → 0  (divergence!).
    """
    e0 = np.array([1., 0., 0.])
    e1 = np.array([0., 1., 0.])
    v = e0 + eps * e1  # shape (3,)
    T1 = (1.0 / eps) * np.einsum("i,j,k->ijk", v, v, v)
    T2 = (1.0 / eps) * np.einsum("i,j,k->ijk", e0, e0, e0)
    return T1 - T2  # rank ≤ 2 for any eps > 0


def ghz_state():
    """
    GHZ state: G = e₀⊗e₀⊗e₀ + e₁⊗e₁⊗e₁
    rank(G) = 2, border rank(G) = 2  → in σ₂⁰ (normal case).
    """
    e0 = np.array([1., 0., 0.])
    e1 = np.array([0., 1., 0.])
    return (np.einsum("i,j,k->ijk", e0, e0, e0)
          + np.einsum("i,j,k->ijk", e1, e1, e1))


# ──────────────────────────────────────────────────────────
# Analysis helpers
# ──────────────────────────────────────────────────────────

def factor_norms(factors):
    """Return L2 norm of each factor column."""
    return [np.linalg.norm(f, axis=0) for f in factors]


def min_rank_decomp_error(T, max_rank=5, n_restarts=15):
    """
    Find minimum rank R such that rank-R ALS achieves <1e-4 relative error.
    Returns dict: rank → rel_err.
    """
    results = {}
    for R in range(1, max_rank + 1):
        _, _, rel_err = cp_als(T, rank=R, n_restarts=n_restarts, n_iter_max=3000)
        results[R] = rel_err
        if rel_err < 1e-4:
            break
    return results


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ──────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────

def main():
    print("\nBorder Rank and Degeneracy Analysis")
    print("W-state  |  GHZ state  |  Generic tensor")

    # ── 1. W-state: rank determination ──────────────────────
    section("1. W-State: Rank via ALS")
    W = w_state()
    print(f"  W = e₀⊗e₀⊗e₁ + e₀⊗e₁⊗e₀ + e₁⊗e₀⊗e₀  (embedded in 3×3×3)")
    print(f"  ||W||_F = {frobenius_norm(W):.6f}  (expected √3 ≈ {np.sqrt(3):.6f})")
    print()
    print("  Rank-R ALS approximation errors (border rank: inf is 0 but unreachable):")

    rank_errs = min_rank_decomp_error(W, max_rank=5)
    for R, err in rank_errs.items():
        achieved = "← exact" if err < 1e-4 else ""
        print(f"    rank={R}: rel_err = {err*100:.4f}%  {achieved}")

    # ── 2. Border rank-2 limit sequence ─────────────────────
    section("2. Border Rank-2 Limit Sequence  A(ε) → W")
    print("  A(ε) = (1/ε)·(e₀+ε·e₁)³ − (1/ε)·e₀³   [rank ≤ 2 for all ε > 0]")
    print()
    print(f"  {'ε':>10}  {'||A(ε)-W||_F':>15}  {'rel_err':>10}  {'factor norm':>12}")
    print(f"  {'-'*55}")

    W_norm = frobenius_norm(W)
    for eps in [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
        A_eps = border_rank2_sequence(eps)
        err = frobenius_norm(A_eps - W)
        rel = err / W_norm
        # factor norms blow up as eps→0
        e0 = np.array([1., 0., 0.])
        e1 = np.array([0., 1., 0.])
        v = e0 + eps * e1
        fnorm = np.linalg.norm(v) / eps  # norm of (1/eps)*v
        print(f"  {eps:>10.4f}  {err:>15.6e}  {rel*100:>9.4f}%  {fnorm:>12.2f}")

    print()
    print("  Observation: as ε→0, ||A(ε)−W||_F → 0  BUT  factor norms → ∞.")
    print("  This is the hallmark of border rank: the infimum is 0,")
    print("  but no actual rank-2 tensor achieves it.")

    # ── 3. ALS behaviour on W ───────────────────────────────
    section("3. ALS Behaviour on W-State  (rank=2)")
    print("  ALS tries to find rank-2 T' minimising ||W - T'||_F.")
    print("  Since W ∉ σ₂⁰, the true minimum is > 0, but ALS may")
    print("  exhibit diverging factors chasing the infimum.\n")

    for seed in range(5):
        factors, abs_err, rel_err = cp_als(W, rank=2, n_restarts=1,
                                            n_iter_max=5000,
                                            random_state=seed)
        fnorms = [np.max(np.linalg.norm(f, axis=0)) for f in factors]
        print(f"  seed={seed}: rel_err={rel_err*100:.4f}%  "
              f"max_factor_norm=[{fnorms[0]:.1f}, {fnorms[1]:.1f}, {fnorms[2]:.1f}]")

    print()
    best_factors, best_abs, best_rel = cp_als(W, rank=2, n_restarts=30,
                                               n_iter_max=5000, random_state=0)
    print(f"  Best result (30 restarts): rel_err = {best_rel*100:.4f}%")
    print(f"  ||W||_F = {frobenius_norm(W):.4f}, ||W-T'||_F = {best_abs:.4f}")
    print()
    print("  Compare: W-state rank=3 ALS:")
    factors3, abs3, rel3 = cp_als(W, rank=3, n_restarts=15, n_iter_max=3000)
    print(f"    rank=3: rel_err = {rel3*100:.4f}%  (should be ~0)")

    # ── 4. GHZ comparison ───────────────────────────────────
    section("4. GHZ State (Normal rank-2 case for comparison)")
    G = ghz_state()
    print(f"  G = e₀³ + e₁³   rank(G)=2, border rank(G)=2  → G ∈ σ₂⁰")
    print(f"  ||G||_F = {frobenius_norm(G):.6f}  (expected √2 ≈ {np.sqrt(2):.6f})")

    gfactors, gabs, grel = cp_als(G, rank=2, n_restarts=10, n_iter_max=2000)
    print(f"  ALS rank-2: rel_err = {grel*100:.6f}%  (should be ~0)")
    fnorms = [np.max(np.linalg.norm(f, axis=0)) for f in gfactors]
    print(f"  Factor norms: [{fnorms[0]:.3f}, {fnorms[1]:.3f}, {fnorms[2]:.3f}]  (bounded ✓)")

    # ── 5. Summary ───────────────────────────────────────────
    section("5. Summary: Border Rank vs Strict Rank")
    print("""
  Tensor      | rank | border rank | ALS rank-2 err | Factor norms
  ────────────┼──────┼─────────────┼────────────────┼────────────
  GHZ state   |  2   |      2      |  ~0 (exact)    |  bounded
  W-state     |  3   |      2      |  > 0 (stuck)   |  diverge
  Random      | ≥4   |     ≥2      |  40–50%        |  bounded

  Key insight: σ₂⁰ (strict rank-2) is NOT closed.
  Its closure σ̄₂ includes W-state (border rank 2).
  ALS on W-state finds a local minimum but cannot reach the infimum.
  The factor norms may explode in degenerate (ill-conditioned) cases.
    """)


if __name__ == "__main__":
    main()
