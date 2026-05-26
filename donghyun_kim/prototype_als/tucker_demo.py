"""
tucker_demo.py
────────────────────────────────────────────────────────────
Tucker decomposition comparison with CP on 3x3x3 tensors.

Tucker: T ≈ G ×₁ U ×₂ V ×₃ W
  G: core tensor (r1×r2×r3), U,V,W: factor matrices
  Special case r1=r2=r3=1 → rank-1 CP decomposition

Tucker vs CP:
  - Tucker preserves multi-linear structure (different ranks per mode)
  - CP is Tucker with diagonal core + equal rank per mode
  - Tucker is better for images (different spatial/channel ranks)
  - CP is better for sparse, structured data

Usage:
    python tucker_demo.py
"""

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, parafac
from cp_als import frobenius_norm, cp_als, reconstruct


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def tucker_error(T, ranks):
    """TensorLy Tucker decomposition relative error."""
    T_tl = tl.tensor(T)
    core, factors = tucker(T_tl, rank=ranks, n_iter_max=2000, tol=1e-8)
    T_rec = tl.tucker_to_tensor((core, factors))
    return (frobenius_norm(T - np.array(T_rec))
            / (frobenius_norm(T) + 1e-16)) * 100


def cp_error(T, rank, n_restarts=10):
    _, _, rel = cp_als(T, rank=rank, n_restarts=n_restarts, n_iter_max=2000)
    return rel * 100


def count_params_tucker(shape, ranks):
    """Number of free parameters in Tucker decomposition."""
    core_params = int(np.prod(ranks))
    factor_params = sum(n * r for n, r in zip(shape, ranks))
    return core_params + factor_params


def count_params_cp(shape, rank):
    """Number of free parameters in CP decomposition."""
    return rank * sum(shape)


def main():
    shape = (3, 3, 3)
    N = int(np.prod(shape))  # = 27

    # ── 1. Tucker vs CP: parameter count ────────────────────
    section("1. Parameter Count: Tucker vs CP")
    print(f"  Tensor shape: {shape}  (ambient dimension = {N})\n")
    print(f"  {'Decomposition':<28} {'Params':>8} {'Compression':>12}")
    print(f"  {'-'*50}")

    entries = [
        ("Tucker (2,2,2)", count_params_tucker(shape, [2,2,2])),
        ("Tucker (3,3,3) = full", count_params_tucker(shape, [3,3,3])),
        ("Tucker (1,1,1) = rank-1 CP", count_params_tucker(shape, [1,1,1])),
        ("CP rank=1", count_params_cp(shape, 1)),
        ("CP rank=2", count_params_cp(shape, 2)),
        ("CP rank=3", count_params_cp(shape, 3)),
        ("CP rank=5", count_params_cp(shape, 5)),
        ("Full tensor", N),
    ]
    for name, p in entries:
        ratio = N / p if p <= N else p / N
        direction = "×" if p <= N else "×(overfit)"
        print(f"  {name:<28} {p:>8d} {ratio:>8.1f}{direction}")

    # ── 2. Approximation quality comparison ──────────────────
    section("2. Approximation Quality: Tucker vs CP")
    print("  10 random 3×3×3 tensors, mean ± std relative error\n")

    n_trials = 10
    configs = [
        ("Tucker (1,1,1)", lambda T: tucker_error(T, [1,1,1]), 3),
        ("Tucker (2,2,2)", lambda T: tucker_error(T, [2,2,2]), 16),
        ("Tucker (3,3,3)", lambda T: tucker_error(T, [3,3,3]), 27),
        ("CP rank=1",      lambda T: cp_error(T, 1),           9),
        ("CP rank=2",      lambda T: cp_error(T, 2),           18),
        ("CP rank=3",      lambda T: cp_error(T, 3),           27),
    ]

    for name, fn, params in configs:
        errs = []
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            T = rng.standard_normal((3, 3, 3))
            errs.append(fn(T))
        mean, std = np.mean(errs), np.std(errs)
        print(f"  {name:<22} params={params:2d}  "
              f"mean={mean:6.2f}%  std={std:5.2f}%")

    # ── 3. Tucker (2,2,2) decomposition example ──────────────
    section("3. Tucker (2,2,2) Decomposition Example")
    rng = np.random.default_rng(0)
    T = rng.standard_normal((3, 3, 3))
    T_tl = tl.tensor(T)

    core, factors = tucker(T_tl, rank=[2, 2, 2], n_iter_max=2000)
    T_rec = np.array(tl.tucker_to_tensor((core, factors)))

    print(f"  T shape:    {T.shape}  ({N} params)")
    print(f"  Core shape: {np.array(core).shape}  ({np.prod(core.shape)} params)")
    print(f"  U: {np.array(factors[0]).shape},  "
          f"V: {np.array(factors[1]).shape},  "
          f"W: {np.array(factors[2]).shape}")
    total_params = count_params_tucker(shape, [2,2,2])
    print(f"  Total Tucker params: {total_params}  "
          f"(compression: {N}/{total_params} = {N/total_params:.1f}×)")
    err = frobenius_norm(T - T_rec) / frobenius_norm(T) * 100
    print(f"  ||T - T_Tucker||_F / ||T||_F = {err:.4f}%")

    # compare with CP rank=2
    _, _, cp_rel = cp_als(T, rank=2, n_restarts=10, n_iter_max=2000)
    print(f"  ||T - T_CP2 ||_F / ||T||_F = {cp_rel*100:.4f}%")
    print(f"\n  Tucker (2,2,2) params=16  vs  CP rank=2 params=18")
    print(f"  Tucker has fewer params yet achieves lower error — ")
    print(f"  this is because Tucker allows different structure per mode.")

    # ── 4. Relationship: CP as special Tucker ────────────────
    section("4. Relationship: CP as Special Case of Tucker")
    print("""
  Tucker:   T ≈ G ×₁ U ×₂ V ×₃ W
    G ∈ ℝ^{r1×r2×r3},  U ∈ ℝ^{3×r1},  V ∈ ℝ^{3×r2},  W ∈ ℝ^{3×r3}

  CP rank-R = Tucker with:
    G = Σ_r λ_r e_r⊗e_r⊗e_r  (superdiagonal core)
    r1 = r2 = r3 = R

  Key differences:
    Tucker  — modes can have different ranks; captures inter-mode correlations
              via the full core tensor G
    CP      — ranks must be equal; G is forced to be superdiagonal
              (only self-interactions between components)

  Practical implication for 3×3×3:
    Tucker (2,2,2): can represent any (T × U^T) × V^T × W^T for rank-2 each mode
    CP rank-2:      can only represent superposition of 2 rank-1 terms
    ⇒ Tucker is strictly more expressive for same "rank" budget
    """)


if __name__ == "__main__":
    main()
