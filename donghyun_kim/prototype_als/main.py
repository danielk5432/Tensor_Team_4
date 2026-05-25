"""
Prototype: Rank-2 CP approximation of 3x3x3 complex tensors
Problem: Find T' in sigma_2(Seg(C^3 x C^3 x C^3)) minimizing ||T - T'||_F

Usage:
    python main.py
    python main.py --real       # real-valued only
    python main.py --seed 42
"""

import argparse
import numpy as np
from donghyun_kim.prototype.cp_als import rank2_approximation, frobenius_norm, reconstruct


def print_tensor(T, name="T"):
    print(f"\n{name} (shape {T.shape}):")
    for i in range(T.shape[0]):
        print(f"  slice [{i},:,:]:")
        for row in T[i]:
            formatted = "  ".join(f"{v.real:+.4f}{v.imag:+.4f}j" for v in row)
            print(f"    [{formatted}]")


def run_demo(T, label=""):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print_tensor(T, "T")

    T_prime, factors, abs_err, rel_err = rank2_approximation(T, n_restarts=10, random_state=0)

    print(f"\n--- Result ---")
    print(f"  ||T - T'||_F  = {abs_err:.6f}")
    print(f"  ||T - T'||_F / ||T||_F = {rel_err:.6f}  ({rel_err*100:.2f}%)")
    print(f"\nFactor matrices (rank-2):")
    for i, (f, name) in enumerate(zip(factors, ["A", "B", "C"])):
        print(f"  {name} (shape {f.shape}):")
        for row in f:
            print("    ", "  ".join(f"{v.real:+.4f}{v.imag:+.4f}j" for v in row))

    # Verify rank-2 structure: T' = a1⊗b1⊗c1 + a2⊗b2⊗c2
    A, B, C = factors
    T_check = (np.einsum("i,j,k->ijk", A[:,0], B[:,0], C[:,0])
             + np.einsum("i,j,k->ijk", A[:,1], B[:,1], C[:,1]))
    err_check = frobenius_norm(T_prime - T_check)
    print(f"\n  Rank-2 structure verified (residual {err_check:.2e})")
    return T_prime, abs_err, rel_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real-valued tensors only")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- Case 1: Random tensor ---
    if args.real:
        T_random = rng.standard_normal((3, 3, 3))
    else:
        T_random = rng.standard_normal((3, 3, 3)) + 1j * rng.standard_normal((3, 3, 3))
    run_demo(T_random, "Case 1: Random tensor")

    # --- Case 2: Exact rank-1 tensor (should give ~0 error with rank 2) ---
    a = rng.standard_normal(3) + (0 if args.real else 1j * rng.standard_normal(3))
    b = rng.standard_normal(3) + (0 if args.real else 1j * rng.standard_normal(3))
    c = rng.standard_normal(3) + (0 if args.real else 1j * rng.standard_normal(3))
    T_rank1 = np.einsum("i,j,k->ijk", a, b, c)
    run_demo(T_rank1, "Case 2: Exact rank-1 tensor (rank-2 approx should be near-exact)")

    # --- Case 3: Exact rank-2 tensor (error should be ~0) ---
    a2 = rng.standard_normal(3) + (0 if args.real else 1j * rng.standard_normal(3))
    b2 = rng.standard_normal(3) + (0 if args.real else 1j * rng.standard_normal(3))
    c2 = rng.standard_normal(3) + (0 if args.real else 1j * rng.standard_normal(3))
    T_rank2 = np.einsum("i,j,k->ijk", a, b, c) + np.einsum("i,j,k->ijk", a2, b2, c2)
    run_demo(T_rank2, "Case 3: Exact rank-2 tensor (error should be ~0)")

    # --- Case 4: Zero tensor ---
    T_zero = np.zeros((3, 3, 3), dtype=complex)
    run_demo(T_zero, "Case 4: Zero tensor")

    print("\n" + "="*60)
    print("  Done. Use rank2_approximation(T) from cp_als.py for custom inputs.")


if __name__ == "__main__":
    main()
