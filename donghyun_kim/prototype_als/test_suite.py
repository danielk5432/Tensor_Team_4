"""
Test suite for rank-2 CP approximation of 3x3x3 tensors.

Three test categories:
  1. Exact rank-2     — error should be ~0 (sanity check)
  2. Noisy rank-2     — error should grow proportionally with noise (stability)
  3. Random tensor    — approximation quality on generic inputs

Usage:
    python test_suite.py
    python test_suite.py --trials 20
    python test_suite.py --complex
    python test_suite.py --noise-levels 0.01 0.1 0.5 1.0
"""

import argparse
import numpy as np
from donghyun_kim.prototype.cp_als import rank2_approximation, frobenius_norm


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def rand_vec(rng, n, is_complex):
    v = rng.standard_normal(n)
    if is_complex:
        v = v + 1j * rng.standard_normal(n)
    return v


def rank1(rng, is_complex):
    a, b, c = [rand_vec(rng, 3, is_complex) for _ in range(3)]
    return np.einsum("i,j,k->ijk", a, b, c)


def rank2_exact(rng, is_complex):
    return rank1(rng, is_complex) + rank1(rng, is_complex)


def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def stats(values, label, unit=""):
    arr = np.array(values)
    print(f"  {label}:")
    print(f"    mean = {arr.mean():.6f}{unit}")
    print(f"    std  = {arr.std():.6f}{unit}")
    print(f"    min  = {arr.min():.6f}{unit}   max = {arr.max():.6f}{unit}")


# ──────────────────────────────────────────────
# Test 1: Exact rank-2
# ──────────────────────────────────────────────

def test_exact_rank2(n_trials, is_complex, als_kwargs):
    section(f"Test 1 — Exact Rank-2  (n_trials={n_trials}, complex={is_complex})")
    print("  Sanity check: ALS should recover T exactly (error ≈ 0).")

    rel_errors, abs_errors = [], []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        T = rank2_exact(rng, is_complex)
        T_norm = frobenius_norm(T)
        _, _, abs_err, rel_err = rank2_approximation(T, **als_kwargs)
        rel_errors.append(rel_err)
        abs_errors.append(abs_err)
        status = "OK" if rel_err < 1e-4 else "FAIL"
        print(f"  seed={seed:2d}  ||T-T'||_F = {abs_err:.2e}  "
              f"rel = {rel_err:.2e}  ||T||_F = {T_norm:.3f}  [{status}]")

    n_pass = sum(e < 1e-4 for e in rel_errors)
    print(f"\n  Passed: {n_pass}/{n_trials}")
    stats(rel_errors, "Relative error")
    return rel_errors


# ──────────────────────────────────────────────
# Test 2: Noisy rank-2
# ──────────────────────────────────────────────

def test_noisy_rank2(n_trials, is_complex, noise_levels, als_kwargs):
    section(f"Test 2 — Noisy Rank-2  (n_trials={n_trials}, complex={is_complex})")
    print("  Stability test: error should scale proportionally with noise level.")
    print(f"  Noise levels: {noise_levels}\n")

    for noise in noise_levels:
        rel_errors = []
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            T_clean = rank2_exact(rng, is_complex)
            noise_tensor = rng.standard_normal((3, 3, 3))
            if is_complex:
                noise_tensor = noise_tensor + 1j * rng.standard_normal((3, 3, 3))
            noise_tensor /= frobenius_norm(noise_tensor)   # unit noise
            T = T_clean + noise * noise_tensor

            T_norm = frobenius_norm(T)
            noise_rel = noise / (frobenius_norm(T_clean) + 1e-16)

            _, _, abs_err, rel_err = rank2_approximation(T, **als_kwargs)
            rel_errors.append(rel_err)

        arr = np.array(rel_errors)
        print(f"  noise={noise:.3f}  =>  rel_err  mean={arr.mean():.4f}  "
              f"std={arr.std():.4f}  [expected ≲ {noise:.4f}]")

    print()


# ──────────────────────────────────────────────
# Test 3: Random tensor
# ──────────────────────────────────────────────

def test_random_tensor(n_trials, is_complex, als_kwargs):
    section(f"Test 3 — Random Tensor  (n_trials={n_trials}, complex={is_complex})")
    print("  Approximation quality on generic inputs.")
    print("  (Generic 3x3x3 tensors have rank up to 5 real / unknown complex,")
    print("   so rank-2 approximation will not be perfect.)\n")

    rel_errors = []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        if is_complex:
            T = rng.standard_normal((3, 3, 3)) + 1j * rng.standard_normal((3, 3, 3))
        else:
            T = rng.standard_normal((3, 3, 3))

        _, _, abs_err, rel_err = rank2_approximation(T, **als_kwargs)
        rel_errors.append(rel_err)
        print(f"  seed={seed:2d}  ||T-T'||_F = {abs_err:.4f}  "
              f"rel = {rel_err:.4f}  ({rel_err*100:.1f}%)")

    stats(rel_errors, "Relative error")
    return rel_errors


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of random trials per test (default: 10)")
    parser.add_argument("--complex", action="store_true",
                        help="Use complex-valued tensors")
    parser.add_argument("--restarts", type=int, default=10,
                        help="ALS random restarts (default: 10)")
    parser.add_argument("--max-iter", type=int, default=2000,
                        help="Max ALS iterations (default: 2000)")
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
                        help="Noise levels for Test 2")
    args = parser.parse_args()

    als_kwargs = dict(n_restarts=args.restarts, n_iter_max=args.max_iter)
    is_complex = args.complex

    print(f"\nCP-ALS Rank-2 Approximation  |  3x3x3 tensors")
    print(f"complex={is_complex}  trials={args.trials}  "
          f"restarts={args.restarts}  max_iter={args.max_iter}")

    test_exact_rank2(args.trials, is_complex, als_kwargs)
    test_noisy_rank2(args.trials, is_complex, args.noise_levels, als_kwargs)
    test_random_tensor(args.trials, is_complex, als_kwargs)

    print(f"\n{'='*62}")
    print("  Done.")


if __name__ == "__main__":
    main()
