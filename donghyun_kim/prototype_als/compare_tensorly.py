"""
compare_tensorly.py
────────────────────────────────────────────────────────────
Our CP-ALS  vs.  TensorLy parafac  —  accuracy & speed comparison

Three scenarios, 10 trials each:
  1. Exact rank-2 tensor       → both should get ~0 error
  2. Noisy rank-2 tensor       → stability comparison
  3. Generic random tensor     → approximation quality

Usage:
    python compare_tensorly.py
    python compare_tensorly.py --trials 20 --rank 2
"""

import argparse
import time
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from cp_als import rank2_approximation, cp_als, frobenius_norm, reconstruct


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def make_rank2(rng):
    def rv(): return rng.standard_normal(3)
    return (np.einsum("i,j,k->ijk", rv(), rv(), rv())
          + np.einsum("i,j,k->ijk", rv(), rv(), rv()))


def tensorly_approx(T, rank=2, n_iter_max=2000, tol=1e-8, n_init=10):
    """Best-of-n_init TensorLy parafac runs."""
    T_tl = tl.tensor(T)
    best_err = np.inf
    best_rec = None
    for seed in range(n_init):
        try:
            cp = parafac(T_tl, rank=rank, n_iter_max=n_iter_max,
                         tol=tol, random_state=seed, init="random",
                         normalize_factors=False)
            rec = tl.cp_to_tensor(cp)
            err = frobenius_norm(T - np.array(rec)) / (frobenius_norm(T) + 1e-16)
            if err < best_err:
                best_err = err
                best_rec = rec
        except Exception:
            pass
    return best_err, best_rec


def our_approx(T, rank=2, n_restarts=10, n_iter_max=2000):
    _, abs_err, rel_err = cp_als(T, rank=rank, n_restarts=n_restarts,
                                  n_iter_max=n_iter_max)
    return rel_err


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def row(label, ours, theirs, t_ours, t_theirs):
    winner = "<-- ours" if ours < theirs else ("     <-- TensorLy" if theirs < ours else "tie")
    print(f"  {label:<12}  ours={ours*100:6.3f}%  TensorLy={theirs*100:6.3f}%  "
          f"t_ours={t_ours:.3f}s  t_tly={t_theirs:.3f}s  {winner}")


# ──────────────────────────────────────────────────────────
# Scenario 1 — Exact rank-2
# ──────────────────────────────────────────────────────────

def scenario_exact(n_trials, rank, als_kwargs, tly_kwargs):
    section(f"Scenario 1 — Exact Rank-2  (rank={rank}, trials={n_trials})")
    print("  Both implementations should recover T exactly (error ≈ 0).\n")

    errs_ours, errs_tly = [], []
    t_ours_total = t_tly_total = 0.0

    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        T = make_rank2(rng)

        t0 = time.perf_counter()
        e_ours = our_approx(T, rank=rank, **als_kwargs)
        t_ours = time.perf_counter() - t0

        t0 = time.perf_counter()
        e_tly, _ = tensorly_approx(T, rank=rank, **tly_kwargs)
        t_tly = time.perf_counter() - t0

        errs_ours.append(e_ours)
        errs_tly.append(e_tly)
        t_ours_total += t_ours
        t_tly_total += t_tly

        row(f"seed={seed:2d}", e_ours, e_tly, t_ours, t_tly)

    print(f"\n  Summary ({n_trials} trials):")
    print(f"    Ours    — mean={np.mean(errs_ours)*100:.4f}%  "
          f"max={np.max(errs_ours)*100:.4f}%  total_time={t_ours_total:.2f}s")
    print(f"    TensorLy— mean={np.mean(errs_tly)*100:.4f}%  "
          f"max={np.max(errs_tly)*100:.4f}%  total_time={t_tly_total:.2f}s")
    return errs_ours, errs_tly


# ──────────────────────────────────────────────────────────
# Scenario 2 — Noisy rank-2
# ──────────────────────────────────────────────────────────

def scenario_noisy(n_trials, rank, noise_levels, als_kwargs, tly_kwargs):
    section(f"Scenario 2 — Noisy Rank-2  (rank={rank})")
    print("  Stability: error should scale with noise level.\n")
    print(f"  {'noise':>8}  {'ours_mean':>10}  {'tly_mean':>10}  {'ours_std':>10}  {'tly_std':>10}")
    print(f"  {'-'*60}")

    results = []
    for noise in noise_levels:
        errs_ours, errs_tly = [], []
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            T_clean = make_rank2(rng)
            N = rng.standard_normal((3, 3, 3))
            N /= frobenius_norm(N)
            T = T_clean + noise * N

            errs_ours.append(our_approx(T, rank=rank, **als_kwargs))
            errs_tly.append(tensorly_approx(T, rank=rank, **tly_kwargs)[0])

        mo, mt = np.mean(errs_ours)*100, np.mean(errs_tly)*100
        so, st = np.std(errs_ours)*100,  np.std(errs_tly)*100
        print(f"  {noise:>7.2f}%  {mo:>9.3f}%  {mt:>9.3f}%  {so:>9.3f}%  {st:>9.3f}%")
        results.append((noise, mo, mt, so, st))
    return results


# ──────────────────────────────────────────────────────────
# Scenario 3 — Generic random tensor
# ──────────────────────────────────────────────────────────

def scenario_random(n_trials, rank, als_kwargs, tly_kwargs):
    section(f"Scenario 3 — Generic Random Tensor  (rank={rank}, trials={n_trials})")
    print("  Approximation quality on unstructured inputs.\n")

    errs_ours, errs_tly = [], []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        T = rng.standard_normal((3, 3, 3))

        e_ours = our_approx(T, rank=rank, **als_kwargs)
        e_tly, _ = tensorly_approx(T, rank=rank, **tly_kwargs)

        errs_ours.append(e_ours)
        errs_tly.append(e_tly)
        row(f"seed={seed:2d}", e_ours, e_tly, 0, 0)

    print(f"\n  Summary ({n_trials} trials):")
    print(f"    Ours    — mean={np.mean(errs_ours)*100:.2f}%  "
          f"std={np.std(errs_ours)*100:.2f}%")
    print(f"    TensorLy— mean={np.mean(errs_tly)*100:.2f}%  "
          f"std={np.std(errs_tly)*100:.2f}%")
    return errs_ours, errs_tly


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=10)
    args = parser.parse_args()

    als_kwargs = dict(n_restarts=args.restarts, n_iter_max=2000)
    tly_kwargs = dict(n_init=args.restarts, n_iter_max=2000)
    noise_levels = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

    print(f"\nCP-ALS: Ours  vs  TensorLy parafac")
    print(f"rank={args.rank}  trials={args.trials}  restarts={args.restarts}")

    scenario_exact(args.trials, args.rank, als_kwargs, tly_kwargs)
    scenario_noisy(args.trials, args.rank, noise_levels, als_kwargs, tly_kwargs)
    scenario_random(args.trials, args.rank, als_kwargs, tly_kwargs)

    print(f"\n{'='*65}")
    print("  Done.")


if __name__ == "__main__":
    main()
