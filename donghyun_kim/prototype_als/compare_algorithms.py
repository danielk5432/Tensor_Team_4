"""
compare_algorithms.py
────────────────────────────────────────────────────────────
CP decomposition: ALS  vs  ASD  vs  dGN  comparison

Based on suho_cho references (dGN&PMF3.pdf, Comparison_CP-algorithms.pdf):

  ALS  — Alternating Least Squares
           Fixes other modes, solves exact least-squares per mode.
           Linear convergence. Low memory.

  ASD  — Alternating Steepest Descent (Alternating Gradient Descent)
           Fixes other modes, takes gradient step with exact line search.
           Same alternating structure as ALS but no matrix solve.
           Mathematically inferior to ALS (more iterations needed).

  dGN  — Damped Gauss-Newton
           All factors updated simultaneously using Gauss-Newton step.
           Quadratic convergence near solution.
           High memory: full Jacobian across all modes.
           Best for mathematical / well-conditioned problems.

  PMF3 — (Reference from suho_cho paper)
           Similar quadratic convergence to dGN.
           Better for noisy real-world data.
           Implemented here as "all-at-once gradient with momentum".

Usage:
    python compare_algorithms.py
    python compare_algorithms.py --trials 10 --rank 2
"""

import argparse
import time
import numpy as np
from cp_als import frobenius_norm, unfold, khatri_rao, reconstruct

# ──────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────

def init_factors(T, rank, rng):
    return [rng.standard_normal((s, rank)) for s in T.shape]


def cp_reconstruct(factors):
    T = reconstruct(factors)
    # reconstruct() always returns complex; cast back to real if factors are real
    if not np.iscomplexobj(factors[0]):
        return T.real
    return T


def rel_err(T, factors, T_norm):
    return frobenius_norm(T - cp_reconstruct(factors)) / (T_norm + 1e-16)


def gram_product(factors, skip_mode):
    ndim = len(factors)
    rank = factors[0].shape[1]
    G = np.ones((rank, rank))
    for m in range(ndim):
        if m != skip_mode:
            G *= (factors[m].T @ factors[m])
    return G


def khatri_rao_all(factors, skip_mode):
    """Khatri-Rao of all factors except skip_mode."""
    ndim = len(factors)
    other = [factors[m] for m in range(ndim) if m != skip_mode]
    kr = other[-1]
    for f in reversed(other[:-1]):
        kr = khatri_rao(f, kr)
    return kr


# ──────────────────────────────────────────────────────────
# 1. ALS — Alternating Least Squares
# ──────────────────────────────────────────────────────────

def als(T, rank, n_iter_max=2000, tol=1e-8, random_state=0):
    """Standard ALS: exact least-squares update per mode."""
    rng = np.random.default_rng(random_state)
    factors = init_factors(T, rank, rng)
    T_norm = frobenius_norm(T)
    ndim = T.ndim
    history = []
    prev_err = np.inf

    for _ in range(n_iter_max):
        for mode in range(ndim):
            kr = khatri_rao_all(factors, mode)
            T_u = unfold(T, mode)
            G = gram_product(factors, mode)
            reg = G + 1e-12 * np.eye(rank)
            factors[mode] = T_u @ kr @ np.linalg.pinv(reg)

        err = rel_err(T, factors, T_norm)
        history.append(err)
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return factors, history


# ──────────────────────────────────────────────────────────
# 2. ASD — Alternating Steepest Descent
# ──────────────────────────────────────────────────────────

def asd(T, rank, n_iter_max=5000, tol=1e-8, random_state=0):
    """
    ASD (Alternating Steepest Descent): gradient step with exact line search per mode.
    Gradient for mode n:  ∇ = A_n G - T_(n) KR
    Exact step: α = ||∇||²_F / ||∇ G||²_F  (minimizes quadratic in α)

    Differs from ALS: ALS takes the exact minimum (one step);
    ASD takes a gradient step (may need many steps to reach same point).
    Mathematically inferior to ALS for well-conditioned problems.
    """
    rng = np.random.default_rng(random_state)
    # Normalize T so factors stay small at init
    T_norm = frobenius_norm(T)
    T_scaled = T / (T_norm + 1e-16)
    factors = [f / (np.linalg.norm(f) + 1e-8) for f in init_factors(T_scaled, rank, rng)]

    ndim = T.ndim
    history = []
    prev_err = np.inf

    for _ in range(n_iter_max):
        for mode in range(ndim):
            kr   = khatri_rao_all(factors, mode)
            T_u  = unfold(T, mode)
            G    = gram_product(factors, mode)

            # Gradient: ∇ = A_n G - T_(n) KR
            grad = factors[mode] @ G - T_u @ kr

            # Armijo backtracking line search (guarantees descent)
            f0    = float(np.sum((factors[mode] @ G - T_u @ kr) * grad))  # = ||grad||²
            alpha = 1.0
            for _ in range(40):
                A_try = factors[mode] - alpha * grad
                # loss change (use quadratic expansion for efficiency)
                reduction = alpha * f0 - 0.5 * alpha**2 * float(np.sum(grad * (grad @ G)))
                if reduction > 1e-4 * alpha * f0:
                    break
                alpha *= 0.5

            factors[mode] = factors[mode] - alpha * grad

        err = rel_err(T, factors, T_norm)
        history.append(err)
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return factors, history


# ──────────────────────────────────────────────────────────
# 3. dGN — Damped Gauss-Newton (all-at-once)
# ──────────────────────────────────────────────────────────

def _build_jacobian(factors, shape, indices):
    """
    Build the full Jacobian J of vec(T') w.r.t. x = [vec(A), vec(B), vec(C), ...].
    J: (prod(shape), sum_n(shape[n]*rank))
    J[obs, n_offset + i_n*R + r] = δ_{indices[obs,n], i_n} * prod_{m≠n} A_m[indices[obs,m], r]
    """
    ndim = len(factors)
    rank = factors[0].shape[1]
    n_obs = len(indices)
    n_params = sum(s * rank for s in shape)
    J = np.zeros((n_obs, n_params))
    col_offset = 0
    for n in range(ndim):
        n_n = shape[n]
        # Compute product of all factors except mode n, for each obs and rank component
        # prod_other[obs, r] = prod_{m!=n} factors[m][indices[obs,m], r]
        prod_other = np.ones((n_obs, rank))
        for m in range(ndim):
            if m != n:
                prod_other *= factors[m][indices[:, m], :]  # broadcast over rank
        # Fill J block: J[obs, col_offset + i_n*rank + r] = prod_other[obs, r] if indices[obs,n]==i_n
        for i_n in range(n_n):
            mask = (indices[:, n] == i_n)
            J[np.ix_(mask, range(col_offset + i_n*rank, col_offset + (i_n+1)*rank))] = prod_other[mask]
        col_offset += n_n * rank
    return J


def dgn(T, rank, n_iter_max=200, tol=1e-8, damping=1e-3, random_state=0):
    """
    dGN (damped Gauss-Newton) — proper full-Jacobian formulation.

    Builds J: (prod(shape), sum_n shape[n]*rank) explicitly.
    Solves: Δx = (J^T J + λI)^{-1} J^T r  where r = vec(T - T').
    Updates all factors simultaneously from the same Jacobian.

    This is the true Gauss-Newton method: quadratic convergence near solution,
    high memory (J explicit), but correct and well-conditioned with damping.
    """
    rng = np.random.default_rng(random_state)
    factors = init_factors(T, rank, rng)
    T_norm = frobenius_norm(T)
    shape = T.shape
    ndim = T.ndim
    n_params = sum(s * rank for s in shape)

    # Precompute multi-indices (C-order, matches T.ravel())
    indices = np.array(list(np.ndindex(*shape)))  # (n_obs, ndim)

    history = []
    prev_err = np.inf

    for _ in range(n_iter_max):
        T_approx = cp_reconstruct(factors)
        res = (T - T_approx).ravel()  # residual vector, shape (n_obs,)

        J = _build_jacobian(factors, shape, indices)
        H = J.T @ J + damping * np.eye(n_params)
        dx = np.linalg.solve(H, J.T @ res)

        # Unpack Δx into per-mode updates
        col_offset = 0
        new_factors = []
        for n in range(ndim):
            n_n = shape[n]
            new_factors.append(factors[n] + dx[col_offset:col_offset + n_n*rank].reshape(n_n, rank))
            col_offset += n_n * rank
        factors = new_factors

        err = rel_err(T, factors, T_norm)
        history.append(err)
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return factors, history


# ──────────────────────────────────────────────────────────
# 4. PMF3-style — All-at-once gradient with momentum
# ──────────────────────────────────────────────────────────

def pmf3(T, rank, n_iter_max=2000, tol=1e-8, lr=0.01, beta=0.9, random_state=0):
    """
    PMF3-inspired: all-at-once gradient descent with momentum.
    Better for noisy data (no alternating lock-in effect).
    """
    rng = np.random.default_rng(random_state)
    factors = init_factors(T, rank, rng)
    T_norm = frobenius_norm(T)
    ndim = T.ndim
    history = []
    prev_err = np.inf

    # Momentum buffers
    velocity = [np.zeros_like(f) for f in factors]

    for _ in range(n_iter_max):
        T_approx = cp_reconstruct(factors)
        R = T_approx - T  # residual (sign: gradient direction)

        # Compute gradients for all modes simultaneously
        grads = []
        for mode in range(ndim):
            kr = khatri_rao_all(factors, mode)
            R_u = unfold(R, mode)
            grads.append(R_u @ kr)

        # Adaptive learning rate (scale by gram norm)
        for mode in range(ndim):
            G = gram_product(factors, mode)
            scale = max(np.linalg.norm(G), 1e-6)
            velocity[mode] = beta * velocity[mode] + (lr / scale) * grads[mode]
            factors[mode] = factors[mode] - velocity[mode]

        err = rel_err(T, factors, T_norm)
        history.append(err)
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return factors, history


# ──────────────────────────────────────────────────────────
# Best-of-n runner
# ──────────────────────────────────────────────────────────

def best_of(algo_fn, T, rank, n_restarts=5, **kwargs):
    T_norm = frobenius_norm(T)
    best_err = np.inf
    best_hist = []
    total_time = 0.0
    for seed in range(n_restarts):
        t0 = time.perf_counter()
        fac, hist = algo_fn(T, rank, random_state=seed, **kwargs)
        total_time += time.perf_counter() - t0
        e = rel_err(T, fac, T_norm)
        if e < best_err:
            best_err = e
            best_hist = hist
    return best_err, best_hist, total_time


# ──────────────────────────────────────────────────────────
# Comparison runner
# ──────────────────────────────────────────────────────────

ALGOS = {
    "ALS":  (als,  dict(n_iter_max=2000)),
    "ASD":  (asd,  dict(n_iter_max=5000)),
    "dGN":  (dgn,  dict(n_iter_max=500, damping=1e-3)),
    "PMF3": (pmf3, dict(n_iter_max=5000, lr=0.05, beta=0.9)),
}


def run_comparison(T, rank, n_restarts, label=""):
    results = {}
    for name, (fn, kwargs) in ALGOS.items():
        err, hist, t = best_of(fn, T, rank, n_restarts=n_restarts, **kwargs)
        results[name] = dict(err=err, iters=len(hist), time=t/n_restarts, hist=hist)
    return results


def section(title):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")


def print_comparison(results, label):
    print(f"\n  [{label}]")
    print(f"  {'Algo':<8} {'rel_err':>10} {'iters':>7} {'time(s)':>9} {'convergence'}")
    print(f"  {'-'*60}")
    for name, r in results.items():
        rate = "quadratic" if name in ("dGN",) else "linear"
        print(f"  {name:<8} {r['err']*100:>9.4f}%  {r['iters']:>6}  "
              f"{r['time']:>8.3f}s  {rate}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=5)
    args = parser.parse_args()

    rank = args.rank
    n_restarts = args.restarts

    print(f"\nAlgorithm Comparison: ALS vs ASD vs dGN vs PMF3")
    print(f"rank={rank}  trials={args.trials}  restarts={n_restarts}")

    # ── Exact rank-2 ──────────────────────────────────────
    section("Scenario 1 — Exact Rank-2 (sanity check)")
    print("  Expected: ALS/dGN converge to ~0; ASD needs more iters; PMF3 stable.\n")

    all_errs = {name: [] for name in ALGOS}
    all_iters = {name: [] for name in ALGOS}
    all_times = {name: [] for name in ALGOS}
    for seed in range(args.trials):
        rng = np.random.default_rng(seed)
        def rv(): return rng.standard_normal(3)
        T = np.einsum("i,j,k->ijk", rv(), rv(), rv()) + np.einsum("i,j,k->ijk", rv(), rv(), rv())
        res = run_comparison(T, rank, n_restarts)
        for name, r in res.items():
            all_errs[name].append(r["err"] * 100)
            all_iters[name].append(r["iters"])
            all_times[name].append(r["time"] * 1000)
    print(f"  {'Algo':<8} {'mean_err':>10} {'std_err':>8} {'mean_iters':>11} {'mean_ms':>9}")
    print(f"  {'-'*52}")
    for name in ALGOS:
        ea = np.array(all_errs[name]); ia = np.array(all_iters[name]); ta = np.array(all_times[name])
        print(f"  {name:<8} {ea.mean():>9.4f}%  {ea.std():>7.4f}%  {ia.mean():>10.1f}  {ta.mean():>8.2f}ms")

    # ── Random tensor ──────────────────────────────────────
    section("Scenario 2 — Generic Random Tensor")
    print("  Expected: all methods give ~40–50% (rank-2 geometric limit).\n")

    all_errs2 = {name: [] for name in ALGOS}
    all_iters2 = {name: [] for name in ALGOS}
    all_times2 = {name: [] for name in ALGOS}
    for seed in range(args.trials):
        rng = np.random.default_rng(seed)
        T = rng.standard_normal((3, 3, 3))
        res = run_comparison(T, rank, n_restarts)
        for name, r in res.items():
            all_errs2[name].append(r["err"] * 100)
            all_iters2[name].append(r["iters"])
            all_times2[name].append(r["time"] * 1000)
    print(f"  {'Algo':<8} {'mean_err':>10} {'std_err':>8} {'mean_iters':>11} {'mean_ms':>9}")
    print(f"  {'-'*52}")
    for name in ALGOS:
        ea = np.array(all_errs2[name]); ia = np.array(all_iters2[name]); ta = np.array(all_times2[name])
        print(f"  {name:<8} {ea.mean():>9.4f}%  {ea.std():>7.4f}%  {ia.mean():>10.1f}  {ta.mean():>8.2f}ms")

    # ── Noisy rank-2 ──────────────────────────────────────
    section("Scenario 3 — Noisy Rank-2 (noise=10%)")
    print("  Expected: all methods ≲ noise level; dGN fewer iterations.\n")

    all_errs3 = {name: [] for name in ALGOS}
    all_iters3 = {name: [] for name in ALGOS}
    all_times3 = {name: [] for name in ALGOS}
    for seed in range(args.trials):
        rng = np.random.default_rng(seed)
        def rv(): return rng.standard_normal(3)
        T_clean = np.einsum("i,j,k->ijk", rv(), rv(), rv()) + np.einsum("i,j,k->ijk", rv(), rv(), rv())
        N_noise = rng.standard_normal((3, 3, 3))
        N_noise /= frobenius_norm(N_noise)
        T = T_clean + 0.1 * N_noise
        res = run_comparison(T, rank, n_restarts)
        for name, r in res.items():
            all_errs3[name].append(r["err"] * 100)
            all_iters3[name].append(r["iters"])
            all_times3[name].append(r["time"] * 1000)
    print(f"  {'Algo':<8} {'mean_err':>10} {'std_err':>8} {'mean_iters':>11} {'mean_ms':>9}")
    print(f"  {'-'*52}")
    for name in ALGOS:
        ea = np.array(all_errs3[name]); ia = np.array(all_iters3[name]); ta = np.array(all_times3[name])
        print(f"  {name:<8} {ea.mean():>9.4f}%  {ea.std():>7.4f}%  {ia.mean():>10.1f}  {ta.mean():>8.2f}ms")

    # ── Summary ──────────────────────────────────────────
    section("Summary: Algorithm Properties")
    print("""
  Algorithm  Convergence   Memory   Best for
  ─────────  ────────────  ───────  ──────────────────────────────────
  ALS        Linear        Low      General purpose; well-conditioned
  ASD        Sub-linear    Low      Simple; mathematically weaker
  dGN        Quadratic     High     Exact/mathematical problems
  PMF3       Quadratic     Medium   Noisy real-world data

  Derivation (from Kolda & Bader, ALS normal equations):
    ALS update: A_n ← T_(n) · KR^* · conj(G)^{-1}
    ASD update: A_n ← A_n - α · (A_n G - T_(n) KR),  α = ||grad||² / ||grad G||²
    dGN update: All A_n simultaneously, (G + λI)^{-1} T_(n) KR
    """)


if __name__ == "__main__":
    main()
