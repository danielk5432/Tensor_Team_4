"""
plot_analysis.py
────────────────────────────────────────────────────────────
Graphical analysis of rank-2 CP approximation.

Generates 4 figures saved to ./figures/:
  fig1_rank_vs_error.png   — rel_err vs CP rank (1..5)
  fig2_convergence.png     — rel_err per ALS iteration
  fig3_noise_stability.png — error vs noise level (ours vs TensorLy)
  fig4_tensorly_compare.png— accuracy comparison: ours vs TensorLy

Usage:
    python plot_analysis.py
    python plot_analysis.py --trials 20 --no-tensorly
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from cp_als import cp_als, frobenius_norm, reconstruct

try:
    import tensorly as tl
    from tensorly.decomposition import parafac
    HAS_TLY = True
except ImportError:
    HAS_TLY = False

os.makedirs("figures", exist_ok=True)

STYLE = {
    "ours":    dict(color="#1759A8", marker="o", linewidth=2, label="Our CP-ALS"),
    "tensorly":dict(color="#D04020", marker="s", linewidth=2, label="TensorLy parafac"),
}

# ──────────────────────────────────────────────────────────
# Data generators
# ──────────────────────────────────────────────────────────

def rand_rank2(rng):
    v = lambda: rng.standard_normal(3)
    return np.einsum("i,j,k->ijk", v(), v(), v()) + np.einsum("i,j,k->ijk", v(), v(), v())


def tly_error(T, rank, n_init=10):
    if not HAS_TLY:
        return np.nan
    T_tl = tl.tensor(T)
    best = np.inf
    for seed in range(n_init):
        try:
            cp = parafac(T_tl, rank=rank, n_iter_max=2000, tol=1e-8,
                         random_state=seed, normalize_factors=False)
            err = frobenius_norm(T - np.array(tl.cp_to_tensor(cp)))
            best = min(best, err / (frobenius_norm(T) + 1e-16))
        except Exception:
            pass
    return best


# ──────────────────────────────────────────────────────────
# Fig 1 — Error vs rank
# ──────────────────────────────────────────────────────────

def fig_rank_vs_error(n_trials=10, max_rank=5):
    print("  Generating fig1: error vs rank …")
    ranks = list(range(1, max_rank + 1))
    all_errs_ours = {r: [] for r in ranks}
    all_errs_tly  = {r: [] for r in ranks}

    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        T = rng.standard_normal((3, 3, 3))
        for r in ranks:
            _, _, e = cp_als(T, rank=r, n_restarts=10, n_iter_max=2000)
            all_errs_ours[r].append(e * 100)
            all_errs_tly[r].append(tly_error(T, rank=r) * 100)

    mean_o = [np.mean(all_errs_ours[r]) for r in ranks]
    std_o  = [np.std(all_errs_ours[r])  for r in ranks]
    mean_t = [np.mean(all_errs_tly[r])  for r in ranks]
    std_t  = [np.std(all_errs_tly[r])   for r in ranks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ranks, mean_o, yerr=std_o, capsize=5, **STYLE["ours"])
    if HAS_TLY:
        ax.errorbar(ranks, mean_t, yerr=std_t, capsize=5, **STYLE["tensorly"])

    ax.set_xlabel("CP rank $R$", fontsize=13)
    ax.set_ylabel("Relative error  $\\|T - T'\\|_F / \\|T\\|_F$ (%)", fontsize=12)
    ax.set_title(f"Approximation error vs CP rank\n"
                 f"(random $3\\times3\\times3$ tensors, {n_trials} trials, mean ± std)",
                 fontsize=13)
    ax.set_xticks(ranks)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("figures/fig1_rank_vs_error.png", dpi=150)
    plt.close()
    print("  → figures/fig1_rank_vs_error.png")
    return mean_o, mean_t


# ──────────────────────────────────────────────────────────
# Fig 2 — Convergence curve
# ──────────────────────────────────────────────────────────

def _als_with_history(T, rank, n_iter_max=500, random_state=0):
    """Run ALS and record rel_err at every iteration."""
    rng = np.random.default_rng(random_state)
    shape = T.shape
    ndim = T.ndim
    T_norm = frobenius_norm(T)
    is_complex = np.iscomplexobj(T)

    factors = [rng.standard_normal((s, rank)) for s in shape]

    history = []
    from cp_als import khatri_rao, unfold
    for iteration in range(n_iter_max):
        for mode in range(ndim):
            other = [factors[i] for i in range(ndim) if i != mode]
            kr = other[-1]
            for f in reversed(other[:-1]):
                kr = khatri_rao(f, kr)
            T_unfold = unfold(T, mode)
            gram = np.ones((rank, rank))
            for f in other:
                gram *= (f.conj().T @ f)
            reg = gram.conj() + 1e-12 * np.eye(rank)
            factors[mode] = T_unfold @ kr.conj() @ np.linalg.pinv(reg)

        T_approx = reconstruct(factors)
        err = frobenius_norm(T - T_approx) / (T_norm + 1e-16)
        history.append(err * 100)

        if len(history) > 1 and abs(history[-2] - history[-1]) < 1e-8:
            break

    return history


def fig_convergence(n_examples=5, max_iter=300):
    print("  Generating fig2: convergence curves …")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: random tensors (different seeds)
    ax = axes[0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_examples))
    for seed in range(n_examples):
        rng = np.random.default_rng(seed)
        T = rng.standard_normal((3, 3, 3))
        hist = _als_with_history(T, rank=2, n_iter_max=max_iter, random_state=seed)
        ax.semilogy(range(1, len(hist) + 1), hist,
                    color=colors[seed], alpha=0.85,
                    label=f"seed={seed}")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Relative error (%)", fontsize=12)
    ax.set_title("Convergence: random tensor (rank-2 ALS)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: exact rank-2 tensors — should drop to ~0
    ax = axes[1]
    colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, n_examples))
    for seed in range(n_examples):
        rng = np.random.default_rng(seed)
        T = rand_rank2(rng)
        hist = _als_with_history(T, rank=2, n_iter_max=max_iter, random_state=seed)
        ax.semilogy(range(1, len(hist) + 1), hist,
                    color=colors2[seed], alpha=0.85,
                    label=f"seed={seed}")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Relative error (%)", fontsize=12)
    ax.set_title("Convergence: exact rank-2 tensor (sanity check)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("ALS Convergence Curves  ($3\\times3\\times3$ tensors)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("figures/fig2_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → figures/fig2_convergence.png")


# ──────────────────────────────────────────────────────────
# Fig 3 — Noise stability
# ──────────────────────────────────────────────────────────

def fig_noise_stability(n_trials=10):
    print("  Generating fig3: noise stability …")
    noise_levels = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    mean_o, std_o, mean_t, std_t = [], [], [], []
    for noise in noise_levels:
        eo, et = [], []
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            T_clean = rand_rank2(rng)
            N = rng.standard_normal((3, 3, 3))
            N /= frobenius_norm(N)
            T = T_clean + noise * N

            _, _, e = cp_als(T, rank=2, n_restarts=8, n_iter_max=2000)
            eo.append(e * 100)
            et.append(tly_error(T, rank=2, n_init=5) * 100)

        mean_o.append(np.mean(eo)); std_o.append(np.std(eo))
        mean_t.append(np.mean(et)); std_t.append(np.std(et))

    noise_pct = [n * 100 for n in noise_levels]
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(noise_pct, mean_o, yerr=std_o, capsize=4, **STYLE["ours"])
    if HAS_TLY:
        ax.errorbar(noise_pct, mean_t, yerr=std_t, capsize=4, **STYLE["tensorly"])

    # Reference line: error = noise (ideal)
    ax.plot(noise_pct, noise_pct, "k--", alpha=0.4, linewidth=1.2, label="error = noise (reference)")

    ax.set_xlabel("Noise level $\\varepsilon$ (%)", fontsize=13)
    ax.set_ylabel("Relative error $\\varepsilon_\\mathrm{rel}$ (%)", fontsize=12)
    ax.set_title(f"Noise stability: $T = T_{{\\mathrm{{clean}}}} + \\varepsilon N$\n"
                 f"({n_trials} trials, mean ± std)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig3_noise_stability.png", dpi=150)
    plt.close()
    print("  → figures/fig3_noise_stability.png")


# ──────────────────────────────────────────────────────────
# Fig 4 — TensorLy accuracy comparison
# ──────────────────────────────────────────────────────────

def fig_tensorly_compare(n_trials=10):
    if not HAS_TLY:
        print("  Skipping fig4 (TensorLy not available).")
        return

    print("  Generating fig4: TensorLy comparison …")
    labels = ["Exact rank-2", "Noisy rank-2\n(ε=10%)", "Random"]
    errs_ours = [[], [], []]
    errs_tly  = [[], [], []]

    for seed in range(n_trials):
        rng = np.random.default_rng(seed)

        # exact rank-2
        T = rand_rank2(rng)
        _, _, e = cp_als(T, rank=2, n_restarts=10, n_iter_max=2000)
        errs_ours[0].append(e * 100)
        errs_tly[0].append(tly_error(T, rank=2) * 100)

        # noisy
        N = rng.standard_normal((3, 3, 3)); N /= frobenius_norm(N)
        Tn = T + 0.1 * N
        _, _, e = cp_als(Tn, rank=2, n_restarts=10, n_iter_max=2000)
        errs_ours[1].append(e * 100)
        errs_tly[1].append(tly_error(Tn, rank=2) * 100)

        # random
        Tr = rng.standard_normal((3, 3, 3))
        _, _, e = cp_als(Tr, rank=2, n_restarts=10, n_iter_max=2000)
        errs_ours[2].append(e * 100)
        errs_tly[2].append(tly_error(Tr, rank=2) * 100)

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    mo = [np.mean(e) for e in errs_ours]; so = [np.std(e) for e in errs_ours]
    mt = [np.mean(e) for e in errs_tly];  st = [np.std(e) for e in errs_tly]

    b1 = ax.bar(x - w/2, mo, w, yerr=so, capsize=5,
                color=STYLE["ours"]["color"], alpha=0.85, label="Our CP-ALS")
    b2 = ax.bar(x + w/2, mt, w, yerr=st, capsize=5,
                color=STYLE["tensorly"]["color"], alpha=0.85, label="TensorLy parafac")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Relative error (%)", fontsize=12)
    ax.set_title(f"Our CP-ALS vs TensorLy parafac\n({n_trials} trials, mean ± std)",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("figures/fig4_tensorly_compare.png", dpi=150)
    plt.close()
    print("  → figures/fig4_tensorly_compare.png")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--no-tensorly", action="store_true")
    args = parser.parse_args()

    if args.no_tensorly:
        global HAS_TLY
        HAS_TLY = False

    print(f"\nGenerating analysis plots  (trials={args.trials}, TensorLy={HAS_TLY})")
    print(f"Output directory: ./figures/\n")

    fig_rank_vs_error(n_trials=args.trials)
    fig_convergence()
    fig_noise_stability(n_trials=args.trials)
    fig_tensorly_compare(n_trials=args.trials)

    print(f"\nAll figures saved to ./figures/")


if __name__ == "__main__":
    main()
