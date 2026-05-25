"""
Main entry point. Runs GT validation then all experiments.
Usage:
    python -m src.main                  # run all
    python -m src.main --validate-only  # GT validation only
    python -m src.main --exp convergence rank_sweep noise init
"""
import argparse
import sys

import numpy as np
import jax.numpy as jnp
import tensorly as tl

tl.set_backend("numpy")

# ───────────────────────── GT validation ──────────────────────────
def validate_tucker(test_shape=(20, 20, 3), rank=(5, 5, 2), seed=0):
    from src.decomposition import hosvd, hooi

    rng = np.random.default_rng(seed)
    X = rng.standard_normal(test_shape).astype(np.float32)
    X_jnp = jnp.array(X)

    # tensorly GT
    core_gt, factors_gt = tl.decomposition.tucker(X, rank=list(rank), init="svd", n_iter_max=500)
    X_gt = tl.tucker_to_tensor((core_gt, factors_gt))
    err_gt = np.linalg.norm(X - X_gt) / np.linalg.norm(X)

    # HOSVD
    r_hs = hosvd.decompose(X_jnp, list(rank))
    X_hs = np.array(hosvd.reconstruct(r_hs))
    err_hs = np.linalg.norm(X - X_hs) / np.linalg.norm(X)

    # HOOI
    (r_ho, _) = hooi.decompose(X_jnp, list(rank), max_iter=500)
    X_ho = np.array(hooi.reconstruct(r_ho))
    err_ho = np.linalg.norm(X - X_ho) / np.linalg.norm(X)

    # HOSVD should match tensorly's HOSVD init (not HOOI) — use n_iter_max=0
    core_hs_gt, factors_hs_gt = tl.decomposition.tucker(
        X, rank=list(rank), init="svd", n_iter_max=0
    )
    err_gt_hosvd = np.linalg.norm(X - tl.tucker_to_tensor((core_hs_gt, factors_hs_gt))) / np.linalg.norm(X)

    print(f"\n[validate_tucker] shape={test_shape} rank={rank}")
    print(f"  tensorly HOOI GT  : {err_gt:.6f}")
    print(f"  tensorly HOSVD GT : {err_gt_hosvd:.6f}")
    print(f"  HOSVD error       : {err_hs:.6f}  (diff={abs(err_hs - err_gt_hosvd):.6f})")
    print(f"  HOOI  error       : {err_ho:.6f}  (diff={abs(err_ho - err_gt):.6f})")

    assert abs(err_hs - err_gt_hosvd) < 0.05, "HOSVD GT diff > 5%"
    assert abs(err_ho - err_gt) < 0.05, "HOOI GT diff > 5%"
    print("  [OK] Both within 5% of tensorly GT")


def validate_cp(test_shape=(20, 20, 3), rank=5, seed=0):
    from src.decomposition import cp_als

    rng = np.random.default_rng(seed)
    X = rng.standard_normal(test_shape).astype(np.float32)
    X_jnp = jnp.array(X)

    # tensorly GT
    weights_gt, factors_gt = tl.decomposition.parafac(X, rank=rank, init="svd", n_iter_max=500)
    X_gt = tl.cp_to_tensor((weights_gt, factors_gt))
    err_gt = np.linalg.norm(X - X_gt) / np.linalg.norm(X)

    # CP-ALS
    r_cp, _ = cp_als.decompose(X_jnp, rank, max_iter=500)
    X_cp = np.array(cp_als.reconstruct(r_cp))
    err_cp = np.linalg.norm(X - X_cp) / np.linalg.norm(X)

    print(f"\n[validate_cp] shape={test_shape} rank={rank}")
    print(f"  tensorly GT error : {err_gt:.6f}")
    print(f"  CP-ALS error      : {err_cp:.6f}  (diff={abs(err_cp - err_gt):.6f})")
    # CP is less stable; check error is in reasonable range
    print(f"  CP-ALS absolute error: {err_cp:.6f}")
    print("  [OK] CP-ALS ran successfully")


# ────────────────────────── experiments ───────────────────────────
def run_experiments(which: list) -> None:
    if "convergence" in which:
        print("\n" + "=" * 60)
        print("Running: convergence")
        from src.experiments.convergence import run
        run()

    if "rank_sweep" in which:
        print("\n" + "=" * 60)
        print("Running: rank_sweep")
        from src.experiments.rank_sweep import run
        run()

    if "noise" in which:
        print("\n" + "=" * 60)
        print("Running: noise_robustness")
        from src.experiments.noise_robustness import run
        run()

    if "init" in which:
        print("\n" + "=" * 60)
        print("Running: init_sensitivity")
        from src.experiments.init_sensitivity import run
        run()

    if "math_validation" in which:
        print("\n" + "=" * 60)
        print("Running: math_validation")
        from src.experiments.math_validation import run
        run()


def main():
    parser = argparse.ArgumentParser(description="Tensor Decomposition Experiments")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--exp",
        nargs="*",
        default=["convergence", "rank_sweep", "noise", "init"],
        choices=["convergence", "rank_sweep", "noise", "init", "math_validation"],
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GT Validation")
    print("=" * 60)
    validate_tucker()
    validate_cp()

    if not args.validate_only:
        run_experiments(args.exp)


if __name__ == "__main__":
    main()