"""
Initialization sensitivity experiment.
Compares HOSVD init vs random init for HOOI and CP-ALS.
Shows convergence variance across multiple random seeds.
"""
from pathlib import Path
from typing import Optional, List

import numpy as np
import jax.numpy as jnp

from src.decomposition import hooi, cp_als
from src.decomposition.hosvd import decompose as hosvd_decompose
from src.utils.visualization import plot_error_curve


_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "init_sensitivity"
_TUCKER_RANK = [10, 10, 3]
_CP_RANK = 10
_MAX_ITER = 200
_N_RANDOM = 5


def run(output_dir: Optional[str] = None, seed: int = 0) -> None:
    out = Path(output_dir) if output_dir else _OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    X_np = rng.standard_normal((50, 50, 3)).astype(np.float32)
    X = jnp.array(X_np)

    print("\n=== Init Sensitivity: HOOI ===")
    hooi_curves = {}

    # HOSVD initialization (default)
    _, errs_svd = hooi.decompose(X, _TUCKER_RANK, max_iter=_MAX_ITER)
    hooi_curves["HOSVD init"] = errs_svd
    print(f"  HOSVD init: final={errs_svd[-1]:.4f} ({len(errs_svd)} iters)")

    # Random initializations
    for i in range(_N_RANDOM):
        rs = int(rng.integers(0, 10000))
        key = jnp.array(rs)
        import jax
        factors_rand = [
            jax.random.normal(jax.random.PRNGKey(rs + n), (X.shape[n], _TUCKER_RANK[n]))
            for n in range(X.ndim)
        ]
        from src.decomposition.hooi import _update_factor
        from src.utils.tensor_ops import multi_mode_product
        factors = factors_rand
        errors_rand: List[float] = []
        prev_err = float("inf")
        for _ in range(_MAX_ITER):
            for n in range(X.ndim):
                factors[n] = _update_factor(X, factors, n, _TUCKER_RANK[n])
            core = multi_mode_product(X, [A.T for A in factors])
            X_hat = multi_mode_product(core, factors)
            e = float(jnp.linalg.norm(X - X_hat) / jnp.linalg.norm(X))
            errors_rand.append(e)
            if abs(prev_err - e) < 1e-6:
                break
            prev_err = e
        hooi_curves[f"Random init {i+1}"] = errors_rand
        print(f"  Random init {i+1}: final={errors_rand[-1]:.4f} ({len(errors_rand)} iters)")

    plot_error_curve(
        hooi_curves,
        title=f"HOOI Init Sensitivity — Tucker rank={_TUCKER_RANK}",
        save_path=str(out / "hooi_init_sensitivity.png"),
    )

    print("\n=== Init Sensitivity: CP-ALS ===")
    cp_curves = {}

    _, errs_svd_cp = cp_als.decompose(X, _CP_RANK, max_iter=_MAX_ITER, init="svd")
    cp_curves["SVD init"] = errs_svd_cp
    print(f"  SVD init: final={errs_svd_cp[-1]:.4f} ({len(errs_svd_cp)} iters)")

    for i in range(_N_RANDOM):
        rs = int(rng.integers(0, 10000))
        _, errs_rand_cp = cp_als.decompose(
            X, _CP_RANK, max_iter=_MAX_ITER, init="random", random_seed=rs
        )
        cp_curves[f"Random init {i+1}"] = errs_rand_cp
        print(f"  Random init {i+1}: final={errs_rand_cp[-1]:.4f} ({len(errs_rand_cp)} iters)")

    plot_error_curve(
        cp_curves,
        title=f"CP-ALS Init Sensitivity — rank={_CP_RANK}",
        save_path=str(out / "cp_als_init_sensitivity.png"),
    )
    print(f"[init_sensitivity] saved to {out}")


if __name__ == "__main__":
    run()
