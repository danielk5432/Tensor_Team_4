"""
Convergence experiment: HOSVD vs HOOI vs CP-ALS.

- Input: random tensors (50, 50, 3) x 3 trials
- Tucker rank: [10, 10, 3]
- CP rank: 10
- max_iter: 500
- Output: iteration vs relative error plots
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import jax.numpy as jnp

from src.decomposition import hosvd, hooi, cp_als
from src.utils.visualization import plot_error_curve


_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "convergence"
_TUCKER_RANK = [10, 10, 3]
_CP_RANK = 10
_MAX_ITER = 500
_N_TRIALS = 3

def run(output_dir: Optional[str] = None, seed: int = 42) -> None:
    out = Path(output_dir) if output_dir else _OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    all_hosvd_errors = []
    all_hooi_errors = []
    all_cp_errors = []

    for trial in range(_N_TRIALS):
        X_np = rng.standard_normal((50, 50, 3)).astype(np.float32)
        X = jnp.array(X_np)

        # HOSVD: single pass → one error point
        hosvd_result = hosvd.decompose(X, _TUCKER_RANK)
        X_hosvd = hosvd.reconstruct(hosvd_result)
        hosvd_err = float(jnp.linalg.norm(X - X_hosvd) / jnp.linalg.norm(X))
        all_hosvd_errors.append(hosvd_err)

        # HOOI: iterative
        _, hooi_errors = hooi.decompose(X, _TUCKER_RANK, max_iter=_MAX_ITER)
        all_hooi_errors.append(hooi_errors)

        # CP-ALS
        _, cp_errors = cp_als.decompose(X, _CP_RANK, max_iter=_MAX_ITER)
        all_cp_errors.append(cp_errors)

        print(
            f"Trial {trial + 1}: HOSVD={hosvd_err:.4f}, "
            f"HOOI_final={hooi_errors[-1]:.4f} ({len(hooi_errors)} iters), "
            f"CP_final={cp_errors[-1]:.4f} ({len(cp_errors)} iters)"
        )

    # Average curves across trials
    max_hooi = max(len(e) for e in all_hooi_errors)
    max_cp = max(len(e) for e in all_cp_errors)

    avg_hooi = [
        np.mean([e[i] if i < len(e) else e[-1] for e in all_hooi_errors])
        for i in range(max_hooi)
    ]
    avg_cp = [
        np.mean([e[i] if i < len(e) else e[-1] for e in all_cp_errors])
        for i in range(max_cp)
    ]
    avg_hosvd = np.mean(all_hosvd_errors)

    errors_dict = {
        f"HOSVD (init, err={avg_hosvd:.4f})": [avg_hosvd],
        "HOOI (avg)": avg_hooi,
        "CP-ALS (avg)": avg_cp,
    }
    plot_error_curve(
        errors_dict,
        title=f"Convergence: Tucker[{_TUCKER_RANK}] vs CP-rank={_CP_RANK}, 50×50×3",
        save_path=str(out / "convergence_curves.png"),
    )
    print(f"[convergence] saved to {out / 'convergence_curves.png'}")


if __name__ == "__main__":
    run()
