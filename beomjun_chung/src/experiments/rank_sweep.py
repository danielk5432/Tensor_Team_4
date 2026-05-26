"""
Rank sweep experiment: rank vs PSNR / compression ratio.
Compares HOSVD, HOOI, CP-ALS on Lena and Baboon (256×256×3).
Falls back to synthetic images if real ones are unavailable.
"""
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import jax.numpy as jnp

from src.decomposition import hosvd, hooi, cp_als
from src.objectives.metrics import psnr, ssim, compression_ratio, cp_compression_ratio, relative_error
from src.utils.data_loader import load_all_benchmark_images, make_synthetic_image
from src.utils.visualization import plot_rank_sweep, plot_reconstructions


_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "rank_sweep"
_DATA_DIR = Path(__file__).parent.parent.parent / "data"

_TUCKER_RANKS = [
    [128, 128, 3],
    [64,  64,  3],
    [32,  32,  3],
    [16,  16,  2],
    [8,   8,   2],
]
_CP_RANKS = [5, 10, 20, 50, 100]


def _run_for_image(name: str, X: jnp.ndarray, out: Path) -> None:
    print(f"\n=== {name.upper()} {X.shape} ===")
    out_img = out / name
    out_img.mkdir(parents=True, exist_ok=True)

    psnr_hosvd, psnr_hooi, psnr_cp = [], [], []
    cr_tucker, cr_cp = [], []

    for rank in _TUCKER_RANKS:
        # HOSVD
        r_hs = hosvd.decompose(X, rank)
        X_hs = hosvd.reconstruct(r_hs)
        psnr_hosvd.append(psnr(X, X_hs))
        cr_tucker.append(compression_ratio(X.shape, rank))

        # HOOI
        r_ho, _ = hooi.decompose(X, rank, max_iter=100)
        X_ho = hooi.reconstruct(r_ho)
        psnr_hooi.append(psnr(X, X_ho))

        print(
            f"  Tucker rank={rank}: HOSVD={psnr_hosvd[-1]:.2f}dB "
            f"HOOI={psnr_hooi[-1]:.2f}dB CR={cr_tucker[-1]:.4f}"
        )

    for rank in _CP_RANKS:
        r_cp, _ = cp_als.decompose(X, rank, max_iter=100)
        X_cp = cp_als.reconstruct(r_cp)
        psnr_cp.append(psnr(X, X_cp))
        cr_cp.append(cp_compression_ratio(X.shape, rank))
        print(f"  CP rank={rank}: PSNR={psnr_cp[-1]:.2f}dB CR={cr_cp[-1]:.4f}")

    # Tucker: rank vs PSNR
    plot_rank_sweep(
        ranks=_TUCKER_RANKS,
        psnr_dict={"HOSVD": psnr_hosvd, "HOOI": psnr_hooi},
        cr_dict={"Tucker": cr_tucker},
        image_name=name,
        save_path=str(out_img / "tucker_rank_sweep.png"),
    )
    # CP: rank vs PSNR
    plot_rank_sweep(
        ranks=_CP_RANKS,
        psnr_dict={"CP-ALS": psnr_cp},
        cr_dict={"CP-ALS": cr_cp},
        image_name=name,
        save_path=str(out_img / "cp_rank_sweep.png"),
    )

    # Reconstruction panel at mid rank
    mid_tucker = _TUCKER_RANKS[2]  # [32, 32, 3]
    r_hs = hosvd.decompose(X, mid_tucker)
    r_ho, _ = hooi.decompose(X, mid_tucker, max_iter=100)
    r_cp, _ = cp_als.decompose(X, _CP_RANKS[2], max_iter=100)  # rank=20
    plot_reconstructions(
        {
            "Original": X,
            f"HOSVD {mid_tucker}": hosvd.reconstruct(r_hs),
            f"HOOI {mid_tucker}": hooi.reconstruct(r_ho),
            f"CP rank={_CP_RANKS[2]}": cp_als.reconstruct(r_cp),
        },
        title=f"{name} — mid-rank reconstruction",
        save_path=str(out_img / "reconstruction_panel.png"),
    )
    print(f"  Saved figures to {out_img}")


def run(output_dir: Optional[str] = None) -> None:
    out = Path(output_dir) if output_dir else _OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    images = load_all_benchmark_images(str(_DATA_DIR))
    targets = ["lena", "baboon"]

    for name in targets:
        if name in images:
            X = images[name]
        else:
            print(f"[rank_sweep] '{name}' not found, using synthetic image.")
            X = make_synthetic_image(256, 256, seed=hash(name) % 100)
        _run_for_image(name, X, out)


if __name__ == "__main__":
    run()
