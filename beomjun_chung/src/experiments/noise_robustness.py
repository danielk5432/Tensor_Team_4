"""
Noise robustness experiment: Frobenius (L2) vs L1 Tucker under noise.
Image: Peppers (256×256×3), Tucker rank [32, 32, 3].
Noise: Gaussian σ∈{0.05, 0.1, 0.2}, Salt-and-pepper p∈{0.1, 0.3}.
"""
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import jax.numpy as jnp

from src.decomposition import tucker_grad, tucker_l1
from src.objectives.metrics import psnr, ssim
from src.utils.data_loader import load_all_benchmark_images, make_synthetic_image
from src.utils.visualization import plot_reconstructions, plot_noise_comparison


_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "noise_robustness"
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_RANK = [32, 32, 3]
_MAX_ITER = 500


def add_gaussian_noise(X: jnp.ndarray, sigma: float, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, X.shape).astype(np.float32)
    return jnp.clip(X + jnp.array(noise), 0.0, 1.0)


def add_salt_pepper_noise(X: jnp.ndarray, prob: float, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    X_np = np.array(X, dtype=np.float32)
    mask = rng.random(X_np.shape[:2])
    X_np[mask < prob / 2] = 0.0
    X_np[(mask >= prob / 2) & (mask < prob)] = 1.0
    return jnp.array(X_np)


def _evaluate(X_clean: jnp.ndarray, X_noisy: jnp.ndarray, noise_label: str, out: Path) -> Dict:
    out.mkdir(parents=True, exist_ok=True)

    # L2 Tucker
    r_l2, errs_l2 = tucker_grad.decompose(X_noisy, _RANK, max_iter=_MAX_ITER)
    X_l2 = tucker_grad.reconstruct(r_l2)

    # L1 Tucker
    r_l1, errs_l1 = tucker_l1.decompose(X_noisy, _RANK, max_iter=_MAX_ITER)
    X_l1 = tucker_l1.reconstruct(r_l1)

    p_l2 = psnr(X_clean, X_l2)
    p_l1 = psnr(X_clean, X_l1)
    s_l2 = ssim(X_clean, X_l2)
    s_l1 = ssim(X_clean, X_l1)

    print(
        f"  {noise_label:30s}  L2: PSNR={p_l2:.2f}dB SSIM={s_l2:.4f}  "
        f"L1: PSNR={p_l1:.2f}dB SSIM={s_l1:.4f}"
    )

    tag = noise_label.replace(" ", "_").replace("=", "").replace(",", "")
    plot_reconstructions(
        {
            "Clean": X_clean,
            f"Noisy ({noise_label})": X_noisy,
            f"L2 Tucker ({p_l2:.1f}dB)": X_l2,
            f"L1 Tucker ({p_l1:.1f}dB)": X_l1,
        },
        title=f"Noise Robustness — {noise_label}",
        save_path=str(out / f"recon_{tag}.png"),
    )
    return {"label": noise_label, "psnr_l2": p_l2, "psnr_l1": p_l1}


def run(output_dir: Optional[str] = None) -> None:
    out = Path(output_dir) if output_dir else _OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    images = load_all_benchmark_images(str(_DATA_DIR))
    if "peppers" in images:
        X = images["peppers"]
    else:
        print("[noise_robustness] 'peppers' not found, using synthetic image.")
        X = make_synthetic_image(256, 256, seed=7)

    print(f"\n=== Noise Robustness: Peppers {X.shape} rank={_RANK} ===")
    results = []

    # Gaussian noise
    for sigma in [0.05, 0.1, 0.2]:
        X_noisy = add_gaussian_noise(X, sigma)
        res = _evaluate(X, X_noisy, f"Gaussian σ={sigma}", out)
        results.append(res)

    # Salt-and-pepper noise
    for prob in [0.1, 0.3]:
        X_noisy = add_salt_pepper_noise(X, prob)
        res = _evaluate(X, X_noisy, f"S&P p={prob}", out)
        results.append(res)

    labels = [r["label"] for r in results]
    psnr_l2 = [r["psnr_l2"] for r in results]
    psnr_l1 = [r["psnr_l1"] for r in results]

    plot_noise_comparison(
        labels, psnr_l2, psnr_l1,
        title=f"Noise Robustness — Tucker rank={_RANK}",
        save_path=str(out / "noise_comparison.png"),
    )
    print(f"[noise_robustness] saved to {out}")


if __name__ == "__main__":
    run()
