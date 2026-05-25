"""
Evaluation metrics: PSNR, SSIM, relative error, compression ratio.
Ma et al. (2025) Table 4/5 standard metrics.
"""
import jax.numpy as jnp
import numpy as np


def relative_error(original: jnp.ndarray, reconstructed: jnp.ndarray) -> float:
    """||X - X_hat||_F / ||X||_F"""
    return float(jnp.linalg.norm(original - reconstructed) / jnp.linalg.norm(original))


def psnr(original: jnp.ndarray, reconstructed: jnp.ndarray, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB). Pixels in [0, 1].
    PSNR = 20 * log10(max_val / RMSE)
    """
    mse = float(jnp.mean((original - reconstructed) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(max_val / np.sqrt(mse))


def ssim(original: jnp.ndarray, reconstructed: jnp.ndarray) -> float:
    """
    Structural Similarity Index.
    Uses skimage.metrics.structural_similarity (requires numpy conversion).
    """
    from skimage.metrics import structural_similarity as _ssim

    orig_np = np.array(original).clip(0.0, 1.0)
    rec_np = np.array(reconstructed).clip(0.0, 1.0)
    if orig_np.ndim == 3:
        return float(_ssim(orig_np, rec_np, data_range=1.0, channel_axis=2))
    return float(_ssim(orig_np, rec_np, data_range=1.0))


def compression_ratio(original_shape: tuple, rank: list) -> float:
    """
    Tucker compression ratio = storage_cost / original_cost.
    storage_cost = prod(rank) + sum(I_n * R_n)
    original_cost = prod(original_shape)
    """
    core_size = 1
    for r in rank:
        core_size *= r
    factor_size = sum(s * r for s, r in zip(original_shape, rank))
    original_size = 1
    for s in original_shape:
        original_size *= s
    return (core_size + factor_size) / original_size


def cp_compression_ratio(original_shape: tuple, rank: int) -> float:
    """CP compression ratio: R * (sum of dims) / prod(dims)."""
    factor_size = rank * sum(original_shape)
    original_size = 1
    for s in original_shape:
        original_size *= s
    return factor_size / original_size
