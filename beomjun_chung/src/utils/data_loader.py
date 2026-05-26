"""
USC SIPI Image Database loader.
Images (House, Peppers, Lena, Airplane, Baboon, Boats) should be placed in data/.
Supported formats: .tif, .tiff, .png, .jpg, .bmp
"""
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import jax.numpy as jnp
from PIL import Image

SIPI_NAMES = ["house", "peppers", "lena", "airplane", "baboon", "boats"]

_EXTENSIONS = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"]


def _find_image(data_dir: str, name: str) -> Optional[str]:
    """Search data_dir for an image file matching the given name (case-insensitive)."""
    data_path = Path(data_dir)
    for ext in _EXTENSIONS:
        for candidate in [name, name.capitalize(), name.upper()]:
            p = data_path / (candidate + ext)
            if p.exists():
                return str(p)
    # Partial match
    for f in data_path.iterdir():
        if f.suffix.lower() in _EXTENSIONS and name.lower() in f.stem.lower():
            return str(f)
    return None


def load_sipi_image(path: str, target_size: Optional[tuple] = None) -> jnp.ndarray:
    """
    Load a single image from disk.
    Returns float32 JAX array of shape (H, W, C) with pixels in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return jnp.array(arr)


def load_all_benchmark_images(
    data_dir: str,
    target_size: Optional[tuple] = (256, 256),
) -> Dict[str, jnp.ndarray]:
    """
    Load all available SIPI benchmark images from data_dir.
    Returns dict mapping image name -> (H, W, 3) float32 tensor in [0, 1].
    Missing images are skipped with a warning.
    """
    images = {}
    for name in SIPI_NAMES:
        path = _find_image(data_dir, name)
        if path is None:
            print(f"[data_loader] WARNING: '{name}' not found in {data_dir}, skipping.")
            continue
        images[name] = load_sipi_image(path, target_size)
        print(f"[data_loader] Loaded '{name}' from {path}, shape={images[name].shape}")
    return images


def make_synthetic_image(height: int = 256, width: int = 256, seed: int = 0) -> jnp.ndarray:
    """Generate a smooth synthetic RGB image for testing when real images are unavailable."""
    rng = np.random.default_rng(seed)
    # Low-frequency components to resemble natural images
    img = np.zeros((height, width, 3), dtype=np.float32)
    for _ in range(10):
        freq_y = rng.integers(1, 8)
        freq_x = rng.integers(1, 8)
        phase = rng.uniform(0, 2 * np.pi, 3)
        amp = rng.uniform(0.05, 0.2, 3)
        ys = np.linspace(0, 2 * np.pi * freq_y, height)
        xs = np.linspace(0, 2 * np.pi * freq_x, width)
        pattern = np.sin(ys[:, None] + phase[None, :1]) * np.cos(xs[None, :] + phase[None, 1:2])
        img += amp[None, None, :] * pattern[:, :, None]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return jnp.array(img.astype(np.float32))
