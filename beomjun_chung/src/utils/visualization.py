"""Visualization utilities: reconstruction panels, error curves, rank-sweep plots."""
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp


def _to_np(img) -> np.ndarray:
    return np.array(img, dtype=np.float32).clip(0.0, 1.0)


def plot_reconstructions(
    images: Dict[str, jnp.ndarray],
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side image comparison. images = {'label': tensor_HxWxC}."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, img) in zip(axes, images.items()):
        ax.imshow(_to_np(img))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_curve(
    errors: Dict[str, List[float]],
    title: str = "Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Relative Error",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Semilogy convergence plot. errors = {'label': [float, ...]}."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, vals in errors.items():
        if len(vals) == 1:
            ax.scatter([0], vals, label=label, zorder=5, s=60)
        else:
            ax.semilogy(vals, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rank_sweep(
    ranks: List,
    psnr_dict: Dict[str, List[float]],
    cr_dict: Dict[str, List[float]],
    image_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Dual-panel: rank vs PSNR and rank vs compression ratio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    rank_labels = [str(r) for r in ranks]
    xs = list(range(len(ranks)))

    for label, vals in psnr_dict.items():
        ax1.plot(xs, vals, marker="o", label=label)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(rank_labels, rotation=30, ha="right", fontsize=7)
    ax1.set_xlabel("Rank")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"Rank vs PSNR — {image_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for label, vals in cr_dict.items():
        ax2.plot(xs, vals, marker="o", label=label)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(rank_labels, rotation=30, ha="right", fontsize=7)
    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Compression Ratio")
    ax2.set_title(f"Rank vs Compression Ratio — {image_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_noise_comparison(
    noise_types: List[str],
    psnr_l2: List[float],
    psnr_l1: List[float],
    title: str = "Noise Robustness: L2 vs L1",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing L2 vs L1 Tucker PSNR under different noise conditions."""
    x = np.arange(len(noise_types))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, psnr_l2, width, label="L2 (Frobenius)")
    ax.bar(x + width / 2, psnr_l1, width, label="L1")
    ax.set_xticks(x)
    ax.set_xticklabels(noise_types, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
