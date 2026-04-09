#!/usr/bin/env python3
"""
Varying Hashmap Size sweep.

Filters to folders where n_levels==23 and n_features_per_level==4,
varying log2_hashmap_size across 17–23. Plots PSNR and SSIM at epoch
250 as a function of log2_hashmap_size.

Output: results/hashmap_psnr_epoch250.png
        results/hashmap_ssim_epoch250.png
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

# ── Paths ─────────────────────────────────────────────────────────────────────
SIZEDIFF_DIR = os.path.join(os.path.dirname(__file__), "sizediff")
PERFECT_PATH = os.path.join(SIZEDIFF_DIR, "perfect", "0500_1400.png")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
EPOCH_FILE   = "0250_1400.png"

# ── Sweep constraints ─────────────────────────────────────────────────────────
FIXED_N_LEVELS   = 23
FIXED_N_FEATURES = 4

# ── Crop regions ──────────────────────────────────────────────────────────────
_CROPS_FILE = os.path.join(os.path.dirname(__file__), "crops.json")
if not os.path.exists(_CROPS_FILE):
    raise FileNotFoundError("crops.json not found — run: python crop_tool.py <image.png>")
with open(_CROPS_FILE) as _f:
    PANEL_CROPS: list[dict] = json.load(_f)["crops"]
print(f"Loaded {len(PANEL_CROPS)} panel crops from crops.json")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def compute_metrics(ref: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Return (PSNR, SSIM) averaged over all panel crops."""
    psnr_vals, ssim_vals = [], []
    for c in PANEL_CROPS:
        y0, y1 = max(0, c["y0"]), min(ref.shape[0], c["y1"])
        x0, x1 = max(0, c["x0"]), min(ref.shape[1], c["x1"])
        rp = ref[y0:y1, x0:x1]
        tp = test[y0:y1, x0:x1]
        m = float(np.mean((rp - tp) ** 2))
        psnr_vals.append(10.0 * math.log10(1.0 / m) if m > 0 else float("inf"))
        ssim_vals.append(float(structural_similarity(rp, tp, data_range=1.0)))
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ref = load_gray(PERFECT_PATH)
    results: list[dict] = []

    for folder in sorted(os.listdir(SIZEDIFF_DIR)):
        if folder == "perfect":
            continue
        folder_path = os.path.join(SIZEDIFF_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        parts = folder.split("_")
        if len(parts) != 3:
            continue
        try:
            n_levels, n_features, log2_hash = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue

        if n_levels != FIXED_N_LEVELS or n_features != FIXED_N_FEATURES:
            continue

        img_path = os.path.join(folder_path, EPOCH_FILE)
        if not os.path.exists(img_path):
            print(f"[warn] missing {img_path}")
            continue

        psnr_val, ssim_val = compute_metrics(ref, load_gray(img_path))
        results.append({"log2_hash": log2_hash, "psnr": psnr_val, "ssim": ssim_val})
        print(f"{folder:15s}  log2_hash={log2_hash}  PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}")

    if not results:
        print("No matching folders found.")
        return

    results.sort(key=lambda r: r["log2_hash"])
    xs    = [r["log2_hash"] for r in results]
    psnrs = [r["psnr"]     for r in results]
    ssims = [r["ssim"]     for r in results]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for metric_key, ys, ylabel, filename in [
        ("psnr", psnrs, "PSNR (dB)  ↑ better", "hashmap_psnr_epoch250.png"),
        ("ssim", ssims, "SSIM  ↑ better",        "hashmap_ssim_epoch250.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, "o-", color="steelblue", linewidth=2, markersize=7)
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.3f}" if metric_key == "ssim" else f"{y:.2f}",
                xy=(x, y), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8,
            )
        ax.set_title(
            f"Varying log₂ hashmap size  (L={FIXED_N_LEVELS}, F={FIXED_N_FEATURES}, epoch 250)",
            fontsize=11,
        )
        ax.set_xlabel("log₂ hashmap size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
