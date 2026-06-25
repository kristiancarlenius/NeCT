#!/usr/bin/env python3
"""
Compare CombinedCubes variants at epoch 150.

Reads every subfolder of sizediff/combinedcube/, loads the 0150_1400.png
image, and computes MSE and PSNR against the perfect reference using the
6 panel crops defined in crops.json.

Folder name format: NL1_NF1_NL2_NF2_NEURONS
  NL1   – n_levels for encoder 1
  NF1   – n_features_per_level for encoder 1
  NL2   – n_levels for encoder 2
  NF2   – n_features_per_level for encoder 2
  NEURONS – MLP width

Output: results/combinedcube_mse.png  and  results/combinedcube_psnr.png
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(__file__)
COMBINEDCUBE_DIR = os.path.join(_HERE, "sizediff", "combinedcubes")
PERFECT_PATH   = os.path.join(_HERE, "sizediff", "perfect", "0525_1400.png")
RESULTS_DIR    = os.path.join(_HERE, "results")
EPOCH_FILE     = "0150_1400.png"

# ── Crop regions ──────────────────────────────────────────────────────────────
_crops_file = os.path.join(_HERE, "crops.json")
if not os.path.exists(_crops_file):
    raise FileNotFoundError("crops.json not found — run: python crop_tool.py <image.png>")
with open(_crops_file) as _f:
    PANEL_CROPS: list[dict] = json.load(_f)["crops"]
print(f"Loaded {len(PANEL_CROPS)} panel crops from crops.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def get_roi(img: np.ndarray) -> np.ndarray:
    return np.concatenate([
        img[max(0, c["y0"]):min(img.shape[0], c["y1"]),
            max(0, c["x0"]):min(img.shape[1], c["x1"])].ravel()
        for c in PANEL_CROPS
    ])


def mse(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.mean((ref - test) ** 2))


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    m = mse(ref, test)
    return float("inf") if m == 0 else 10.0 * math.log10(1.0 / m)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ref = get_roi(load_gray(PERFECT_PATH))

    results = []
    for folder in sorted(os.listdir(COMBINEDCUBE_DIR)):
        folder_path = os.path.join(COMBINEDCUBE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        img_path = os.path.join(folder_path, EPOCH_FILE)
        if not os.path.exists(img_path):
            print(f"[skip] {folder}: {EPOCH_FILE} not found")
            continue

        test = get_roi(load_gray(img_path))
        m = mse(ref, test)
        p = psnr(ref, test)
        results.append({"name": folder, "mse": m, "psnr": p})
        print(f"{folder:25s}  MSE={m:.6f}  PSNR={p:.2f} dB")

    if not results:
        print("No results found.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    names  = [r["name"]  for r in results]
    mse_v  = [r["mse"]   for r in results]
    psnr_v = [r["psnr"]  for r in results]
    x      = np.arange(len(names))
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(names)))

    for metric, values, ylabel, filename, better in [
        ("MSE",  mse_v,  "MSE  ↓ better",       "combinedcube_mse.png",  "min"),
        ("PSNR", psnr_v, "PSNR (dB)  ↑ better", "combinedcube_psnr.png", "max"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)

        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.5f}" if metric == "MSE" else f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

        # Highlight best bar
        best_idx = int(np.argmin(values) if better == "min" else np.argmax(values))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"CombinedCubes - {metric} at epoch 150", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
