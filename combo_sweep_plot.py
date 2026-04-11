#!/usr/bin/env python3
"""
Full combination sweep plots.

For n_features=4 (the middle component of folder name {N}_{F}_{H}):

  n_levels sweep  — for each unique log2_hashmap_size H, plot PSNR/SSIM
                    vs n_levels for every N that has data at that H.
                    Folder pattern:  X_4_H  (X varies, H is fixed)

  hashmap sweep   — for each unique n_levels N, plot PSNR/SSIM
                    vs log2_hashmap_size for every H that has data at that N.
                    Folder pattern:  N_4_X  (X varies, N is fixed)

Outputs:
  results/nlev_sweep/nlev_psnr_H<H>.png
  results/nlev_sweep/nlev_ssim_H<H>.png
  results/hash_sweep/hash_psnr_N<N>.png
  results/hash_sweep/hash_ssim_N<N>.png
"""

import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
SIZEDIFF_DIR = os.path.join(BASE_DIR, "sizediff")
PERFECT_PATH = os.path.join(SIZEDIFF_DIR, "perfect", "0650_1400.png")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
EPOCH_FILE   = "0250_1400.png"

# Only folders with this n_features value are considered.
FIXED_N_FEATURES = 4

# ── Crop regions ──────────────────────────────────────────────────────────────
_CROPS_FILE = os.path.join(BASE_DIR, "crops.json")
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
        y0 = max(0, c["y0"]); y1 = min(ref.shape[0], c["y1"])
        x0 = max(0, c["x0"]); x1 = min(ref.shape[1], c["x1"])
        rp = ref[y0:y1, x0:x1]
        tp = test[y0:y1, x0:x1]
        m = float(np.mean((rp - tp) ** 2))
        psnr_vals.append(10.0 * math.log10(1.0 / m) if m > 0 else float("inf"))
        ssim_vals.append(float(structural_similarity(rp, tp, data_range=1.0)))
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def save_plot(
    xs: list[int],
    ys: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    color: str,
    out_path: str,
    metric_key: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=7)
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.3f}" if metric_key == "ssim" else f"{y:.2f}",
            xy=(x, y), xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=8,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Scan all folders once
# ─────────────────────────────────────────────────────────────────────────────

def scan_folders(ref: np.ndarray) -> list[dict]:
    """
    Walk sizediff/ and return a list of dicts:
      { n_levels, n_features, log2_hash, psnr, ssim }
    for every folder that matches N_F_H and has EPOCH_FILE.
    """
    records = []
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
        if n_features != FIXED_N_FEATURES:
            continue
        img_path = os.path.join(folder_path, EPOCH_FILE)
        if not os.path.exists(img_path):
            print(f"  [warn] missing {img_path}")
            continue
        psnr_val, ssim_val = compute_metrics(ref, load_gray(img_path))
        records.append({
            "n_levels":   n_levels,
            "n_features": n_features,
            "log2_hash":  log2_hash,
            "psnr":       psnr_val,
            "ssim":       ssim_val,
        })
        print(f"  {folder:16s}  PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ref = load_gray(PERFECT_PATH)

    print("\nScanning folders …")
    records = scan_folders(ref)
    if not records:
        print("No matching folders found.")
        return

    # ── Group for n_levels sweep (X_4_H): fixed H, varying n_levels ──────────
    # key = log2_hash
    by_hash: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_hash[r["log2_hash"]].append(r)

    print(f"\nGenerating n_levels sweep plots ({len(by_hash)} fixed-H values) …")
    for h_val, group in sorted(by_hash.items()):
        group.sort(key=lambda r: r["n_levels"])
        xs    = [r["n_levels"] for r in group]
        psnrs = [r["psnr"]     for r in group]
        ssims = [r["ssim"]     for r in group]

        for metric_key, ys, ylabel, suffix in [
            ("psnr", psnrs, "PSNR (dB)  ↑ better", "psnr"),
            ("ssim", ssims, "SSIM  ↑ better",        "ssim"),
        ]:
            save_plot(
                xs=xs, ys=ys,
                xlabel="Number of levels (L)",
                ylabel=ylabel,
                title=f"n_levels sweep  (log₂T={h_val}, F={FIXED_N_FEATURES}, epoch 250)",
                color="darkorange",
                out_path=os.path.join(RESULTS_DIR, "nlev_sweep", f"nlev_{suffix}_H{h_val}.png"),
                metric_key=metric_key,
            )

    # ── Group for hashmap sweep (N_4_X): fixed N, varying log2_hash ──────────
    # key = n_levels
    by_levels: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_levels[r["n_levels"]].append(r)

    print(f"\nGenerating hashmap sweep plots ({len(by_levels)} fixed-N values) …")
    for n_val, group in sorted(by_levels.items()):
        group.sort(key=lambda r: r["log2_hash"])
        xs    = [r["log2_hash"] for r in group]
        psnrs = [r["psnr"]      for r in group]
        ssims = [r["ssim"]      for r in group]

        for metric_key, ys, ylabel, suffix in [
            ("psnr", psnrs, "PSNR (dB)  ↑ better", "psnr"),
            ("ssim", ssims, "SSIM  ↑ better",        "ssim"),
        ]:
            save_plot(
                xs=xs, ys=ys,
                xlabel="log₂ hashmap size",
                ylabel=ylabel,
                title=f"hashmap sweep  (L={n_val}, F={FIXED_N_FEATURES}, epoch 250)",
                color="steelblue",
                out_path=os.path.join(RESULTS_DIR, "hash_sweep", f"hash_{suffix}_N{n_val}.png"),
                metric_key=metric_key,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
