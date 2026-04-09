#!/usr/bin/env python3
"""
Varying Features Per Level sweep.

Groups all sizediff/ folders by n_features_per_level (F=2 vs F=4),
plotting PSNR and SSIM at epoch 250 as a function of total encoder
parameter count. Each value of F appears as a separate colored series.

Note: the baseline comparison is n_levels=23, log2_hash=23.
If 23_2_23 exists it will appear as the F=2 data point at that param count.

Output: results/features_psnr_epoch250.png
        results/features_ssim_epoch250.png
"""

import json
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

# ── Paths ─────────────────────────────────────────────────────────────────────
SIZEDIFF_DIR = os.path.join(os.path.dirname(__file__), "sizediff")
PERFECT_PATH = os.path.join(SIZEDIFF_DIR, "perfect", "0500_1400.png")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
EPOCH_FILE   = "0250_1400.png"

# ── Encoder geometry ──────────────────────────────────────────────────────────
BASE_RESOLUTION = 16

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


def encoder_param_count(n_levels: int, n_features: int, log2_hash: int, b: float = 2.0) -> int:
    """Parameter count per encoder using the thesis formula N_l = floor(N_min * b^l)."""
    T = 2 ** log2_hash
    return sum(
        min(math.floor(BASE_RESOLUTION * (b ** l)) ** 3, T) * n_features
        for l in range(n_levels)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ref = load_gray(PERFECT_PATH)
    # groups[n_features] = list of dicts
    groups: dict[int, list[dict]] = {}

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

        img_path = os.path.join(folder_path, EPOCH_FILE)
        if not os.path.exists(img_path):
            print(f"[warn] missing {img_path}")
            continue

        n_params = encoder_param_count(n_levels, n_features, log2_hash) * 4
        psnr_val, ssim_val = compute_metrics(ref, load_gray(img_path))
        groups.setdefault(n_features, []).append({
            "name":     folder,
            "n_params": n_params,
            "psnr":     psnr_val,
            "ssim":     ssim_val,
        })
        print(
            f"{folder:15s}  F={n_features}  params={n_params:>12,}  "
            f"PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}"
        )

    if not groups:
        print("No data found.")
        return

    for f in groups:
        groups[f].sort(key=lambda r: r["n_params"])

    sorted_features = sorted(groups.keys())
    colors = {2: "tomato", 4: "steelblue"}
    # fallback colors for unexpected F values
    _fallback = plt.cm.get_cmap("tab10", len(sorted_features))
    for i, f in enumerate(sorted_features):
        if f not in colors:
            colors[f] = _fallback(i)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for metric_key, ylabel, filename in [
        ("psnr", "PSNR (dB)  ↑ better", "features_psnr_epoch250.png"),
        ("ssim", "SSIM  ↑ better",        "features_ssim_epoch250.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for n_features in sorted_features:
            entries = groups[n_features]
            xs     = [e["n_params"]      for e in entries]
            ys     = [e[metric_key]      for e in entries]
            names  = [e["name"]          for e in entries]
            color  = colors[n_features]

            ax.plot(
                xs, ys, "o-", color=color, linewidth=1.8, markersize=6,
                label=f"F={n_features} features/level",
            )
            for x, y, name in zip(xs, ys, names):
                ax.annotate(
                    name, xy=(x, y), xytext=(5, 3),
                    textcoords="offset points", fontsize=7, color=color,
                )

        ax.set_title(f"Varying features per level — epoch 250", fontsize=11)
        ax.set_xlabel("Encoder parameters (×4 encoders)")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
