#!/usr/bin/env python3
"""
Plot reconstruction quality at the equalized-time checkpoint.

For each XX_4_YY folder in sizediff/, finds the NNNN_1400.png image that is
NOT 0100 or 0250 (the equalized wall-clock checkpoint), compares it against
the perfect reference, and plots MSE and PSNR grouped by n_levels (XX).

Output: results/equalized_mse.png  and  results/equalized_psnr.png
"""

import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
SIZEDIFF_DIR = os.path.join(os.path.dirname(__file__), "sizediff")
PERFECT_PATH = os.path.join(SIZEDIFF_DIR, "perfect", "1300_1400.png")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")

# ── Crop region ───────────────────────────────────────────────────────────────
CROP_X0, CROP_Y0 = 5400, 1800
CROP_X1, CROP_Y1 = 6000, 2600

# ── Encoder geometry ──────────────────────────────────────────────────────────
BASE_RESOLUTION      = 16
MAX_RESOLUTION_FACTOR = 2
N_DETECTOR           = [1880, 1496]

# ── Epochs to skip ────────────────────────────────────────────────────────────
SKIP_EPOCHS = {"0100", "0250"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def crop(img):
    h, w = img.shape
    x0, x1 = max(0, CROP_X0), min(w, CROP_X1)
    y0, y1 = max(0, CROP_Y0), min(h, CROP_Y1)
    return img[y0:y1, x0:x1]


def mse(ref, test):
    return float(np.mean((ref - test) ** 2))


def psnr(ref, test):
    m = mse(ref, test)
    return float("inf") if m == 0 else 10.0 * math.log10(1.0 / m)


def encoder_param_count(n_levels, n_features, log2_hash):
    size = max(N_DETECTOR)
    per_level_scale = (MAX_RESOLUTION_FACTOR * size / BASE_RESOLUTION) ** (
        1.0 / (n_levels - 1)
    ) if n_levels > 1 else 1.0
    T, F = 2 ** log2_hash, n_features
    return sum(min(math.floor(BASE_RESOLUTION * per_level_scale ** l) ** 3, T) * F
               for l in range(n_levels))


def find_equalized_image(folder_path):
    """Return path to the NNNN_1400.png that is not 0100 or 0250, or None."""
    for fname in os.listdir(folder_path):
        if not fname.endswith("_1400.png"):
            continue
        epoch_tag = fname[:4]
        if epoch_tag in SKIP_EPOCHS:
            continue
        if epoch_tag == "0000":
            continue
        return os.path.join(folder_path, fname), int(epoch_tag)
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ref = crop(load_gray(PERFECT_PATH))
    groups = {}   # n_levels -> list of dicts

    for folder in sorted(os.listdir(SIZEDIFF_DIR)):
        if folder == "perfect":
            continue
        parts = folder.split("_")
        if len(parts) != 3:
            continue
        try:
            n_levels, n_features, log2_hash = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if n_features != 4:
            continue

        folder_path = os.path.join(SIZEDIFF_DIR, folder)
        img_path, epoch = find_equalized_image(folder_path)
        if img_path is None:
            print(f"[skip] {folder}: no equalized-time image found")
            continue

        n_params = encoder_param_count(n_levels, n_features, log2_hash) * 4
        test = crop(load_gray(img_path))

        entry = {
            "name":     folder,
            "epoch":    epoch,
            "n_params": n_params,
            "mse":      mse(ref, test),
            "psnr":     psnr(ref, test),
        }
        groups.setdefault(n_levels, []).append(entry)
        print(f"{folder:12s}  epoch={epoch:4d}  params={n_params:>12,}  "
              f"MSE={entry['mse']:.6f}  PSNR={entry['psnr']:.2f} dB")

    if not groups:
        print("No data found.")
        return

    for nl in groups:
        groups[nl].sort(key=lambda r: r["n_params"])

    sorted_levels = sorted(groups.keys())
    cmap      = plt.cm.get_cmap("tab10", len(sorted_levels))
    color_map = {nl: cmap(i) for i, nl in enumerate(sorted_levels)}

    os.makedirs(RESULTS_DIR, exist_ok=True)

    METRICS = [
        ("mse",  "MSE  ↓ better",       "equalized_mse.png"),
        ("psnr", "PSNR (dB)  ↑ better", "equalized_psnr.png"),
    ]

    for metric_key, ylabel, filename in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))

        for n_levels in sorted_levels:
            entries = groups[n_levels]
            xs     = [e["n_params"]    for e in entries]
            ys     = [e[metric_key]    for e in entries]
            labels = [f"{e['name']}\n(ep {e['epoch']})" for e in entries]
            color  = color_map[n_levels]

            ax.plot(xs, ys, "o-", color=color, linewidth=1.8,
                    markersize=6, label=f"n_levels={n_levels}")
            for x, y, lbl in zip(xs, ys, labels):
                ax.annotate(lbl, xy=(x, y), xytext=(5, 3),
                            textcoords="offset points", fontsize=6.5, color=color)

        ax.set_title(f"{metric_key.upper()} at equalized wall-clock time  (XX_4_YY)", fontsize=11)
        ax.set_xlabel("Encoder parameters (×4 encoders)")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()

        out_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
