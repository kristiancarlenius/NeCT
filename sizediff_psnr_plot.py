#!/usr/bin/env python3
"""
Compare sizediff reconstructions against the perfect reference image.

Only folders matching XX_4_YY (n_features_per_level == 4) are included.
Each distinct XX (n_levels) is plotted as one colored series; points within
a series are connected by a line and sorted by encoder parameter count.

A single 2×2 figure is produced:
  rows  → metric  (MSE, PSNR)
  cols  → epoch   (100, 250)

Parameter count formula (multi-resolution hash grid, 3-D inputs):
  per_level_scale = (max_resolution_factor * max(nDetector) / base_resolution)
                    ^ (1 / (n_levels - 1))
  For each level l: N_l = floor(base_resolution * per_level_scale^l)
  params_l = min(N_l^3, 2^log2_hashmap_size) * n_features_per_level
  total_params = sum over all levels
"""

import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
SIZEDIFF_DIR = os.path.join(os.path.dirname(__file__), "sizediff")
PERFECT_PATH = os.path.join(SIZEDIFF_DIR, "perfect", "1300_1400.png")

# ── Crop region (same as nect/comp_slice.py) ─────────────────────────────────
CROP_X0 = 5400
CROP_Y0 = 1800
CROP_X1 = 6000
CROP_Y1 = 2600

# ── Encoder defaults shared across all sizediff runs ─────────────────────────
BASE_RESOLUTION = 16
MAX_RESOLUTION_FACTOR = 2
N_DETECTOR = [1880, 1496]

# ── Epoch checkpoints present in each subfolder ──────────────────────────────
EPOCH_FILES = {
    "100": "0100_1400.png",
    "250": "0250_1400.png",
}

EPOCH_LABELS = ["100", "250"]   # column order in figure


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_grayscale(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def crop(img: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
    h, w = img.shape
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop: ({x0},{y0}) → ({x1},{y1}) on ({w},{h})")
    return img[y0:y1, x0:x1]


def mse(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.mean((ref - test) ** 2))


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    m = mse(ref, test)
    if m == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / m)


def encoder_param_count(
    n_levels: int,
    n_features_per_level: int,
    log2_hashmap_size: int,
    base_resolution: int = BASE_RESOLUTION,
    max_resolution_factor: float = MAX_RESOLUTION_FACTOR,
    n_detector: list = N_DETECTOR,
) -> int:
    size = max(n_detector)
    if n_levels == 1:
        per_level_scale = 1.0
    else:
        per_level_scale = (max_resolution_factor * size / base_resolution) ** (
            1.0 / (n_levels - 1)
        )
    T = 2 ** log2_hashmap_size
    F = n_features_per_level
    total = 0
    for level in range(n_levels):
        N_l = math.floor(base_resolution * (per_level_scale ** level))
        total += min(N_l ** 3, T) * F
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ref_crop = crop(load_grayscale(PERFECT_PATH), CROP_X0, CROP_Y0, CROP_X1, CROP_Y1)

    # Collect results, keyed by (n_levels, log2_hash)
    # groups[n_levels] = list of dicts
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

        # Only include XX_4_YY
        if n_features != 4:
            continue

        n_params = encoder_param_count(n_levels, n_features, log2_hash) * 4

        entry = {
            "name": folder,
            "n_levels": n_levels,
            "log2_hash": log2_hash,
            "n_params": n_params,
        }
        for epoch_label, filename in EPOCH_FILES.items():
            img_path = os.path.join(folder_path, filename)
            if not os.path.exists(img_path):
                print(f"[warn] missing {img_path}")
                continue
            test_crop = crop(load_grayscale(img_path), CROP_X0, CROP_Y0, CROP_X1, CROP_Y1)
            entry[f"mse_{epoch_label}"] = mse(ref_crop, test_crop)
            entry[f"psnr_{epoch_label}"] = psnr(ref_crop, test_crop)

        groups.setdefault(n_levels, []).append(entry)
        print(
            f"{folder:15s}  params={n_params:>12,}  "
            + "  ".join(
                f"MSE@{k}={entry[f'mse_{k}']:.6f}  PSNR@{k}={entry[f'psnr_{k}']:.2f} dB"
                for k in EPOCH_FILES
                if f"mse_{k}" in entry
            )
        )

    if not groups:
        print("No XX_4_YY folders found.")
        return

    # Sort each group by param count
    for nl in groups:
        groups[nl].sort(key=lambda r: r["n_params"])

    sorted_levels = sorted(groups.keys())
    cmap = plt.cm.get_cmap("tab10", len(sorted_levels))
    color_map = {nl: cmap(i) for i, nl in enumerate(sorted_levels)}

    METRICS = [
        ("mse",  "MSE  ↓ better"),
        ("psnr", "PSNR (dB)  ↑ better"),
    ]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    for metric_key, ylabel in METRICS:
        for epoch_label in EPOCH_LABELS:
            key = f"{metric_key}_{epoch_label}"

            fig, ax = plt.subplots(figsize=(10, 6))

            for n_levels in sorted_levels:
                entries = groups[n_levels]
                pts = [(e["n_params"], e[key], e["name"]) for e in entries if key in e]
                if not pts:
                    continue
                xs, ys, names = zip(*pts)
                color = color_map[n_levels]
                ax.plot(
                    xs, ys,
                    "o-",
                    color=color,
                    linewidth=1.8,
                    markersize=6,
                    label=f"n_levels={n_levels}",
                )
                for x, y, name in zip(xs, ys, names):
                    ax.annotate(
                        name,
                        xy=(x, y),
                        xytext=(5, 3),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                    )

            ax.set_title(f"{metric_key.upper()} — epoch {epoch_label}  (XX_4_YY series)", fontsize=11)
            ax.set_xlabel("Encoder parameters (×4 encoders)")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
            plt.tight_layout()

            out_path = os.path.join(results_dir, f"sizediff_{metric_key}_epoch{epoch_label}.png")
            plt.savefig(out_path, dpi=150)
            print(f"Saved plot → {out_path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
