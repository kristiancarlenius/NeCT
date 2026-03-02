#!/usr/bin/env python3
"""
Compare sizediff reconstructions against the perfect reference image and plot
MSE vs. encoder parameter count for each model.

Folder names encode encoder hyperparameters as: n_levels_n_features_per_level_log2_hashmap_size
Two plots are generated — one per epoch checkpoint (100 and 250 epochs).

Geometry used (same as training runs):
  nDetector: [1880, 1496]
  base_resolution: 16
  max_resolution_factor: 2

Parameter count formula (multi-resolution hash grid, 3-D inputs):
  per_level_scale = (max_resolution_factor * max(nDetector) / base_resolution) ^ (1 / (n_levels - 1))
  For each level l: N_l = floor(base_resolution * per_level_scale^l)
  params_l = min(N_l^3, 2^log2_hashmap_size) * n_features_per_level
  total_params = sum over all levels
"""

import math  # still used by encoder_param_count
import os

import matplotlib.pyplot as plt
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
N_DETECTOR = [1880, 1496]   # geometry: nDetector

# ── Epoch checkpoints present in each subfolder ──────────────────────────────
EPOCH_FILES = {
    "100":  "0100_1400.png",
    "250":  "0250_1400.png",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_grayscale(path: str) -> np.ndarray:
    """Load image as float32 in [0, 1]."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def crop(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
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
    """
    Exact parameter count for a multi-resolution hash-grid encoder with 3-D input.

    Mirrors the NeCT HashEncoderConfig.per_level_scale property:
      per_level_scale = (max_resolution_factor * max(nDetector) / base_resolution)
                        ^ (1 / (n_levels - 1))
    """
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
    # Load and crop the reference once
    ref_crop = crop(load_grayscale(PERFECT_PATH), CROP_X0, CROP_Y0, CROP_X1, CROP_Y1)

    results: list[dict] = []

    for folder in sorted(os.listdir(SIZEDIFF_DIR)):
        if folder == "perfect":
            continue
        folder_path = os.path.join(SIZEDIFF_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        # Parse n_levels_n_features_log2hash
        parts = folder.split("_")
        if len(parts) != 3:
            print(f"[skip] {folder}: expected 3 underscore-separated parts")
            continue
        try:
            n_levels, n_features, log2_hash = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            print(f"[skip] {folder}: non-integer parts")
            continue

        n_params = encoder_param_count(n_levels, n_features, log2_hash) * 4

        entry = {"name": folder, "n_params": n_params}
        for epoch_label, filename in EPOCH_FILES.items():
            img_path = os.path.join(folder_path, filename)
            if not os.path.exists(img_path):
                print(f"[warn] missing {img_path}")
                continue
            test_crop = crop(load_grayscale(img_path), CROP_X0, CROP_Y0, CROP_X1, CROP_Y1)
            entry[f"mse_{epoch_label}"] = mse(ref_crop, test_crop)
            entry[f"psnr_{epoch_label}"] = psnr(ref_crop, test_crop)

        results.append(entry)
        print(
            f"{folder:15s}  params={n_params:>12,}  "
            + "  ".join(
                f"MSE@{k}={entry[f'mse_{k}']:.6f}  PSNR@{k}={entry[f'psnr_{k}']:.2f} dB"
                for k in EPOCH_FILES
                if f"mse_{k}" in entry
            )
        )

    results.sort(key=lambda r: r["n_params"])

    # ── One plot per metric × epoch ──────────────────────────────────────────
    METRICS = [
        ("mse",  "MSE  ↓ better",    ".6f"),
        ("psnr", "PSNR (dB)  ↑ better", ".2f"),
    ]

    for metric_key, ylabel, fmt in METRICS:
        for epoch_label in EPOCH_FILES:
            key = f"{metric_key}_{epoch_label}"
            pts = [(r["name"], r["n_params"], r[key]) for r in results if key in r]
            if not pts:
                continue

            names, params, vals = zip(*pts)

            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(params, vals, "o-", linewidth=1.5, markersize=7)

            for name, x, y in zip(names, params, vals):
                ax.annotate(
                    name,
                    xy=(x, y),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

            ax.set_xlabel("Encoder parameters (×4 encoders)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Reconstruction quality vs. encoder size — {epoch_label} epochs")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            out_path = os.path.join(
                os.path.dirname(__file__),
                f"sizediff_{metric_key}_epoch{epoch_label}.png",
            )
            plt.savefig(out_path, dpi=150)
            print(f"Saved plot → {out_path}")
            plt.show()


if __name__ == "__main__":
    main()
