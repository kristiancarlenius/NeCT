#!/usr/bin/env python3
"""
Plot PSNR vs epoch for sizediff XX_4_YY models.
Images are read from each folder's epoch/ subfolder.
Filename format: EEEE_1400.png where EEEE is the epoch number.

Produces three plots:
  1. 23_4_XX  — fixed n_levels=23, varying log2_hash
  2. XX_4_23  — varying n_levels, fixed log2_hash=23
  3. All XX_4_YY models combined
"""

import json
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
SIZEDIFF_DIR = os.path.join(os.path.dirname(__file__), "sizediff")
PERFECT_PATH = os.path.join(SIZEDIFF_DIR, "perfect", "1300_1400.png")
CROPS_FILE   = os.path.join(os.path.dirname(__file__), "crops.json")

with open(CROPS_FILE) as f:
    PANEL_CROPS: list[dict] = json.load(f)["crops"]
print(f"Loaded {len(PANEL_CROPS)} panel crops from crops.json")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_grayscale(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def get_roi(img: np.ndarray) -> np.ndarray:
    pieces = []
    h, w = img.shape
    for c in PANEL_CROPS:
        x0, x1 = max(0, c["x0"]), min(w, c["x1"])
        y0, y1 = max(0, c["y0"]), min(h, c["y1"])
        pieces.append(img[y0:y1, x0:x1].ravel())
    return np.concatenate(pieces)


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    m = float(np.mean((ref - test) ** 2))
    return float("inf") if m == 0 else 10.0 * math.log10(1.0 / m)


def epoch_from_filename(name: str) -> int | None:
    m = re.match(r"^(\d+)_\d+\.png$", name)
    return int(m.group(1)) if m else None


# ── Data collection ───────────────────────────────────────────────────────────

def mse(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.mean((ref - test) ** 2))


def collect() -> dict[str, dict]:
    """Returns {folder_name: {"n_levels": int, "n_features": int, "log2_hash": int,
                               "epochs": [int, ...], "psnrs": [float, ...], "mses": [float, ...]}}"""
    ref_roi = get_roi(load_grayscale(PERFECT_PATH))
    data: dict[str, dict] = {}

    for folder in sorted(os.listdir(SIZEDIFF_DIR)):
        parts = folder.split("_")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            continue
        n_levels, n_features, log2_hash = int(parts[0]), int(parts[1]), int(parts[2])
        if n_features != 4:
            continue

        epoch_dir = os.path.join(SIZEDIFF_DIR, folder, "epoch")
        if not os.path.isdir(epoch_dir):
            continue

        epochs, psnrs, mses = [], [], []
        for fname in sorted(os.listdir(epoch_dir)):
            ep = epoch_from_filename(fname)
            if ep is None:
                continue
            path = os.path.join(epoch_dir, fname)
            try:
                roi = get_roi(load_grayscale(path))
                m = mse(ref_roi, roi)
                p = float("inf") if m == 0 else 10.0 * math.log10(1.0 / m)
                epochs.append(ep)
                psnrs.append(p)
                mses.append(m)
            except Exception as e:
                print(f"[warn] {path}: {e}")

        if epochs:
            data[folder] = {
                "n_levels": n_levels,
                "n_features": n_features,
                "log2_hash": log2_hash,
                "epochs": epochs,
                "psnrs": psnrs,
                "mses": mses,
            }
            print(f"{folder:15s}  {len(epochs)} checkpoints  "
                  f"PSNR [{min(psnrs):.2f}, {max(psnrs):.2f}] dB  "
                  f"MSE [{min(mses):.6f}, {max(mses):.6f}]")

    return data


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plot(entries: dict[str, dict], metric: str,
              ylabel: str, title: str, out_path: str) -> None:
    if not entries:
        print(f"[skip] no data for: {title}")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.get_cmap("tab10", max(len(entries), 1))

    for i, (name, d) in enumerate(sorted(entries.items())):
        ax.plot(d["epochs"], d[metric], "o-", color=cmap(i),
                linewidth=1.8, markersize=5, label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


METRICS = [
    ("psnrs", "PSNR (dB)  ↑ better", "psnr"),
    ("mses",  "MSE  ↓ better",        "mse"),
]


def main() -> None:
    data = collect()
    if not data:
        print("No data found — make sure epoch/ subfolders contain EEEE_1400.png images.")
        return

    fixed_levels_23 = {k: v for k, v in data.items() if v["n_levels"] == 23}
    fixed_hash_23   = {k: v for k, v in data.items() if v["log2_hash"] == 23}

    subsets = [
        (fixed_levels_23, "23_4_XX", "23_4_XX  (n_levels=23, n_features=4, varying log2_hash)"),
        (fixed_hash_23,   "XX_4_23", "XX_4_23  (varying n_levels, n_features=4, log2_hash=23)"),
        (data,            "all",     "all XX_4_YY models"),
    ]

    for metric_key, ylabel, metric_short in METRICS:
        for entries, tag, desc in subsets:
            make_plot(
                entries,
                metric_key,
                ylabel,
                f"{metric_short.upper()} vs Epoch — {desc}",
                os.path.join(RESULTS_DIR, f"epoch_{metric_short}_{tag}.png"),
            )


if __name__ == "__main__":
    main()
