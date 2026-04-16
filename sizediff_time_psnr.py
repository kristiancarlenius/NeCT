#!/usr/bin/env python3
"""
Plot PSNR vs training time for sizediff XX_4_YY models.
Images are read from each folder's time/ subfolder.
Files are sorted by name; each successive image is 6 hours apart.
X-axis is therefore: index * 6  (hours).

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

TIME_INTERVAL_HOURS = 6   # hours between successive images in time/

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


def mse(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.mean((ref - test) ** 2))


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    m = mse(ref, test)
    return float("inf") if m == 0 else 10.0 * math.log10(1.0 / m)


def is_image(name: str) -> bool:
    return bool(re.match(r"^\d+_\d+\.png$", name))


# ── Data collection ───────────────────────────────────────────────────────────

def collect() -> dict[str, dict]:
    """Returns {folder_name: {"n_levels": int, "n_features": int, "log2_hash": int,
                               "hours": [float, ...], "psnrs": [float, ...]}}"""
    ref_roi = get_roi(load_grayscale(PERFECT_PATH))
    data: dict[str, dict] = {}

    for folder in sorted(os.listdir(SIZEDIFF_DIR)):
        parts = folder.split("_")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            continue
        n_levels, n_features, log2_hash = int(parts[0]), int(parts[1]), int(parts[2])
        if n_features != 4:
            continue

        time_dir = os.path.join(SIZEDIFF_DIR, folder, "time")
        if not os.path.isdir(time_dir):
            continue

        image_files = sorted(f for f in os.listdir(time_dir) if is_image(f))
        if not image_files:
            continue

        hours, psnrs, mses = [], [], []
        for idx, fname in enumerate(image_files):
            path = os.path.join(time_dir, fname)
            try:
                roi = get_roi(load_grayscale(path))
                m = mse(ref_roi, roi)
                p = float("inf") if m == 0 else 10.0 * math.log10(1.0 / m)
                hours.append((idx + 1) * TIME_INTERVAL_HOURS)
                psnrs.append(p)
                mses.append(m)
            except Exception as e:
                print(f"[warn] {path}: {e}")

        if hours:
            data[folder] = {
                "n_levels": n_levels,
                "n_features": n_features,
                "log2_hash": log2_hash,
                "hours": hours,
                "psnrs": psnrs,
                "mses": mses,
            }
            print(f"{folder:15s}  {len(hours)} snapshots  "
                  f"up to {max(hours):.0f}h  "
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
        ax.plot(d["hours"], d[metric], "o-", color=cmap(i),
                linewidth=1.8, markersize=5, label=name)

    ax.set_xlabel("Training time (hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Tick every 6 hours; label every 24 hours
    max_h = max((max(d["hours"]) for d in entries.values()), default=0)
    ax.set_xticks(range(0, int(max_h) + TIME_INTERVAL_HOURS, TIME_INTERVAL_HOURS))
    ax.xaxis.set_tick_params(which="major", labelbottom=True)
    if max_h > 48:
        for tick, val in zip(ax.xaxis.get_major_ticks(),
                              range(0, int(max_h) + TIME_INTERVAL_HOURS, TIME_INTERVAL_HOURS)):
            tick.label1.set_visible(val % 24 == 0)

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
        print("No data found — make sure time/ subfolders contain images.")
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
                f"{metric_short.upper()} vs Training Time — {desc}",
                os.path.join(RESULTS_DIR, f"time_{metric_short}_{tag}.png"),
            )


if __name__ == "__main__":
    main()
