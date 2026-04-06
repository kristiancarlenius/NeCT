#!/usr/bin/env python3
"""
Compare selected sizediff and combinedcube variants at their "extra" checkpoints.

sizediff folders (skips epoch 100 and 250):
    18_4_21, 18_4_23, 22_4_20, 19_4_20, 20_4_21, 23_4_23

combinedcube folders (skips epoch 150):
    all subfolders of sizediff/combinedcube/

Both groups are plotted on the same bar chart with different colours so you can
compare across architectures.  The epoch number is shown on each bar.

Output: results/equalized2_mse.png  and  results/equalized2_psnr.png
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE            = os.path.dirname(__file__)
SIZEDIFF_DIR     = os.path.join(_HERE, "sizediff")
COMBINEDCUBE_DIR = os.path.join(SIZEDIFF_DIR, "combinedcube")
PERFECT_PATH     = os.path.join(SIZEDIFF_DIR, "perfect", "0550_1400.png")
RESULTS_DIR      = os.path.join(_HERE, "results")

SIZEDIFF_FOLDERS     = ["18_4_21", "18_4_23", "19_4_20", "23_4_23"]
COMBINEDCUBE_FOLDERS = ["23_4_23_4_128", "24_4_24_6_64"]

SIZEDIFF_SKIP     = {"0100", "0250"}
COMBINEDCUBE_SKIP = {"0150"}

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


def find_image(folder_path: str, skip_epochs: set[str]):
    """Return (path, epoch_int) for the first NNNN_1400.png not in skip_epochs."""
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith("_1400.png"):
            continue
        tag = fname[:4]
        if tag in skip_epochs or tag == "0000":
            continue
        return os.path.join(folder_path, fname), int(tag)
    return None, None


# ── Collect results ───────────────────────────────────────────────────────────

def collect() -> tuple[list[dict], list[dict]]:
    ref = get_roi(load_gray(PERFECT_PATH))

    sizediff_results   = []
    combinedcube_results = []

    for name in SIZEDIFF_FOLDERS:
        folder_path = os.path.join(SIZEDIFF_DIR, name)
        if not os.path.isdir(folder_path):
            print(f"[skip] {name}: folder not found")
            continue
        img_path, epoch = find_image(folder_path, SIZEDIFF_SKIP)
        if img_path is None:
            print(f"[skip] {name}: no qualifying image found")
            continue
        test = get_roi(load_gray(img_path))
        m, p = mse(ref, test), psnr(ref, test)
        sizediff_results.append({"name": name, "epoch": epoch, "mse": m, "psnr": p})
        print(f"[sizediff]     {name:12s}  ep={epoch:4d}  MSE={m:.6f}  PSNR={p:.2f} dB")

    for folder in COMBINEDCUBE_FOLDERS:
        folder_path = os.path.join(COMBINEDCUBE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        img_path, epoch = find_image(folder_path, COMBINEDCUBE_SKIP)
        if img_path is None:
            print(f"[skip] combinedcube/{folder}: no qualifying image found")
            continue
        test = get_roi(load_gray(img_path))
        m, p = mse(ref, test), psnr(ref, test)
        combinedcube_results.append({"name": folder, "epoch": epoch, "mse": m, "psnr": p})
        print(f"[combinedcube] {folder:25s}  ep={epoch:4d}  MSE={m:.6f}  PSNR={p:.2f} dB")

    return sizediff_results, combinedcube_results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(sizediff_results: list[dict], combinedcube_results: list[dict]):
    all_results = sizediff_results + combinedcube_results
    n_sd  = len(sizediff_results)
    n_cc  = len(combinedcube_results)
    n_all = len(all_results)

    names  = [r["name"]  for r in all_results]
    epochs = [r["epoch"] for r in all_results]

    # Two colours: one per architecture group
    colors = (
        [plt.cm.tab10(0.0)] * n_sd +   # blue for sizediff (hash grid)
        [plt.cm.tab10(0.2)] * n_cc      # orange for combinedcube
    )

    x = np.arange(n_all)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for metric, ylabel, filename, better in [
        ("mse",  "MSE  ↓ better",       "equalized2_mse.png",  "min"),
        ("psnr", "PSNR (dB)  ↑ better", "equalized2_psnr.png", "max"),
    ]:
        values = [r[metric] for r in all_results]
        best_idx = int(np.argmin(values) if better == "min" else np.argmax(values))

        fig, ax = plt.subplots(figsize=(max(8, n_all * 1.3), 5))
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)

        # Epoch label inside / on top of each bar
        for bar, val, ep in zip(bars, values, epochs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"ep {ep}\n{'%.5f' % val if metric == 'mse' else '%.2f' % val}",
                ha="center", va="bottom", fontsize=7.5,
            )

        # Gold border on winner
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        # Separator line between groups
        if n_sd > 0 and n_cc > 0:
            ax.axvline(n_sd - 0.5, color="gray", linestyle="--", linewidth=0.8)
            ax.text(n_sd / 2 - 0.5, ax.get_ylim()[1], "hash grid",
                    ha="center", va="top", fontsize=8, color=plt.cm.tab10(0.0))
            ax.text(n_sd + n_cc / 2 - 0.5, ax.get_ylim()[1], "combinedcube",
                    ha="center", va="top", fontsize=8, color=plt.cm.tab10(0.2))

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric.upper()} — equalized checkpoint (hash grid vs combinedcube)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
        plt.close(fig)


def main():
    sizediff_results, combinedcube_results = collect()
    plot(sizediff_results, combinedcube_results)


if __name__ == "__main__":
    main()
