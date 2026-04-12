#!/usr/bin/env python3
"""
Compare two reconstruction images against a baseline using crops from crops.json.

Usage:
    python compare_two.py <baseline> <comp1> <comp2> [options]

Positional arguments:
    baseline    Path to the reference / ground-truth image.
    comp1       Path to the first comparison image.
    comp2       Path to the second comparison image.

Options:
    --crops     Path to crops.json  (default: crops.json next to this script)
    --labels    Two labels for the bar chart, e.g. --labels "Model A" "Model B"
    --out       Save the figure to this path instead of showing it interactively.
    --title     Overall figure title.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_grayscale(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def crop(img: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
    h, w = img.shape
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop ({x0},{y0})→({x1},{y1}) for image of shape {img.shape}")
    return img[y0:y1, x0:x1]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    mse = float(np.mean((ref - test) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    return float(ssim_fn(ref, test, data_range=1.0))


def mae(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.mean(np.abs(ref - test)))


def compute_metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    return {
        "PSNR": psnr(ref, test),
        "SSIM": ssim(ref, test),
        "MAE":  mae(ref, test),
    }


# ── crops.json ────────────────────────────────────────────────────────────────

def load_crops(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    crops = data.get("crops", [])
    if not crops:
        raise ValueError(f"No crops found in {path}")
    return crops


# ── Core logic ────────────────────────────────────────────────────────────────

def mean_metrics_over_crops(baseline: np.ndarray, test: np.ndarray, crops: list[dict]) -> dict:
    """Compute each metric for every crop, return the mean across all crops."""
    per_crop = []
    for c in crops:
        ref_c  = crop(baseline, c["x0"], c["y0"], c["x1"], c["y1"])
        test_c = crop(test,     c["x0"], c["y0"], c["x1"], c["y1"])
        per_crop.append(compute_metrics(ref_c, test_c))

    keys = list(per_crop[0].keys())
    return {k: float(np.mean([m[k] for m in per_crop])) for k in keys}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(
    metrics1: dict,
    metrics2: dict,
    labels: tuple[str, str],
    title: str,
    out: str | None,
):
    metric_names = ["PSNR", "SSIM", "MAE"]
    # Axis labels and whether higher-is-better (used only for annotation)
    ylabels = {
        "PSNR": "dB",
        "SSIM": "(0–1)",
        "MAE":  "(0–1)",
    }
    higher_better = {"PSNR": True, "SSIM": True, "MAE": False}

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    bar_width = 0.35
    x = np.array([0])
    colors = ["#4C72B0", "#DD8452"]  # muted blue / orange

    for ax, metric in zip(axes, metric_names):
        v1 = metrics1[metric]
        v2 = metrics2[metric]

        b1 = ax.bar(x - bar_width / 2, v1, bar_width, label=labels[0], color=colors[0])
        b2 = ax.bar(x + bar_width / 2, v2, bar_width, label=labels[1], color=colors[1])

        # Value labels on top of bars
        for bar in (b1, b2):
            h = bar[0].get_height()
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                h * 1.01,
                f"{h:.4f}" if metric in ("SSIM", "MAE") else f"{h:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

        direction = "↑ higher better" if higher_better[metric] else "↓ lower better"
        ax.set_title(f"{metric}  {ylabels[metric]}\n{direction}", fontsize=10)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "PSNR":
            ax.set_ylabel("dB")
        ax.legend(fontsize=8)

        # Start y-axis slightly below the smaller value so bars have visual height
        lo = min(v1, v2)
        hi = max(v1, v2)
        margin = (hi - lo) * 0.4 if hi != lo else hi * 0.05
        ax.set_ylim(max(0, lo - margin * 2), hi + margin * 3)

    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {out}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", help="Baseline / ground-truth image path")
    parser.add_argument("comp1",    help="First comparison image path")
    parser.add_argument("comp2",    help="Second comparison image path")
    parser.add_argument("--crops",  default=os.path.join(os.path.dirname(__file__), "..", "crops.json"),
                        help="Path to crops.json (default: ../crops.json relative to this script)")
    parser.add_argument("--labels", nargs=2, default=["Comp 1", "Comp 2"],
                        metavar=("LABEL1", "LABEL2"), help="Bar chart labels for comp1 and comp2")
    parser.add_argument("--out",    default=None, help="Save figure to this file path")
    parser.add_argument("--title",  default="", help="Figure title")
    args = parser.parse_args()

    # Load images
    print(f"Loading baseline : {args.baseline}")
    baseline = load_grayscale(args.baseline)
    print(f"Loading comp1    : {args.comp1}")
    img1 = load_grayscale(args.comp1)
    print(f"Loading comp2    : {args.comp2}")
    img2 = load_grayscale(args.comp2)

    # Load crops
    crops_path = os.path.abspath(args.crops)
    crops = load_crops(crops_path)
    print(f"Using crops from : {crops_path}  ({len(crops)} crops)")

    # Compute metrics
    m1 = mean_metrics_over_crops(baseline, img1, crops)
    m2 = mean_metrics_over_crops(baseline, img2, crops)

    # Print summary
    header = f"{'Metric':<8}  {args.labels[0]:>14}  {args.labels[1]:>14}"
    print(f"\n{header}")
    print("-" * len(header))
    for k in ("PSNR", "SSIM", "MAE"):
        print(f"{k:<8}  {m1[k]:>14.4f}  {m2[k]:>14.4f}")

    # Plot
    plot_comparison(m1, m2, tuple(args.labels), args.title, args.out)


if __name__ == "__main__":
    main()
