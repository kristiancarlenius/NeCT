#!/usr/bin/env python3
"""
Static continuous scan quality sweep.

Baseline: sizediff/static_continous/ac1_0250_1400.png
Comparison images: all other ac{N}_*.png files in the same directory.

Filename convention: ac{ac_num}_{tag_a}_{tag_b}.png
  ac_num — number of angular cycles (x-axis)
  tag_a  — run/epoch identifier
  tag_b  — variant identifier

Files are grouped by (tag_a, tag_b) into series and plotted as separate
lines so that both variants across ac values are visible.

Crop regions are loaded from crop_static.json.
Output: results/static_continous_psnr.png
"""

import json
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
SCAN_DIR   = os.path.join(BASE_DIR, "sizediff", "static_continous")
BASELINE   = os.path.join(SCAN_DIR, "ac1_0250_1400.png")
CROPS_FILE = os.path.join(BASE_DIR, "crop_static.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Crop regions ──────────────────────────────────────────────────────────────
with open(CROPS_FILE) as _f:
    _crops_data = json.load(_f)
PANEL_CROPS: list[dict] = _crops_data["crops"]
print(f"Loaded {len(PANEL_CROPS)} crop region(s) from crop_static.json")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def compute_psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """PSNR averaged over all crop regions."""
    psnr_vals = []
    for c in PANEL_CROPS:
        y0 = max(0, c["y0"]); y1 = min(ref.shape[0], c["y1"])
        x0 = max(0, c["x0"]); x1 = min(ref.shape[1], c["x1"])
        rp = ref[y0:y1, x0:x1]
        tp = test[y0:y1, x0:x1]
        m = float(np.mean((rp - tp) ** 2))
        psnr_vals.append(10.0 * math.log10(1.0 / m) if m > 0 else float("inf"))
    return float(np.mean(psnr_vals))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ref = load_gray(BASELINE)
    print(f"Baseline: {os.path.basename(BASELINE)}, shape={ref.shape}")

    # series_data[variant_label] = list of (ac_num, psnr)
    series_data: dict[str, list[tuple[int, float]]] = defaultdict(list)

    pattern = re.compile(r"^ac(\d+)_(.+)\.png$")

    for fname in sorted(os.listdir(SCAN_DIR)):
        m = pattern.match(fname)
        if m is None:
            continue
        ac_num = int(m.group(1))
        variant = m.group(2)           # e.g. "1025_0360" or "3700_0100"

        # Skip the baseline file itself
        full_path = os.path.join(SCAN_DIR, fname)
        if os.path.abspath(full_path) == os.path.abspath(BASELINE):
            continue

        psnr_val = compute_psnr(ref, load_gray(full_path))
        series_data[variant].append((ac_num, psnr_val))
        print(f"  ac={ac_num}  variant={variant:12s}  PSNR={psnr_val:.2f} dB")

    if not series_data:
        print("No comparison images found.")
        return

    # Sort each series by ac number
    for key in series_data:
        series_data[key].sort(key=lambda t: t[0])

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for idx, (variant, points) in enumerate(sorted(series_data.items())):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color  = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(xs, ys, marker=marker, linestyle="-", color=color,
                linewidth=2, markersize=8, label=variant)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=8, color=color)

    # Collect all unique ac values across all series for x-ticks
    all_ac = sorted({p[0] for pts in series_data.values() for p in pts})
    ax.set_xticks(all_ac)
    ax.set_xlabel("Number of angular cycles (ac)")
    ax.set_ylabel("PSNR (dB)  ↑ better")
    ax.set_title("Static continuous scan: reconstruction quality vs angular cycles\n"
                 f"(baseline: {os.path.basename(BASELINE)})")
    ax.legend(title="variant", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "static_continous_psnr.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
