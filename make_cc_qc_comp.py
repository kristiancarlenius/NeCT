#!/usr/bin/env python3
"""
Bar chart comparing CombinedCubes (24_4_24_6_128) vs QuadCubes (23_4_23_4_128)
at epoch 150, evaluated on the crop region in crop_img_comparison.json.

Output: docs/images/cc_qc_comp.png
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path(__file__).parent

REFERENCE   = ROOT / "sizediff" / "perfect" / "0525_1400.png"
COMBINEDCUBES = ROOT / "sizediff" / "combinedcube" / "24_4_24_6_128" / "0150_1400.png"
QUADCUBES   = ROOT / "sizediff" / "quadcubes" / "23_4_23_4_128" / "epoch" / "0150_1400.png"
CROPS_FILE  = ROOT / "crops.json"
OUT_PATH    = ROOT / "docs" / "images" / "cc_qc_comp.png"


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def crop(img: np.ndarray, c: dict) -> np.ndarray:
    return img[c["y0"]:c["y1"], c["x0"]:c["x1"]]


def compute_metrics(ref: np.ndarray, cand: np.ndarray) -> dict:
    # normalise candidate to reference intensity range
    if ref.std() > 0 and cand.std() > 0:
        cand_n = (cand - cand.mean()) / cand.std() * ref.std() + ref.mean()
    else:
        cand_n = cand

    mse = float(np.mean((ref - cand_n) ** 2))
    psnr = 10.0 * math.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
    ssim = float(ssim_fn(ref, cand, data_range=255.0))
    mae  = float(np.mean(np.abs(ref - cand_n)))
    return {"PSNR": psnr, "SSIM": ssim, "MAE": mae}


def main():
    with open(CROPS_FILE) as f:
        crops = json.load(f)["crops"]

    ref_full  = load_gray(REFERENCE)
    cc_full   = load_gray(COMBINEDCUBES)
    qc_full   = load_gray(QUADCUBES)

    # average metrics over all crop regions
    cc_metrics_list, qc_metrics_list = [], []
    for c in crops:
        ref_c = crop(ref_full, c)
        cc_metrics_list.append(compute_metrics(ref_c, crop(cc_full, c)))
        qc_metrics_list.append(compute_metrics(ref_c, crop(qc_full, c)))

    def mean_metrics(lst):
        keys = lst[0].keys()
        return {k: float(np.mean([m[k] for m in lst])) for k in keys}

    cc = mean_metrics(cc_metrics_list)
    qc = mean_metrics(qc_metrics_list)

    print(f"{'Metric':<8}  {'CombinedCubes':>15}  {'QuadCubes':>12}")
    print("-" * 40)
    for k in ("PSNR", "SSIM", "MAE"):
        print(f"{k:<8}  {cc[k]:>15.4f}  {qc[k]:>12.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    metrics   = ["PSNR", "SSIM", "MAE"]
    ylabels   = {"PSNR": "dB", "SSIM": "(0–1)", "MAE": "intensity units"}
    higher_better = {"PSNR": True, "SSIM": True, "MAE": False}

    CC_COLOR = "#2ca02c"   # green  – CombinedCubes
    QC_COLOR = "#1f77b4"   # blue   – QuadCubes

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("CombinedCubes vs QuadCubes — epoch 150", fontsize=13, fontweight="bold")

    bar_w = 0.32
    x = np.array([0])

    for ax, metric in zip(axes, metrics):
        v_cc = cc[metric]
        v_qc = qc[metric]

        b_cc = ax.bar(x - bar_w / 2, v_cc, bar_w, label="CombinedCubes\n24_4_24_6_128", color=CC_COLOR)
        b_qc = ax.bar(x + bar_w / 2, v_qc, bar_w, label="QuadCubes\n23_4_23_4_128",    color=QC_COLOR)

        fmt = ".4f" if metric in ("SSIM", "MAE") else ".2f"
        for bar, val in [(b_cc, v_cc), (b_qc, v_qc)]:
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                bar[0].get_height() * 1.01,
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=9,
            )

        direction = "↑ better" if higher_better[metric] else "↓ better"
        ax.set_title(f"{metric}  [{ylabels[metric]}]\n{direction}", fontsize=10)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="lower right" if higher_better[metric] else "upper right")

        lo, hi = min(v_cc, v_qc), max(v_cc, v_qc)
        margin = (hi - lo) * 0.5 if hi != lo else hi * 0.05
        ax.set_ylim(max(0, lo - margin * 2), hi + margin * 3)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
