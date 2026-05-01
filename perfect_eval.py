#!/usr/bin/env python3
"""
Evaluate which perfect/ reference image produces the most self-consistent
metric rankings across experiments:
  1. Epoch monotonicity  — later epochs should have higher PSNR/SSIM, lower MSE
  2. Param ordering      — larger configs (same model) should have higher PSNR/SSIM, lower MSE

Runs for PSNR, MSE, and SSIM independently.
"""

from pathlib import Path
import re
import json
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim_fn
from PIL import Image

ROOT     = Path(__file__).parent
SIZEDIFF = ROOT / "sizediff"
CROPS_FILE = ROOT / "crops.json"
RESULTS  = ROOT / "results" / "perfect_eval"
RESULTS.mkdir(parents=True, exist_ok=True)

MODELS = [
    "quadcubes",
    "mixedcubes",
    "combinedcube",
    "quadcubes_large_spatial",
    "quadcubes_large_temporal",
]

# (metric_name, higher_is_better)
METRICS = [
    ("psnr", True),
    ("mse",  False),
    ("ssim", True),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_crops():
    with open(CROPS_FILE) as f:
        return json.load(f)["crops"]


def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def compute_metrics(ref, cand, crops):
    """Returns (psnr, mse, ssim) averaged across crops."""
    psnrs, mses, ssims = [], [], []
    for c in crops:
        r = ref[c["y0"]:c["y1"], c["x0"]:c["x1"]]
        k = cand[c["y0"]:c["y1"], c["x0"]:c["x1"]]
        # normalize candidate to reference intensity so MSE/PSNR measure structure, not brightness offset
        r_std = r.std()
        if r_std > 0 and k.std() > 0:
            k_norm = (k - k.mean()) / k.std() * r_std + r.mean()
        else:
            k_norm = k
        mse = float(np.mean((r - k_norm) ** 2))
        mses.append(mse)
        psnrs.append(10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else 100.0)
        ssims.append(float(ssim_fn(r, k, data_range=255.0)))
    return float(np.mean(psnrs)), float(np.mean(mses)), float(np.mean(ssims))


def config_key(label):
    return tuple(int(x) for x in re.findall(r"\d+", label))


def epoch_from_name(name):
    m = re.match(r"(\d+)_", name)
    return int(m.group(1)) if m else None


# ── data loading ──────────────────────────────────────────────────────────────

def load_epoch_metrics(ref_arr, crops):
    """
    Returns {model: {config_label: [(epoch, psnr, mse, ssim), ...]}}
    """
    data = {}
    for model in MODELS:
        model_dir = SIZEDIFF / model
        if not model_dir.exists():
            continue
        model_data = {}
        for param_dir in sorted(model_dir.iterdir()):
            if not param_dir.is_dir():
                continue
            epoch_dir = param_dir / "epoch"
            if not epoch_dir.exists():
                continue
            pts = []
            for img_path in sorted(epoch_dir.glob("*_1400.png")):
                ep = epoch_from_name(img_path.name)
                if ep is None:
                    continue
                cand = load_gray(img_path)
                psnr, mse, ssim = compute_metrics(ref_arr, cand, crops)
                pts.append((ep, psnr, mse, ssim))
            if pts:
                model_data[param_dir.name] = sorted(pts)
        if model_data:
            data[model] = model_data
    return data


# ── scoring ───────────────────────────────────────────────────────────────────

METRIC_IDX = {"psnr": 1, "mse": 2, "ssim": 3}  # index into (ep, psnr, mse, ssim)

def score_epoch_monotonicity(data, metric, higher_is_better):
    """Spearman ρ between epoch and metric value (negated for lower-is-better)."""
    rhos = []
    idx = METRIC_IDX[metric]
    for model, configs in data.items():
        for label, pts in configs.items():
            if len(pts) < 3:
                continue
            epochs  = [p[0]   for p in pts]
            values  = [p[idx] for p in pts]
            rho, _ = spearmanr(epochs, values)
            rhos.append(rho if higher_is_better else -rho)
    return float(np.mean(rhos)) if rhos else 0.0


def score_param_ordering(data, metric, higher_is_better):
    """Fraction of (large, small) config pairs where large config is better."""
    correct = total = 0
    idx = METRIC_IDX[metric]
    for model, configs in data.items():
        if len(configs) < 2:
            continue
        labels = sorted(configs.keys(), key=config_key)

        ep_vals = {}
        for label, pts in configs.items():
            for pt in pts:
                ep_vals.setdefault(pt[0], {})[label] = pt[idx]

        for ep, lv in ep_vals.items():
            available = [l for l in labels if l in lv]
            if len(available) < 2:
                continue
            for a, b in itertools.combinations(available, 2):
                # a < b by config_key → expect b is "better"
                if higher_is_better:
                    if lv[b] >= lv[a]:
                        correct += 1
                else:
                    if lv[b] <= lv[a]:
                        correct += 1
                total += 1

    return correct / total if total else 0.0


# ── plotting ──────────────────────────────────────────────────────────────────

METRIC_COLORS = {
    "psnr": {"mono": "#1f77b4", "order": "#aec7e8"},
    "mse":  {"mono": "#d62728", "order": "#f5a9a9"},
    "ssim": {"mono": "#2ca02c", "order": "#98df8a"},
}

def make_chart(all_results, sorted_names):
    """
    all_results: {ref_name: {metric: (mono, order, combined)}}
    sorted_names: ordered list of ref names (by overall rank)
    """
    n = len(sorted_names)
    n_metrics = len(METRICS)
    # two bars per metric (mono + order), grouped by reference image
    group_w = 0.8
    bar_w = group_w / (n_metrics * 2)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    titles = ["Epoch monotonicity (Spearman ρ)", "Param ordering (fraction correct)"]

    for ax_i, (ax, score_idx, title) in enumerate(zip(axes, [0, 1], titles)):
        for m_i, (metric, _) in enumerate(METRICS):
            offset = (m_i - (n_metrics - 1) / 2) * bar_w * 2
            vals = [all_results[name][metric][score_idx] for name in sorted_names]
            ax.bar(x + offset, vals, bar_w * 1.8,
                   label=metric.upper(),
                   color=METRIC_COLORS[metric]["mono" if score_idx == 0 else "order"])
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9)
        if ax_i == 0:
            ax.set_ylabel("Score (higher = better)")

    fig.suptitle("Perfect reference quality — PSNR / MSE / SSIM", fontsize=12)
    fig.tight_layout()
    out = RESULTS / "perfect_eval.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    crops = load_crops()
    candidates = sorted((SIZEDIFF / "perfect").glob("*.png"))
    print(f"Evaluating {len(candidates)} reference candidates ...\n")

    all_results = {}  # {ref_name: {metric: (mono, order, combined)}}

    for ref_path in candidates:
        print(f"  {ref_path.name} ...", flush=True)
        ref_arr = load_gray(ref_path)
        data    = load_epoch_metrics(ref_arr, crops)

        scores = {}
        for metric, higher in METRICS:
            mono  = score_epoch_monotonicity(data, metric, higher)
            order = score_param_ordering(data, metric, higher)
            scores[metric] = (mono, order, (mono + order) / 2)
            print(f"    {metric:>4}: mono={mono:.4f}  order={order:.4f}  combined={(mono+order)/2:.4f}")
        all_results[ref_path.name] = scores

    # rank by average combined score across all metrics
    def overall(name):
        return np.mean([all_results[name][m][2] for m, _ in METRICS])

    sorted_names = sorted(all_results.keys(), key=overall, reverse=True)

    # print table
    col_w = 10
    header = f"\n{'Reference':<22}" + "".join(
        f"  {m.upper():>{col_w*2+2}}" for m, _ in METRICS
    ) + f"  {'Overall':>{col_w}}"
    print(header)
    subheader = " " * 22 + "".join(
        f"  {'mono':>{col_w}} {'order':>{col_w}}" for _ in METRICS
    )
    print(subheader)
    print("-" * (22 + (col_w * 2 + 4) * len(METRICS) + col_w + 2))

    for name in sorted_names:
        row = f"{name:<22}"
        for metric, _ in METRICS:
            mono, order, _ = all_results[name][metric]
            row += f"  {mono:>{col_w}.4f} {order:>{col_w}.4f}"
        row += f"  {overall(name):>{col_w}.4f}"
        if name == sorted_names[0]:
            row += "  ◀ best"
        print(row)

    out = make_chart(all_results, sorted_names)
    print(f"\nChart saved to {out}")
    print(f"Best reference: {sorted_names[0]}")


if __name__ == "__main__":
    main()
