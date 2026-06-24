#!/usr/bin/env python3
"""SSIM/PSNR/MAE vs epoch/time plots for sizediff experiments, plus VRAM efficiency."""

from pathlib import Path
import copy
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path(__file__).parent
SIZEDIFF = ROOT / "sizediff"
CROPS_FILE = ROOT / "crops.json"
PERFECT_EPOCH = SIZEDIFF / "perfect" / "0525_1400.png"
PERFECT_TIME = SIZEDIFF / "perfect" / "0525_1400.png"
RESULTS = ROOT / "results" / "test_plots"
RESULTS.mkdir(exist_ok=True)

MODELS = [
    "quadcubes",
    "mixedcubes",
    "combinedcubes",
    "quadcubes_large_spatial",
    "quadcubes_large_temporal",
]

MODEL_COLORS = {
    "quadcubes": "#1f77b4",
    "mixedcubes": "#ff7f0e",
    "combinedcubes": "#2ca02c",
    "quadcubes_large_spatial": "#9467bd",
    "quadcubes_large_temporal": "#d62728",
}

TARGET_24H = 24.0  # hours

GPU_LINES = [
    ("A100 40 GB", 40),
    ("5090 32 GB", 32),
    ("4090 24 GB", 24),
    ("5080 16 GB", 16),
    ("5070 12 GB", 12),
    ("3070  8 GB",  8),
]

QUADCUBES_FOCUSED_CONFIG = "23_4_23"  # config prefix for focused subset

METRICS = {
    "ssim": "SSIM",
    "psnr": "PSNR (dB)",
    "mae":  "MAE",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_crops():
    with open(CROPS_FILE) as f:
        return json.load(f)["crops"]


def load_image_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)



def compute_all_metrics(ref, cand, crops):
    ssim_vals, mae_vals, mse_vals = [], [], []
    for c in crops:
        r_crop = ref[c["y0"]:c["y1"], c["x0"]:c["x1"]]
        c_crop = cand[c["y0"]:c["y1"], c["x0"]:c["x1"]]

        # normalize candidate to reference intensity so MAE/PSNR measure structure, not brightness offset
        r_std = r_crop.std()
        if r_std > 0 and c_crop.std() > 0:
            c_norm = (c_crop - c_crop.mean()) / c_crop.std() * r_std + r_crop.mean()
        else:
            c_norm = c_crop

        ssim_vals.append(float(ssim_fn(r_crop, c_crop, data_range=255.0)))
        mae_vals.append(float(np.mean(np.abs(r_crop - c_norm))))
        mse_vals.append(float(np.mean((r_crop - c_norm) ** 2)))

    ssim = float(np.mean(ssim_vals))
    mae  = float(np.mean(mae_vals))
    mse  = float(np.mean(mse_vals))
    psnr = float(10.0 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")
    return ssim, psnr, mae


def parse_epoch_losses(path):
    """Return {epoch: time_seconds} or {} if no time column."""
    result = {}
    has_time = False
    with open(path) as f:
        for line in f:
            em = re.search(r"epoch=(\d+)", line)
            tm = re.search(r"time=([\d.]+)s", line)
            if em:
                ep = int(em.group(1))
                if tm:
                    result[ep] = float(tm.group(1))
                    has_time = True
    return result if has_time else {}


def parse_vram_gb(path):
    if not path or not path.exists():
        return None
    with open(path) as f:
        for line in f:
            if "Peak reserved" in line:
                m = re.search(r"([\d.]+)\s*GB", line)
                if m:
                    return float(m.group(1))
    return None


def epoch_from_name(name):
    m = re.match(r"(\d+)_", name)
    return int(m.group(1)) if m else None


def nearest_lookup(mapping, epoch):
    if epoch in mapping:
        return mapping[epoch]
    nearest = min(mapping.keys(), key=lambda k: abs(k - epoch))
    return mapping[nearest]


# ── data loading ─────────────────────────────────────────────────────────────

def load_param_data(param_dir, ref_epoch_arr, ref_time_arr, crops):
    losses_file = param_dir / "epoch_losses.txt"
    vram_file = param_dir / "vram.txt"
    epoch_dir = param_dir / "epoch"
    time_dir = param_dir / "time"

    epoch_time_map = parse_epoch_losses(losses_file) if losses_file.exists() else {}
    vram_gb = parse_vram_gb(vram_file if vram_file.exists() else None)

    epoch_ssim, epoch_psnr, epoch_mae = [], [], []
    if epoch_dir.exists():
        for img_path in sorted(epoch_dir.glob("*_1400.png")):
            ep = epoch_from_name(img_path.name)
            if ep is None:
                continue
            cand = load_image_gray(img_path)
            s, p, m = compute_all_metrics(ref_epoch_arr, cand, crops)
            epoch_ssim.append((ep, s))
            epoch_psnr.append((ep, p))
            epoch_mae.append((ep, m))

    time_ssim, time_psnr, time_mae = [], [], []
    if time_dir.exists() and epoch_time_map:
        for img_path in sorted(time_dir.glob("*_1400.png")):
            ep = epoch_from_name(img_path.name)
            if ep is None:
                continue
            t_h = nearest_lookup(epoch_time_map, ep) / 3600.0
            cand = load_image_gray(img_path)
            s, p, m = compute_all_metrics(ref_time_arr, cand, crops)
            time_ssim.append((t_h, s))
            time_psnr.append((t_h, p))
            time_mae.append((t_h, m))

    return {
        "epoch_ssim": sorted(epoch_ssim),
        "epoch_psnr": sorted(epoch_psnr),
        "epoch_mae":  sorted(epoch_mae),
        "time_ssim":  sorted(time_ssim),
        "time_psnr":  sorted(time_psnr),
        "time_mae":   sorted(time_mae),
        "vram_gb":    vram_gb,
    }


def collect_all(crops, ref_epoch_arr, ref_time_arr):
    """Returns {model: {param_label: data_dict}}."""
    all_data = {}
    for model in MODELS:
        model_dir = SIZEDIFF / model
        if not model_dir.exists():
            continue
        model_data = {}
        for param_dir in sorted(model_dir.iterdir()):
            if not param_dir.is_dir():
                continue
            label = param_dir.name
            print(f"  Loading {model}/{label} ...", flush=True)
            model_data[label] = load_param_data(param_dir, ref_epoch_arr, ref_time_arr, crops)
        if model_data:
            all_data[model] = model_data
    return all_data


# ── capping ──────────────────────────────────────────────────────────────────

def cap_to_minimum(all_data):
    """Per-model: cap every config to the fewest images any config in that model has."""
    for model, configs in all_data.items():
        epoch_lens = [len(d["epoch_ssim"]) for d in configs.values() if d["epoch_ssim"]]
        time_lens  = [len(d["time_ssim"])  for d in configs.values() if d["time_ssim"]]

        min_epoch = min(epoch_lens) if epoch_lens else 0
        min_time  = min(time_lens)  if time_lens  else 0

        for d in configs.values():
            for key in ("epoch_ssim", "epoch_psnr", "epoch_mae"):
                d[key] = d[key][:min_epoch]
            for key in ("time_ssim", "time_psnr", "time_mae"):
                d[key] = d[key][:min_time]

        print(f"  {model}: capped to {min_epoch} epoch images, {min_time} time images")
    return all_data


# ── plotting ─────────────────────────────────────────────────────────────────

def _finish(fig, path, title):
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_model_epoch(model, model_data, metric):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    key = f"epoch_{metric}"
    for i, (label, data) in enumerate(sorted(model_data.items())):
        pts = data[key]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=label, color=cmap(i % 10), linewidth=1.5, marker="o",
                markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(METRICS[metric])
    ax.legend(fontsize=8, title="Config", loc="lower right")
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / f"{model}_epoch_{metric}.png",
            f"{model} — {METRICS[metric]} vs Epoch")


def plot_model_time(model, model_data, metric):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    key = f"time_{metric}"
    plotted = 0
    for i, (label, data) in enumerate(sorted(model_data.items())):
        pts = data[key]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=label, color=cmap(i % 10), linewidth=1.5, marker="o",
                markersize=3)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.axvline(TARGET_24H, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="24 h")
    ax.set_xlabel("Wall-clock time (hours)")
    ax.set_ylabel(METRICS[metric])
    ax.legend(fontsize=8, title="Config", loc="lower right")
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / f"{model}_time_{metric}.png",
            f"{model} — {METRICS[metric]} vs Time")


def plot_combined_epoch(all_data, metric, filename=None, title=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    key = f"epoch_{metric}"
    for model, model_data in all_data.items():
        color = MODEL_COLORS.get(model, "gray")
        for i, (label, data) in enumerate(sorted(model_data.items())):
            pts = data[key]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.75,
                    label=f"{model}/{label}" if i > 0 else model,
                    linestyle=["-", "--", ":", "-."][i % 4])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(METRICS[metric])
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / (filename or f"combined_epoch_{metric}.png"),
            title or f"All Models — {METRICS[metric]} vs Epoch")


def plot_combined_time(all_data, metric, filename=None, title=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    key = f"time_{metric}"
    has_data = False
    for model, model_data in all_data.items():
        color = MODEL_COLORS.get(model, "gray")
        for i, (label, data) in enumerate(sorted(model_data.items())):
            pts = data[key]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.75,
                    label=f"{model}/{label}" if i > 0 else model,
                    linestyle=["-", "--", ":", "-."][i % 4])
            has_data = True
    if not has_data:
        plt.close(fig)
        return
    ax.axvline(TARGET_24H, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="24 h")
    ax.set_xlabel("Wall-clock time (hours)")
    ax.set_ylabel(METRICS[metric])
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / (filename or f"combined_time_{metric}.png"),
            title or f"All Models — {METRICS[metric]} vs Time")


def plot_vram_efficiency(all_data, metric):
    fig, ax = plt.subplots(figsize=(9, 6))
    key = f"time_{metric}"
    plotted = []
    for model, model_data in all_data.items():
        color = MODEL_COLORS.get(model, "gray")
        for label, data in sorted(model_data.items()):
            vram = data["vram_gb"]
            val_24h = None
            if data[key]:
                nearest = min(data[key], key=lambda p: abs(p[0] - TARGET_24H))
                val_24h = nearest[1]
            if vram is None or val_24h is None:
                continue
            plotted.append((vram, val_24h, model, label, color))

    if not plotted:
        plt.close(fig)
        print(f"  No VRAM efficiency data available for {metric}.")
        return

    seen_models = set()
    for vram, val, model, label, color in plotted:
        first = model not in seen_models
        ax.scatter(vram, val, color=color, s=70, zorder=3,
                   label=model if first else None)
        seen_models.add(model)
        ax.annotate(label, (vram, val), textcoords="offset points",
                    xytext=(5, 3), fontsize=6.5, color=color)

    x_max = max(ax.get_xlim()[1], max(gb for _, gb in GPU_LINES) * 1.08)
    for gpu_name, gpu_gb in GPU_LINES:
        ax.axvline(gpu_gb, color="dimgray", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.text(gpu_gb, ax.get_ylim()[0], gpu_name,
                ha="center", va="bottom", fontsize=9, color="black", rotation=90)
    ax.set_xlim(left=0, right=x_max)

    ax.set_xlabel("Peak Reserved VRAM (GB)")
    ax.set_ylabel(f"{METRICS[metric]} at ~{TARGET_24H:.0f} h")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / f"vram_efficiency_{metric}.png",
            f"VRAM Efficiency — {METRICS[metric]} at {TARGET_24H:.0f} h vs Peak Reserved VRAM")


# ── top-2 diverging comparison ───────────────────────────────────────────────

def plot_top2_comparison(all_data):
    """Diverging bar chart: top-2 configs per architecture, all 3 metrics.

    Layout
    ------
    Y-axis  : 5 architectures × 3 metrics (PSNR / SSIM / MAE), grouped.
    Left    : top-1 config bars (extend left / negative x).
    Right   : top-2 config bars (extend right / positive x).
    Values are normalised per-metric to [0, 1] across all plotted data so
    that PSNR, SSIM, and MAE share one axis.  MAE is flipped (lower → longer
    bar) so longer always means better.  Raw values are printed on each bar.
    """
    METRIC_KEYS   = ["PSNR",   "SSIM",   "MAE"]
    METRIC_SERIES = ["time_psnr", "time_ssim", "time_mae"]
    METRIC_FLIP   = [False, False, True]   # MAE: lower is better → flip
    METRIC_FMT    = [".2f", ".4f", ".2f"]
    METRIC_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # ── select top-2 configs per architecture by last-time PSNR ──────────────
    top2 = {}
    for model in MODELS:
        if model not in all_data:
            continue
        entries = []
        for label, data in all_data[model].items():
            if not data["time_psnr"]:
                continue
            metrics = {
                "PSNR": data["time_psnr"][-1][1],
                "SSIM": data["time_ssim"][-1][1] if data["time_ssim"] else None,
                "MAE":  data["time_mae"][-1][1]  if data["time_mae"]  else None,
            }
            entries.append((metrics["PSNR"], label, metrics))
        entries.sort(reverse=True)
        if entries:
            top2[model] = [(e[1], e[2]) for e in entries[:2]]

    if not top2:
        print("  No time data for top-2 comparison — skipping.")
        return

    # ── normalise each metric across all selected values ─────────────────────
    all_vals = {m: [] for m in METRIC_KEYS}
    for pairs in top2.values():
        for _, metrics in pairs:
            for m in METRIC_KEYS:
                if metrics.get(m) is not None:
                    all_vals[m].append(metrics[m])

    def norm(val, metric):
        vals = all_vals[metric]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return 0.5
        n = (val - lo) / (hi - lo)
        return (1 - n) if METRIC_FLIP[METRIC_KEYS.index(metric)] else n

    # ── layout geometry ───────────────────────────────────────────────────────
    arch_list    = [m for m in MODELS if m in top2]
    n_arch       = len(arch_list)
    n_metrics    = len(METRIC_KEYS)
    bar_h        = 0.18          # height of each bar
    metric_gap   = 0.02          # gap between metrics within a group
    arch_spacing = n_metrics * (bar_h + metric_gap) + 0.35  # gap between arches

    fig, ax = plt.subplots(figsize=(14, max(5, n_arch * arch_spacing * 1.6)))

    y_arch_centers = []
    for ai, model in enumerate(arch_list):
        arch_center = ai * arch_spacing
        y_arch_centers.append(arch_center)
        pairs = top2[model]

        for mi, (mkey, mcolor, mfmt) in enumerate(zip(METRIC_KEYS, METRIC_COLORS, METRIC_FMT)):
            y = arch_center + (mi - (n_metrics - 1) / 2) * (bar_h + metric_gap)

            # top-1: extends LEFT (negative)
            if len(pairs) >= 1:
                val1 = pairs[0][1].get(mkey)
                if val1 is not None:
                    w = -norm(val1, mkey)
                    ax.barh(y, w, bar_h, color=mcolor, alpha=0.85, left=0)
                    ax.text(w - 0.01, y, f"{val1:{mfmt}}",
                            ha="right", va="center", fontsize=7, color="white",
                            fontweight="bold")

            # top-2: extends RIGHT (positive)
            if len(pairs) >= 2:
                val2 = pairs[1][1].get(mkey)
                if val2 is not None:
                    w = norm(val2, mkey)
                    ax.barh(y, w, bar_h, color=mcolor, alpha=0.50, left=0)
                    ax.text(w + 0.01, y, f"{val2:{mfmt}}",
                            ha="left", va="center", fontsize=7)

    # ── central spine ─────────────────────────────────────────────────────────
    ax.axvline(0, color="black", linewidth=1.2)

    # ── architecture labels ───────────────────────────────────────────────────
    ax.set_yticks(y_arch_centers)
    ax.set_yticklabels(arch_list, fontsize=10)

    # model name annotations above/below each arch group
    x_pad = 0.04
    for ai, model in enumerate(arch_list):
        pairs = top2[model]
        yc = y_arch_centers[ai]
        half = (n_metrics * (bar_h + metric_gap)) / 2
        if len(pairs) >= 1:
            ax.text(-x_pad, yc + half + 0.05, pairs[0][0],
                    ha="right", va="bottom", fontsize=7.5, color="#1a1a8c",
                    style="italic")
        if len(pairs) >= 2:
            ax.text(x_pad, yc - half - 0.05, pairs[1][0],
                    ha="left", va="top", fontsize=7.5, color="#8c1a1a",
                    style="italic")

    # ── metric legend ─────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=f"{m} {'(flip)' if f else ''}")
                      for m, c, f in zip(METRIC_KEYS, METRIC_COLORS, METRIC_FLIP)]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right",
              title="Metric  (normalised 0–1 per metric,\nlonger = better)")

    # direction labels
    ax.text(-0.02, ax.get_ylim()[1], "← top-1 model", ha="right", va="top",
            fontsize=9, color="#1a1a8c")
    ax.text(0.02, ax.get_ylim()[1], "top-2 model →", ha="left", va="top",
            fontsize=9, color="#8c1a1a")

    ax.set_xlabel("Normalised score  (0 = worst, 1 = best within metric)")
    ax.set_xlim(-1.15, 1.15)
    ax.set_title("Top-2 configs per architecture — PSNR / SSIM / MAE at last time point",
                 fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    _finish(fig, RESULTS / "top2_comparison.png", "")


# ── summary printing ─────────────────────────────────────────────────────────

def print_summary(all_data: dict) -> None:
    """Print per-config and cross-model statistics for thesis reference."""
    BASELINE_MODEL  = "quadcubes"
    BASELINE_CONFIG = "23_4_23_4_128"

    def last(series):
        return series[-1][1] if series else None

    def at_time(series, target):
        if not series:
            return None
        return min(series, key=lambda p: abs(p[0] - target))[1]

    def _row(model, label, data):
        ep_p = last(data["epoch_psnr"])
        ep_s = last(data["epoch_ssim"])
        ep_m = last(data["epoch_mae"])
        t_p  = last(data["time_psnr"])
        t_s  = last(data["time_ssim"])
        t_m  = last(data["time_mae"])
        c_p  = at_time(data["time_psnr"], TARGET_24H)
        c_s  = at_time(data["time_ssim"], TARGET_24H)
        c_m  = at_time(data["time_mae"],  TARGET_24H)
        vram = data["vram_gb"]
        ep_n = data["epoch_psnr"][-1][0] if data["epoch_psnr"] else "?"
        t_n  = f"{data['time_psnr'][-1][0]:.0f}h" if data["time_psnr"] else "?"
        psnr_str  = f"{ep_p:.2f}" if ep_p is not None else "  --  "
        ssim_str  = f"{ep_s:.4f}" if ep_s is not None else "  -- "
        mae_str   = f"{ep_m:.1f}"  if ep_m is not None else "  --  "
        tpsnr_str = f"{t_p:.2f}"  if t_p  is not None else "  --  "
        tssim_str = f"{t_s:.4f}"  if t_s  is not None else "  -- "
        tmae_str  = f"{t_m:.1f}"  if t_m  is not None else "  --  "
        cpsnr_str = f"{c_p:.2f}"  if c_p  is not None else "  --  "
        cssim_str = f"{c_s:.4f}"  if c_s  is not None else "  -- "
        cmae_str  = f"{c_m:.1f}"  if c_m  is not None else "  --  "
        vram_str  = f"{vram:.1f}" if vram is not None else " -- "
        return (f"  {model:30s} {label:22s}  "
                f"VRAM={vram_str:>5}GB  "
                f"epoch{ep_n}: PSNR={psnr_str} SSIM={ssim_str} MAE={mae_str}  "
                f"@{TARGET_24H:.0f}h: PSNR={cpsnr_str} SSIM={cssim_str} MAE={cmae_str}  "
                f"@{t_n}: PSNR={tpsnr_str} SSIM={tssim_str} MAE={tmae_str}")

    # ── Per-model tables ──────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("SUMMARY — per config, last capped epoch and last capped time snapshot")
    print("=" * 110)
    for model in MODELS:
        if model not in all_data:
            continue
        print(f"\n  [{model}]")
        for label, data in sorted(all_data[model].items()):
            marker = " ◀ baseline" if (model == BASELINE_MODEL and label == BASELINE_CONFIG) else ""
            print(_row(model, label, data) + marker)

    # ── Cross-model comparison at 24 h ───────────────────────────────────────
    print("\n" + "=" * 120)
    print(f"CROSS-MODEL — all configs sorted by PSNR at {TARGET_24H:.0f} h")
    print("=" * 120)
    rows = []
    for model, model_data in all_data.items():
        for label, data in model_data.items():
            c_p = at_time(data["time_psnr"], TARGET_24H)
            c_s = at_time(data["time_ssim"], TARGET_24H)
            c_m = at_time(data["time_mae"],  TARGET_24H)
            t_h = data["time_psnr"][-1][0] if data["time_psnr"] else None
            if c_p is not None:
                rows.append((c_p, model, label, data["vram_gb"], c_p, c_s, c_m, t_h))
    rows.sort(reverse=True)
    baseline_psnr = None
    bl = all_data.get(BASELINE_MODEL, {}).get(BASELINE_CONFIG)
    if bl:
        baseline_psnr = at_time(bl["time_psnr"], TARGET_24H)
    for _, model, label, vram, c_p, c_s, c_m, t_h in rows:
        delta = f"({c_p - baseline_psnr:+.2f} dB)" if baseline_psnr is not None else ""
        marker = " ◀ baseline" if (model == BASELINE_MODEL and label == BASELINE_CONFIG) else ""
        vram_s = f"{vram:.1f}" if vram is not None else "--"
        t_h_s  = f"{t_h:.0f}h" if t_h is not None else "?"
        print(f"  {model:30s} {label:22s}  VRAM={vram_s:>5}GB  "
              f"@{TARGET_24H:.0f}h: PSNR={c_p:.2f} SSIM={c_s:.4f} MAE={c_m:.1f}  "
              f"{delta}{marker}")

    # ── Baseline vs best alternative ─────────────────────────────────────────
    if baseline_psnr is not None:
        print(f"\n  Baseline PSNR at {TARGET_24H:.0f} h: {baseline_psnr:.2f} dB")
        best_non_base = next(
            (r for r in rows if not (r[1] == BASELINE_MODEL and r[2] == BASELINE_CONFIG)), None)
        if best_non_base:
            _, m, l, v, tp, ts, tm, th = best_non_base
            print(f"  Best alternative:  {m}/{l}  VRAM={v:.1f}GB  "
                  f"PSNR={tp:.2f} ({tp - baseline_psnr:+.2f} dB)  "
                  f"SSIM={ts:.4f}  MAE={tm:.1f}")


# ── focused subset ───────────────────────────────────────────────────────────

def build_focused_data(all_data):
    """quadcubes configs matching 23_4_23 + all large-spatial/temporal configs.

    Deep-copies the series so the focused cap below does not affect all_data.
    """
    focused = {}
    for model, model_data in all_data.items():
        if model == "quadcubes":
            filtered = {label: copy.deepcopy(data)
                        for label, data in model_data.items()
                        if label.startswith(QUADCUBES_FOCUSED_CONFIG)}
            if filtered:
                focused[model] = filtered
        elif model in ("quadcubes_large_spatial", "quadcubes_large_temporal"):
            focused[model] = {label: copy.deepcopy(data)
                              for label, data in model_data.items()}
    return focused


def cap_focused_data(focused_data):
    """Cap all configs across every model in focused_data to the same minimum count.

    Needed because cap_to_minimum operates per model family, so large_temporal
    may have a longer series than the quadcubes subset after that cap.
    """
    all_configs = [d for md in focused_data.values() for d in md.values()]
    time_lens  = [len(d["time_ssim"])  for d in all_configs if d["time_ssim"]]
    epoch_lens = [len(d["epoch_ssim"]) for d in all_configs if d["epoch_ssim"]]
    min_time  = min(time_lens)  if time_lens  else 0
    min_epoch = min(epoch_lens) if epoch_lens else 0
    for d in all_configs:
        for key in ("time_ssim", "time_psnr", "time_mae"):
            d[key] = d[key][:min_time]
        for key in ("epoch_ssim", "epoch_psnr", "epoch_mae"):
            d[key] = d[key][:min_epoch]
    print(f"  focused: cross-model cap → {min_epoch} epoch images, {min_time} time images")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading reference images and crops ...")
    crops = load_crops()
    ref_epoch = load_image_gray(PERFECT_EPOCH)
    ref_time = load_image_gray(PERFECT_TIME)

    print("Collecting experiment data ...")
    all_data = collect_all(crops, ref_epoch, ref_time)
    cap_to_minimum(all_data)

    print_summary(all_data)

    print("\nGenerating plots ...")

    for metric in METRICS:
        for model, model_data in all_data.items():
            plot_model_epoch(model, model_data, metric)
            plot_model_time(model, model_data, metric)

        plot_combined_epoch(all_data, metric)
        plot_combined_time(all_data, metric)
        plot_vram_efficiency(all_data, metric)

    plot_top2_comparison(all_data)

    focused_data = build_focused_data(all_data)
    if focused_data:
        print("\nGenerating focused QuadCubes plots ...")
        cap_focused_data(focused_data)
        for metric in METRICS:
            plot_combined_epoch(
                focused_data, metric,
                filename=f"focused_epoch_{metric}.png",
                title=f"QuadCubes (23_4_23) vs Large Variants — {METRICS[metric]} vs Epoch",
            )
            plot_combined_time(
                focused_data, metric,
                filename=f"focused_time_{metric}.png",
                title=f"QuadCubes (23_4_23) vs Large Variants — {METRICS[metric]} vs Time",
            )

    print("Done. Results in", RESULTS)


if __name__ == "__main__":
    main()
