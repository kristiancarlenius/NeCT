#!/usr/bin/env python3
"""SSIM vs epoch/time plots for sizediff experiments, plus VRAM efficiency."""

from pathlib import Path
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
PERFECT_EPOCH = SIZEDIFF / "perfect" / "epoch" / "0525_1400.png"
PERFECT_TIME = SIZEDIFF / "perfect" / "time" / "0525_1400.png"
RESULTS = ROOT / "results" / "ssim_arc"
RESULTS.mkdir(exist_ok=True)

MODELS = [
    "quadcubes",
    "mixedcubes",
    "combinedcube",
    "quadcubes_large_spatial",
    "quadcubes_large_temporal",
]

MODEL_COLORS = {
    "quadcubes": "#1f77b4",
    "mixedcubes": "#ff7f0e",
    "combinedcube": "#2ca02c",
    "quadcubes_large_spatial": "#9467bd",
    "quadcubes_large_temporal": "#d62728",
}

TARGET_18H = 18.0  # hours


# ── helpers ──────────────────────────────────────────────────────────────────

def load_crops():
    with open(CROPS_FILE) as f:
        return json.load(f)["crops"]


def load_image_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def compute_ssim(ref, cand, crops):
    vals = []
    for c in crops:
        r_crop = ref[c["y0"]:c["y1"], c["x0"]:c["x1"]]
        c_crop = cand[c["y0"]:c["y1"], c["x0"]:c["x1"]]
        vals.append(float(ssim_fn(r_crop, c_crop, data_range=255.0)))
    return float(np.mean(vals))


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

    epoch_data = []
    if epoch_dir.exists():
        for img_path in sorted(epoch_dir.glob("*_1400.png")):
            ep = epoch_from_name(img_path.name)
            if ep is None:
                continue
            cand = load_image_gray(img_path)
            epoch_data.append((ep, compute_ssim(ref_epoch_arr, cand, crops)))

    time_data = []
    if time_dir.exists() and epoch_time_map:
        for img_path in sorted(time_dir.glob("*_1400.png")):
            ep = epoch_from_name(img_path.name)
            if ep is None:
                continue
            t_sec = nearest_lookup(epoch_time_map, ep)
            cand = load_image_gray(img_path)
            time_data.append((t_sec / 3600.0, compute_ssim(ref_time_arr, cand, crops)))

    return {
        "epoch_data": sorted(epoch_data),
        "time_data": sorted(time_data),
        "vram_gb": vram_gb,
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
    """Trim every epoch/time series to the shortest one across all models/configs."""
    epoch_lens = [
        len(d["epoch_data"])
        for configs in all_data.values()
        for d in configs.values()
        if d["epoch_data"]
    ]
    time_lens = [
        len(d["time_data"])
        for configs in all_data.values()
        for d in configs.values()
        if d["time_data"]
    ]
    min_epoch = min(epoch_lens) if epoch_lens else 0
    min_time  = min(time_lens)  if time_lens  else 0
    for configs in all_data.values():
        for d in configs.values():
            d["epoch_data"] = d["epoch_data"][:min_epoch]
            d["time_data"]  = d["time_data"][:min_time]
    print(f"  Capped to {min_epoch} epoch images, {min_time} time images")
    return all_data


# ── plotting ─────────────────────────────────────────────────────────────────

def _finish(fig, path, title):
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_model_epoch(model, model_data):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for i, (label, data) in enumerate(sorted(model_data.items())):
        pts = data["epoch_data"]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=label, color=cmap(i % 10), linewidth=1.5, marker="o",
                markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.legend(fontsize=8, title="Config", loc="lower right")
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / f"{model}_epoch_ssim.png", f"{model} — SSIM vs Epoch")


def plot_model_time(model, model_data):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    plotted = 0
    for i, (label, data) in enumerate(sorted(model_data.items())):
        pts = data["time_data"]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=label, color=cmap(i % 10), linewidth=1.5, marker="o",
                markersize=3)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.axvline(TARGET_18H, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="18 h")
    ax.set_xlabel("Wall-clock time (hours)")
    ax.set_ylabel("SSIM")
    ax.legend(fontsize=8, title="Config", loc="lower right")
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / f"{model}_time_ssim.png", f"{model} — SSIM vs Time")


def plot_combined_epoch(all_data):
    fig, ax = plt.subplots(figsize=(11, 6))
    for model, model_data in all_data.items():
        color = MODEL_COLORS.get(model, "gray")
        for i, (label, data) in enumerate(sorted(model_data.items())):
            pts = data["epoch_data"]
            if not pts:
                continue
            xs, ys = zip(*pts)
            is_first = i == 0
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.75,
                    label=model if is_first else None,
                    linestyle=["-", "--", ":", "-."][i % 4])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / "combined_epoch_ssim.png", "All Models — SSIM vs Epoch")


def plot_combined_time(all_data):
    fig, ax = plt.subplots(figsize=(11, 6))
    has_data = False
    for model, model_data in all_data.items():
        color = MODEL_COLORS.get(model, "gray")
        for i, (label, data) in enumerate(sorted(model_data.items())):
            pts = data["time_data"]
            if not pts:
                continue
            xs, ys = zip(*pts)
            is_first = i == 0
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.75,
                    label=model if is_first else None,
                    linestyle=["-", "--", ":", "-."][i % 4])
            has_data = True
    if not has_data:
        plt.close(fig)
        return
    ax.axvline(TARGET_18H, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="18 h")
    ax.set_xlabel("Wall-clock time (hours)")
    ax.set_ylabel("SSIM")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / "combined_time_ssim.png", "All Models — SSIM vs Time")


def plot_vram_efficiency(all_data):
    fig, ax = plt.subplots(figsize=(9, 6))

    plotted = []
    for model, model_data in all_data.items():
        color = MODEL_COLORS.get(model, "gray")
        for label, data in sorted(model_data.items()):
            vram = data["vram_gb"]
            ssim_18h = None
            if data["time_data"]:
                # pick the point nearest to 18 h
                nearest = min(data["time_data"], key=lambda p: abs(p[0] - TARGET_18H))
                ssim_18h = nearest[1]
            if vram is None or ssim_18h is None:
                continue
            plotted.append((vram, ssim_18h, model, label, color))

    if not plotted:
        plt.close(fig)
        print("  No VRAM efficiency data available.")
        return

    # scatter + annotate
    seen_models = set()
    for vram, ssim, model, label, color in plotted:
        first = model not in seen_models
        ax.scatter(vram, ssim, color=color, s=70, zorder=3,
                   label=model if first else None)
        seen_models.add(model)
        ax.annotate(label, (vram, ssim), textcoords="offset points",
                    xytext=(5, 3), fontsize=6.5, color=color)

    ax.set_xlabel("Peak Reserved VRAM (GB)")
    ax.set_ylabel(f"SSIM at ~{TARGET_18H:.0f} h")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / "vram_efficiency.png",
            f"VRAM Efficiency — SSIM at {TARGET_18H:.0f} h vs Peak Reserved VRAM")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading reference images and crops ...")
    crops = load_crops()
    ref_epoch = load_image_gray(PERFECT_EPOCH)
    ref_time = load_image_gray(PERFECT_TIME)

    print("Collecting experiment data ...")
    all_data = collect_all(crops, ref_epoch, ref_time)
    cap_to_minimum(all_data)

    print("Generating plots ...")

    for model, model_data in all_data.items():
        plot_model_epoch(model, model_data)
        plot_model_time(model, model_data)

    plot_combined_epoch(all_data)
    plot_combined_time(all_data)
    plot_vram_efficiency(all_data)

    print("Done. Results in", RESULTS)


if __name__ == "__main__":
    main()
