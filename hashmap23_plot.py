#!/usr/bin/env python3
"""SSIM/PSNR/MAE vs epoch/time plots for hashmap sweep (23_4_XX folders)."""

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
PERFECT_EPOCH = SIZEDIFF / "perfect" / "0525_1400.png"
PERFECT_TIME = SIZEDIFF / "perfect" / "0525_1400.png"
RESULTS = ROOT / "results" / "hashmap23_plots"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET_18H = 18.0  # hours

METRICS = {
    "ssim": "SSIM",
    "psnr": "PSNR (dB)",
    "mae":  "MAE",
}

FOLDER_RE = re.compile(r"^23_4_(\d+)$")


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── data loading ──────────────────────────────────────────────────────────────

def build_fallback_timing(timing_maps):
    """Average epoch→time mappings from all runs that have timing data."""
    if not timing_maps:
        return {}
    all_epochs = set()
    for m in timing_maps.values():
        all_epochs.update(m.keys())
    fallback = {}
    for ep in all_epochs:
        vals = [m[ep] for m in timing_maps.values() if ep in m]
        fallback[ep] = float(np.mean(vals))
    return fallback


def load_config_data(config_dir, ref_epoch_arr, ref_time_arr, crops, epoch_time_map):
    vram_file = config_dir / "vram.txt"
    epoch_dir = config_dir / "epoch"
    time_dir = config_dir / "time"

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
    """Returns {label: data_dict} for every 23_4_XX folder, sorted by hashmap size."""
    dirs = sorted(
        (d for d in SIZEDIFF.iterdir() if d.is_dir() and FOLDER_RE.match(d.name)),
        key=lambda d: int(FOLDER_RE.match(d.name).group(1)),
    )

    # Build epoch→time maps from runs that recorded timing; average as fallback
    timing_maps = {}
    for d in dirs:
        lf = d / "epoch_losses.txt"
        if lf.exists():
            m = parse_epoch_losses(lf)
            if m:
                timing_maps[d.name] = m
    fallback = build_fallback_timing(timing_maps)
    if fallback:
        print(f"  Timing data from: {list(timing_maps.keys())} (used as fallback for others)")

    all_data = {}
    for d in dirs:
        label = d.name
        timing = timing_maps.get(label, fallback)
        print(f"  Loading {label} ...", flush=True)
        all_data[label] = load_config_data(d, ref_epoch_arr, ref_time_arr, crops, timing)
    return all_data


def cap_to_minimum(all_data):
    epoch_lens = [len(d["epoch_ssim"]) for d in all_data.values() if d["epoch_ssim"]]
    time_lens  = [len(d["time_ssim"])  for d in all_data.values() if d["time_ssim"]]
    min_epoch = min(epoch_lens) if epoch_lens else 0
    min_time  = min(time_lens)  if time_lens  else 0
    for d in all_data.values():
        for key in ("epoch_ssim", "epoch_psnr", "epoch_mae"):
            d[key] = d[key][:min_epoch]
        for key in ("time_ssim", "time_psnr", "time_mae"):
            d[key] = d[key][:min_time]
    print(f"  Capped to {min_epoch} epoch images, {min_time} time images")
    return all_data


# ── plotting ──────────────────────────────────────────────────────────────────

def _finish(fig, path, title):
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_epoch(all_data, metric):
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    key = f"epoch_{metric}"
    for i, (label, data) in enumerate(all_data.items()):
        pts = data[key]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=label, color=cmap(i % 10), linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(METRICS[metric])
    ax.legend(fontsize=8, title="23_4_hashmap", loc="lower right")
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / f"hashmap23_epoch_{metric}.png",
            f"hashmap sweep (n_levels=23, 4 layers) — {METRICS[metric]} vs Epoch")


def plot_time(all_data, metric):
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    key = f"time_{metric}"
    plotted = 0
    for i, (label, data) in enumerate(all_data.items()):
        pts = data[key]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=label, color=cmap(i % 10), linewidth=1.5, marker="o", markersize=3)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.axvline(TARGET_18H, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="24 h")
    ax.set_xlabel("Wall-clock time (hours)")
    ax.set_ylabel(METRICS[metric])
    ax.legend(fontsize=8, title="23_4_hashmap", loc="lower right")
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / f"hashmap23_time_{metric}.png",
            f"hashmap sweep (n_levels=23, 4 layers) — {METRICS[metric]} vs Time")


def plot_vram_efficiency(all_data, metric):
    fig, ax = plt.subplots(figsize=(9, 6))
    key = f"time_{metric}"
    cmap = plt.get_cmap("tab10")
    plotted = []
    for i, (label, data) in enumerate(all_data.items()):
        vram = data["vram_gb"]
        val_24h = None
        if data[key]:
            nearest = min(data[key], key=lambda p: abs(p[0] - TARGET_18H))
            val_24h = nearest[1]
        if vram is None or val_24h is None:
            continue
        plotted.append((vram, val_24h, label, cmap(i % 10)))

    if not plotted:
        plt.close(fig)
        print(f"  No VRAM efficiency data available for {metric}.")
        return

    for vram, val, label, color in plotted:
        ax.scatter(vram, val, color=color, s=70, zorder=3, label=label)
        ax.annotate(label, (vram, val), textcoords="offset points",
                    xytext=(5, 3), fontsize=6.5, color=color)

    ax.set_xlabel("Peak Reserved VRAM (GB)")
    ax.set_ylabel(f"{METRICS[metric]} at ~{TARGET_18H:.0f} h")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _finish(fig, RESULTS / f"hashmap23_vram_{metric}.png",
            f"VRAM Efficiency — {METRICS[metric]} at {TARGET_18H:.0f} h vs Peak Reserved VRAM")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading reference images and crops ...")
    crops = load_crops()
    ref_epoch = load_image_gray(PERFECT_EPOCH)
    ref_time = load_image_gray(PERFECT_TIME)

    print("Collecting experiment data ...")
    all_data = collect_all(crops, ref_epoch, ref_time)
    if not all_data:
        print("No 23_4_XX folders found in", SIZEDIFF)
        return
    cap_to_minimum(all_data)

    print("Generating plots ...")
    for metric in METRICS:
        plot_epoch(all_data, metric)
        plot_time(all_data, metric)
        plot_vram_efficiency(all_data, metric)

    print("Done. Results in", RESULTS)


if __name__ == "__main__":
    main()
