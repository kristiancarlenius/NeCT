#!/usr/bin/env python3
"""Produce 6 PNGs comparing the top-2 configs from each model family.

All-models set  (5 families × top-2 configs):
  results/all_psnr_epoch.png
  results/all_psnr_time.png
  results/all_vram.png

Quad-only set  (quadcubes / large_spatial / large_temporal):
  results/quad_psnr_epoch.png
  results/quad_psnr_time.png
  results/quad_vram.png
"""

from pathlib import Path
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT       = Path(__file__).parent
SIZEDIFF   = ROOT / "sizediff"
CROPS_FILE = ROOT / "crops.json"
PERFECT    = SIZEDIFF / "perfect" / "0525_1400.png"
RESULTS    = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ALL_MODELS = [
    "combinedcubes",
    "mixedcubes",
    "quadcubes",
    "quadcubes_large_spatial",
    "quadcubes_large_temporal",
]

QUAD_MODELS = [
    "quadcubes",
    "quadcubes_large_spatial",
    "quadcubes_large_temporal",
]

MODEL_DISPLAY = {
    "combinedcubes":             "Combinedcubes",
    "mixedcubes":               "MixedCubes",
    "quadcubes":                "QuadCubes",
    "quadcubes_large_spatial":  "Large Spatial",
    "quadcubes_large_temporal": "Large Temporal",
}

# Manually chosen configs for the all-models plots.
# None  → fall back to auto top-2.
# Families absent here are excluded from the all-model plots entirely.
MANUAL_ALL_CONFIGS = {
    "quadcubes":               ["23_4_23_4_128"],
    "quadcubes_large_spatial": None,   # keep auto top-2
    "mixedcubes":              ["18_4_22_4_64", "18_4_23_6_128", "24_4_24_4_128", "22_4_25_4_128"],
    "combinedcubes":            ["18_4_23_6_64", "18_4_24_4_128", "24_4_24_6_128", "22_4_25_4_128"],
    # quadcubes_large_temporal excluded
}

# One color per family; solid = rank-1 config, dashed = rank-2 config
FAMILY_COLORS = {
    "combinedcubes":             "#2ca02c",
    "mixedcubes":               "#ff7f0e",
    "quadcubes":                "#1f77b4",
    "quadcubes_large_spatial":  "#9467bd",
    "quadcubes_large_temporal": "#d62728",
}

GPU_LINES = [
    ("A100 40 GB", 40),
    ("5090 32 GB", 32),
    ("4090 24 GB", 24),
    ("5080 16 GB", 16),
    ("5070 12 GB", 12),
    ("3070  8 GB",  8),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_crops():
    with open(CROPS_FILE) as f:
        return json.load(f)["crops"]


def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def compute_psnr(ref, cand, crops):
    mse_vals = []
    for c in crops:
        r = ref[c["y0"]:c["y1"], c["x0"]:c["x1"]]
        q = cand[c["y0"]:c["y1"], c["x0"]:c["x1"]]
        if r.std() > 0 and q.std() > 0:
            q = (q - q.mean()) / q.std() * r.std() + r.mean()
        mse_vals.append(float(np.mean((r - q) ** 2)))
    mse = float(np.mean(mse_vals))
    return 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")


def parse_epoch_times(path):
    result = {}
    with open(path) as f:
        for line in f:
            em = re.search(r"epoch=(\d+)", line)
            tm = re.search(r"time=([\d.]+)s", line)
            if em and tm:
                result[int(em.group(1))] = float(tm.group(1))
    return result


def nearest_time(epoch_time_map, epoch):
    if epoch in epoch_time_map:
        return epoch_time_map[epoch]
    return epoch_time_map[min(epoch_time_map, key=lambda k: abs(k - epoch))]


def parse_vram_gb(path):
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


# ── data loading ──────────────────────────────────────────────────────────────

def load_config(cfg_dir, ref, crops):
    losses_file = cfg_dir / "epoch_losses.txt"
    vram_file   = cfg_dir / "vram.txt"
    epoch_dir   = cfg_dir / "epoch"
    time_dir    = cfg_dir / "time"

    epoch_time_map = parse_epoch_times(losses_file) if losses_file.exists() else {}
    vram_gb = parse_vram_gb(vram_file) if vram_file.exists() else None

    epoch_psnr, time_psnr = [], []

    if epoch_dir.exists():
        for img_path in sorted(epoch_dir.glob("*_1400.png")):
            ep = epoch_from_name(img_path.name)
            if ep is None:
                continue
            epoch_psnr.append((ep, compute_psnr(ref, load_gray(img_path), crops)))

    if time_dir.exists() and epoch_time_map:
        for img_path in sorted(time_dir.glob("*_1400.png")):
            ep = epoch_from_name(img_path.name)
            if ep is None:
                continue
            t_h = nearest_time(epoch_time_map, ep) / 3600.0
            time_psnr.append((t_h, compute_psnr(ref, load_gray(img_path), crops)))

    return {
        "epoch_psnr": sorted(epoch_psnr),
        "time_psnr":  sorted(time_psnr),
        "vram_gb":    vram_gb,
    }


def collect_top2(crops, ref):
    """Return {model: [(cfg_label, data), (cfg_label, data)]} — top-2 by final time PSNR.

    Requires epoch_psnr, time_psnr, and vram_gb so every selected config
    appears in all three plots.
    """
    result = {}
    for model in ALL_MODELS:
        model_dir = SIZEDIFF / model
        if not model_dir.exists():
            continue
        scored = []
        for cfg_dir in sorted(model_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            data = load_config(cfg_dir, ref, crops)
            if not data["epoch_psnr"] or not data["time_psnr"] or data["vram_gb"] is None:
                continue
            score = data["time_psnr"][-1][1]
            scored.append((score, cfg_dir.name, data))
        scored.sort(reverse=True)
        if scored:
            result[model] = [(name, data) for _, name, data in scored[:2]]
            for rank, (_, name, _) in enumerate(scored[:2], 1):
                print(f"  {MODEL_DISPLAY[model]:22s} #{rank}  {name}  "
                      f"PSNR={scored[rank-1][0]:.2f} dB")
    return result


def collect_manual_all(crops, ref, top2_auto):
    """Build the config dict for the all-models plots using MANUAL_ALL_CONFIGS.

    For each family:
      - If the value is None, use the auto top-2 from top2_auto.
      - Otherwise load the explicitly named config directories.
    Families not present in MANUAL_ALL_CONFIGS are excluded.
    """
    result = {}
    for model, cfg_names in MANUAL_ALL_CONFIGS.items():
        if cfg_names is None:
            if model in top2_auto:
                result[model] = top2_auto[model]
            continue
        model_dir = SIZEDIFF / model
        if not model_dir.exists():
            print(f"  [skip] {model}: directory not found")
            continue
        entries = []
        for name in cfg_names:
            cfg_dir = model_dir / name
            if not cfg_dir.exists():
                print(f"  [skip] {model}/{name}: not found")
                continue
            data = load_config(cfg_dir, ref, crops)
            if not data["epoch_psnr"] and not data["time_psnr"]:
                print(f"  [skip] {model}/{name}: no data")
                continue
            final_psnr = data["time_psnr"][-1][1] if data["time_psnr"] else 0.0
            print(f"  {MODEL_DISPLAY.get(model, model):22s}  {name}  "
                  f"PSNR={final_psnr:.2f} dB")
            entries.append((name, data))
        if entries:
            result[model] = entries
    return result


def collect_family_smallest(model, n, top2, crops, ref):
    """Return up to n configs from a model family sorted by hash-table size (smallest first).

    Configs already selected as top-2 and configs missing time or vram data are skipped.
    Returns list of (cfg_label, data).
    """
    model_dir = SIZEDIFF / model
    if not model_dir.exists():
        return []
    top2_labels = {label for label, _ in top2.get(model, [])}
    candidates = []
    for cfg_dir in sorted(model_dir.iterdir()):
        if not cfg_dir.is_dir() or cfg_dir.name in top2_labels:
            continue
        data = load_config(cfg_dir, ref, crops)
        if not data["time_psnr"] or data["vram_gb"] is None:
            continue
        parts = cfg_dir.name.split("_")
        try:
            size = int(parts[0]) * int(parts[1]) * (2 ** int(parts[2]))
        except (IndexError, ValueError):
            continue
        candidates.append((size, cfg_dir.name, data))
    candidates.sort()
    return [(name, data) for _, name, data in candidates[:n]]


# ── plot functions ────────────────────────────────────────────────────────────

# Marker shapes per rank within a family.
# All lines stay solid and full-colour; configs are told apart by marker shape.
MARKER_SHAPES = ["o", "s", "^", "D"]


def plot_psnr_epoch(models, top2, filename, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in models:
        if model not in top2:
            continue
        color = FAMILY_COLORS[model]
        display = MODEL_DISPLAY[model]
        for rank, (cfg_label, data) in enumerate(top2[model]):
            pts = data["epoch_psnr"]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=1.8,
                    linestyle="-", marker=MARKER_SHAPES[rank % len(MARKER_SHAPES)],
                    markersize=5, label=f"{display} — {cfg_label}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = RESULTS / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def _cap_time_series(entries):
    """Cap all (label, pts) series to the same number of points as the shortest one."""
    valid = [(lab, pts) for lab, pts in entries if pts]
    if not valid:
        return valid
    min_count = min(len(pts) for _, pts in valid)
    return [(lab, pts[:min_count]) for lab, pts in valid]


def plot_psnr_time(models, top2, filename, title):
    # collect all series first so we can cap them together
    entries = []  # (shaded_color, label_str, pts)
    for model in models:
        if model not in top2:
            continue
        color = FAMILY_COLORS[model]
        display = MODEL_DISPLAY[model]
        for rank, (cfg_label, data) in enumerate(top2[model]):
            entries.append((color,
                            MARKER_SHAPES[rank % len(MARKER_SHAPES)],
                            f"{display} — {cfg_label}",
                            data["time_psnr"]))

    # cap: stop when first series runs out, same number of points for all
    labeled = _cap_time_series([(e[2], e[3]) for e in entries])
    capped = {lab: pts for lab, pts in labeled}

    fig, ax = plt.subplots(figsize=(10, 5))
    for color, marker, label, _ in entries:
        pts = capped.get(label)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=1.8, linestyle="-",
                marker=marker, markersize=5, label=label)
    ax.set_xlabel("Wall-clock time (hours)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = RESULTS / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


TARGET_H = 24.0  # hours — PSNR evaluated at this wall-clock time


def _psnr_at(time_psnr, target_h):
    if not time_psnr:
        return None
    return min(time_psnr, key=lambda p: abs(p[0] - target_h))[1]


def plot_vram(models, top2, filename, title):
    """Scatter: x = VRAM, y = PSNR at 24 h. One circle per config, colored by family."""
    fig, ax = plt.subplots(figsize=(9, 5))

    seen_models = set()
    any_point = False
    for model in models:
        if model not in top2:
            continue
        color = FAMILY_COLORS[model]
        display = MODEL_DISPLAY[model]
        for cfg_label, data in top2[model]:
            vram = data["vram_gb"]
            psnr = _psnr_at(data["time_psnr"], TARGET_H)
            if vram is None or psnr is None:
                continue
            legend_label = display if model not in seen_models else None
            seen_models.add(model)
            ax.scatter(vram, psnr, color=color, marker="o",
                       s=80, zorder=3, label=legend_label)
            ax.annotate(cfg_label, (vram, psnr),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, color=color)
            any_point = True

    if not any_point:
        print(f"  No VRAM/time data for {filename} — skipping.")
        plt.close(fig)
        return

    x_max = max(ax.get_xlim()[1], max(v for _, v in GPU_LINES) * 1.08)
    for gpu_name, gpu_gb in GPU_LINES:
        ax.axvline(gpu_gb, color="dimgray", linewidth=0.9,
                   linestyle="--", alpha=0.7)
        ax.text(gpu_gb, 0, gpu_name,
                ha="center", va="top", fontsize=9, color="black", rotation=90,
                transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_xlabel("Peak Reserved VRAM (GB)", loc="left")
    ax.set_ylabel(f"PSNR (dB) at ~{TARGET_H:.0f} h")
    ax.set_title(title)
    ax.set_xlim(left=0, right=x_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = RESULTS / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading crops and reference image ...")
    crops = load_crops()
    ref   = load_gray(PERFECT)

    print("\nCollecting top-2 configs per model ...")
    top2 = collect_top2(crops, ref)

    print("\nCollecting manual all-models configs ...")
    all_top2 = collect_manual_all(crops, ref, top2)

    print("\nGenerating all-models plots ...")
    all_models_ordered = list(MANUAL_ALL_CONFIGS.keys())
    plot_psnr_epoch(all_models_ordered, all_top2, "all_psnr_epoch.png",
                    "PSNR vs Epoch — selected configs")
    plot_psnr_time(all_models_ordered, all_top2, "all_psnr_time.png",
                   "PSNR vs Time — selected configs")
    plot_vram(all_models_ordered, all_top2, "all_vram.png",
              "PSNR at 24 h vs VRAM — selected configs")

    print("\nGenerating quad-only plots ...")
    plot_psnr_epoch(QUAD_MODELS, top2, "quad_psnr_epoch.png",
                    "PSNR vs Epoch — QuadCubes variants (top-2 configs each)")
    plot_psnr_time(QUAD_MODELS, top2, "quad_psnr_time.png",
                   "PSNR vs Time — QuadCubes variants (top-2 configs each)")
    plot_vram(QUAD_MODELS, top2, "quad_vram.png",
              "PSNR at 24 h vs VRAM — QuadCubes variants (top-2 configs each)")


    print("\nDone. Results in", RESULTS)


if __name__ == "__main__":
    main()
