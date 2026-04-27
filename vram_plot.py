#!/usr/bin/env python3
"""
Plot peak allocated VRAM for all architectures found in outputs/.
Deduplicates to one run per model config (latest timestamp).
Saves results/vram_by_arch.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def parse_vram(path):
    with open(path) as f:
        for line in f:
            if line.startswith("Peak allocated:"):
                return float(line.split(":")[1].strip().split()[0])
    return None


def collect():
    # model_config -> {timestamp -> vram_gb}
    runs = defaultdict(dict)

    for root, _, files in os.walk(OUTPUTS_DIR):
        if "vram.txt" not in files:
            continue
        parts = os.path.normpath(root).split(os.sep)
        try:
            mi = parts.index("model")
        except ValueError:
            continue
        timestamp = parts[mi - 1]
        model_config = parts[mi - 2]
        vram = parse_vram(os.path.join(root, "vram.txt"))
        if vram is not None:
            runs[model_config][timestamp] = vram

    records = []
    for model_config, ts_map in runs.items():
        latest = sorted(ts_map)[-1]
        arch = model_config.split("_")[0]
        records.append({"config": model_config, "arch": arch, "vram": ts_map[latest]})
    return records


def main():
    records = collect()
    if not records:
        print("No vram.txt files found under outputs/")
        return

    arch_data: dict[str, list[float]] = defaultdict(list)
    for r in records:
        arch_data[r["arch"]].append(r["vram"])

    archs = sorted(arch_data)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(archs)))
    color_map = dict(zip(archs, colors))

    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(42)

    for i, arch in enumerate(archs):
        vals = sorted(arch_data[arch])
        xs = rng.uniform(i - 0.25, i + 0.25, len(vals))
        ax.scatter(xs, vals, color=color_map[arch], alpha=0.75, s=60, zorder=3)
        ax.hlines(np.mean(vals), i - 0.35, i + 0.35,
                  color=color_map[arch], linewidth=2.5, zorder=4)

    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs, rotation=20, ha="right")
    ax.set_ylabel("Peak Allocated VRAM (GB)")
    ax.set_title("Peak Allocated VRAM by Architecture  (thick bar = mean)")
    gpu_limits = [
        (80, "H100/A100 80GB"),
        (40, "A100 40GB"),
        (32, "RTX 5090"),
        (24, "RTX 4090"),
        (16, "RTX 5080"),
        (12, "RTX 5070"),
        ( 8, "RTX 3070"),
    ]
    for vram_limit, label in gpu_limits:
        ax.axhline(vram_limit, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
        ax.text(len(archs) - 0.5, vram_limit + 0.3, label,
                fontsize=7, color="gray", ha="right", va="bottom")

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "vram_by_arch.png")
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.close(fig)

    print()
    print(f"{'Architecture':<22}  {'n':>3}  {'min':>6}  {'mean':>6}  {'max':>6}  GB")
    print("-" * 55)
    for arch in archs:
        vals = arch_data[arch]
        print(f"{arch:<22}  {len(vals):>3}  {min(vals):>6.2f}  "
              f"{np.mean(vals):>6.2f}  {max(vals):>6.2f}")


if __name__ == "__main__":
    main()
