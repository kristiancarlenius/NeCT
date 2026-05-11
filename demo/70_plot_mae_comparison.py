"""
Compare sand-volume MAE metrics across multiple runs, grouped by acquisition count.

Run names are expected to follow the pattern: {fps}_{steps}_ac{N}[_re]
e.g. 4fps_11000_ac2, 8fps_5500_ac6_re

Groups: one line per {fps}_{steps} combination.
X-axis: ac number (2, 3, 4, 6, ...).
Four subplots, one per MAE metric.

Usage:
    Set BASE_DIR below and run locally (no GPU needed).
    mae.txt files must have been produced by 60_hourglass_sand_volume.py.
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────── CONFIG ──────────────────────────────────────────

BASE_DIR = Path(
    "/cluster/home/kristiac/NeCT/outputs/dynamic_continious"
    "/quadcubes_22_4_22_16_2_4_128_L1"
)

# Output file written next to this script
OUTPUT_PNG = Path(__file__).parent / "mae_comparison.png"

# ─────────────────────────────────────────────────────────────────────────────

# Matches e.g. "4fps_11000_ac2", "8fps_5500_ac6_re"
RUN_RE = re.compile(r"^(\d+fps_\d+)_ac(\d+)")

METRICS = ["total", "1to1", "top", "bot"]
METRIC_TITLES = [
    "Total conservation MAE",
    "1:1 tracking MAE",
    "Top chamber MAE",
    "Bottom chamber MAE",
]


def parse_mae(path: Path) -> dict[str, float] | None:
    text = path.read_text()
    values = re.findall(r"MAE\s*:\s*([\d.]+)", text)
    if len(values) < 4:
        print(f"  WARNING: expected 4 MAE values in {path}, found {len(values)} — skipping")
        return None
    return {k: float(v) for k, v in zip(METRICS, values)}


def collect(base: Path) -> dict[str, dict[int, dict[str, float]]]:
    """Return {group: {ac: metrics}} from all subdirectories with a mae.txt."""
    groups: dict[str, dict[int, dict]] = defaultdict(dict)
    for sub in sorted(base.iterdir()):
        mae_file = sub / "mae.txt"
        if not (sub.is_dir() and mae_file.exists()):
            continue
        m = RUN_RE.match(sub.name)
        if not m:
            print(f"  Skipping (name doesn't match pattern): {sub.name}")
            continue
        group, ac = m.group(1), int(m.group(2))
        metrics = parse_mae(mae_file)
        if metrics is not None:
            if ac in groups[group]:
                print(f"  WARNING: duplicate ac{ac} in group {group} — keeping first")
            else:
                groups[group][ac] = metrics
    return dict(groups)


def plot(groups: dict[str, dict[int, dict]], out: Path) -> None:
    sorted_groups = sorted(groups)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_groups)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
    axes = axes.flatten()

    for ax, metric, title in zip(axes, METRICS, METRIC_TITLES):
        for group, color in zip(sorted_groups, colors):
            ac_data = groups[group]
            ac_nums = sorted(ac_data)
            vals = [ac_data[ac][metric] for ac in ac_nums]
            ax.plot(ac_nums, vals, marker="o", label=group, color=color)
            for ac, val in zip(ac_nums, vals):
                ax.annotate(f"{val:.0f}", (ac, val),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Acquisition count (ac)")
        ax.set_ylabel("MAE (mm³)")
        ax.set_xticks(sorted({ac for g in groups.values() for ac in g}))
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    # Shared legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Sand volume MAE by acquisition count", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    plt.show()


def main():
    groups = collect(BASE_DIR)
    if not groups:
        print(f"No parseable mae.txt files found under {BASE_DIR}")
        return

    print(f"Found {sum(len(v) for v in groups.values())} run(s) in {len(groups)} group(s):")
    for group in sorted(groups):
        for ac in sorted(groups[group]):
            m = groups[group][ac]
            print(f"  {group}_ac{ac}: total={m['total']:.2f}  1to1={m['1to1']:.2f}  "
                  f"top={m['top']:.2f}  bot={m['bot']:.2f}  mm³")

    plot(groups, OUTPUT_PNG)


if __name__ == "__main__":
    main()
