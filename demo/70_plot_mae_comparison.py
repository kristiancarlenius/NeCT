"""
Compare sand-volume MASE (MAE²) metrics across multiple runs, grouped by acquisition count.

Run names are expected to follow the pattern: {fps}_{steps}_ac{N}[_re]
e.g. 4fps_11000_ac2, 8fps_5500_ac6_re

Outputs (all saved to OUT_DIR):
  mase_total.png, mase_1to1.png, mase_top.png, mase_bot.png
      One image per metric.  All {fps}_{steps} groups as lines, x = ac number.

  mase_group_4fps_11000.png, mase_group_8fps_5500.png, ...
      One image per {fps}_{steps} group.  All four metrics as lines, x = ac number.

  mase_4fps.png, mase_8fps.png
      One image per fps tier (4fps / 8fps), all metrics as subplots,
      filtered to groups of that fps only.

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

# Directory where all PNGs are written
OUT_DIR = Path(__file__).parent / "mae_plots"

# ─────────────────────────────────────────────────────────────────────────────

RUN_RE = re.compile(r"^(\d+fps_\d+)_ac(\d+)")

METRICS = ["total", "1to1", "top", "bot"]
METRIC_TITLES = {
    "total": "Total conservation MASE (mm⁶)",
    "1to1":  "1:1 tracking MASE (mm⁶)",
    "top":   "Top chamber MASE (mm⁶)",
    "bot":   "Bottom chamber MASE (mm⁶)",
}
METRIC_SHORT = {
    "total": "Total conservation",
    "1to1":  "1:1 tracking",
    "top":   "Top chamber",
    "bot":   "Bottom chamber",
}
METRIC_COLORS = {
    "total": "mediumpurple",
    "1to1":  "darkorange",
    "top":   "steelblue",
    "bot":   "firebrick",
}


def parse_mase(path: Path) -> dict[str, float] | None:
    """Read MAE values from mae.txt and return their squares (MASE = MAE²)."""
    text = path.read_text()
    values = re.findall(r"MAE\s*:\s*([\d.]+)", text)
    if len(values) < 4:
        print(f"  WARNING: expected 4 MAE values in {path}, found {len(values)} — skipping")
        return None
    return {k: float(v) ** 2 for k, v in zip(METRICS, values)}


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
        metrics = parse_mase(mae_file)
        if metrics is not None:
            if ac in groups[group]:
                print(f"  WARNING: duplicate ac{ac} in group {group} — keeping first")
            else:
                groups[group][ac] = metrics
    return dict(groups)


def _annotate(ax, ac_nums, vals):
    for ac, val in zip(ac_nums, vals):
        ax.annotate(f"{val:.0f}", (ac, val),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)


def all_ac_ticks(groups):
    return sorted({ac for g in groups.values() for ac in g})


# ── One image per metric (all groups as lines) ────────────────────────────────

def plot_per_metric(groups: dict, out_dir: Path) -> None:
    sorted_groups = sorted(groups)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_groups)))
    ac_ticks = all_ac_ticks(groups)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for group, color in zip(sorted_groups, colors):
            ac_data = groups[group]
            ac_nums = sorted(ac_data)
            vals = [ac_data[ac][metric] for ac in ac_nums]
            ax.plot(ac_nums, vals, marker="o", label=group, color=color)
            _annotate(ax, ac_nums, vals)

        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("Acquisition count (ac)")
        ax.set_ylabel("MASE (mm⁶)")
        ax.set_xticks(ac_ticks)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

        path = out_dir / f"mase_{metric}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


# ── One image per group (all metrics as lines) ────────────────────────────────

def plot_per_group(groups: dict, out_dir: Path) -> None:
    ac_ticks = all_ac_ticks(groups)

    for group in sorted(groups):
        ac_data = groups[group]
        ac_nums = sorted(ac_data)

        fig, ax = plt.subplots(figsize=(8, 5))
        for metric in METRICS:
            vals = [ac_data[ac][metric] for ac in ac_nums]
            ax.plot(ac_nums, vals, marker="o",
                    label=METRIC_SHORT[metric],
                    color=METRIC_COLORS[metric])
            _annotate(ax, ac_nums, vals)

        ax.set_title(f"{group} — MASE by acquisition count")
        ax.set_xlabel("Acquisition count (ac)")
        ax.set_ylabel("MASE (mm⁶)")
        ax.set_xticks(ac_ticks)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

        path = out_dir / f"mase_group_{group}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


# ── One image per fps tier (4fps / 8fps), all metrics as subplots ─────────────

def plot_per_fps(groups: dict, fps_prefix: str, out_dir: Path) -> None:
    subset = {g: v for g, v in groups.items() if g.startswith(fps_prefix)}
    if not subset:
        print(f"  No groups found for prefix '{fps_prefix}'")
        return

    sorted_groups = sorted(subset)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_groups)))
    ac_ticks = all_ac_ticks(subset)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        for group, color in zip(sorted_groups, colors):
            ac_data = subset[group]
            ac_nums = sorted(ac_data)
            vals = [ac_data[ac][metric] for ac in ac_nums]
            ax.plot(ac_nums, vals, marker="o", label=group, color=color)
            _annotate(ax, ac_nums, vals)

        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("Acquisition count (ac)")
        ax.set_ylabel("MASE (mm⁶)")
        ax.set_xticks(ac_ticks)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(sorted_groups),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{fps_prefix} — MASE by acquisition count", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    path = out_dir / f"mase_{fps_prefix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


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
                  f"top={m['top']:.2f}  bot={m['bot']:.2f}  mm⁶")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_per_metric(groups, OUT_DIR)
    plot_per_group(groups, OUT_DIR)
    plot_per_fps(groups, "4fps", OUT_DIR)
    plot_per_fps(groups, "8fps", OUT_DIR)
    print(f"\nAll plots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
