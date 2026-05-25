"""
Bar-chart comparison of new vs _OLD model variants.

For every run that has both a {base} and a {base}_OLD directory with a
mae.txt, plots grouped bars showing RMAE (√MAE) side by side.

Outputs (all saved to OUT_DIR):
  old_cmp_4fps.png, old_cmp_8fps.png
      One figure per fps tier.  Each figure has a 2×2 grid of subplots
      (one per metric).  Every matched pair is a group of two bars:
        ■ New  (steelblue)
        ■ OLD  (firebrick)

Usage:
    Set BASE_DIR below and run locally (no GPU needed).
    mae.txt files must have been produced by 60_hourglass_sand_volume.py.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────── CONFIG ──────────────────────────────────────────

BASE_DIR = Path(
    "/cluster/home/kristiac/NeCT/outputs/dynamic_continious"
    "/quadcubes_22_4_22_16_2_4_128_L1"
)

OUT_DIR = Path(__file__).parent / "mae_plots"

# ─────────────────────────────────────────────────────────────────────────────

RUN_RE = re.compile(r"^(\d+fps_\d+_ac\d+)(_OLD)?$")

METRICS = ["total", "1to1", "top", "bot"]
METRIC_TITLES = {
    "total": "Total conservation",
    "1to1":  "1:1 tracking",
    "top":   "Top chamber",
    "bot":   "Bottom chamber",
}

COLOR_NEW = "steelblue"
COLOR_OLD = "firebrick"
BAR_WIDTH = 0.35


def parse_rmae(path: Path) -> dict[str, float] | None:
    text = path.read_text()
    values = re.findall(r"MAE\s*:\s*([\d.]+)", text)
    if len(values) < 4:
        print(f"  WARNING: expected 4 MAE values in {path}, found {len(values)} — skipping")
        return None
    return {k: float(v) ** 0.5 for k, v in zip(METRICS, values)}


def collect(base: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Return {base_name: {"new": metrics, "old": metrics}} for paired runs."""
    all_data: dict[str, dict] = {}

    for sub in sorted(base.iterdir()):
        mae_file = sub / "mae.txt"
        if not (sub.is_dir() and mae_file.exists()):
            continue
        m = RUN_RE.match(sub.name)
        if not m:
            continue
        base_name = m.group(1)   # e.g. "4fps_11000_ac3"
        is_old    = m.group(2) is not None

        metrics = parse_rmae(mae_file)
        if metrics is None:
            continue

        if base_name not in all_data:
            all_data[base_name] = {}
        key = "old" if is_old else "new"
        if key in all_data[base_name]:
            print(f"  WARNING: duplicate {sub.name} — keeping first")
        else:
            all_data[base_name][key] = metrics

    # Keep only pairs where both new and old are present
    pairs = {k: v for k, v in all_data.items() if "new" in v and "old" in v}
    missing = {k: v for k, v in all_data.items() if "new" not in v or "old" not in v}
    if missing:
        print("  Runs with no counterpart (skipped):")
        for name, d in sorted(missing.items()):
            print(f"    {name}  (have: {list(d)})")
    return pairs


def plot_fps_tier(pairs: dict, fps_prefix: str, out_dir: Path) -> None:
    subset = {k: v for k, v in pairs.items() if k.startswith(fps_prefix)}
    if not subset:
        print(f"  No paired runs for prefix '{fps_prefix}'")
        return

    labels = sorted(subset)                  # sorted base names
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(max(10, len(labels) * 1.5), 8), sharey=False)
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        new_vals = [subset[lbl]["new"][metric] for lbl in labels]
        old_vals = [subset[lbl]["old"][metric] for lbl in labels]

        bars_new = ax.bar(x - BAR_WIDTH / 2, new_vals, BAR_WIDTH,
                          label="New", color=COLOR_NEW, alpha=0.85)
        bars_old = ax.bar(x + BAR_WIDTH / 2, old_vals, BAR_WIDTH,
                          label="OLD", color=COLOR_OLD, alpha=0.85)

        # Value labels above each bar
        for bar in bars_new:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)
        for bar in bars_old:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(METRIC_TITLES[metric])
        ax.set_ylabel("RMAE (mm^1.5)")
        ax.set_xticks(x)
        # Strip the fps prefix from tick labels to save space
        short_labels = [lbl.replace(fps_prefix + "_", "") for lbl in labels]
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{fps_prefix} — New vs OLD  (RMAE = √MAE)", fontsize=12)
    plt.tight_layout()

    path = out_dir / f"old_cmp_{fps_prefix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    pairs = collect(BASE_DIR)
    if not pairs:
        print(f"No paired (new + OLD) runs with mae.txt found under {BASE_DIR}")
        return

    print(f"Found {len(pairs)} matched pair(s):")
    for name in sorted(pairs):
        m = pairs[name]
        print(f"  {name}:  new total={m['new']['total']:.3f}  old total={m['old']['total']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_fps_tier(pairs, "4fps", OUT_DIR)
    plot_fps_tier(pairs, "8fps", OUT_DIR)
    print(f"\nAll plots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
