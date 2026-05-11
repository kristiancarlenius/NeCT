"""
Compare sand-volume MAE metrics across multiple runs.

Reads mae.txt from every immediate subdirectory of BASE_DIR, parses the four
MAE values (total conservation, 1:1 tracking, top chamber, bottom chamber),
and produces a grouped bar chart saved alongside the script.

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

# Output file written next to this script
OUTPUT_PNG = Path(__file__).parent / "mae_comparison.png"

# ─────────────────────────────────────────────────────────────────────────────

METRIC_LABELS = [
    "Total\nconservation",
    "1:1\ntracking",
    "Top\nchamber",
    "Bottom\nchamber",
]


def parse_mae(path: Path) -> dict[str, float] | None:
    """Extract the four MAE values from a mae.txt file."""
    text = path.read_text()
    values = re.findall(r"MAE\s*:\s*([\d.]+)", text)
    if len(values) < 4:
        print(f"  WARNING: expected 4 MAE values in {path}, found {len(values)} — skipping")
        return None
    return {
        "total": float(values[0]),
        "1to1":  float(values[1]),
        "top":   float(values[2]),
        "bot":   float(values[3]),
    }


def collect(base: Path) -> list[tuple[str, dict]]:
    """Return [(run_name, metrics), ...] sorted by run name."""
    results = []
    for sub in sorted(base.iterdir()):
        mae_file = sub / "mae.txt"
        if sub.is_dir() and mae_file.exists():
            metrics = parse_mae(mae_file)
            if metrics is not None:
                results.append((sub.name, metrics))
    return results


def plot(runs: list[tuple[str, dict]], out: Path) -> None:
    n_runs = len(runs)
    n_metrics = len(METRIC_LABELS)

    names = [r[0] for r in runs]
    data = np.array([[r[1]["total"], r[1]["1to1"], r[1]["top"], r[1]["bot"]]
                     for r in runs])  # shape (n_runs, n_metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_runs
    offsets = (np.arange(n_runs) - (n_runs - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_runs + 4), 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_runs))

    for i, (name, color, offset) in enumerate(zip(names, colors, offsets)):
        bars = ax.bar(x + offset, data[i], width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, data[i]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("MAE (mm³)")
    ax.set_title("Sand volume MAE comparison across runs")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.show()


def main():
    runs = collect(BASE_DIR)
    if not runs:
        print(f"No mae.txt files found under {BASE_DIR}")
        return
    print(f"Found {len(runs)} run(s):")
    for name, m in runs:
        print(f"  {name}: total={m['total']:.2f}  1to1={m['1to1']:.2f}  "
              f"top={m['top']:.2f}  bot={m['bot']:.2f}  mm³")
    plot(runs, OUTPUT_PNG)


if __name__ == "__main__":
    main()
