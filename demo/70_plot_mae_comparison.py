"""
Compare sand-volume MAE metrics across multiple runs, grouped by acquisition count.

Run names are expected to follow the pattern: {fps}_{steps}_ac{N}[_re][_OLD]
e.g. 4fps_11000_ac2, 8fps_5500_ac6_re

Outputs (all saved to OUT_DIR):
  mae_total.png, mae_top.png, mae_bot.png          — line plots, one per metric
  mae_total_bar.png, mae_top_bar.png, mae_bot_bar.png — bar plots, one per metric
  mae_group_{group}.png                            — line plot per group, all metrics
  mae_group_{group}_bar.png                        — bar plot per group, all metrics
  mae_4fps.png, mae_8fps.png                       — fps-tier line plots
  mae_4fps_bar.png, mae_8fps_bar.png               — fps-tier bar plots

USE_NPZ=True: loads sand_volume.npz and recomputes MAE with glitch filtering,
              skipping pseudo-periodic buggy timesteps (mirrors 60_hourglass_sand_volume.py).
USE_NPZ=False: reads pre-computed mae.txt files.

Usage:
    Set BASE_DIR below and run locally (no GPU needed).
    mae.txt / sand_volume.npz must have been produced by 60_hourglass_sand_volume.py.
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

OUT_DIR = Path(__file__).parent / "mae_plots"

# When both a plain run and its _OLD counterpart exist for the same ac number,
# True  → keep the _OLD version
# False → keep the non-OLD version
PREFER_OLD = False

# Set True to load sand_volume.npz and recompute MAE with glitch-skip filtering.
# Set False to read pre-computed mae.txt files.
USE_NPZ = True

# Glitch-filter sigma (only used when USE_NPZ=True)
FILTER_SIGMA = 300

# ─────────────────────────────────────────────────────────────────────────────

RUN_RE = re.compile(r"^(\d+fps_\d+)_ac(\d+)(_OLD)?$")

METRICS = ["total", "1to1", "top", "bot"]
PLOT_METRICS = ["total", "top", "bot"]
METRIC_TITLES = {
    "total": "Total conservation MAE (mm³)",
    "top":   "Top chamber MAE (mm³)",
    "bot":   "Bottom chamber MAE (mm³)",
}
METRIC_SHORT = {
    "total": "Total conservation",
    "top":   "Top chamber",
    "bot":   "Bottom chamber",
}
METRIC_COLORS = {
    "total": "mediumpurple",
    "top":   "steelblue",
    "bot":   "firebrick",
}


# ── Glitch filter (mirrors 60_hourglass_sand_volume.py) ──────────────────────

def _filter_glitches(arr: np.ndarray, t_axis: np.ndarray, sigma: float):
    """Return (clean_arr, bad_mask, linear_trend). Replaces outliers with trend."""
    K = 5
    v0 = float(np.median(arr[:K]))
    v1 = float(np.median(arr[-K:]))
    t0, t1 = float(t_axis[0]), float(t_axis[-1])
    trend = v0 + (v1 - v0) * (t_axis - t0) / (t1 - t0)
    residuals = arr - trend
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad == 0:
        return arr.copy(), np.zeros(len(arr), dtype=bool), trend
    bad = np.abs(residuals) > sigma * mad * 1.4826
    clean = arr.copy()
    clean[bad] = trend[bad]
    return clean, bad, trend


def parse_mae_from_npz(path: Path) -> dict[str, float] | None:
    """Recompute MAE from sand_volume.npz, skipping pseudo-periodic buggy timesteps."""
    try:
        data = np.load(path)
    except Exception as e:
        print(f"  WARNING: cannot load {path}: {e} — skipping")
        return None

    top_vols = data["top_volume_mm3"]
    bot_vols = data["bottom_volume_mm3"]
    t_axis   = data["projection_indices"]

    _, top_bad, _ = _filter_glitches(top_vols, t_axis, FILTER_SIGMA)
    _, bot_bad, _ = _filter_glitches(bot_vols, t_axis, FILTER_SIGMA)

    keep = ~(top_bad | bot_bad)
    n_bad = int((~keep).sum())
    if n_bad:
        print(f"  {path.parent.name}: skipped {n_bad} buggy timestep(s)")
    if keep.sum() < 10:
        print(f"  WARNING: only {keep.sum()} clean timesteps in {path} — skipping")
        return None

    top_c   = top_vols[keep]
    bot_c   = bot_vols[keep]
    total_c = top_c + bot_c

    total_truth = float(total_c.mean())
    mae_total   = float(np.mean(np.abs(total_c - total_truth)))

    mae_1to1 = float(np.mean(np.abs((top_c - top_c[0]) + (bot_c - bot_c[0]))))

    t_idx     = np.arange(len(top_c), dtype=float)
    top_trend = np.polyval(np.polyfit(t_idx, top_c, 1), t_idx)
    bot_trend = np.polyval(np.polyfit(t_idx, bot_c, 1), t_idx)
    mae_top   = float(np.mean(np.abs(top_c - top_trend)))
    mae_bot   = float(np.mean(np.abs(bot_c - bot_trend)))

    return {"total": mae_total, "1to1": mae_1to1, "top": mae_top, "bot": mae_bot}


def parse_mae(path: Path) -> dict[str, float] | None:
    """Read MAE values from mae.txt."""
    text = path.read_text()
    values = re.findall(r"MAE\s*:\s*([\d.]+)", text)
    if len(values) < 4:
        print(f"  WARNING: expected 4 MAE values in {path}, found {len(values)} — skipping")
        return None
    return {k: float(v) for k, v in zip(METRICS, values)}


def collect(base: Path) -> dict[str, dict[int, dict[str, float]]]:
    """Return {group: {ac: metrics}} from all subdirectories."""
    candidates: dict[tuple[str, int], list[tuple[dict, bool]]] = defaultdict(list)
    for sub in sorted(base.iterdir()):
        data_file = sub / ("sand_volume.npz" if USE_NPZ else "mae.txt")
        if not (sub.is_dir() and data_file.exists()):
            continue
        m = RUN_RE.match(sub.name)
        if not m:
            print(f"  Skipping (name doesn't match pattern): {sub.name}")
            continue
        group, ac, old_suffix = m.group(1), int(m.group(2)), m.group(3)
        is_old = old_suffix is not None
        metrics = parse_mae_from_npz(data_file) if USE_NPZ else parse_mae(data_file)
        if metrics is not None:
            candidates[(group, ac)].append((metrics, is_old))

    groups: dict[str, dict[int, dict]] = defaultdict(dict)
    for (group, ac), entries in candidates.items():
        if len(entries) == 1:
            groups[group][ac] = entries[0][0]
        else:
            preferred = [e for e in entries if e[1] == PREFER_OLD]
            chosen_metrics, chosen_is_old = preferred[0] if preferred else entries[0]
            tag = "_OLD" if chosen_is_old else "non-OLD"
            print(f"  INFO: duplicate ac{ac} in {group} — kept {tag} (PREFER_OLD={PREFER_OLD})")
            groups[group][ac] = chosen_metrics

    return dict(groups)


def _annotate_line(ax, ac_nums, vals):
    for ac, val in zip(ac_nums, vals):
        ax.annotate(f"{val:.0f}", (ac, val),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)


def _bar_layout(n_groups: int):
    """Return (offsets array of length n_groups, bar_width)."""
    bar_width = min(0.8 / max(n_groups, 1), 0.4)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_width
    return offsets, bar_width


def all_ac_ticks(groups):
    return sorted({ac for g in groups.values() for ac in g})


# ── Line plots ────────────────────────────────────────────────────────────────

def plot_per_metric(groups: dict, out_dir: Path) -> None:
    sorted_groups = sorted(groups)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_groups)))
    ac_ticks = all_ac_ticks(groups)

    for metric in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for group, color in zip(sorted_groups, colors):
            ac_data = groups[group]
            ac_nums = sorted(ac_data)
            vals = [ac_data[ac][metric] for ac in ac_nums]
            ax.plot(ac_nums, vals, marker="o", label=group, color=color)
            _annotate_line(ax, ac_nums, vals)

        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("Acquisition count (ac)")
        ax.set_ylabel("MAE (mm³)")
        ax.set_xticks(ac_ticks)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

        path = out_dir / f"mae_{metric}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def plot_per_group(groups: dict, out_dir: Path) -> None:
    ac_ticks = all_ac_ticks(groups)

    for group in sorted(groups):
        ac_data = groups[group]
        ac_nums = sorted(ac_data)

        fig, ax = plt.subplots(figsize=(8, 5))
        for metric in PLOT_METRICS:
            vals = [ac_data[ac][metric] for ac in ac_nums]
            ax.plot(ac_nums, vals, marker="o",
                    label=METRIC_SHORT[metric],
                    color=METRIC_COLORS[metric])
            _annotate_line(ax, ac_nums, vals)

        ax.set_title(f"{group} — MAE by acquisition count")
        ax.set_xlabel("Acquisition count (ac)")
        ax.set_ylabel("MAE (mm³)")
        ax.set_xticks(ac_ticks)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

        path = out_dir / f"mae_group_{group}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def plot_per_fps(groups: dict, fps_prefix: str, out_dir: Path) -> None:
    subset = {g: v for g, v in groups.items() if g.startswith(fps_prefix)}
    if not subset:
        print(f"  No groups found for prefix '{fps_prefix}'")
        return

    sorted_groups = sorted(subset)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_groups)))
    ac_ticks = all_ac_ticks(subset)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, metric in zip(axes, PLOT_METRICS):
        for group, color in zip(sorted_groups, colors):
            ac_data = subset[group]
            ac_nums = sorted(ac_data)
            vals = [ac_data[ac][metric] for ac in ac_nums]
            ax.plot(ac_nums, vals, marker="o", label=group, color=color)
            _annotate_line(ax, ac_nums, vals)

        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("Acquisition count (ac)")
        ax.set_ylabel("MAE (mm³)")
        ax.set_xticks(ac_ticks)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(sorted_groups),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{fps_prefix} — MAE by acquisition count", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    path = out_dir / f"mae_{fps_prefix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Bar plots ─────────────────────────────────────────────────────────────────

def plot_per_metric_bar(groups: dict, out_dir: Path) -> None:
    sorted_groups = sorted(groups)
    n_groups = len(sorted_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    ac_ticks = all_ac_ticks(groups)
    offsets, bw = _bar_layout(n_groups)

    for metric in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(max(8, len(ac_ticks) * n_groups * 0.5 + 2), 5))

        for gi, (group, color) in enumerate(zip(sorted_groups, colors)):
            ac_data = groups[group]
            xs   = [ac + offsets[gi] for ac in ac_ticks if ac in ac_data]
            vals = [ac_data[ac][metric] for ac in ac_ticks if ac in ac_data]
            ax.bar(xs, vals, width=bw, label=group, color=color, alpha=0.85)
            for x, v in zip(xs, vals):
                ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("Acquisition count (ac)")
        ax.set_ylabel("MAE (mm³)")
        ax.set_xticks(ac_ticks)
        ax.set_xticklabels([str(a) for a in ac_ticks])
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

        path = out_dir / f"mae_{metric}_bar.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def plot_per_group_bar(groups: dict, out_dir: Path) -> None:
    n_metrics = len(PLOT_METRICS)
    offsets, bw = _bar_layout(n_metrics)

    for group in sorted(groups):
        ac_data = groups[group]
        ac_nums = sorted(ac_data)

        fig, ax = plt.subplots(figsize=(max(8, len(ac_nums) * n_metrics * 0.5 + 2), 5))

        for mi, metric in enumerate(PLOT_METRICS):
            xs   = [ac + offsets[mi] for ac in ac_nums]
            vals = [ac_data[ac][metric] for ac in ac_nums]
            ax.bar(xs, vals, width=bw, label=METRIC_SHORT[metric],
                   color=METRIC_COLORS[metric], alpha=0.85)
            for x, v in zip(xs, vals):
                ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(f"{group} — MAE by acquisition count")
        ax.set_xlabel("Acquisition count (ac)")
        ax.set_ylabel("MAE (mm³)")
        ax.set_xticks(ac_nums)
        ax.set_xticklabels([str(a) for a in ac_nums])
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

        path = out_dir / f"mae_group_{group}_bar.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def plot_per_fps_bar(groups: dict, fps_prefix: str, out_dir: Path) -> None:
    subset = {g: v for g, v in groups.items() if g.startswith(fps_prefix)}
    if not subset:
        print(f"  No groups found for prefix '{fps_prefix}'")
        return

    sorted_groups = sorted(subset)
    n_groups = len(sorted_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    ac_ticks = all_ac_ticks(subset)
    offsets, bw = _bar_layout(n_groups)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, metric in zip(axes, PLOT_METRICS):
        for gi, (group, color) in enumerate(zip(sorted_groups, colors)):
            ac_data = subset[group]
            xs   = [ac + offsets[gi] for ac in ac_ticks if ac in ac_data]
            vals = [ac_data[ac][metric] for ac in ac_ticks if ac in ac_data]
            ax.bar(xs, vals, width=bw, label=group, color=color, alpha=0.85)
            for x, v in zip(xs, vals):
                ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("Acquisition count (ac)")
        ax.set_ylabel("MAE (mm³)")
        ax.set_xticks(ac_ticks)
        ax.set_xticklabels([str(a) for a in ac_ticks])
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_groups,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{fps_prefix} — MAE by acquisition count", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    path = out_dir / f"mae_{fps_prefix}_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    groups = collect(BASE_DIR)
    if not groups:
        src = "sand_volume.npz" if USE_NPZ else "mae.txt"
        print(f"No parseable {src} files found under {BASE_DIR}")
        return

    print(f"Found {sum(len(v) for v in groups.values())} run(s) in {len(groups)} group(s):")
    for group in sorted(groups):
        for ac in sorted(groups[group]):
            m = groups[group][ac]
            print(f"  {group}_ac{ac}: total={m['total']:.2f}  "
                  f"top={m['top']:.2f}  bot={m['bot']:.2f}  mm³")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_per_metric(groups, OUT_DIR)
    plot_per_group(groups, OUT_DIR)
    plot_per_fps(groups, "4fps", OUT_DIR)
    plot_per_fps(groups, "8fps", OUT_DIR)
    plot_per_metric_bar(groups, OUT_DIR)
    plot_per_group_bar(groups, OUT_DIR)
    plot_per_fps_bar(groups, "4fps", OUT_DIR)
    plot_per_fps_bar(groups, "8fps", OUT_DIR)
    print(f"\nAll plots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
