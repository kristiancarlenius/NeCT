"""
Soundness check: verify MAE metrics follow expected monotonic orderings.

Two checks run for each metric (total / top / bot):

  1. AC monotonicity — for each (fps, steps) group, MAE should decrease as ac increases.
     More acquisitions → better temporal coverage → lower error.

  2. Steps ordering — for each (fps, ac), MAE should increase with steps:
       MAE(2750) < MAE(5500) < MAE(11000)
     Fewer steps per acquisition → less overfitting per frame → lower temporal MAE.

Reports PASS / FAIL per case and prints a summary.

Usage:
    Configure BASE_DIR / USE_NPZ / FILTER_SIGMA to match your 70_ run, then execute.
    No GPU needed.
"""

import re
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np

# ─────────────────────────── CONFIG ──────────────────────────────────────────

BASE_DIR = Path(
    "/cluster/home/kristiac/NeCT/outputs/dynamic_continious"
    "/quadcubes_22_4_22_16_2_4_128_L1"
)

PREFER_OLD = False

USE_NPZ = True
FILTER_SIGMA = 30

OUT_DIR = Path(__file__).parent / "mae_plots"

# ─────────────────────────────────────────────────────────────────────────────

RUN_RE   = re.compile(r"^(\d+fps_\d+)_ac(\d+)(_OLD)?$")
GROUP_RE = re.compile(r"^(\d+)fps_(\d+)$")
METRICS  = ["total", "1to1", "top", "bot"]
CHECK_METRICS = ["total", "top", "bot"]


# ── Data loading (mirrors 70_plot_mae_comparison.py) ─────────────────────────

def _filter_glitches(arr: np.ndarray, t_axis: np.ndarray, sigma: float):
    K = 5
    v0, v1 = float(np.median(arr[:K])), float(np.median(arr[-K:]))
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


def _parse_npz(path: Path) -> dict | None:
    try:
        data = np.load(path)
    except Exception:
        return None
    top_vols = data["top_volume_mm3"]
    bot_vols = data["bottom_volume_mm3"]
    t_axis   = data["projection_indices"]
    _, top_bad, _ = _filter_glitches(top_vols, t_axis, FILTER_SIGMA)
    _, bot_bad, _ = _filter_glitches(bot_vols, t_axis, FILTER_SIGMA)
    keep = ~(top_bad | bot_bad)
    if keep.sum() < 10:
        return None
    top_c, bot_c = top_vols[keep], bot_vols[keep]
    total_c = top_c + bot_c
    total_truth = float(total_c.mean())
    mae_total = float(np.mean(np.abs(total_c - total_truth)))
    mae_1to1  = float(np.mean(np.abs((top_c - top_c[0]) + (bot_c - bot_c[0]))))
    t_idx     = np.arange(len(top_c), dtype=float)
    top_trend = np.polyval(np.polyfit(t_idx, top_c, 1), t_idx)
    bot_trend = np.polyval(np.polyfit(t_idx, bot_c, 1), t_idx)
    mae_top   = float(np.mean(np.abs(top_c - top_trend)))
    mae_bot   = float(np.mean(np.abs(bot_c - bot_trend)))
    return {"total": mae_total, "1to1": mae_1to1, "top": mae_top, "bot": mae_bot}


def _parse_txt(path: Path) -> dict | None:
    values = re.findall(r"MAE\s*:\s*([\d.]+)", path.read_text())
    if len(values) < 4:
        return None
    return {k: float(v) for k, v in zip(METRICS, values)}


def collect(base: Path) -> dict:
    candidates: dict = defaultdict(list)
    for sub in sorted(base.iterdir()):
        data_file = sub / ("sand_volume.npz" if USE_NPZ else "mae.txt")
        if not (sub.is_dir() and data_file.exists()):
            continue
        m = RUN_RE.match(sub.name)
        if not m:
            continue
        group, ac, old_suffix = m.group(1), int(m.group(2)), m.group(3)
        metrics = _parse_npz(data_file) if USE_NPZ else _parse_txt(data_file)
        if metrics is not None:
            candidates[(group, ac)].append((metrics, old_suffix is not None))

    groups: dict = defaultdict(dict)
    for (group, ac), entries in candidates.items():
        if len(entries) == 1:
            groups[group][ac] = entries[0][0]
        else:
            preferred = [e for e in entries if e[1] == PREFER_OLD]
            groups[group][ac] = (preferred[0] if preferred else entries[0])[0]
    return dict(groups)


# ── Checks ────────────────────────────────────────────────────────────────────

def _pass_fail(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def check_ac_monotonicity(groups: dict) -> tuple[int, int]:
    """MAE should decrease (or stay flat) as ac increases."""
    print(f"\n{'='*65}")
    print("CHECK 1 — AC MONOTONICITY")
    print("  Expect: MAE decreases as ac count increases")
    print(f"{'='*65}")

    total_pass = total_fail = 0
    for metric in CHECK_METRICS:
        print(f"\n  [{metric}]")
        for group in sorted(groups):
            ac_data = groups[group]
            ac_nums = sorted(ac_data)
            if len(ac_nums) < 2:
                print(f"    --    {group}: only 1 ac point")
                continue
            vals = [ac_data[ac][metric] for ac in ac_nums]
            viols = [(ac_nums[i], ac_nums[i+1], vals[i], vals[i+1])
                     for i in range(len(vals)-1) if vals[i+1] > vals[i]]
            ok = len(viols) == 0
            summary = "  ".join(f"ac{a}={v:.1f}" for a, v in zip(ac_nums, vals))
            print(f"    {_pass_fail(ok)}  {group}: {summary}")
            for a0, a1, v0, v1 in viols:
                print(f"          ↑ ac{a0}={v0:.1f} → ac{a1}={v1:.1f}  (+{v1-v0:.1f})")
            if ok:
                total_pass += 1
            else:
                total_fail += 1

    return total_pass, total_fail


def check_steps_ordering(groups: dict) -> tuple[int, int]:
    """MAE should increase with steps: MAE(2750) < MAE(5500) < MAE(11000)."""
    print(f"\n{'='*65}")
    print("CHECK 2 — STEPS ORDERING")
    print("  Expect: MAE(fewer steps) < MAE(more steps)")
    print(f"{'='*65}")

    # Restructure to {(fps, ac): {steps: {metric: value}}}
    by_fps_ac: dict = defaultdict(lambda: defaultdict(dict))
    for group, ac_data in groups.items():
        m = GROUP_RE.match(group)
        if not m:
            continue
        fps, steps = int(m.group(1)), int(m.group(2))
        for ac, metrics in ac_data.items():
            by_fps_ac[(fps, ac)][steps] = metrics

    total_pass = total_fail = 0
    for metric in CHECK_METRICS:
        print(f"\n  [{metric}]")
        for (fps, ac) in sorted(by_fps_ac):
            steps_data = by_fps_ac[(fps, ac)]
            sorted_steps = sorted(steps_data)
            if len(sorted_steps) < 2:
                continue
            vals = [steps_data[s][metric] for s in sorted_steps]
            # expect ascending: fewer steps → lower MAE
            viols = [(sorted_steps[i], sorted_steps[i+1], vals[i], vals[i+1])
                     for i in range(len(vals)-1) if vals[i+1] < vals[i]]
            ok = len(viols) == 0
            summary = "  ".join(f"{s}steps={v:.1f}" for s, v in zip(sorted_steps, vals))
            print(f"    {_pass_fail(ok)}  {fps}fps ac{ac}: {summary}")
            for s0, s1, v0, v1 in viols:
                print(f"          ↓ {s0}steps={v0:.1f} → {s1}steps={v1:.1f}  (-{v0-v1:.1f}, expected ↑)")
            if ok:
                total_pass += 1
            else:
                total_fail += 1

    return total_pass, total_fail


def main():
    print(f"Base dir : {BASE_DIR}")
    print(f"Source   : {'sand_volume.npz (recomputed)' if USE_NPZ else 'mae.txt'}")
    if USE_NPZ:
        print(f"Sigma    : {FILTER_SIGMA}")

    groups = collect(BASE_DIR)
    if not groups:
        src = "sand_volume.npz" if USE_NPZ else "mae.txt"
        print(f"\nNo parseable {src} files found under {BASE_DIR}")
        return

    n_runs = sum(len(v) for v in groups.values())
    print(f"Loaded   : {n_runs} run(s) in {len(groups)} group(s)\n")
    for group in sorted(groups):
        acs = sorted(groups[group])
        print(f"  {group}: ac = {acs}")

    p1, f1 = check_ac_monotonicity(groups)
    p2, f2 = check_steps_ordering(groups)

    total_p = p1 + p2
    total_f = f1 + f2
    print(f"\n{'='*65}")
    print(f"SUMMARY: {total_p} PASS  /  {total_f} FAIL  "
          f"({'all good' if total_f == 0 else 'issues found'})")
    print(f"{'='*65}")


class _Tee:
    """Write to both stdout and a StringIO buffer simultaneously."""
    def __init__(self):
        self._buf = StringIO()
    def write(self, s):
        sys.stdout.write(s)
        self._buf.write(s)
    def flush(self):
        sys.stdout.flush()
    def getvalue(self):
        return self._buf.getvalue()


if __name__ == "__main__":
    tee = _Tee()
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = old_stdout
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "soundness_report.txt"
    out_path.write_text(tee.getvalue())
    print(f"Report saved to {out_path}")
