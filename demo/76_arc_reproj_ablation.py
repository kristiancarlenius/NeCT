"""
Arc-integrated reprojection RMSE ablation — tests whether training accumulation_steps
is sufficient to model the angular blur of each projection.

For each run, computes RMSE between the model's arc-averaged forward projection and the
actual measured projection at several evaluation resolutions (N_eval sub-angles).

Interpretation:
    ratio = RMSE(N_eval=N_HIGH) / RMSE(N_eval=1)

    ratio ≈ 1.0  →  N_eval doesn't matter, model already handles blur correctly
                    (training accumulation_steps was sufficient)
    ratio > 1.0  →  accurate arc integration gives WORSE fit than single-angle
                    evaluation — the model absorbed the blur into its weights by
                    learning a single-angle approximation.  Training with more
                    accumulation_steps should drive this ratio toward 1.

Experiment design to test accumulation_steps:
    1. Run this script on existing models at various ac values.
       Expect ratio to grow with ac (more blur → single-angle approx breaks down).
    2. Retrain the high-ac run with larger accumulation_steps.
    3. Rerun this script.  Ratio should drop back toward 1 if accumulation_steps helps.

Usage:
    Set MODEL_DIRS to the list of run directories to evaluate, then run on a GPU node.
    Results are cached in each run dir as arc_reproj_metrics.txt.
    A summary table is printed and saved to OUT_DIR/arc_reproj_report.txt.
"""

from __future__ import annotations

import gc
import math
import sys
from io import StringIO
from pathlib import Path

import numpy as np

# ─────────────────────────── CONFIG ──────────────────────────────────────────

# Base directory containing run subdirectories.
BASE_DIR = Path(
    "/cluster/home/kristiac/NeCT/outputs/dynamic_continious"
    "/quadcubes_22_4_22_16_2_4_128_L1"
)

# Override to evaluate specific subdirs only; None = scan all subdirs matching RUN_RE.
MODEL_DIRS: list[str] | None = None

OUT_DIR = Path(__file__).parent / "mae_plots"

# Evaluation resolutions: RMSE is computed independently at each N.
# The ratio RMSE(N_EVAL_HIGH) / RMSE(1) is the key diagnostic.
N_EVAL_LIST: list[int] = [1, 3, 10]
N_EVAL_HIGH = max(N_EVAL_LIST)

# Stride when sampling projections from the dataset (1 = all, 10 = every 10th).
EVAL_EVERY_N = 5

# If arc_reproj_metrics.txt already exists and SKIP_CACHED=True, reuse it.
SKIP_CACHED = True

# ─────────────────────────────────────────────────────────────────────────────

import re
RUN_RE = re.compile(r"^(\d+fps_\d+)_ac(\d+)(_OLD)?$")


class _Tee:
    def __init__(self, real):
        self._real = real
        from io import StringIO
        self._buf = StringIO()

    def write(self, s):
        self._real.write(s)
        self._buf.write(s)

    def flush(self):
        self._real.flush()

    def getvalue(self):
        return self._buf.getvalue()


# ── Core computation ──────────────────────────────────────────────────────────

def _compute_arc_rmse_at_n(
    model,
    dataset,
    projector,
    geometry,
    device,
    n_eval: int,
    indices: list[int],
) -> float | None:
    """
    Returns RMSE across all (projection, pixel) pairs, using n_eval sub-angles
    spanning the full arc [angle_start, angle_stop] for each projection.
    """
    import torch

    total_sq = 0.0
    total_n = 0

    with torch.no_grad():
        for idx in indices:
            item = dataset[idx]
            if len(item) == 4:
                proj_data, angle_start, angle_stop, timestep = item
            else:
                proj_data, angle_center, timestep = item
                # Non-continuous dataset: no arc info — treat as zero-width arc
                angle_start = angle_center
                angle_stop = angle_center

            proj_flat = proj_data.flatten().to(device)

            # Set up pixel batches at the arc start angle (establishes batch_per_epoch,
            # pixel ordering, etc.).  update_angle() changes only the 3D ray geometry.
            projector.update(
                angle=float(angle_start),
                detector_binning=1,
                points_per_ray=projector.points_per_ray,
                random_offset_detector=0.0,
            )
            projector.random_indexes = torch.arange(
                projector.total_detector_pixels, dtype=torch.int64, device=device
            )

            # Sub-angles: midpoints of n_eval equal sub-intervals spanning the arc.
            end_lsp = np.linspace(float(angle_start), float(angle_stop), n_eval + 1, endpoint=True)
            sub_angles = [(end_lsp[k] + end_lsp[k + 1]) / 2 for k in range(n_eval)]

            for batch_num in range(projector.batch_per_epoch):
                # Accumulate arc-averaged prediction across sub-angles.
                y_pred_sum = None
                y_measured = None

                for ang in sub_angles:
                    projector.update_angle(ang)
                    points, y = projector(batch_num=batch_num, proj=proj_flat)
                    if points is None or y is None:
                        continue
                    if y_measured is None:
                        y_measured = y  # same pixels for every sub-angle

                    n_rays, ppr = points.shape[0], points.shape[1]
                    zero_mask = torch.all(points.view(-1, 3) == 0, dim=-1)
                    pts = points.view(-1, 3)[~zero_mask]
                    if pts.size(0) == 0:
                        continue

                    chunks = []
                    for p0 in range(0, pts.size(0), 5_000_000):
                        chunk = model(pts[p0:p0 + 5_000_000], float(timestep)).squeeze(0)
                        chunks.append(chunk)
                    atten = torch.cat(chunks)

                    processed = torch.zeros(
                        (n_rays * ppr, 1), dtype=torch.float32, device=device
                    )
                    src = (atten if atten.dim() == 2 else atten.unsqueeze(-1)).float()
                    processed[~zero_mask] = src
                    contrib = (
                        processed.view(n_rays, ppr).sum(dim=1)
                        * (projector.distances / geometry.max_distance_traveled)
                        / n_eval
                    )

                    y_pred_sum = contrib if y_pred_sum is None else y_pred_sum + contrib

                if y_pred_sum is None or y_measured is None:
                    continue

                total_sq += ((y_pred_sum - y_measured) ** 2).sum().item()
                total_n += y_measured.numel()

    if total_n == 0:
        return None
    return float((total_sq / total_n) ** 0.5)


def compute_arc_reproj_metrics(run_dir: Path, device) -> dict[int, float] | None:
    """
    Load model from run_dir, compute arc_reproj_rmse for each N in N_EVAL_LIST.
    Returns {n_eval: rmse} or None on failure.
    """
    import torch
    import nect
    from nect.config import get_cfg
    from nect.data import NeCTDataset
    from nect.sampling import Geometry

    model_path = run_dir / "model"

    try:
        config = get_cfg(model_path / "config.yaml")
    except Exception as e:
        print(f"    config load failed: {e}")
        return None

    if config.geometry is None or config.mode != "dynamic":
        print("    skipping: not a dynamic run or no geometry")
        return None

    if config.continous_scanning is not True:
        print("    skipping: not a continuous-scanning run")
        return None

    # Resolve points_per_ray (mirrors base_trainer logic)
    if isinstance(config.points_per_ray.end, str):
        if config.points_per_ray.end == "auto":
            config.points_per_ray.end = math.ceil(max(config.geometry.nDetector) * 1.5)
        elif "x" in config.points_per_ray.end:
            fac = float(config.points_per_ray.end.split("x")[0])
            config.points_per_ray.end = math.ceil(max(config.geometry.nDetector) * fac)
    points_per_ray = int(config.points_per_ray.end)

    if isinstance(config.points_per_batch, str):
        print("    points_per_batch unresolved — skipping")
        return None
    points_per_batch = int(config.points_per_batch)

    try:
        model = config.get_model()
        ckpt = torch.load(model_path / "checkpoints" / "last.ckpt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"    model load failed: {e}")
        return None

    try:
        dataset = NeCTDataset(config=config, device="cpu")
        geometry = Geometry.from_cfg(
            config.geometry,
            reconstruction_mode=config.reconstruction_mode,
            sample_outside=config.sample_outside,
        )
    except Exception as e:
        print(f"    dataset load failed: {e}")
        del model
        return None

    projector = nect.sampling.Projector(
        geometry=geometry,
        points_per_batch=points_per_batch // 2,
        points_per_ray=points_per_ray,
        device=device,
        uniform_ray_spacing=config.uniform_ray_spacing,
    )

    indices = list(range(0, len(dataset), EVAL_EVERY_N))
    print(f"    evaluating {len(indices)} / {len(dataset)} projections …")

    results: dict[int, float] = {}
    for n_eval in N_EVAL_LIST:
        print(f"      N_eval={n_eval} …", end="", flush=True)
        rmse = _compute_arc_rmse_at_n(model, dataset, projector, geometry, device, n_eval, indices)
        if rmse is not None:
            results[n_eval] = rmse
            print(f" RMSE={rmse:.6f}")
        else:
            print(" N/A")

    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results if results else None


# ── Caching ───────────────────────────────────────────────────────────────────

def _load_cached(run_dir: Path) -> dict[int, float] | None:
    cache_file = run_dir / "arc_reproj_metrics.txt"
    if not cache_file.exists():
        return None
    results: dict[int, float] = {}
    for line in cache_file.read_text().splitlines():
        m = re.match(r"N=(\d+)\s+RMSE=([\d.eE+\-]+)", line.strip())
        if m:
            results[int(m.group(1))] = float(m.group(2))
    return results if results else None


def _save_cached(run_dir: Path, label: str, results: dict[int, float]) -> None:
    lines = [f"arc_reproj_metrics — {label}", ""]
    for n, rmse in sorted(results.items()):
        lines.append(f"N={n:<4}  RMSE={rmse:.6f}")
    if 1 in results and N_EVAL_HIGH in results:
        ratio = results[N_EVAL_HIGH] / results[1]
        lines.append(f"ratio(N={N_EVAL_HIGH}/N=1) = {ratio:.4f}")
    lines.append("")
    (run_dir / "arc_reproj_metrics.txt").write_text("\n".join(lines))


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(rows: list[tuple[str, dict[int, float] | None]]) -> None:
    n_cols = sorted(N_EVAL_LIST)
    n_col_hdrs = "  ".join(f"RMSE(N={n})" for n in n_cols)
    ratio_hdr = f"ratio({N_EVAL_HIGH}/1)"
    hdr = f"{'run':<32}  {n_col_hdrs}  {ratio_hdr:>12}"
    sep = "-" * len(hdr)

    print("\n" + "=" * len(hdr))
    print("ARC REPROJ ABLATION SUMMARY")
    print(hdr)
    print(sep)

    for label, results in rows:
        if results is None:
            vals = "  ".join(f"{'N/A':>12}" for _ in n_cols)
            ratio_str = f"{'N/A':>12}"
        else:
            vals = "  ".join(
                f"{results[n]:.6f}" if n in results else f"{'N/A':>12}"
                for n in n_cols
            )
            if 1 in results and N_EVAL_HIGH in results:
                ratio = results[N_EVAL_HIGH] / results[1]
                flag = "  ← INSUFFICIENT" if ratio > 1.05 else ""
                ratio_str = f"{ratio:>12.4f}{flag}"
            else:
                ratio_str = f"{'N/A':>12}"
        print(f"{label:<32}  {vals}  {ratio_str}")

    print("=" * len(hdr))
    print()
    print("ratio interpretation:")
    print("  ≈ 1.0  →  accumulation_steps sufficient (blur model is correct)")
    print("  > 1.05 →  accumulation_steps too low; retrain with higher value")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch

    if not torch.cuda.is_available():
        print("No CUDA device found — Tier 3b requires GPU.  Exiting.")
        return

    device = torch.device("cuda")

    if MODEL_DIRS is not None:
        candidates = [(BASE_DIR / d, d) for d in MODEL_DIRS]
    else:
        candidates = []
        for sub in sorted(BASE_DIR.iterdir()):
            if sub.is_dir() and RUN_RE.match(sub.name):
                candidates.append((sub, sub.name))

    if not candidates:
        print(f"No matching run directories found under {BASE_DIR}")
        return

    print(f"Found {len(candidates)} run(s)")
    rows: list[tuple[str, dict[int, float] | None]] = []

    for run_dir, label in candidates:
        print(f"\n{'─'*60}")
        print(f"Run: {label}")

        if SKIP_CACHED:
            cached = _load_cached(run_dir)
            if cached is not None:
                print(f"    loaded from cache: {sorted(cached.items())}")
                rows.append((label, cached))
                continue

        results = compute_arc_reproj_metrics(run_dir, device)
        if results is not None:
            _save_cached(run_dir, label, results)
        rows.append((label, results))

    _print_summary(rows)


if __name__ == "__main__":
    old_stdout = sys.stdout
    tee = _Tee(old_stdout)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = old_stdout

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "arc_reproj_report.txt"
    out_path.write_text(tee.getvalue())
    print(f"Report saved to {out_path}")
