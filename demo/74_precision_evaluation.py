"""
Reconstruction precision evaluation — three tiers, no ground truth needed.

Tier 1  (no GPU, from sand_volume.npz):
    delta_mono  — fraction of steps where (V_top−V_bot) decreases  higher = better (max 1.0)
    delta_snr   — flow signal / noise: range(trend) / std(residual) higher = better

    Both use the DIFFERENTIAL signal V_top−V_bot, which cancels the common-mode
    total-volume oscillations that dominated the previous metrics.

Tier 2  (no GPU, from sand_volume.npz, ac ≥ 2 only):
    boundary_jump — mean |V(t_k+) − V(t_k−)| at each acquisition-period seam
                    lower = smoother reconstruction across acquisition boundaries

Tier 3a (no GPU, from epoch_losses.txt):
    final_loss  — last-epoch average training loss     lower = better

Tier 3b (GPU required):
    reproj_rmse — reprojection RMSE on a held-out subset of projection angles
                  lower = better fit to measured data

Outputs
    {run_dir}/precision_metrics.txt — per-run results (cached for Tier 3b)
    OUT_DIR/precision_report.txt    — summary table

Usage
    Set BASE_DIR to match your 70_ / 72_ config and run.
    On a CPU-only machine, Tier 3b is skipped automatically.
    On a GPU node, set SKIP_CACHED_REPROJ=True to reuse previous Tier 3b results.
"""

import gc
import math
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

OUT_DIR = Path(__file__).parent / "mae_plots"

PREFER_OLD = False

# Glitch-filter sigma (Tier 1 & 2)
FILTER_SIGMA = 30

# Tier 3b: stride when selecting which projection angles to evaluate
# 1 = all angles, 10 = every 10th angle (~10% sample, much faster)
EVAL_EVERY_N = 10

# Tier 3b: if precision_metrics.txt already contains a reproj_rmse, reuse it
SKIP_CACHED_REPROJ = True

# ─────────────────────────────────────────────────────────────────────────────

RUN_RE   = re.compile(r"^(\d+fps_\d+)_ac(\d+)(_OLD)?$")


# ── Shared glitch filter ──────────────────────────────────────────────────────

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


# ── Tier 1 + 2 ───────────────────────────────────────────────────────────────

def compute_volume_metrics(run_dir: Path, ac: int) -> dict | None:
    """Compute dimensionless quality metrics from existing sand_volume.npz."""
    npz_path = run_dir / "sand_volume.npz"
    if not npz_path.exists():
        return None

    data     = np.load(npz_path)
    top_raw  = data["top_volume_mm3"]
    bot_raw  = data["bottom_volume_mm3"]
    t_axis   = data["projection_indices"]

    _, top_bad, _ = _filter_glitches(top_raw, t_axis, FILTER_SIGMA)
    _, bot_bad, _ = _filter_glitches(bot_raw, t_axis, FILTER_SIGMA)
    keep = ~(top_bad | bot_bad)
    if keep.sum() < 10:
        return None

    top   = top_raw[keep]
    bot   = bot_raw[keep]

    # ── Tier 1: differential signal metrics ──────────────────────────────────
    # delta_V = V_top − V_bot cancels common-mode total-volume oscillations.
    # For a correct hourglass reconstruction delta_V decreases monotonically.

    delta_V = top - bot
    d_delta = np.diff(delta_V)

    delta_mono = float(np.mean(d_delta < 0))

    t_idx  = np.arange(len(delta_V), dtype=float)
    trend  = np.polyval(np.polyfit(t_idx, delta_V, 1), t_idx)
    signal_range = float(np.abs(trend[-1] - trend[0]))
    noise_std    = float(np.std(delta_V - trend))
    delta_snr    = signal_range / (noise_std + 1e-9)

    # ── Tier 2: acquisition-boundary jump ────────────────────────────────────
    boundary_jump = float("nan")
    if ac >= 2:
        N = len(top_raw)
        t_norm = np.linspace(0.0, 1.0, N, endpoint=False)
        t_kept = t_norm[keep]

        jumps = []
        for k in range(1, ac):
            bt = k / ac
            idx = int(np.argmin(np.abs(t_kept - bt)))
            if 0 < idx < len(top) - 1:
                j_top = abs(float(top[idx]) - float(top[idx - 1]))
                j_bot = abs(float(bot[idx]) - float(bot[idx - 1]))
                jumps.append((j_top + j_bot) / 2.0)
        if jumps:
            boundary_jump = float(np.mean(jumps))

    return {
        "delta_mono":    delta_mono,
        "delta_snr":     delta_snr,
        "boundary_jump": boundary_jump,
    }


# ── Tier 3a: final training loss ─────────────────────────────────────────────

def read_final_loss(run_dir: Path) -> float | None:
    """Read final-epoch average loss from epoch_losses.txt."""
    loss_file = run_dir / "model" / "epoch_losses.txt"
    if not loss_file.exists():
        return None
    lines = [l.strip() for l in loss_file.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    m = re.search(r"avg_loss=([\d.eE+\-]+)", lines[-1])
    return float(m.group(1)) if m else None


# ── Tier 3b: reprojection RMSE (GPU) ─────────────────────────────────────────

def compute_reproj_rmse(run_dir: Path, device) -> float | None:
    """
    Forward-project the trained model through a subset of projection angles and
    compute RMSE against the actual measured detector values.
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
        return None

    # Resolve points_per_ray.end (mirrors base_trainer logic)
    if isinstance(config.points_per_ray.end, str):
        if config.points_per_ray.end == "auto":
            config.points_per_ray.end = math.ceil(max(config.geometry.nDetector) * 1.5)
        elif "x" in config.points_per_ray.end:
            fac = float(config.points_per_ray.end.split("x")[0])
            config.points_per_ray.end = math.ceil(max(config.geometry.nDetector) * fac)
    points_per_ray = int(config.points_per_ray.end)

    if isinstance(config.points_per_batch, str):
        print(f"    points_per_batch unresolved — skipping Tier 3b")
        return None
    points_per_batch = int(config.points_per_batch)

    try:
        model = config.get_model()
        ckpt  = torch.load(model_path / "checkpoints" / "last.ckpt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"    model load failed: {e}")
        return None

    try:
        dataset  = NeCTDataset(config=config, device="cpu")
        geometry = Geometry.from_cfg(
            config.geometry,
            reconstruction_mode=config.reconstruction_mode,
            sample_outside=config.sample_outside,
        )
    except Exception as e:
        print(f"    dataset load failed: {e}")
        return None

    projector = nect.sampling.Projector(
        geometry=geometry,
        points_per_batch=points_per_batch // 2,
        points_per_ray=points_per_ray,
        device=device,
        uniform_ray_spacing=config.uniform_ray_spacing,
    )

    total_sq = 0.0
    total_n  = 0
    indices  = list(range(0, len(dataset), EVAL_EVERY_N))

    with torch.no_grad():
        for idx in indices:
            item = dataset[idx]
            proj, angle, timestep = item[0], item[1], item[-1]

            projector.update(
                angle=float(angle),
                detector_binning=1,
                points_per_ray=points_per_ray,
                random_offset_detector=0.0,
            )
            # Sequential pixel order so all pixels are covered exactly once
            projector.random_indexes = torch.arange(
                projector.total_detector_pixels, dtype=torch.int64, device=device
            )
            proj_flat = proj.flatten().to(device)

            for batch_num in range(projector.batch_per_epoch):
                points, y = projector(batch_num=batch_num, proj=proj_flat)
                if points is None or y is None:
                    continue

                n_rays, ppr = points.shape[0], points.shape[1]
                zero_mask = torch.all(points.view(-1, 3) == 0, dim=-1)
                pts_flat  = points.view(-1, 3)[~zero_mask]
                if pts_flat.size(0) == 0:
                    continue

                chunks = []
                for p0 in range(0, pts_flat.size(0), 5_000_000):
                    chunks.append(
                        model(pts_flat[p0:p0 + 5_000_000], float(timestep)).squeeze(0)
                    )
                atten = torch.cat(chunks)

                processed = torch.zeros(
                    (n_rays * ppr, 1), dtype=torch.float32, device=device
                )
                src = (atten if atten.dim() == 2 else atten.unsqueeze(-1)).float()
                processed[~zero_mask] = src
                y_pred = processed.view(n_rays, ppr).sum(dim=1) * (
                    projector.distances / geometry.max_distance_traveled
                )

                total_sq += ((y_pred - y) ** 2).sum().item()
                total_n  += len(y)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if total_n == 0:
        return None
    return float((total_sq / total_n) ** 0.5)


# ── Run discovery ─────────────────────────────────────────────────────────────

def find_runs(base: Path) -> list[tuple[Path, str, int, bool]]:
    """Return deduplicated list of (run_dir, group, ac, is_old)."""
    seen: dict[tuple[str, int], list] = defaultdict(list)
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        m = RUN_RE.match(sub.name)
        if not m:
            continue
        group, ac, old_sfx = m.group(1), int(m.group(2)), m.group(3)
        seen[(group, ac)].append((sub, group, ac, old_sfx is not None))

    results = []
    for entries in seen.values():
        if len(entries) == 1:
            results.append(entries[0])
        else:
            preferred = [e for e in entries if e[3] == PREFER_OLD]
            results.append(preferred[0] if preferred else entries[0])
    return sorted(results, key=lambda x: (x[1], x[2]))


# ── Output helpers ────────────────────────────────────────────────────────────

def _write_run_metrics(
    run_dir: Path, label: str,
    vol: dict | None, loss: float | None, rmse: float | None,
) -> None:
    lines = [f"Precision metrics — {label}", "=" * 52, ""]
    if vol:
        lines += [
            "Tier 1 — differential flow signal quality (no GPU):",
            f"  delta_mono    = {vol['delta_mono']:.4f}     higher = better  (max 1.0)",
            f"  delta_snr     = {vol['delta_snr']:.2f}       higher = better",
            "",
            "Tier 2 — inter-acquisition consistency (no GPU):",
        ]
        if not math.isnan(vol["boundary_jump"]):
            lines.append(f"  boundary_jump = {vol['boundary_jump']:.2f} mm³     lower  = better")
        else:
            lines.append("  boundary_jump = N/A (ac < 2)")
    else:
        lines.append("Tier 1/2 — sand_volume.npz not found")

    lines += [
        "",
        "Tier 3a — training fit (no GPU):",
        f"  final_loss    = {loss:.6f}   lower  = better" if loss is not None
        else "  final_loss    = N/A (epoch_losses.txt not found)",
        "",
        "Tier 3b — reprojection RMSE (GPU):",
        f"  reproj_rmse   = {rmse:.6f}   lower  = better" if rmse is not None
        else "  reproj_rmse   = N/A",
        "",
    ]
    (run_dir / "precision_metrics.txt").write_text("\n".join(lines))


def _print_summary(rows: list) -> None:
    hdr = (f"{'run':<30}  {'delta_mono':>10}  {'delta_snr':>9}"
           f"  {'bdry_jump':>10}  {'final_loss':>10}  {'reproj_rmse':>12}")
    print("\n" + "=" * len(hdr))
    print("PRECISION SUMMARY")
    print(hdr)
    print("-" * len(hdr))
    for label, vol, loss, rmse in rows:
        dm = f"{vol['delta_mono']:.4f}"      if vol                                             else "       N/A"
        ds = f"{vol['delta_snr']:.2f}"       if vol                                             else "      N/A"
        bj = f"{vol['boundary_jump']:.1f}"   if vol and not math.isnan(vol['boundary_jump'])    else "       N/A"
        fl = f"{loss:.5f}"                   if loss is not None                                else "       N/A"
        rr = f"{rmse:.5f}"                   if rmse is not None                                else "          N/A"
        print(f"{label:<30}  {dm:>10}  {ds:>9}  {bj:>10}  {fl:>10}  {rr:>12}")
    print("=" * len(hdr))
    print()
    print("Ideal: delta_mono↑  delta_snr↑  boundary_jump↓  final_loss↓  reproj_rmse↓")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    device = torch.device(0) if has_gpu else None
    print(f"Base dir     : {BASE_DIR}")
    print(f"GPU          : {'yes — Tier 3b enabled' if has_gpu else 'no — Tier 3b skipped'}")
    print(f"Filter sigma : {FILTER_SIGMA}   eval_every_n : {EVAL_EVERY_N}\n")

    runs = find_runs(BASE_DIR)
    if not runs:
        print("No matching run directories found.")
        return
    print(f"Found {len(runs)} run(s)\n")

    summary_rows = []
    for run_dir, group, ac, is_old in runs:
        label = f"{group}_ac{ac}{'_OLD' if is_old else ''}"
        print(f"── {label}")

        vol  = compute_volume_metrics(run_dir, ac)
        loss = read_final_loss(run_dir)

        rmse = None
        if has_gpu:
            # Try to read cached value first
            metrics_file = run_dir / "precision_metrics.txt"
            if SKIP_CACHED_REPROJ and metrics_file.exists():
                m = re.search(r"reproj_rmse\s*=\s*([\d.eE+\-]+)", metrics_file.read_text())
                if m:
                    rmse = float(m.group(1))
                    print(f"   Tier 3b: cached  reproj_rmse={rmse:.6f}")
            if rmse is None:
                print(f"   Tier 3b: computing reproj RMSE (every {EVAL_EVERY_N}th angle) …")
                rmse = compute_reproj_rmse(run_dir, device)
                if rmse is not None:
                    print(f"   Tier 3b: reproj_rmse={rmse:.6f}")
                else:
                    print(f"   Tier 3b: failed (see above)")

        if vol:
            bj_str = f"{vol['boundary_jump']:.1f}" if not math.isnan(vol['boundary_jump']) else "N/A"
            print(f"   Tier 1 : delta_mono={vol['delta_mono']:.4f}  delta_snr={vol['delta_snr']:.2f}"
                  f"  boundary_jump={bj_str}")
        if loss is not None:
            print(f"   Tier 3a: final_loss={loss:.6f}")

        _write_run_metrics(run_dir, label, vol, loss, rmse)
        summary_rows.append((label, vol, loss, rmse))

    _print_summary(summary_rows)


# ── Tee stdout → file ─────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, real_stdout):
        self._real = real_stdout
        self._buf  = StringIO()

    def write(self, s):
        self._real.write(s)
        self._buf.write(s)

    def flush(self):
        self._real.flush()

    def getvalue(self):
        return self._buf.getvalue()


if __name__ == "__main__":
    old_stdout = sys.stdout
    tee = _Tee(old_stdout)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = old_stdout
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "precision_report.txt"
    out_path.write_text(tee.getvalue())
    print(f"Report saved to {out_path}")
