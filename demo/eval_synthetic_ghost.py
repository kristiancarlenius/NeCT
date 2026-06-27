"""
Evaluate synthetic ghost reconstructions against ground truth.

For each outputs/synthetic_ghost_deg{dd}_ac{aa}/ experiment:
  - Load the trained model from last.ckpt
  - Reconstruct volumes at all 11 GT timesteps
  - Compute PSNR / SSIM / MSE over the full volume
  - Compute target-region metrics: PSNR / MSE restricted to voxels around the
    static target sphere (the region where ghost artifacts appear)
  - Save summary to results/synthetic_ghost_eval.csv and .json

The target sphere is static at (x=0.5, y=0.35, z=0.5), radius=0.06.
The target mask includes a margin so ghost copies at nearby positions are captured.
The key metric is target_psnr: if K-step accumulation helps, this improves
because the obstructor's blocking transition is modelled correctly.

Usage:
  python -m demo.eval_synthetic_ghost
  python -m demo.eval_synthetic_ghost --outputs-dir /cluster/home/kristiac/NeCT/outputs
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path(__file__).parent.parent

# Target sphere centre and mask bounds (must match make_synthetic_ghost.py)
TARGET_X   = 0.50
TARGET_Y   = 0.35
TARGET_Z   = 0.50
MASK_MARGIN = 0.15   # extra margin beyond target radius to catch ghost copies

TARGET_X_LO = TARGET_X - MASK_MARGIN
TARGET_X_HI = TARGET_X + MASK_MARGIN
TARGET_Y_LO = TARGET_Y - MASK_MARGIN
TARGET_Y_HI = TARGET_Y + MASK_MARGIN
TARGET_Z_LO = TARGET_Z - MASK_MARGIN
TARGET_Z_HI = TARGET_Z + MASK_MARGIN

N_VOX    = 128
DATA_RANGE = 0.50   # max attenuation (obstructor=0.45 + headroom)


# ── Target mask (computed once) ───────────────────────────────────────────────

def _build_target_mask() -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.linspace(0, 1, N_VOX),
        np.linspace(0, 1, N_VOX),
        np.linspace(0, 1, N_VOX),
        indexing="ij",
    )
    return (
        (xx >= TARGET_X_LO) & (xx <= TARGET_X_HI)
        & (yy >= TARGET_Y_LO) & (yy <= TARGET_Y_HI)
        & (zz >= TARGET_Z_LO) & (zz <= TARGET_Z_HI)
    )

TARGET_MASK = _build_target_mask()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(recon: np.ndarray, gt: np.ndarray) -> dict:
    mse  = float(np.mean((recon - gt) ** 2))
    psnr = float(10.0 * np.log10(DATA_RANGE ** 2 / mse)) if mse > 0 else 100.0
    ssim = float(ssim_fn(gt, recon, data_range=DATA_RANGE))

    tgt_mse  = float(np.mean((recon[TARGET_MASK] - gt[TARGET_MASK]) ** 2))
    tgt_psnr = float(10.0 * np.log10(DATA_RANGE ** 2 / tgt_mse)) if tgt_mse > 0 else 100.0

    return {
        "psnr": psnr, "mse": mse, "ssim": ssim,
        "target_psnr": tgt_psnr, "target_mse": tgt_mse,
    }


# ── Experiment discovery ──────────────────────────────────────────────────────

def _find_config_near_ckpt(ckpt_path: Path) -> Path | None:
    """Walk up from the checkpoint file looking for config.yaml in the run directory."""
    for parent in ckpt_path.parents:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
        # Stop after leaving the run-timestamp directory (don't walk all the way to outputs/).
        if parent.name.startswith("synthetic_ghost_"):
            break
    return None


def find_experiments(outputs_dir: Path) -> list[dict]:
    experiments = []
    for exp_dir in sorted(outputs_dir.glob("synthetic_ghost_deg*_ac*")):
        if not exp_dir.is_dir():
            continue
        m = re.search(r"deg(\d+)_ac(\d+)", exp_dir.name)
        if not m:
            continue
        deg, ac = int(m.group(1)), int(m.group(2))

        # Pick the most recently modified last.ckpt anywhere under this experiment dir.
        ckpt_candidates = sorted(exp_dir.rglob("last.ckpt"), key=lambda p: p.stat().st_mtime)
        ckpt_path = ckpt_candidates[-1] if ckpt_candidates else None

        # Find config.yaml co-located with the chosen checkpoint (walk up its directory tree).
        config_path = _find_config_near_ckpt(ckpt_path) if ckpt_path is not None else None

        if config_path is None or ckpt_path is None:
            missing = []
            if config_path is None: missing.append("config.yaml")
            if ckpt_path  is None:  missing.append("last.ckpt")
            print(f"  [skip] {exp_dir.name}: missing {', '.join(missing)}", flush=True)
            continue

        experiments.append({
            "exp_dir": exp_dir, "deg": deg, "ac": ac,
            "config_path": config_path, "ckpt_path": ckpt_path,
        })

    return experiments


# ── Model loading ─────────────────────────────────────────────────────────────

def load_trainer(config_path: Path, ckpt_path: Path):
    from nect.config import get_cfg
    from nect.trainers.continous_scanning_trainer_batch import ContinousScanningTrainerBatch

    config  = get_cfg(str(config_path))
    trainer = ContinousScanningTrainerBatch(
        config=config,
        output_directory=None,
        save_ckpt=False,
        save_last=False,
        log=False,
        verbose=False,
    )
    # Use fabric.load() — the same API used by the training code for checkpoint
    # resumption, ensuring the state-dict format matches what fabric.save() wrote.
    checkpoint_data = trainer.fabric.load(str(ckpt_path))
    state_dict = checkpoint_data["model"]

    # Snapshot a parameter before load to verify the checkpoint actually changes
    # the weights (catches the case where load_state_dict silently no-ops).
    sample_key = next(iter(state_dict))
    before = trainer.model.state_dict()[sample_key].detach().cpu().clone()

    missing, unexpected = trainer.model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    [WARN] missing keys in ckpt ({len(missing)}): {missing[:3]}", flush=True)
    if unexpected:
        print(f"    [WARN] unexpected keys in ckpt ({len(unexpected)}): {unexpected[:3]}", flush=True)

    after = trainer.model.state_dict()[sample_key].detach().cpu()
    if torch.allclose(before, after):
        print("    [WARN] model weights did NOT change after load_state_dict — "
              "checkpoint may not have loaded correctly!", flush=True)
    else:
        print(f"    Checkpoint loaded: {len(state_dict)} param tensors", flush=True)

    trainer.model.eval()
    return trainer


# ── Slice visualisation ───────────────────────────────────────────────────────

# XY at z=0.5: the only view where target (y=0.35), obstructor (y=0.65) and
# reference spheres (y=0.5) are all visible simultaneously (all at z=0.5).
PLOT_TYPE = "XY"


def _build_xy_grid(trainer, device):
    """Return (grid, slice_shape) for the XY slice at z=0.5."""
    nVoxel = list(trainer.config.geometry.nVoxel)
    rm = trainer.config.sample_outside
    sample_size = [nVoxel[0], nVoxel[1] + 2 * rm, nVoxel[2] + 2 * rm]

    z, y, x = torch.meshgrid(
        [
            torch.tensor([0.5]),   # 1-D tensor, shape (1,)
            torch.linspace(0, 1, steps=sample_size[1])[slice(rm, -rm) if rm > 0 else slice(None)],
            torch.linspace(0, 1, steps=sample_size[2])[slice(rm, -rm) if rm > 0 else slice(None)],
        ],
        indexing="ij",
    )
    grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t().to(device)
    return grid, list(z.shape)


def save_training_style_images(
    exp: dict,
    trainer,
    gt_volumes: np.ndarray,
    gt_timesteps: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Save comparison images using XY slice at z=0.5.

    XY at z=0.5 shows all objects: target sphere (y=0.35), moving obstructor
    (y=0.65) and both reference spheres (y=0.5) — matching the training's
    generate_image() normalization (vmin=dataset.min, vmax=99th-percentile).

    Row 0 (top): difference from t=0
    Row 1 (bot): normalised attenuation
    Columns: t=0.25, t=0.50, t=0.75
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    grid, slice_shape = _build_xy_grid(trainer, trainer.fabric.device)

    max_dist   = trainer.geometry.max_distance_traveled
    dset_min   = trainer.dataset.minimum.item()
    dset_range = trainer.dataset.maximum.item() - dset_min

    eval_times = [0.25, 0.50, 0.75]

    # Pre-compute all reconstruction slices so we can derive a shared adaptive
    # colour scale (same logic as save_model_gif).
    with torch.no_grad():
        avg_raw = (trainer.model(grid, 0.0)
                   .squeeze().reshape(slice_shape).squeeze()
                   .detach().cpu().numpy())
        raws = []
        for t_val in eval_times:
            raws.append(
                trainer.model(grid, float(t_val))
                .squeeze().reshape(slice_shape).squeeze()
                .detach().cpu().numpy()
            )

    recon_norms = [r / max_dist * dset_range + dset_min for r in raws]

    all_recon_vals = np.concatenate([n.flatten() for n in recon_norms])
    print(f"    [diag] raw model output at t=0.0  min={avg_raw.min():.4f}  "
          f"max={avg_raw.max():.4f}  mean={avg_raw.mean():.4f}", flush=True)
    print(f"    [diag] normalised recon (all t)  min={all_recon_vals.min():.4f}  "
          f"max={all_recon_vals.max():.4f}  "
          f"p99={float(np.percentile(all_recon_vals, 99)):.4f}  "
          f"(DATA_RANGE={DATA_RANGE})", flush=True)

    # Adaptive vmin/vmax: same logic as save_model_gif so images and GIF match.
    vmin_r = max(0.0, float(np.percentile(all_recon_vals, 1)))
    vmax_r_adapt = float(np.percentile(all_recon_vals, 99))
    vmax_r = vmax_r_adapt if vmax_r_adapt > 1e-6 else DATA_RANGE

    # GT z=0.5 slice (all objects visible here)
    z_mid = gt_volumes[0].shape[0] // 2
    gt_t0_sl = gt_volumes[0][z_mid, :, :]

    fig_r, axes_r = plt.subplots(2, 3, figsize=(24, 10))
    fig_r.suptitle(
        f"Reconstruction  deg={exp['deg']:02d}  ac={exp['ac']:02d}  —  XY (z=0.5)"
        f"  [vmax={vmax_r:.3f}]",
        fontsize=13,
    )
    fig_g, axes_g = plt.subplots(2, 3, figsize=(24, 10))
    fig_g.suptitle(
        f"Ground truth  deg={exp['deg']:02d}  ac={exp['ac']:02d}  —  XY (z=0.5)",
        fontsize=13,
    )

    for col, (t_val, raw, recon_norm) in enumerate(zip(eval_times, raws, recon_norms)):
        axes_r[0, col].imshow(raw - avg_raw, cmap="gray", interpolation="none")
        axes_r[0, col].set_title(f"t={t_val:.2f}  (diff from t=0)", fontsize=9)
        axes_r[0, col].axis("off")

        axes_r[1, col].imshow(recon_norm, cmap="gray", interpolation="none",
                               vmin=vmin_r, vmax=vmax_r)
        axes_r[1, col].set_title(f"t={t_val:.2f}  (attenuation)", fontsize=9)
        axes_r[1, col].axis("off")

        t_idx  = int(np.argmin(np.abs(gt_timesteps - t_val)))
        gt_sl  = gt_volumes[t_idx][z_mid, :, :]
        t_real = float(gt_timesteps[t_idx])

        axes_g[0, col].imshow(gt_sl - gt_t0_sl, cmap="gray", interpolation="none")
        axes_g[0, col].set_title(f"GT t={t_real:.2f}  (diff from t=0)", fontsize=9)
        axes_g[0, col].axis("off")

        axes_g[1, col].imshow(gt_sl, cmap="gray", vmin=0.0, vmax=DATA_RANGE,
                               interpolation="none")
        axes_g[1, col].set_title(f"GT t={t_real:.2f}  (attenuation)", fontsize=9)
        axes_g[1, col].axis("off")

    for fig in (fig_r, fig_g):
        fig.tight_layout()

    prefix = f"deg{exp['deg']:02d}_ac{exp['ac']:02d}_XY"
    fig_r.savefig(out_dir / f"{prefix}_recon.png", dpi=300)
    fig_g.savefig(out_dir / f"{prefix}_gt.png",   dpi=300)
    plt.close(fig_r)
    plt.close(fig_g)
    print(f"    images → {prefix}_recon.png  +  {prefix}_gt.png", flush=True)


# ── GIF generation ────────────────────────────────────────────────────────────

def save_model_gif(
    exp: dict,
    trainer,
    out_dir: Path,
    n_frames: int = 100,
    fps: int = 10,
) -> None:
    """
    Save a GIF of the XY (z=0.5) reconstruction across n_frames timesteps.

    Each frame shows the normalised attenuation slice at a different t ∈ [0, 1].
    """
    from matplotlib.animation import FuncAnimation

    out_dir.mkdir(parents=True, exist_ok=True)

    grid, slice_shape = _build_xy_grid(trainer, trainer.fabric.device)

    max_dist   = trainer.geometry.max_distance_traveled
    dset_min   = trainer.dataset.minimum.item()
    dset_range = trainer.dataset.maximum.item() - dset_min

    timesteps = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)

    frames = []
    with torch.no_grad():
        for t in timesteps:
            raw = (trainer.model(grid, float(t))
                   .squeeze().reshape(slice_shape).squeeze()
                   .detach().cpu().numpy())
            recon = raw / max_dist * dset_range + dset_min
            frames.append(recon)

    all_vals = np.concatenate([f.flatten() for f in frames])
    print(f"    [GIF diag] recon min={all_vals.min():.4f}  max={all_vals.max():.4f}  "
          f"p99={float(np.percentile(all_vals, 99)):.4f}", flush=True)

    # Adaptive colour scale: show the real signal range.
    # Fall back to [0, DATA_RANGE] if the reconstruction is clearly in the
    # correct physical attenuation range (values reach at least 10 % of DATA_RANGE).
    vmin = max(0.0, float(np.percentile(all_vals, 1)))
    vmax_adaptive = float(np.percentile(all_vals, 99))
    if vmax_adaptive > 0.1 * DATA_RANGE:
        vmax = vmax_adaptive
    else:
        # Model outputs are tiny — still show something meaningful.
        vmax = vmax_adaptive if vmax_adaptive > 1e-6 else DATA_RANGE

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    im = ax.imshow(frames[0], cmap="gray", vmin=vmin, vmax=vmax,
                   interpolation="none", origin="upper")
    title = ax.set_title(
        f"deg={exp['deg']:02d} ac={exp['ac']:02d}  t={timesteps[0]:.3f}  [XY z=0.5]",
        fontsize=11,
    )
    fig.tight_layout()

    def _update(i):
        im.set_data(frames[i])
        title.set_text(
            f"deg={exp['deg']:02d} ac={exp['ac']:02d}  t={timesteps[i]:.3f}  [XY z=0.5]"
        )
        return im, title

    ani = FuncAnimation(fig, _update, frames=n_frames, interval=1000 // fps, repeat=True)

    prefix   = f"deg{exp['deg']:02d}_ac{exp['ac']:02d}"
    gif_path = out_dir / f"{prefix}_XY_video.gif"
    ani.save(str(gif_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"    GIF  → {gif_path}", flush=True)


def save_gt_gif(
    gt_volumes: np.ndarray,
    gt_timesteps: np.ndarray,
    out_dir: Path,
    n_frames: int = 100,
    fps: int = 10,
) -> None:
    """
    Save a single GT GIF of the XY (z=0.5) slice across n_frames timesteps.

    Nearest-neighbour selection from the available GT volumes (typically 11).
    Uses fixed [0, DATA_RANGE] scale so it is directly comparable to GT images.
    Saved once per run as gt_XY_video.gif.
    """
    from matplotlib.animation import FuncAnimation

    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "gt_XY_video.gif"
    if gif_path.exists():
        print(f"GT GIF already exists → {gif_path}", flush=True)
        return

    z_mid = gt_volumes.shape[1] // 2  # gt_volumes: (N_GT, D, H, W)
    timesteps = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)

    frames = []
    for t in timesteps:
        idx = int(np.argmin(np.abs(gt_timesteps - float(t))))
        frames.append(gt_volumes[idx][z_mid, :, :])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    im = ax.imshow(frames[0], cmap="gray", vmin=0.0, vmax=DATA_RANGE,
                   interpolation="none", origin="upper")
    title = ax.set_title(f"GT  t={timesteps[0]:.3f}  [XY z=0.5]", fontsize=11)
    fig.tight_layout()

    def _update(i):
        im.set_data(frames[i])
        title.set_text(f"GT  t={timesteps[i]:.3f}  [XY z=0.5]")
        return im, title

    ani = FuncAnimation(fig, _update, frames=n_frames, interval=1000 // fps, repeat=True)
    ani.save(str(gif_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"GT GIF  → {gif_path}", flush=True)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_experiment(exp: dict, gt_volumes: np.ndarray,
                        gt_timesteps: np.ndarray,
                        slices_dir: Path | None = None,
                        gifs_dir: Path | None = None,
                        gif_frames: int = 100) -> dict | None:
    print(f"  deg={exp['deg']:2d} ac={exp['ac']:2d}  ckpt={exp['ckpt_path']}", flush=True)

    try:
        trainer = load_trainer(exp["config_path"], exp["ckpt_path"])
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"    [ERROR loading model]: {e}", flush=True)
        return None

    per_timestep = []
    try:
        for i, t in enumerate(gt_timesteps):
            with torch.no_grad():
                vol = trainer.create_volume(timestep=float(t), save=False, cpu=True)
            if vol is None:
                print(f"    [WARN] create_volume returned None at t={t:.1f}", flush=True)
                continue

            # create_volume already denormalises to physical attenuation units.
            recon = vol.float().cpu().numpy()
            gt    = gt_volumes[i]

            if i == 0:
                print(f"    [diag] create_volume t={t:.2f}: "
                      f"recon [{recon.min():.4f}, {recon.max():.4f}]  "
                      f"gt [{gt.min():.4f}, {gt.max():.4f}]", flush=True)

            m     = compute_metrics(recon, gt)
            m["timestep"] = float(t)
            per_timestep.append(m)

        # Generate comparison images while the trainer / model are still loaded.
        if slices_dir is not None:
            save_training_style_images(exp, trainer, gt_volumes, gt_timesteps, slices_dir)

        # Generate GIF while the trainer / model are still loaded.
        if gifs_dir is not None:
            save_model_gif(exp, trainer, gifs_dir, n_frames=gif_frames)

    finally:
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

    if not per_timestep:
        return None

    keys  = ("psnr", "mse", "ssim", "target_psnr", "target_mse")
    means = {k: float(np.mean([p[k] for p in per_timestep])) for k in keys}
    stds  = {f"{k}_std": float(np.std([p[k] for p in per_timestep])) for k in keys}

    print(
        f"    PSNR={means['psnr']:.2f}  SSIM={means['ssim']:.4f}  "
        f"target_PSNR={means['target_psnr']:.2f}",
        flush=True,
    )
    return {"deg": exp["deg"], "ac": exp["ac"],
            **means, **stds, "per_timestep": per_timestep}


# ── Output ────────────────────────────────────────────────────────────────────

def save_results(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "synthetic_ghost_eval.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved → {json_path}")

    csv_path  = out_dir / "synthetic_ghost_eval.csv"
    flat_keys = ["deg", "ac",
                 "psnr", "psnr_std", "ssim", "ssim_std", "mse", "mse_std",
                 "target_psnr", "target_psnr_std", "target_mse", "target_mse_std"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in flat_keys})
    print(f"CSV  saved → {csv_path}")


def print_table(results: list[dict]) -> None:
    sorted_results = sorted(results, key=lambda r: (r["deg"], r["ac"]))
    header = (
        f"{'deg':>5} {'ac':>4}  "
        f"{'PSNR':>7} {'±':>5}  "
        f"{'SSIM':>6} {'±':>5}  "
        f"{'tgtPSNR':>9} {'±':>5}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in sorted_results:
        print(
            f"{r['deg']:>5} {r['ac']:>4}  "
            f"{r['psnr']:>7.2f} {r['psnr_std']:>5.2f}  "
            f"{r['ssim']:>6.4f} {r['ssim_std']:>5.4f}  "
            f"{r['target_psnr']:>9.2f} {r['target_psnr_std']:>5.2f}"
        )
    print("=" * len(header))


# ── GPU selection ─────────────────────────────────────────────────────────────

def _gpu_free_mb() -> list[tuple[int, float, float]]:
    """Return [(gpu_idx, free_mb, total_mb), ...] via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        rows = []
        for i, line in enumerate(out.strip().splitlines()):
            free_s, total_s = line.split(",")
            rows.append((i, float(free_s.strip()), float(total_s.strip())))
        return rows
    except Exception:
        return []


def select_gpu(min_free_mb: float = 3000.0) -> int | None:
    """Pick the GPU with the most free memory (>= min_free_mb). Return None if none qualify."""
    rows = _gpu_free_mb()
    if not rows:
        return None
    print("GPU memory (nvidia-smi):")
    for idx, free, total in rows:
        print(f"  GPU {idx}: {free/1024:.1f} GB free / {total/1024:.1f} GB total")
    best = max(rows, key=lambda r: r[1])
    if best[1] < min_free_mb:
        print(f"  WARNING: no GPU has >= {min_free_mb/1024:.1f} GB free — model needs ~2 GB")
        return None
    print(f"  → selecting GPU {best[0]} ({best[1]/1024:.1f} GB free)")
    return best[0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path,
                        default=Path("/cluster/home/kristiac/NeCT/outputs"))
    parser.add_argument("--gt-dir",      type=Path,
                        default=Path("/cluster/home/kristiac/NeCT/Datasets/synthetic_ghost"))
    parser.add_argument("--out-dir",     type=Path,
                        default=ROOT / "results" / "synthetic_ghost_eval")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (default: auto-select by free memory)")
    parser.add_argument("--gif", action="store_true",
                        help="Generate 100-frame GIF for each model (XY z=0.5, t=0..1)")
    parser.add_argument("--gif-frames", type=int, default=100,
                        help="Number of timestep frames in each GIF (default: 100)")
    args = parser.parse_args()

    # Pin to a single GPU before fabric/TCNN initialise CUDA.
    # TCNN allocates hash-grid memory directly on GPU during model __init__,
    # so CUDA_VISIBLE_DEVICES must be set before any trainer is constructed.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU {args.gpu} (--gpu flag)")
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu_idx = select_gpu(min_free_mb=3000.0)
        if gpu_idx is None:
            sys.exit("ERROR: no GPU with >= 3 GB free — use --gpu N or free up GPU memory")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    # Smoke-test: verify TCNN can allocate a tiny hash grid before running 24 experiments.
    print("Verifying TCNN / CUDA ...", flush=True)
    try:
        import tinycudann as tcnn
        _t = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 4,
                               "n_features_per_level": 2, "log2_hashmap_size": 16,
                               "base_resolution": 16, "per_level_scale": 1.5})
        del _t
        torch.cuda.empty_cache()
        print("  TCNN OK", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit("ERROR: TCNN / CUDA smoke-test failed — see traceback above.\n"
                 "Try: rm -rf ~/.cache/torch_extensions  then re-run.\n"
                 "If that fails, the installed TCNN may not support CUDA 12.9.")

    gt_volumes_path   = args.gt_dir / "gt_volumes.npy"
    gt_timesteps_path = args.gt_dir / "gt_timesteps.npy"
    if not gt_volumes_path.exists():
        sys.exit(f"ERROR: GT volumes not found at {gt_volumes_path}")

    gt_volumes   = np.load(gt_volumes_path).astype(np.float32)
    gt_timesteps = np.load(gt_timesteps_path).astype(np.float32)
    print(f"GT volumes: {gt_volumes.shape}  t={gt_timesteps.tolist()}")
    print(f"Target mask: {TARGET_MASK.sum()} voxels  "
          f"(x∈[{TARGET_X_LO:.2f},{TARGET_X_HI:.2f}], "
          f"y∈[{TARGET_Y_LO:.2f},{TARGET_Y_HI:.2f}], "
          f"z∈[{TARGET_Z_LO:.2f},{TARGET_Z_HI:.2f}])")

    experiments = find_experiments(args.outputs_dir)
    if not experiments:
        sys.exit(f"ERROR: no synthetic_ghost experiments found under {args.outputs_dir}")
    print(f"\nFound {len(experiments)} experiments\n")

    slices_dir = args.out_dir / "slices"
    gifs_dir   = (args.out_dir / "gifs") if args.gif else None

    if gifs_dir is not None:
        print(f"GIF output → {gifs_dir}  ({args.gif_frames} frames per model)", flush=True)
        save_gt_gif(gt_volumes, gt_timesteps, gifs_dir, n_frames=args.gif_frames)

    results = []
    for exp in experiments:
        result = evaluate_experiment(
            exp, gt_volumes, gt_timesteps,
            slices_dir=slices_dir,
            gifs_dir=gifs_dir,
            gif_frames=args.gif_frames,
        )
        if result is not None:
            results.append(result)

    if not results:
        sys.exit("ERROR: all experiments failed to evaluate")

    print_table(results)
    save_results(results, args.out_dir)


if __name__ == "__main__":
    main()
