"""
Evaluate synthetic slab reconstructions against ground truth.

For each outputs/synthetic_slab_deg{dd}_ac{aa}/ experiment:
  - Load the trained model from last.ckpt
  - Reconstruct volumes at all 11 GT timesteps
  - Compute PSNR / SSIM / MSE over the full volume
  - Compute slab-region metrics: PSNR / MSE restricted to slab voxels
    (the slab is where angular blur matters; this is the key metric)
  - Save summary to results/synthetic_slab_eval.csv and .json

The slab occupies y ∈ [0.48, 0.52] (thin in Y, extends in XZ).
Its presence in the reconstruction depends on whether the K-step forward model
correctly captures the sharp sinogram spike near the edge-on angle.

Usage:
  python -m demo.eval_synthetic_slab
  python -m demo.eval_synthetic_slab --outputs-dir /cluster/home/kristiac/NeCT/outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path(__file__).parent.parent

# Slab geometry in [0,1]³ normalized coords (must match make_synthetic_slab.py)
SLAB_Y_LO   = 0.48
SLAB_Y_HI   = 0.52
SLAB_X_LO   = 0.10
SLAB_X_HI   = 0.90
SLAB_Z_LO   = 0.10
SLAB_Z_HI   = 0.90
N_VOX       = 128

DATA_RANGE  = 0.20   # max attenuation in slab phantom


# ── Slab mask (computed once) ─────────────────────────────────────────────────

def _build_slab_mask() -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.linspace(0, 1, N_VOX),
        np.linspace(0, 1, N_VOX),
        np.linspace(0, 1, N_VOX),
        indexing="ij",
    )
    return (
        (yy >= SLAB_Y_LO) & (yy <= SLAB_Y_HI)
        & (xx >= SLAB_X_LO) & (xx <= SLAB_X_HI)
        & (zz >= SLAB_Z_LO) & (zz <= SLAB_Z_HI)
    )

SLAB_MASK = _build_slab_mask()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(recon: np.ndarray, gt: np.ndarray) -> dict:
    """Full-volume PSNR, MSE, SSIM; plus slab-region MSE / PSNR."""
    mse  = float(np.mean((recon - gt) ** 2))
    psnr = float(10.0 * np.log10(DATA_RANGE ** 2 / mse)) if mse > 0 else 100.0
    ssim = float(ssim_fn(gt, recon, data_range=DATA_RANGE))

    slab_mse  = float(np.mean((recon[SLAB_MASK] - gt[SLAB_MASK]) ** 2))
    slab_psnr = float(10.0 * np.log10(DATA_RANGE ** 2 / slab_mse)) if slab_mse > 0 else 100.0

    return {
        "psnr": psnr, "mse": mse, "ssim": ssim,
        "slab_psnr": slab_psnr, "slab_mse": slab_mse,
    }


# ── Experiment discovery ──────────────────────────────────────────────────────

def find_experiments(outputs_dir: Path) -> list[dict]:
    experiments = []
    for exp_dir in sorted(outputs_dir.glob("synthetic_slab_deg*_ac*")):
        if not exp_dir.is_dir():
            continue
        m = re.search(r"deg(\d+)_ac(\d+)", exp_dir.name)
        if not m:
            continue
        deg, ac = int(m.group(1)), int(m.group(2))

        config_path = None
        for candidate in [exp_dir / "model" / "config.yaml", exp_dir / "config.yaml"]:
            if candidate.exists():
                config_path = candidate
                break

        ckpt_candidates = sorted(exp_dir.rglob("last.ckpt"), key=lambda p: p.stat().st_mtime)
        ckpt_path = ckpt_candidates[-1] if ckpt_candidates else None

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
    checkpoint_data = trainer.fabric.load(str(ckpt_path))
    trainer.model.load_state_dict(checkpoint_data["model"])
    trainer.model.eval()
    return trainer


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_experiment(exp: dict, gt_volumes: np.ndarray,
                        gt_timesteps: np.ndarray) -> dict | None:
    print(f"  deg={exp['deg']:2d} ac={exp['ac']:2d}  ckpt={exp['ckpt_path']}", flush=True)

    try:
        trainer = load_trainer(exp["config_path"], exp["ckpt_path"])
    except Exception as e:
        print(f"    [ERROR loading model]: {e}", flush=True)
        return None

    per_timestep = []
    for i, t in enumerate(gt_timesteps):
        with torch.no_grad():
            vol = trainer.create_volume(timestep=float(t), save=False, cpu=True)
        if vol is None:
            print(f"    [WARN] create_volume returned None at t={t:.1f}", flush=True)
            continue

        recon = vol.float().cpu().numpy()
        gt    = gt_volumes[i]
        m     = compute_metrics(recon, gt)
        m["timestep"] = float(t)
        per_timestep.append(m)

    if not per_timestep:
        return None

    keys  = ("psnr", "mse", "ssim", "slab_psnr", "slab_mse")
    means = {k: float(np.mean([p[k] for p in per_timestep])) for k in keys}
    stds  = {f"{k}_std": float(np.std([p[k] for p in per_timestep])) for k in keys}

    print(
        f"    PSNR={means['psnr']:.2f}  SSIM={means['ssim']:.4f}  "
        f"slab_PSNR={means['slab_psnr']:.2f}",
        flush=True,
    )
    return {"deg": exp["deg"], "ac": exp["ac"],
            **means, **stds, "per_timestep": per_timestep}


# ── Output ────────────────────────────────────────────────────────────────────

def save_results(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "synthetic_slab_eval.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved → {json_path}")

    csv_path  = out_dir / "synthetic_slab_eval.csv"
    flat_keys = ["deg", "ac",
                 "psnr", "psnr_std", "ssim", "ssim_std", "mse", "mse_std",
                 "slab_psnr", "slab_psnr_std", "slab_mse", "slab_mse_std"]
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
        f"{'slabPSNR':>9} {'±':>5}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in sorted_results:
        print(
            f"{r['deg']:>5} {r['ac']:>4}  "
            f"{r['psnr']:>7.2f} {r['psnr_std']:>5.2f}  "
            f"{r['ssim']:>6.4f} {r['ssim_std']:>5.4f}  "
            f"{r['slab_psnr']:>9.2f} {r['slab_psnr_std']:>5.2f}"
        )
    print("=" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path,
                        default=Path("/cluster/home/kristiac/NeCT/outputs"))
    parser.add_argument("--gt-dir",      type=Path,
                        default=Path("/cluster/home/kristiac/NeCT/Datasets/synthetic_slab"))
    parser.add_argument("--out-dir",     type=Path,
                        default=ROOT / "results" / "synthetic_slab_eval")
    args = parser.parse_args()

    gt_volumes_path   = args.gt_dir / "gt_volumes.npy"
    gt_timesteps_path = args.gt_dir / "gt_timesteps.npy"
    if not gt_volumes_path.exists():
        sys.exit(f"ERROR: GT volumes not found at {gt_volumes_path}")

    gt_volumes   = np.load(gt_volumes_path).astype(np.float32)
    gt_timesteps = np.load(gt_timesteps_path).astype(np.float32)
    print(f"GT volumes: {gt_volumes.shape}  t={gt_timesteps.tolist()}")
    print(f"Slab mask: {SLAB_MASK.sum()} voxels  "
          f"(y∈[{SLAB_Y_LO},{SLAB_Y_HI}], x∈[{SLAB_X_LO},{SLAB_X_HI}], "
          f"z∈[{SLAB_Z_LO},{SLAB_Z_HI}])")

    experiments = find_experiments(args.outputs_dir)
    if not experiments:
        sys.exit(f"ERROR: no synthetic_slab experiments found under {args.outputs_dir}")
    print(f"\nFound {len(experiments)} experiments\n")

    results = []
    for exp in experiments:
        result = evaluate_experiment(exp, gt_volumes, gt_timesteps)
        if result is not None:
            results.append(result)

    if not results:
        sys.exit("ERROR: all experiments failed to evaluate")

    print_table(results)
    save_results(results, args.out_dir)


if __name__ == "__main__":
    main()
