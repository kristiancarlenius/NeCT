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
import json
import re
import sys
from pathlib import Path

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

def find_experiments(outputs_dir: Path) -> list[dict]:
    experiments = []
    for exp_dir in sorted(outputs_dir.glob("synthetic_ghost_deg*_ac*")):
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path,
                        default=Path("/cluster/home/kristiac/NeCT/outputs"))
    parser.add_argument("--gt-dir",      type=Path,
                        default=Path("/cluster/home/kristiac/NeCT/Datasets/synthetic_ghost"))
    parser.add_argument("--out-dir",     type=Path,
                        default=ROOT / "results" / "synthetic_ghost_eval")
    args = parser.parse_args()

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
