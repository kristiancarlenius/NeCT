"""
Evaluate synthetic bead reconstructions against ground truth.

For each outputs/synthetic_bead_deg{dd}_ac{aa}/ experiment:
  - Load the trained model from last.ckpt
  - Reconstruct volumes at all 11 GT timesteps
  - Compute PSNR / SSIM / MSE vs ground truth
  - Save summary to results/synthetic_bead_eval.csv and .json

GT volumes shape: (11, 128, 128, 128), timesteps [0.0, 0.1, ..., 1.0]
The model outputs attenuation values in the same units as the GT phantom.

Usage (run from the NeCT root directory on the cluster):
  python -m demo.eval_synthetic_bead
  python -m demo.eval_synthetic_bead --outputs-dir /cluster/home/kristiac/NeCT/outputs
  python -m demo.eval_synthetic_bead --outputs-dir outputs --gt-dir dataset/synthetic_bead_ga
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


# ── metrics ───────────────────────────────────────────────────────────────────

DATA_RANGE = 0.19  # max attenuation value in the GT phantom (BEAD_ATT=0.15 + sphere=0.08, clipped)


def compute_metrics(recon: np.ndarray, gt: np.ndarray) -> dict:
    """3D PSNR, MSE, SSIM between a reconstructed and GT volume."""
    mse = float(np.mean((recon - gt) ** 2))
    psnr = float(10.0 * np.log10(DATA_RANGE ** 2 / mse)) if mse > 0 else 100.0
    ssim = float(ssim_fn(gt, recon, data_range=DATA_RANGE))
    return {"psnr": psnr, "mse": mse, "ssim": ssim}


# ── experiment discovery ───────────────────────────────────────────────────────

def find_experiments(outputs_dir: Path, pattern: str = "synthetic_bead_deg*_ac*") -> list[dict]:
    """
    Return list of dicts with keys: exp_dir, deg, ac, config_path, ckpt_path.
    Searches for config.yaml in <exp_dir>/model/ and <exp_dir>/,
    and last.ckpt anywhere under <exp_dir>.
    """
    experiments = []
    for exp_dir in sorted(outputs_dir.glob(pattern)):
        if not exp_dir.is_dir():
            continue

        m = re.search(r"deg(\d+)_ac(\d+)", exp_dir.name)
        if not m:
            continue
        deg, ac = int(m.group(1)), int(m.group(2))

        # Config: prefer new layout (model/config.yaml) over old flat layout
        config_path = None
        for candidate in [exp_dir / "model" / "config.yaml", exp_dir / "config.yaml"]:
            if candidate.exists():
                config_path = candidate
                break

        # Checkpoint: find the most recently modified last.ckpt
        ckpt_candidates = sorted(exp_dir.rglob("last.ckpt"), key=lambda p: p.stat().st_mtime)
        ckpt_path = ckpt_candidates[-1] if ckpt_candidates else None

        if config_path is None or ckpt_path is None:
            missing = []
            if config_path is None:
                missing.append("config.yaml")
            if ckpt_path is None:
                missing.append("last.ckpt")
            print(f"  [skip] {exp_dir.name}: missing {', '.join(missing)}", flush=True)
            continue

        experiments.append({
            "exp_dir": exp_dir,
            "deg": deg,
            "ac": ac,
            "config_path": config_path,
            "ckpt_path": ckpt_path,
        })

    return experiments


# ── model loading ──────────────────────────────────────────────────────────────

def load_trainer(config_path: Path, ckpt_path: Path):
    """
    Instantiate a ContinousScanningTrainerBatch in inference-only mode,
    then manually load the checkpoint weights.

    The trainer must be instantiated (not just the model) so that dataset.maximum
    and dataset.minimum are available for create_volume() rescaling.
    """
    from nect.config import get_cfg
    from nect.trainers.continous_scanning_trainer_batch import ContinousScanningTrainerBatch

    config = get_cfg(str(config_path))

    trainer = ContinousScanningTrainerBatch(
        config=config,
        output_directory=None,   # no output; no checkpoint saving
        save_ckpt=False,
        save_last=False,
        log=False,
        verbose=False,
    )

    checkpoint_data = trainer.fabric.load(str(ckpt_path))
    trainer.model.load_state_dict(checkpoint_data["model"])
    trainer.model.eval()

    return trainer


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_experiment(exp: dict, gt_volumes: np.ndarray, gt_timesteps: np.ndarray) -> dict | None:
    """
    Load the model and reconstruct at each GT timestep.
    Returns dict with per-timestep and mean metrics.
    """
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
        gt = gt_volumes[i]
        m = compute_metrics(recon, gt)
        m["timestep"] = float(t)
        per_timestep.append(m)

    if not per_timestep:
        return None

    means = {k: float(np.mean([p[k] for p in per_timestep])) for k in ("psnr", "mse", "ssim")}
    stds  = {f"{k}_std": float(np.std([p[k] for p in per_timestep])) for k in ("psnr", "mse", "ssim")}

    print(f"    PSNR={means['psnr']:.2f}  SSIM={means['ssim']:.4f}  MSE={means['mse']:.6f}", flush=True)
    return {"deg": exp["deg"], "ac": exp["ac"], **means, **stds, "per_timestep": per_timestep}


# ── output ────────────────────────────────────────────────────────────────────

def save_results(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON (full, including per-timestep detail)
    json_path = out_dir / "synthetic_bead_eval.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved → {json_path}")

    # CSV (summary only)
    csv_path = out_dir / "synthetic_bead_eval.csv"
    flat_keys = ["deg", "ac", "psnr", "psnr_std", "ssim", "ssim_std", "mse", "mse_std"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in flat_keys})
    print(f"CSV  saved → {csv_path}")


def print_table(results: list[dict]) -> None:
    sorted_results = sorted(results, key=lambda r: (r["deg"], r["ac"]))
    header = f"{'deg':>5} {'ac':>4}  {'PSNR':>7} {'±':>5}  {'SSIM':>6} {'±':>5}  {'MSE':>10} {'±':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in sorted_results:
        print(
            f"{r['deg']:>5} {r['ac']:>4}  "
            f"{r['psnr']:>7.2f} {r['psnr_std']:>5.2f}  "
            f"{r['ssim']:>6.4f} {r['ssim_std']:>5.4f}  "
            f"{r['mse']:>10.6f} {r['mse_std']:>10.6f}"
        )
    print("=" * len(header))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir", type=Path,
        default=Path("/cluster/home/kristiac/NeCT/outputs"),
        help="Directory containing synthetic_bead_deg*_ac* experiment subdirs",
    )
    parser.add_argument(
        "--gt-dir", type=Path,
        default=Path("/cluster/home/kristiac/NeCT/Datasets/synthetic_bead_ga"),
        help="Directory containing gt_volumes.npy and gt_timesteps.npy",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "synthetic_bead_eval",
        help="Where to write CSV and JSON results",
    )
    args = parser.parse_args()

    # Load ground truth
    gt_volumes_path   = args.gt_dir / "gt_volumes.npy"
    gt_timesteps_path = args.gt_dir / "gt_timesteps.npy"
    if not gt_volumes_path.exists():
        sys.exit(f"ERROR: GT volumes not found at {gt_volumes_path}")
    gt_volumes   = np.load(gt_volumes_path).astype(np.float32)   # (11, 128, 128, 128)
    gt_timesteps = np.load(gt_timesteps_path).astype(np.float32)  # (11,)
    print(f"GT volumes: {gt_volumes.shape}  t={gt_timesteps.tolist()}")

    # Discover experiments
    experiments = find_experiments(args.outputs_dir)
    if not experiments:
        sys.exit(f"ERROR: no experiments found under {args.outputs_dir}")
    print(f"\nFound {len(experiments)} experiments\n")

    # Evaluate
    results = []
    for exp in experiments:
        result = evaluate_experiment(exp, gt_volumes, gt_timesteps)
        if result is not None:
            results.append(result)

    if not results:
        sys.exit("ERROR: all experiments failed to evaluate")

    # Report
    print_table(results)
    save_results(results, args.out_dir)


if __name__ == "__main__":
    main()
