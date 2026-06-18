"""
Synthetic bead continuous reconstruction experiment.

Each run combines two axes:
  --degree  angular window (1, 4, 8, 12, 24):
            how many consecutive projections are treated as from the same timestep,
            emulating what the scanner "sees" when it cannot resolve motion within
            that angular window. degree=1 means no mashing (original dynamic data).
  --ac      accumulation steps (1, 2, 3, 4, 6, 8, 12):
            sub-angle resolution that the reconstruction attempts within each window.

Usage (called from slurm via python -m demo.synthetic_bead):
  python -m demo.synthetic_bead --degree 4 --ac 2
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import nect

DATA = Path("/cluster/home/kristiac/NeCT/Datasets/synthetic_bead_ga/full")


def mash_timesteps(timesteps: list, mash_n: int) -> list:
    """
    Replace each timestep with the mean of its N-projection window.
    All projections within a window share the same t_center, so the
    reconstruction cannot distinguish motion within that window.
    """
    ts = np.array(timesteps, dtype=np.float64)
    n  = len(ts)
    out = ts.copy()
    for k in range(0, n, mash_n):
        window = ts[k : k + mash_n]
        out[k : k + mash_n] = window.mean()
    return out.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, required=True, choices=[1, 4, 8, 12, 24],
                        help="Angular window: how many projections are merged per frame")
    parser.add_argument("--ac", type=int, required=True, choices=[1, 2, 3, 4, 6, 8, 12],
                        help="Accumulation steps (sub-angle resolution)")
    args = parser.parse_args()

    print(f"PyTorch {torch.__version__}  |  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load geometry and apply temporal mashing ─────────────────────────────
    geometry = nect.Geometry.from_yaml(DATA / "geometry.yaml")
    if args.degree > 1:
        geometry.set_timesteps(mash_timesteps(geometry.timesteps, args.degree))

    # ── Common config ─────────────────────────────────────────────────────────
    net = {
        "otype": "FullyFusedMLP",
        "activation": "LeakyReLU",
        "output_activation": "None",
        "n_neurons": 128,
        "n_hidden_layers": 4,
        "include_identity": False,
    }

    exp_name = f"synthetic_bead_deg{args.degree:02d}_ac{args.ac:02d}"
    print(f"Running: {exp_name}")
    nect.reconstruct_continious_scan(
        geometry=geometry,
        projections=str(DATA / "projections.npy"),
        quality="high",
        mode="dynamic",
        exp_name=exp_name,
        config_override={
            "epochs": 100,
            "checkpoint_interval": 0,
            "image_interval": 0,
            "plot_type": "XZ",
            "base_lr": 0.001,
            "warmup": {
                "steps": 1800 * 2,   # ~2 epochs over N_TOTAL=1800
                "lr0": 0.001,
            },
            "encoder": {
                "otype": "HashGrid",
                "n_levels": 18,
                "n_features_per_level": 2,
                "log2_hashmap_size": 22,
                "base_resolution": 16,
                "max_resolution_factor": 2,
            },
            "encoder_2d": {
                "n_levels": 11,
                "n_features_per_level": 4,
                "base_resolution": 16,
                "per_level_scale": 1.5,
            },
            "net": net,
            "tv_spatial": 1e-4,
            "accumulation_steps": args.ac,
            "continous_scanning": True,
        },
        enc_arc="mixedcubes",
        memvstime="batch",
    )


if __name__ == "__main__":
    main()
