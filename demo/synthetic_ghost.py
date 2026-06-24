"""
Synthetic ghost continuous-scan reconstruction experiment.

Tests whether K accumulation steps can recover the static target sphere that
is periodically obstructed by a dense moving sphere during continuous scanning.

  --degree   angular step in degrees (4, 8, 12, 24)
  --ac       accumulation steps K (1, 2, 3, 4, 6, 8)

Usage:
  python -m demo.synthetic_ghost --degree 4 --ac 2
  python -m demo.synthetic_ghost --degree 8 --ac 4
"""

import argparse
from pathlib import Path

import torch

import nect

DATA = Path("/cluster/home/kristiac/NeCT/Datasets/synthetic_ghost")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, required=True, choices=[4, 8, 12, 24],
                        help="Angular step size; selects the matching dataset subdirectory")
    parser.add_argument("--ac", type=int, required=True, choices=[1, 2, 3, 4, 6, 8],
                        help="Accumulation steps K for the K-step forward model")
    args = parser.parse_args()

    print(f"PyTorch {torch.__version__}  |  GPU: {torch.cuda.get_device_name(0)}")

    deg_dir  = DATA / f"deg{args.degree:02d}"
    geometry = nect.Geometry.from_yaml(deg_dir / "geometry.yaml")

    net = {
        "otype": "FullyFusedMLP",
        "activation": "LeakyReLU",
        "output_activation": "None",
        "n_neurons": 128,
        "n_hidden_layers": 4,
        "include_identity": False,
    }

    exp_name = f"synthetic_ghost_deg{args.degree:02d}_ac{args.ac:02d}"
    print(f"Running: {exp_name}")

    nect.reconstruct_continious_scan(
        geometry=geometry,
        projections=str(deg_dir / "projections.npy"),
        quality="high",
        mode="dynamic",
        exp_name=exp_name,
        config_override={
            "epochs": "18x",
            "checkpoint_interval": 0,
            "image_interval": 0,
            "plot_type": "XZ",
            "base_lr": 0.001,
            "warmup": {
                "steps": 270 * 2,
                "lr0": 0.001,
            },
            "encoder": {
                "otype": "HashGrid",
                "n_levels": 20,
                "n_features_per_level": 4,
                "log2_hashmap_size": 23,
                "base_resolution": 16,
                "max_resolution_factor": 2,
            },
            "encoder_2d": {
                "n_levels": 12,
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
