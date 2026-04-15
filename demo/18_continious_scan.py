from pathlib import Path
import yaml
import numpy as np
import nect
import torch
from nect.config import MLPNetConfig

data_path = "/cluster/home/kristiac/NeCT/Datasets/continious_scan_dyn/"
geometry_file = Path(data_path) / "geometry_4fps_2750.yaml"
geometry = nect.Geometry.from_yaml(geometry_file)

reconstruction_path, _ = nect.reconstruct_continious_scan(
    geometry=geometry,
    projections=str(Path(data_path) / "proj_4fps_2750.npy"),
    quality="high",
    mode="dynamic",
    exp_name="dynamic_continious",
    config_override={
        "epochs": "6x",
        "checkpoint_interval": 16000,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.0001,
        "warmup": {
            "steps": 1400*10,
            "lr0": 0.0001,
        },
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 21,
            "n_features_per_level": 4,
            "log2_hashmap_size": 21,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
        "net": MLPNetConfig(
            otype="FullyFusedMLP",
            activation="LeakyReLU",
            output_activation="None",
            n_neurons=128,
            n_hidden_layers=4,
            include_identity=False,
            include_adaptive_skip=False,
        ),
        "accumulation_steps": 2,
        "continous_scanning": True,

    },)

print(reconstruction_path, _)
