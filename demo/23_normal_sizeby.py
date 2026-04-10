from pathlib import Path
import numpy as np
import nect
import torch
from nect.config import MLPNetConfig

data_path = "/cluster/home/kristiac/NeCT/Datasets/bentheimer/"
geometry_file = Path(data_path) / "geometry.yaml"
geometry = nect.Geometry.from_yaml(geometry_file)

reconstruction_path_dynamic, _ = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="dynamic",
    exp_name="sizediff",
    config_override={
        "epochs": "6x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.0001,
        "warmup": {
            "steps": 1400*10,
            "lr0": 0.0005,
        },
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 22,
            "n_features_per_level": 4,
            "log2_hashmap_size": 23,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
        "net": MLPNetConfig(
            otype="FullyFusedMLP",
            activation="LeakyReLU",
            output_activation="ReLU",
            n_neurons=128,
            n_hidden_layers=4,
            include_identity=False,
            include_adaptive_skip=False,
        ),
    },)
print(reconstruction_path_dynamic, _)
