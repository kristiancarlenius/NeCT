from pathlib import Path
import yaml
import numpy as np
import nect
import torch 
from nect.config import MLPNetConfig

data_path = "/cluster/home/kristiac/NeCT/Datasets/simulatedfluidinvasion/"
"""
config_file = Path(data_path) / "config.yaml"
with open(config_file, "r") as f:
    config = yaml.safe_load(f)
config["img_path"] = str(Path(data_path) / "projections")
tmp_config_file = Path(data_path) / "config_tmp.yaml"
with open(tmp_config_file, "w") as f:
    yaml.safe_dump(config, f)
nect.export_dataset_to_npy(tmp_config_file, Path(data_path) / "projections.npy")
"""
geometry_file = Path(data_path) / "geometry.yaml"
geometry = nect.Geometry.from_yaml(geometry_file)

"""
# run reconstruction using the new .npy projections
reconstruction_path_static = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="static",
    exp_name="static_init",
    config_override={
        "epochs": "2x",
        "checkpoint_interval": 0,
        "image_interval": 10,
        "plot_type": "XZ",
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
            output_activation="ReLU",
            n_neurons=128,
            n_hidden_layers=4,
            include_identity=True,
            include_adaptive_skip=False,
        ),
    },
)

"""
reconstruction_path_dynamic = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="dynamic",
    exp_name="dynamic_init",
    static_init = "/cluster/home/kristiac/NeCT/outputs/static_init/hash_grid_21_4_21_16_2_4_128_L1/2025-09-27T13-50-23/model/checkpoints/last.ckpt",
    static_init_config="/cluster/home/kristiac/NeCT/outputs/static_init/hash_grid_21_4_21_16_2_4_128_L1/2025-09-27T13-50-23/model/config.yaml",
    config_override={
        "epochs": "6x",
        "checkpoint_interval": 0,
        "image_interval": 80,
        "plot_type": "XZ",
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
            output_activation="ReLU",
            n_neurons=128,
            n_hidden_layers=4,
            include_identity=True,
            include_adaptive_skip=False,
        ),
    },
)

nect.export_volume(reconstruction_path_dynamic, binning=3)


