from pathlib import Path
import yaml
import numpy as np
import nect
import torch 
from nect.config import MLPNetConfig

data_path = "/cluster/home/kristiac/NeCT/Datasets/bentheimer/"#simulatedfluidinvasion/"#
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


# run reconstruction using the new .npy projections
reconstruction_path_static, output_path_0 = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="static",
    exp_name="static_init",
    config_override={
        "epochs": "3x",
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
            include_identity=False,
            include_adaptive_skip=False,
        ),
    },
)
reconstruction_path_static, output_path_1 = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="static",
    exp_name="static_init",
    config_override={
        "epochs": "6x",
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
            include_identity=False,
            include_adaptive_skip=False,
        ),
    },
)
reconstruction_path_static, output_path_2 = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="static",
    exp_name="static_init",
    config_override={
        "epochs": "9x",
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
            include_identity=False,
            include_adaptive_skip=False,
        ),
    },
)
reconstruction_path_static, output_path_3 = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="static",
    exp_name="static_init",
    config_override={
        "epochs": "12x",
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
            include_identity=False,
            include_adaptive_skip=False,
        ),
    },
)
liste = [output_path_0, output_path_1, output_path_2, output_path_3]
with open("file_overview_longer.txt", "a") as f:
  for write_out in liste:
    f.write(write_out)