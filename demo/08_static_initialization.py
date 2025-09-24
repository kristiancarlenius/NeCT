from pathlib import Path
import yaml
import numpy as np
import nect
import torch 

data_path = "/cluster/home/kristiac/NeCT/Datasets/bentheimer/"
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
reconstruction_path = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="static",
    exp_name="static_initilalization",
    config_override={
        "epochs": "2x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 20,
            "n_features_per_level": 4,
            "log2_hashmap_size": 21,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
    },
)
"""
ckpt = torch.load("last.ckpt")

reconstruction_path = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="dynamic",
    exp_name="dynamic_initilalized",
    config_override={
        "epochs": "8x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 20,
            "n_features_per_level": 4,
            "log2_hashmap_size": 21,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
    },
)
nect.export_volumes(reconstruction_path, binning=3, avg_timesteps=5)