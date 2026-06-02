from pathlib import Path
import yaml
import numpy as np
import nect
import torch 
from nect.config import MLPNetConfig

data_path = "/cluster/home/kristiac/NeCT/Datasets/continious_scans/"#_dyn/"
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
geometry_file = Path(data_path) / "geometry_optimized_100_cont.yaml"#"geometry_8fps_5500.yaml"
geometry = nect.Geometry.from_yaml(geometry_file)


reconstruction_path_static, output_path = nect.reconstruct_continious_scan(
    geometry=geometry,
    projections=str(Path(data_path) / "proj_100_cont.npy"),#"projections.npy"),
    quality="high",
    mode="static",
    exp_name="static_continious",
    config_override={
        "epochs": "3x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 23,
            "n_features_per_level": 4,
            "log2_hashmap_size": 23,
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
        "accumulation_steps": 6,
        "continous_scanning": True,
    },
    memvstime=True,
)
"""

reconstruction_path_dynamic, _ = nect.reconstruct_continious_scan(
    geometry=geometry,
    projections=str(Path(data_path) / "proj_8fps_5500.npy"),
    quality="high",
    mode="dynamic",
    exp_name="dynamic_continious",
    config_override={
        "epochs": "5x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.001,
        "warmup": {
            "steps": 1400*20,
            "lr0": 0.001,
        },
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 23,
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
        "net": {
            "otype": "FullyFusedMLP",
            "activation": "LeakyReLU",
            "output_activation": "None",
            "n_neurons": 128,
            "n_hidden_layers": 4,
            "include_identity": False,
        },
        "tv_temporal": 1e-4,
        "accumulation_steps": 4,
        "continous_scanning": True,
        
    },
    enc_arc="mixedcubes",  
    memvstime=True,
)

print(reconstruction_path_dynamic, _)
"""