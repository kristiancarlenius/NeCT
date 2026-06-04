from pathlib import Path
import yaml
import numpy as np
import nect
import torch 
from nect.config import MLPNetConfig

print(torch.__version__)
print(torch.cuda.get_arch_list())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
print(torch.cuda.is_available())


data_path = "/cluster/home/kristiac/NeCT/Datasets/continious_scan_dyn/"

geometry_file = Path(data_path) / "geometry_4fps_5500.yaml"
geometry = nect.Geometry.from_yaml(geometry_file)

"""
reconstruction_path_static, output_path = nect.reconstruct_continious_scan(
    geometry=geometry,
    projections=str(Path(data_path) / "proj_360_cont.npy"),#"projections.npy"),
    quality="high",
    mode="static",
    exp_name="static_continious",
    config_override={
        "epochs": "4x",
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
)
"""

reconstruction_path_dynamic, _ = nect.reconstruct_continious_scan(
    geometry=geometry,
    projections=str(Path(data_path) / "proj_4fps_5500.npy"),
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
            "steps": 1400*10,
            "lr0": 0.001,
        },
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 18,
            "n_features_per_level": 2,
            "log2_hashmap_size": 23,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
        "encoder_2d": {
            "n_levels": 11,
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
        "tv_spatial": 1e-4,
        "accumulation_steps": 2,
        "continous_scanning": True,
        
    },
    enc_arc="mixedcubes",
    memvstime="batch",)
print(reconstruction_path_dynamic, _)
