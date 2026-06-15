from pathlib import Path
import nect
import torch

print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

DATA = Path("/cluster/home/kristiac/NeCT/Datasets/synthetic_bead_ga/full")
geometry = nect.Geometry.from_yaml(DATA / "geometry.yaml")

nect.reconstruct_continious_scan(
    geometry=geometry,
    projections=str(DATA / "projections.npy"),
    quality="high",
    mode="dynamic",
    exp_name="synthetic_bead_dynamic_ac12",
    config_override={
        "epochs": "5x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.001,
        "warmup": {
            "steps": 360 * 5,
            "lr0": 0.001,
        },
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 20,
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
            "n_neurons": 64,
            "n_hidden_layers": 2,
            "include_identity": False,
        },
        "tv_spatial": 1e-4,
        "accumulation_steps": 12,
        "continous_scanning": True,
    },
    enc_arc="mixedcubes",
    memvstime="batch",
)
