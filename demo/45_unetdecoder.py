from pathlib import Path
import nect
from nect.config import UNetDecoderConfig

data_path = "/cluster/home/kristiac/NeCT/Datasets/bentheimer/"

geometry_file = Path(data_path) / "geometry.yaml"
geometry = nect.Geometry.from_yaml(geometry_file)

reconstruction_path_dynamic, _ = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="dynamic",
    exp_name="quadcubes_unet",
    config_override={
        "epochs": "8x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.0001,
        "lr": 0.004,
        "warmup": {
            "steps": 1400 * 20,
            "lr0": 0.02,
        },
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 23,
            "n_features_per_level": 4,
            "log2_hashmap_size": 23,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
        "net": UNetDecoderConfig(
            hidden_dims=[128, 64, 32],
            levels_coarse=8,
            levels_medium=8,
            dropout=0.0,
        ),
    },
    enc_arc="quadcubes_unet",
)

print(reconstruction_path_dynamic, _)
