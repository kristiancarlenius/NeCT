from pathlib import Path
import yaml
import numpy as np
import nect

data_path = "/cluster/home/kristiac/NeCT/Datasets/bentheimer/"
config_file = Path(data_path) / "config.yaml"
geometry_file = Path(data_path) / "geometry.yaml"

# load config
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# update img_path to point to projections folder
config["img_path"] = str(Path(data_path) / "projections")

# write the updated config to a temporary yaml
tmp_config_file = Path(data_path) / "config_tmp.yaml"
with open(tmp_config_file, "w") as f:
    yaml.safe_dump(config, f)

# export dataset to npy (requires the helper function we discussed earlier)
nect.export_dataset_to_npy(tmp_config_file, Path(data_path) / "projections.npy")

# load geometry
geometry = nect.Geometry.from_yaml(geometry_file)

# run reconstruction using the new .npy projections
reconstruction_path = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="dynamic",
    config_override={
        "epochs": "3x",
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
