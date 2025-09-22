from pathlib import Path
import nect 
import numpy as np
import yaml

data_path = "/cluster/home/kristiac/NeCT/Datasets/bentheimer/"
config_file = data_path+"config.yaml"
geometry_file = data_path+"geometry.yaml"

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

geometry = nect.Geometry.from_yaml(geometry_file)

config["img_path"] = data_path+"projections" # need to change the img_path to point to the path of the projections
nect.export_dataset_to_npy('config.yaml', 'dataset.npy')

reconstruction_path = nect.reconstruct(
    geometry=geometry,
    projections=data_path+"projections.npy",
    quality="high",
    mode="dynamic",
    config_override={
        "epochs": "3x",  # a multiplier of base-epochs. Base-epochs is: floor(49 / num_projections * max(nDetector))
        "checkpoint_interval": 0,  # How often to save the model in seconds
        "image_interval": 0,  # How often to save images in seconds
        "plot_type": "XZ",  # XZ or XY, YZ
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 20,
            "n_features_per_level": 4,
            "log2_hashmap_size": 21,
            "base_resolution": 16,
            "max_resolution_factor": 2
        },
    },
)


