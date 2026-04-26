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
    exp_name="sexcubes_unet",
    config_override={
        "epochs": "8x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.0002,
        "warmup": {
            "steps": 1400 * 20,
            "lr0": 0.001,
        },
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 22,
            "n_features_per_level": 4,
            "log2_hashmap_size": 20,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
        "net": UNetDecoderConfig(                                                                                                                                                                   
            hidden_dims=[84, 42, 21],  # was [168, 84, 42]
            levels_coarse=5,           # was 7                                                                                                                                                      
            levels_medium=5,           # was 7
            dropout=0.1,
        ),
    },
    enc_arc="sexcubes_unet",
)

print(reconstruction_path_dynamic, _)
