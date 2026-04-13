from pathlib import Path
import nect
from nect.config import TransformerDecoderConfig

data_path = "/cluster/home/kristiac/NeCT/Datasets/bentheimer/"

geometry_file = Path(data_path) / "geometry.yaml"
geometry = nect.Geometry.from_yaml(geometry_file)

reconstruction_path_dynamic, _ = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="dynamic",
    exp_name="sexcubes_transformer",
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
            "n_levels": 21,
            "n_features_per_level": 4,
            "log2_hashmap_size": 20,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
        "net": TransformerDecoderConfig(
            d_model=32,
            n_heads=4,
            n_layers=3,
            dropout=0.0,
        ),
    },
    enc_arc="sexcubes_transformer",
)

print(reconstruction_path_dynamic, _)
