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
    exp_name="sexcubes_densegrid_transformer",
    config_override={
        "epochs": "8x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.0001,
        "lr": 0.0004,
        "warmup": {
            "steps": 1400 * 20,
            "lr0": 0.002,
        },
        "encoder": {
            "n_levels": 10,
            "n_features_per_level": 4,
            "base_resolution": 16,
            "per_level_scale": 1.5,
        },
        "net": TransformerDecoderConfig(
            d_model=20,
            n_heads=4,
            n_layers=3,
            dropout=0.0,
        ),
    },
    enc_arc="sexcubes_densegrid_transformer",
)

print(reconstruction_path_dynamic, _)
