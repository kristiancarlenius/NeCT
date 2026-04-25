from pathlib import Path
import nect
from nect.config import MLPNetConfig

data_path = "/cluster/home/kristiac/NeCT/Datasets/bentheimer/"

geometry_file = Path(data_path) / "geometry.yaml"
geometry = nect.Geometry.from_yaml(geometry_file)

reconstruction_path_dynamic, _ = nect.reconstruct(
    geometry=geometry,
    projections=str(Path(data_path) / "projections.npy"),
    quality="high",
    mode="dynamic",
    exp_name="sexcubes_kplanes",
    config_override={
        "epochs": "10x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.0002,
        "warmup": {
            "steps": 1400 * 10,
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
        "net": MLPNetConfig(
            otype="FullyFusedMLP",
            activation="LeakyReLU",
            output_activation="None",
            n_neurons=128,
            n_hidden_layers=8,
            include_identity=False,
            include_adaptive_skip=False,
        ),
    },
    enc_arc="sexcubes_kplanes",
)

print(reconstruction_path_dynamic, _)
