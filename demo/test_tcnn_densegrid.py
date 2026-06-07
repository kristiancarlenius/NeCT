import tinycudann as tcnn

encoding = {
    "otype": "Composite",
    "nested": [
        {"n_dims_to_encode": 2, "otype": "DenseGrid", "n_levels": 11, "n_features_per_level": 4, "base_resolution": 16, "per_level_scale": 1.5},
        {"n_dims_to_encode": 2, "otype": "DenseGrid", "n_levels": 11, "n_features_per_level": 4, "base_resolution": 16, "per_level_scale": 1.5},
        {"n_dims_to_encode": 2, "otype": "DenseGrid", "n_levels": 11, "n_features_per_level": 4, "base_resolution": 16, "per_level_scale": 1.5},
        {"n_dims_to_encode": 3, "otype": "HashGrid", "n_levels": 18, "n_features_per_level": 2, "log2_hashmap_size": 23, "base_resolution": 16, "per_level_scale": 1.44},
    ],
}
network = {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None", "n_neurons": 128, "n_hidden_layers": 4}

print("Creating model...")
model = tcnn.NetworkWithInputEncoding(9, 1, encoding, network)
print("Success — encoding output dims:", model.n_input_dims, "->", model.n_output_dims)
