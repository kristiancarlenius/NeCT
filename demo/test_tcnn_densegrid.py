import tinycudann as tcnn

print("Test 1: standalone 2D DenseGrid encoding (should work)")
try:
    enc = tcnn.Encoding(2, {
        "otype": "DenseGrid",
        "n_levels": 11,
        "n_features_per_level": 4,
        "base_resolution": 16,
        "per_level_scale": 1.5,
    })
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("Test 2: Composite encoding with HashGrids only (no DenseGrid)")
try:
    enc = tcnn.Encoding(6, {
        "otype": "Composite",
        "nested": [
            {"n_dims_to_encode": 3, "otype": "HashGrid", "n_levels": 4, "n_features_per_level": 2, "log2_hashmap_size": 15, "base_resolution": 16, "per_level_scale": 1.5},
            {"n_dims_to_encode": 3, "otype": "HashGrid", "n_levels": 4, "n_features_per_level": 2, "log2_hashmap_size": 15, "base_resolution": 16, "per_level_scale": 1.5},
        ],
    })
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("Test 3: Composite encoding with a single small DenseGrid")
try:
    enc = tcnn.Encoding(9, {
        "otype": "Composite",
        "nested": [
            {"n_dims_to_encode": 2, "otype": "DenseGrid", "n_levels": 4, "n_features_per_level": 2, "base_resolution": 16, "per_level_scale": 1.5},
            {"n_dims_to_encode": 7, "otype": "HashGrid", "n_levels": 4, "n_features_per_level": 2, "log2_hashmap_size": 15, "base_resolution": 16, "per_level_scale": 1.5},
        ],
    })
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("Test 4: full MixedCubes config as NetworkWithInputEncoding")
try:
    model = tcnn.NetworkWithInputEncoding(9, 1, {
        "otype": "Composite",
        "nested": [
            {"n_dims_to_encode": 2, "otype": "DenseGrid", "n_levels": 11, "n_features_per_level": 4, "base_resolution": 16, "per_level_scale": 1.5},
            {"n_dims_to_encode": 2, "otype": "DenseGrid", "n_levels": 11, "n_features_per_level": 4, "base_resolution": 16, "per_level_scale": 1.5},
            {"n_dims_to_encode": 2, "otype": "DenseGrid", "n_levels": 11, "n_features_per_level": 4, "base_resolution": 16, "per_level_scale": 1.5},
            {"n_dims_to_encode": 3, "otype": "HashGrid", "n_levels": 18, "n_features_per_level": 2, "log2_hashmap_size": 23, "base_resolution": 16, "per_level_scale": 1.44},
        ],
    }, {
        "otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
        "n_neurons": 128, "n_hidden_layers": 4,
    })
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
