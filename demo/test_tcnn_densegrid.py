import tinycudann as tcnn

DENSE_CFG = {
    "otype": "Grid",
    "type": "Dense",
    "n_levels": 11,
    "n_features_per_level": 4,
    "base_resolution": 16,
    "per_level_scale": 1.5,
}

HASH_CFG = {
    "otype": "HashGrid",
    "n_levels": 18,
    "n_features_per_level": 2,
    "log2_hashmap_size": 23,
    "base_resolution": 16,
    "per_level_scale": 1.44,
}

NETWORK_CFG = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 4,
}

print("Test 1: standalone 2D Grid/Dense encoding")
try:
    enc = tcnn.Encoding(2, DENSE_CFG)
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("Test 2: standalone 3D HashGrid encoding")
try:
    enc = tcnn.Encoding(3, HASH_CFG)
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("Test 3: Composite with 3 x Grid/Dense + 1 x HashGrid (full MixedCubes encoding)")
try:
    enc = tcnn.Encoding(9, {
        "otype": "Composite",
        "nested": [
            {"n_dims_to_encode": 2, **DENSE_CFG},
            {"n_dims_to_encode": 2, **DENSE_CFG},
            {"n_dims_to_encode": 2, **DENSE_CFG},
            {"n_dims_to_encode": 3, **HASH_CFG},
        ],
    })
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("Test 4: full MixedCubes NetworkWithInputEncoding")
try:
    model = tcnn.NetworkWithInputEncoding(9, 1, {
        "otype": "Composite",
        "nested": [
            {"n_dims_to_encode": 2, **DENSE_CFG},
            {"n_dims_to_encode": 2, **DENSE_CFG},
            {"n_dims_to_encode": 2, **DENSE_CFG},
            {"n_dims_to_encode": 3, **HASH_CFG},
        ],
    }, NETWORK_CFG)
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
