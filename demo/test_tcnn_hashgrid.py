import tinycudann as tcnn

print("Testing standalone 3D HashGrid (QuadCubes-style)...")
try:
    enc = tcnn.Encoding(3, {
        "otype": "HashGrid",
        "n_levels": 4,
        "n_features_per_level": 2,
        "log2_hashmap_size": 15,
        "base_resolution": 16,
        "per_level_scale": 1.5,
    })
    print("PASS — HashGrid works")
except Exception as e:
    print(f"FAIL: {e}")
