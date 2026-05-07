#!/usr/bin/env python3
"""
Generate the three cutout images used in the MixedCubes quality figure in the thesis.

Reads the crop region from crop_img_comparison.json, applies it to the three source
images (ground truth, NeCT baseline at 12 h, MixedCubes at 12 h), applies a
percentile-based intensity stretch so pore structure is visible, then writes the
results to docs/images/.

Usage:
    python make_comparison_crops.py
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent

CROPS_FILE = ROOT / "crop_img_comparison.json"
OUT_DIR = ROOT / "docs" / "images"

SOURCES = [
    # MixedCubes quality figure
    (ROOT / "sizediff" / "perfect" / "0525_1400.png",                              "compairson.png"),
    (ROOT / "sizediff" / "quadcubes" / "23_4_23_4_128" / "time" / "0130_1400.png", "highcontrast_nect_12h.png"),
    (ROOT / "sizediff" / "mixedcubes" / "24_4_24_4_128" / "time" / "0170_1400.png", "highcontrast_mixedcubes_12h.png"),
    # MLP decoder depth figure
    (ROOT / "sizediff" / "sexcubes" / "128_4"  / "0525_1400.png", "sexcubes_128_4.png"),
    (ROOT / "sizediff" / "sexcubes" / "128_7"  / "0525_1400.png", "sexcubes_128_7.png"),
    (ROOT / "sizediff" / "sexcubes" / "128_16" / "0525_1400.png", "sexcubes_128_16.png"),
    # SingleCube figure
    (ROOT / "sizediff" / "singlecube" / "24_4_24_4_128" / "1050_1400.png", "singlecube.png"),
]

UPSCALE = 4        # nearest-neighbour upscale factor for print clarity
P_LOW, P_HIGH = 2, 98  # percentile clip for intensity stretch


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def stretch(img: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    lo = np.percentile(img, lo_pct)
    hi = np.percentile(img, hi_pct)
    if hi <= lo:
        return img.astype(np.uint8)
    stretched = (img - lo) / (hi - lo) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def main():
    with open(CROPS_FILE) as f:
        data = json.load(f)

    crops = data["crops"]
    if not crops:
        raise ValueError("No crops found in crop_img_comparison.json")

    c = crops[0]
    x0, y0, x1, y1 = c["x0"], c["y0"], c["x1"], c["y1"]
    print(f"Crop region: ({x0},{y0}) → ({x1},{y1})  [{x1-x0}×{y1-y0} px]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for src_path, out_name in SOURCES:
        if not src_path.exists():
            print(f"[MISSING] {src_path}")
            continue

        img = load_gray(src_path)
        h, w = img.shape
        cx0, cx1 = max(0, x0), min(w, x1)
        cy0, cy1 = max(0, y0), min(h, y1)
        patch = img[cy0:cy1, cx0:cx1]

        patch_stretched = stretch(patch, P_LOW, P_HIGH)

        pil = Image.fromarray(patch_stretched, mode="L")
        if UPSCALE > 1:
            pil = pil.resize(
                (pil.width * UPSCALE, pil.height * UPSCALE),
                resample=Image.NEAREST,
            )

        out_path = OUT_DIR / out_name
        pil.save(out_path)
        print(f"Saved {out_path}  ({pil.width}×{pil.height} px)")


if __name__ == "__main__":
    main()
