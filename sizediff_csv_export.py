#!/usr/bin/env python3
"""Export per-time-image metrics for all sizediff models to a CSV file."""

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path(__file__).parent
SIZEDIFF = ROOT / "sizediff"
CROPS_FILE = ROOT / "crops.json"
PERFECT = SIZEDIFF / "perfect" / "0525_1400.png"
OUT_CSV = ROOT / "results" / "sizediff_metrics.csv"

HOURS_PER_IMAGE = 6.0


def load_crops():
    with open(CROPS_FILE) as f:
        return json.load(f)["crops"]


def load_image_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def compute_all_metrics(ref, cand, crops):
    ssim_vals, mae_vals, mse_vals = [], [], []
    for c in crops:
        r_crop = ref[c["y0"]:c["y1"], c["x0"]:c["x1"]]
        c_crop = cand[c["y0"]:c["y1"], c["x0"]:c["x1"]]

        r_std = r_crop.std()
        if r_std > 0 and c_crop.std() > 0:
            c_norm = (c_crop - c_crop.mean()) / c_crop.std() * r_std + r_crop.mean()
        else:
            c_norm = c_crop

        ssim_vals.append(float(ssim_fn(r_crop, c_crop, data_range=255.0)))
        mae_vals.append(float(np.mean(np.abs(r_crop - c_norm))))
        mse_vals.append(float(np.mean((r_crop - c_norm) ** 2)))

    ssim = float(np.mean(ssim_vals))
    mae  = float(np.mean(mae_vals))
    mse  = float(np.mean(mse_vals))
    psnr = float(10.0 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")
    return ssim, psnr, mae


def parse_vram_gb(vram_file):
    if not vram_file or not vram_file.exists():
        return None
    with open(vram_file) as f:
        for line in f:
            if "Peak reserved" in line:
                m = re.search(r"([\d.]+)\s*GB", line)
                if m:
                    return float(m.group(1))
    return None


def display_label(model_family, config_label):
    """For quadcubes family, append _4_128 if missing."""
    if model_family == "quadcubes" and "_4_128" not in config_label:
        return config_label + "_4_128"
    return config_label


def main():
    OUT_CSV.parent.mkdir(exist_ok=True)
    crops = load_crops()
    ref = load_image_gray(PERFECT)

    rows = []

    for model_dir in sorted(SIZEDIFF.iterdir()):
        if not model_dir.is_dir():
            continue
        model_family = model_dir.name

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            label = display_label(model_family, config_dir.name)
            vram_gb = parse_vram_gb(config_dir / "vram.txt")
            vram_str = vram_gb if vram_gb is not None else ""

            time_dir = config_dir / "time"
            if time_dir.exists():
                images = sorted(time_dir.glob("*_1400.png"))
                for idx, img_path in enumerate(images):
                    cand = load_image_gray(img_path)
                    ssim, psnr, mae = compute_all_metrics(ref, cand, crops)
                    rows.append({
                        "model": model_family,
                        "config": label,
                        "source": "time",
                        "epoch": "",
                        "time_h": HOURS_PER_IMAGE * idx,
                        "psnr": round(psnr, 4),
                        "ssim": round(ssim, 6),
                        "mae": round(mae, 4),
                        "vram_gb": vram_str,
                    })
                print(f"  {model_family}/{label} time:  {len(images)} images")

            epoch_dir = config_dir / "epoch"
            if epoch_dir.exists():
                images = sorted(epoch_dir.glob("*_1400.png"))
                for img_path in images:
                    m = re.match(r"(\d+)_", img_path.name)
                    epoch_num = int(m.group(1)) if m else ""
                    cand = load_image_gray(img_path)
                    ssim, psnr, mae = compute_all_metrics(ref, cand, crops)
                    rows.append({
                        "model": model_family,
                        "config": label,
                        "source": "epoch",
                        "epoch": epoch_num,
                        "time_h": "",
                        "psnr": round(psnr, 4),
                        "ssim": round(ssim, 6),
                        "mae": round(mae, 4),
                        "vram_gb": vram_str,
                    })
                print(f"  {model_family}/{label} epoch: {len(images)} images")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "config", "source", "epoch", "time_h", "psnr", "ssim", "mae", "vram_gb"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
