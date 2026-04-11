#!/usr/bin/env python3
"""
Compare two grayscale images (CT slices) using cropped regions from crops.json.
- REF_PATH defaults to the source_image recorded in crops.json.
- TEST_PATH is the reconstruction to evaluate.
- All crop regions in crops.json are evaluated; metrics are printed for each.
- CROP_INDEX can be set to a specific integer to visualise only that crop,
  or None to visualise all crops.
"""

import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Try to import SSIM from scikit-image (optional)
try:
    from skimage.metrics import structural_similarity as ssim
    _HAS_SKIMAGE = True
except ImportError:
    ssim = None
    _HAS_SKIMAGE = False

# ================================
# CONFIG: EDIT THESE
# ================================
CROPS_JSON = r"/home/user/Documents/NeCT/crops.json"
# REF_PATH: leave as None to use source_image from crops.json
REF_PATH = None
TEST_PATH = r"/home/user/Documents/NeCT/sizediff/perfect/0500_1400.png"

# Set to an integer index to visualise only that crop, or None for all crops.
CROP_INDEX = None
# ================================


def load_grayscale(path: str) -> np.ndarray:
    """
    Load an image as grayscale float32 NumPy array in [0, 1].
    Assumes 8-bit input (0-255).
    """
    img = Image.open(path).convert("L")  # force grayscale
    arr = np.array(img, dtype=np.float32)
    arr /= 255.0  # scale to [0, 1]
    return arr


def crop_image(img: np.ndarray,
               x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """
    Crop image using the given coordinates.
    img shape: (H, W)
    """
    h, w = img.shape
    # Clamp coordinates to image bounds just in case
    x0_clamped = max(0, min(w, x0))
    x1_clamped = max(0, min(w, x1))
    y0_clamped = max(0, min(h, y0))
    y1_clamped = max(0, min(h, y1))

    if x1_clamped <= x0_clamped or y1_clamped <= y0_clamped:
        raise ValueError("Invalid crop coordinates after clamping.")

    return img[y0_clamped:y1_clamped, x0_clamped:x1_clamped]


def compute_metrics(ref: np.ndarray, test: np.ndarray):
    """
    Compute a set of error / similarity metrics between two images.
    ref, test: shape (H, W), values in [0, 1]
    Returns dict of metrics and an error map (abs diff).
    """
    if ref.shape != test.shape:
        raise ValueError(f"Shapes do not match: {ref.shape} vs {test.shape}")

    diff = ref - test
    abs_err = np.abs(diff)

    mae = float(np.mean(abs_err))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_abs_err = float(np.max(abs_err))

    # PSNR assuming max pixel value = 1.0
    eps = 1e-12
    psnr = float(10.0 * np.log10(1.0**2 / (mse + eps))) if mse > 0 else float("inf")

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MaxAbsErr": max_abs_err,
        "PSNR": psnr,
    }

    ssim_val = None
    ssim_map = None
    if _HAS_SKIMAGE:
        # data_range=1.0 since our data is in [0, 1]
        ssim_val, ssim_map = ssim(ref, test, data_range=1.0, full=True)
        metrics["SSIM"] = float(ssim_val)

    return metrics, abs_err, ssim_map


def visualize(ref_crop: np.ndarray,
              test_crop: np.ndarray,
              err_norm: np.ndarray,
              metrics: dict,
              crop_index: int = 0):
    """
    Show reference, test, error heatmap, and overlay.
    """
    mse = metrics["MSE"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Crop {crop_index}", fontsize=10)

    # 1) Reference
    axes[0].imshow(ref_crop, cmap="gray")
    axes[0].set_title("Reference")
    axes[0].axis("off")

    # 2) Test
    axes[1].imshow(test_crop, cmap="gray")
    axes[1].set_title("Test")
    axes[1].axis("off")

    # 3) Error heatmap
    im2 = axes[2].imshow(err_norm, cmap="hot")
    axes[2].set_title(f"Abs error\nMSE={mse:.6f}")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4) Error overlay on reference
    axes[3].imshow(ref_crop, cmap="gray")
    # alpha proportional to error: tune scale factor if needed
    alpha = np.clip(err_norm ** 0.4, 0.0, 1)
    im3 = axes[3].imshow(err_norm, cmap="hot", alpha=alpha)
    axes[3].set_title("Reference + error overlay")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def load_crops_json(path: str):
    """Load crops.json and return (ref_path, list of crop dicts)."""
    with open(path) as f:
        data = json.load(f)
    source_image = data.get("source_image")
    if source_image and not os.path.isabs(source_image):
        # Resolve relative to the JSON file's directory
        source_image = os.path.join(os.path.dirname(path), source_image)
    crops = data.get("crops", [])
    return source_image, crops


def main():
    # Load crop definitions
    source_image_from_json, crops = load_crops_json(CROPS_JSON)

    ref_path = REF_PATH if REF_PATH is not None else source_image_from_json
    if ref_path is None:
        raise ValueError("No REF_PATH set and crops.json has no source_image.")

    ref_full = load_grayscale(ref_path)
    test_full = load_grayscale(TEST_PATH)

    print(f"Loaded ref:  {ref_path}, shape={ref_full.shape}")
    print(f"Loaded test: {TEST_PATH}, shape={test_full.shape}")
    print(f"Crops loaded from: {CROPS_JSON} ({len(crops)} regions)")

    # Determine which crops to evaluate
    if CROP_INDEX is not None:
        indices = [CROP_INDEX]
    else:
        indices = list(range(len(crops)))

    all_metrics = []
    for i in indices:
        c = crops[i]
        x0, y0, x1, y1 = c["x0"], c["y0"], c["x1"], c["y1"]

        ref_crop = crop_image(ref_full, x0, y0, x1, y1)
        test_crop = crop_image(test_full, x0, y0, x1, y1)

        metrics, abs_err, ssim_map = compute_metrics(ref_crop, test_crop)
        all_metrics.append(metrics)
        err_norm = abs_err / (abs_err.max() + 1e-8)

        print(f"\n=== Crop {i} ({x0},{y0})→({x1},{y1})  shape={ref_crop.shape} ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
        if not _HAS_SKIMAGE:
            print("  (SSIM not computed: scikit-image not installed)")

        visualize(ref_crop, test_crop, err_norm, metrics, crop_index=i)

        if ssim_map is not None:
            plt.figure(figsize=(5, 4))
            plt.imshow(ssim_map, cmap="viridis")
            plt.title(f"SSIM map – crop {i}")
            plt.axis("off")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    # Print aggregate summary across all evaluated crops
    if len(all_metrics) > 1:
        print("\n=== Aggregate (mean across crops) ===")
        keys = list(all_metrics[0].keys())
        for k in keys:
            mean_val = float(np.mean([m[k] for m in all_metrics]))
            print(f"  {k}: {mean_val:.6f}")


if __name__ == "__main__":
    main()