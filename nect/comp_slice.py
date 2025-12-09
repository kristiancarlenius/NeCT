#!/usr/bin/env python3
"""
Compare two grayscale images (CT slices) using a cropped region.
- Paths are defined as strings in the code (no CLI args).
- Both images are cropped with the same (x, y) coordinates.
- Displays:
    - Reference (cropped)
    - Test (cropped)
    - Absolute error heatmap
    - Error overlay on reference
"""

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
REF_PATH = r"/home/user/Documents/img_comp/pr360_ac6/0300_0360.png"   # path to your "perfect" image
TEST_PATH = r"/home/user/Documents/img_comp/pr1400_ac1/0325_1400.png" # path to your reconstruction image

# Crop rectangle (in pixel coordinates)
# (x0, y0) is top-left; (x1, y1) is bottom-right (exclusive)
# Example: crop from x=100..300, y=150..350
CROP_X0 = 4800
CROP_Y0 = 250
CROP_X1 = 5650
CROP_Y1 = 1550
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
              metrics: dict):
    """
    Show reference, test, error heatmap, and overlay.
    """
    mse = metrics["MSE"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1) Reference
    axes[0].imshow(ref_crop, cmap="gray")
    axes[0].set_title("360 projections 400 epochs")
    axes[0].axis("off")

    # 2) Test
    axes[1].imshow(test_crop, cmap="gray")
    axes[1].set_title("1400 projections 350 epochs")
    axes[1].axis("off")

    # 3) Error heatmap
    im2 = axes[2].imshow(err_norm, cmap="hot")
    axes[2].set_title(f"Abs error\nMSE={mse:.6f}")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4) Error overlay on reference
    axes[3].imshow(ref_crop, cmap="gray")
    # alpha proportional to error: tune scale factor if needed
    alpha = np.clip(err_norm * 10.0, 0.0, 1)
    im3 = axes[3].imshow(err_norm, cmap="hot", alpha=alpha)
    axes[3].set_title("360 projections 400 epochs + error overlay")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def main():
    # Load full images
    ref_full = load_grayscale(REF_PATH)
    test_full = load_grayscale(TEST_PATH)

    print(f"Loaded ref:  {REF_PATH}, shape={ref_full.shape}")
    print(f"Loaded test: {TEST_PATH}, shape={test_full.shape}")

    # Crop both images with the same coordinates
    ref_crop = crop_image(ref_full, CROP_X0, CROP_Y0, CROP_X1, CROP_Y1)
    test_crop = crop_image(test_full, CROP_X0, CROP_Y0, CROP_X1, CROP_Y1)

    print(f"Cropped shape: {ref_crop.shape}")

    # Compute metrics
    metrics, abs_err, ssim_map = compute_metrics(ref_crop, test_crop)

    # Normalize error for visualization
    err_norm = 2*abs_err / (abs_err.max() + 1e-8)

    # Print metrics
    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    if not _HAS_SKIMAGE:
        print("(SSIM not computed: scikit-image not installed)")

    # Visualize
    visualize(ref_crop, test_crop, err_norm, metrics)

    # Optional: if you want to also visualize SSIM map when available
    if ssim_map is not None:
        plt.figure(figsize=(5, 4))
        plt.imshow(ssim_map, cmap="viridis")
        plt.title("SSIM map")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()