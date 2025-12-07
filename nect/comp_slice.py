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

# ================================
# CONFIG: EDIT THESE
# ================================
REF_PATH = r"ref.png"   # path to your "perfect" image
TEST_PATH = r"test.png" # path to your reconstruction image

# Crop rectangle (in pixel coordinates)
# (x0, y0) is top-left; (x1, y1) is bottom-right (exclusive)
# Example: crop from x=100..300, y=150..350
CROP_X0 = 100
CROP_Y0 = 150
CROP_X1 = 300
CROP_Y1 = 350
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


def compute_error(ref: np.ndarray, test: np.ndarray):
    """
    Compute absolute error and MSE between two images.
    ref, test: shape (H, W), values in [0, 1]
    """
    if ref.shape != test.shape:
        raise ValueError(f"Shapes do not match: {ref.shape} vs {test.shape}")
    err = np.abs(ref - test)
    mse = np.mean((ref - test) ** 2)
    return err, mse


def visualize(ref_crop: np.ndarray,
              test_crop: np.ndarray,
              err_norm: np.ndarray,
              mse: float):
    """
    Show reference, test, error heatmap, and overlay.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1) Reference
    axes[0].imshow(ref_crop, cmap="gray")
    axes[0].set_title("Reference (cropped)")
    axes[0].axis("off")

    # 2) Test
    axes[1].imshow(test_crop, cmap="gray")
    axes[1].set_title("Test (cropped)")
    axes[1].axis("off")

    # 3) Error heatmap
    im2 = axes[2].imshow(err_norm, cmap="hot")
    axes[2].set_title(f"Abs error\nMSE={mse:.6f}")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4) Error overlay on reference
    axes[3].imshow(ref_crop, cmap="gray")
    # alpha proportional to error: tune scale factor if needed
    alpha = np.clip(err_norm * 2.0, 0.0, 1.0)
    im3 = axes[3].imshow(err_norm, cmap="hot", alpha=alpha)
    axes[3].set_title("Reference + error overlay")
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

    # Compute error
    err, mse = compute_error(ref_crop, test_crop)
    # Normalize error for visualization
    err_norm = err / (err.max() + 1e-8)

    print(f"Mean Squared Error (MSE) on cropped region: {mse:.6f}")

    # Visualize
    visualize(ref_crop, test_crop, err_norm, mse)


if __name__ == "__main__":
    main()

