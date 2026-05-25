#!/usr/bin/env python3
"""
Binary overlay comparison of two neural CT reconstructions at one timestep.

Reconstructs the full 3D volume from both models, thresholds to binary, and
composites them as a false-colour image:
  Model A only  →  Red
  Model B only  →  Blue
  Both present  →  Purple  (agreement)
  Neither       →  Black

Three orthogonal slices are shown: axial (XY), coronal (XZ), sagittal (YZ).

Edit the CONFIG section and run on a GPU node.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.filters import threshold_otsu
from tqdm import tqdm

from nect.config import get_cfg
from nect.data import NeCTDataset
from nect.sampling import Geometry

# ─────────────────────────────────── CONFIG ───────────────────────────────────

# Paths to the two run directories (each must contain config.yaml and
# model/checkpoints/last.ckpt)
MODEL_A_DIR = Path("/cluster/home/kristiac/NeCT/outputs/dynamic_continious/quadcubes_22_4_22_16_2_4_128_L1/4fps_11000_ac2/model")
MODEL_B_DIR = Path("/cluster/home/kristiac/NeCT/outputs/dynamic_continious/quadcubes_22_4_22_16_2_4_128_L1/4fps_11000_ac6/model")

# Labels shown in the legend
NAME_A = "ac2"
NAME_B = "ac6"

# Timestep in [0, 1] through the acquisition (0 = start, 1 = end)
TIMESTEP = 0.5

# Threshold applied to the calibrated attenuation volumes.
# None → Otsu threshold computed from Model A's volume.
# float → manual threshold in the same physical units as the projections.
THRESHOLD = None

# Spatial binning (4 = 4× fewer voxels per axis, much faster)
BINNING = 4

# Optional ROI in full-resolution voxel coordinates [start, end].
# None = full volume.
ROI_Z = None
ROI_Y = None
ROI_X = None

# Fractional position [0, 1] along each axis for the three slice planes.
SLICE_FRAC_Z = 0.5   # axial slice
SLICE_FRAC_Y = 0.5   # coronal slice
SLICE_FRAC_X = 0.5   # sagittal slice

# Where to save the figure (None = show interactively)
SAVE_PATH = None  # e.g. Path("comparison_ac2_ac6.png")

# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device(0)

COLOR_A    = np.array([220,  30,  30], dtype=np.uint8)   # red
COLOR_B    = np.array([ 30,  30, 220], dtype=np.uint8)   # blue
COLOR_BOTH = np.array([210,  30, 210], dtype=np.uint8)   # purple
COLOR_NONE = np.array([ 15,  15,  15], dtype=np.uint8)   # near-black


def load_model(run_dir: Path):
    config = get_cfg(run_dir / "config.yaml")
    model = config.get_model()
    ckpt = torch.load(run_dir / "checkpoints" / "last.ckpt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return config, model.to(DEVICE).eval()


def build_grid(config, binning: int):
    """Return (z_lin, y_lin, x_lin, calibrate_fn) for the configured geometry."""
    geometry = Geometry.from_cfg(
        config.geometry,
        reconstruction_mode=config.reconstruction_mode,
        sample_outside=config.sample_outside,
    )
    dataset = NeCTDataset(config=config, device="cpu")

    nVoxels_raw = list(config.geometry.nVoxel)
    rm = config.sample_outside
    nVoxels = [nVoxels_raw[0], nVoxels_raw[1] + 2 * rm, nVoxels_raw[2] + 2 * rm]

    def roi_coords(roi, n_full, n_voxels, rm_offset=0):
        if roi is None:
            return 0.0, 1.0, n_full // binning
        n_bins = (roi[1] - roi[0]) // binning
        start  = (roi[0] - rm_offset) / n_voxels
        end    = (roi[1] - rm_offset) / n_voxels
        return start, end, n_bins

    start_z, end_z, z_h = roi_coords(ROI_Z, nVoxels_raw[0], nVoxels[0], rm_offset=0)
    start_y, end_y, y_w = roi_coords(ROI_Y, nVoxels_raw[1], nVoxels[1], rm_offset=rm)
    start_x, end_x, x_w = roi_coords(ROI_X, nVoxels_raw[2], nVoxels[2], rm_offset=rm)

    z_lin = torch.linspace(start_z, end_z, steps=z_h, device=DEVICE)
    y_lin = torch.linspace(start_y, end_y, steps=y_w, device=DEVICE)
    x_lin = torch.linspace(start_x, end_x, steps=x_w, device=DEVICE)

    scale    = 1.0 / float(geometry.max_distance_traveled)
    data_min = dataset.minimum.item()
    data_max = dataset.maximum.item()

    def calibrate(raw: np.ndarray) -> np.ndarray:
        return raw * scale * (data_max - data_min) + data_min

    return z_lin, y_lin, x_lin, calibrate


def query_volume(model, t: float, z_lin, y_lin, x_lin, is_dynamic: bool) -> np.ndarray:
    z_h, y_w, x_w = len(z_lin), len(y_lin), len(x_lin)
    output = torch.zeros((z_h, y_w, x_w), device="cpu", dtype=torch.float32)
    t_tensor = torch.tensor(t, device=DEVICE)
    with torch.no_grad():
        for ii, z_ in enumerate(tqdm(z_lin, desc="  z-slices", leave=False)):
            z, y, x = torch.meshgrid([z_.unsqueeze(0), y_lin, x_lin], indexing="ij")
            grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t()
            if is_dynamic:
                out = model(grid, t_tensor).float()
            else:
                out = model(grid).float()
            output[ii] = out.view(y_w, x_w).cpu()
    return output.numpy()


def overlay_rgb(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Build [H, W, 3] uint8 false-colour image from two boolean masks."""
    h, w = mask_a.shape
    img = np.full((h, w, 3), COLOR_NONE, dtype=np.uint8)
    img[mask_a & ~mask_b] = COLOR_A
    img[~mask_a & mask_b] = COLOR_B
    img[mask_a & mask_b]  = COLOR_BOTH
    return img


def main():
    # ── Load both models ──────────────────────────────────────────────────────
    print(f"Loading {NAME_A} from {MODEL_A_DIR} ...")
    cfg_a, model_a = load_model(MODEL_A_DIR)
    print(f"Loading {NAME_B} from {MODEL_B_DIR} ...")
    cfg_b, model_b = load_model(MODEL_B_DIR)

    # ── Build sampling grids (use model A's geometry for both) ────────────────
    z_lin, y_lin, x_lin, calibrate = build_grid(cfg_a, BINNING)
    z_h, y_w, x_w = len(z_lin), len(y_lin), len(x_lin)
    print(f"Volume shape (binned): ({z_h}, {y_w}, {x_w})  t={TIMESTEP:.3f}")

    # ── Query volumes ─────────────────────────────────────────────────────────
    print(f"Querying {NAME_A} ...")
    vol_a = calibrate(query_volume(model_a, TIMESTEP, z_lin, y_lin, x_lin, cfg_a.mode == "dynamic"))

    print(f"Querying {NAME_B} ...")
    # Re-build grid with model B's geometry (same dataset → same values, but
    # sample_outside may differ; safer to use B's own grid)
    z_lin_b, y_lin_b, x_lin_b, calibrate_b = build_grid(cfg_b, BINNING)
    vol_b = calibrate_b(query_volume(model_b, TIMESTEP, z_lin_b, y_lin_b, x_lin_b, cfg_b.mode == "dynamic"))

    # Crop vol_b to match vol_a if shapes differ (different sample_outside)
    if vol_b.shape != vol_a.shape:
        sz = min(vol_a.shape[0], vol_b.shape[0])
        sy = min(vol_a.shape[1], vol_b.shape[1])
        sx = min(vol_a.shape[2], vol_b.shape[2])
        vol_a = vol_a[:sz, :sy, :sx]
        vol_b = vol_b[:sz, :sy, :sx]
        z_h, y_w, x_w = sz, sy, sx

    # ── Threshold → binary masks ──────────────────────────────────────────────
    thresh = THRESHOLD if THRESHOLD is not None else float(threshold_otsu(vol_a))
    print(f"Threshold: {thresh:.4f}{'  (Otsu on ' + NAME_A + ')' if THRESHOLD is None else '  (manual)'}")
    mask_a = vol_a > thresh
    mask_b = vol_b > thresh

    pct_a    = mask_a.mean() * 100
    pct_b    = mask_b.mean() * 100
    pct_both = (mask_a & mask_b).mean() * 100
    print(f"  {NAME_A} foreground: {pct_a:.1f}%")
    print(f"  {NAME_B} foreground: {pct_b:.1f}%")
    print(f"  Agreement (both):    {pct_both:.1f}%")

    # ── Slice indices ─────────────────────────────────────────────────────────
    iz = int(SLICE_FRAC_Z * (z_h - 1))
    iy = int(SLICE_FRAC_Y * (y_w - 1))
    ix = int(SLICE_FRAC_X * (x_w - 1))

    # ── Build overlay images ──────────────────────────────────────────────────
    # axial   — vol[iz, :, :]  → y (rows) × x (cols)
    # coronal — vol[:, iy, :]  → z (rows) × x (cols)
    # sagittal— vol[:, :, ix]  → z (rows) × y (cols)
    panels = [
        (overlay_rgb(mask_a[iz, :, :],  mask_b[iz, :, :]),  f"Axial z={iz}"),
        (overlay_rgb(mask_a[:, iy, :],  mask_b[:, iy, :]),  f"Coronal y={iy}"),
        (overlay_rgb(mask_a[:, :, ix],  mask_b[:, :, ix]),  f"Sagittal x={ix}"),
    ]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"{NAME_A} vs {NAME_B}  |  t={TIMESTEP:.3f}  |  threshold={thresh:.4f}  |  binning={BINNING}×",
        fontsize=11,
    )

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=np.array(COLOR_A)    / 255, label=f"{NAME_A} only"),
        mpatches.Patch(color=np.array(COLOR_B)    / 255, label=f"{NAME_B} only"),
        mpatches.Patch(color=np.array(COLOR_BOTH) / 255, label="Both (agreement)"),
        mpatches.Patch(color=np.array(COLOR_NONE) / 255, label="Neither"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               framealpha=0.9, fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if SAVE_PATH is not None:
        plt.savefig(SAVE_PATH, dpi=150)
        print(f"Saved to {SAVE_PATH}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
