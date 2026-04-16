"""
Hourglass sand volume analysis.

For each timestep, queries the dynamic model to get a calibrated 3D volume,
segments sand via threshold (Otsu or manual), splits into top/bottom chambers
at a user-defined neck z-voxel index, and plots sand volume (mm³) over time.

Geometry for this dataset:
  nVoxel = [1148, 748, 748]   — z is the tall/vertical axis of the hourglass
  dVoxel = ~0.1377 mm (isotropic)
  With BINNING=4: output shape is (287, 187, 187), neck auto-midpoint at z=143

Usage:
    Edit the CONFIG section below and run on a GPU node.
    After a first quick run (BINNING=8, N_TIMESTEPS=10) inspect the Otsu
    threshold and neck position, then re-run with BINNING=4, N_TIMESTEPS=50.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.filters import threshold_otsu
from tqdm import tqdm

from nect.config import get_cfg
from nect.data import NeCTDataset
from nect.sampling import Geometry

# ─────────────────────────── CONFIG ──────────────────────────────────────────

# Directory containing config.yaml and checkpoints/ subfolder
MODEL_PATH = "/cluster/home/kristiac/NeCT/outputs/dynamic_continious/quadcubes_21_4_21_16_2_4_128_L1/2026-04-16T02-30-34/model/"

# How many evenly-spaced timesteps to sample across the full acquisition
N_TIMESTEPS = 50

# Spatial binning factor (4 = 4× faster/lower-res; 1 = full resolution)
BINNING = 1

# Optional ROI in *full-resolution* voxel coordinates [start, end].
# Set to None to use the full volume. Trim air around the hourglass to
# reduce noise contribution to the Otsu threshold.
# Values below are binned coords (BINNING=8) × 8 → full-res.
ROI_Z = [240, 1056]   # binned 30–132  (vertical axis, nVoxel[0]=1148)
ROI_Y = [136, 560]    # binned 17–70   (nVoxel[1]=748)
ROI_X = [184, 600]    # binned 23–75   (nVoxel[2]=748)

# Z-voxel index (in the *binned, ROI-cropped* output) of the hourglass neck.
# With ROI_Z=[240,1056] the cropped volume starts at binned z=30, so the
# original binned neck at z=75 becomes index 75-30 = 45 here.
NECK_Z_VOXEL = 370

# Attenuation threshold for segmenting sand vs air.
# None → Otsu's method applied to the first timestep volume (recommended).
# Override if Otsu picks the wrong region (inspect threshold_check.png).
THRESHOLD = 0.15 # e.g. 0.025

# Output directory (sits next to the model/ folder)
OUTPUT_DIR = Path(MODEL_PATH).parent

# ─────────────────────────────────────────────────────────────────────────────


def query_volume(
    model,
    t: float,
    z_lin: torch.Tensor,
    y_lin: torch.Tensor,
    x_lin: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Returns calibrated 3D volume (z, y, x) for a single timestep t in [0,1]."""
    z_h, y_w, x_w = len(z_lin), len(y_lin), len(x_lin)
    output = torch.zeros((z_h, y_w, x_w), device="cpu", dtype=torch.float32)
    t_tensor = torch.tensor(t, device=device)
    for ii, z_ in enumerate(z_lin):
        z, y, x = torch.meshgrid(
            [z_.unsqueeze(0), y_lin, x_lin],
            indexing="ij",
        )
        grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t().to(device)
        output[ii] = model(grid, t_tensor).view(y_w, x_w).cpu()
    return output.numpy()


def main():
    base_path = Path(MODEL_PATH)
    device = torch.device(0)

    print("Loading config and model...")
    config = get_cfg(base_path / "config.yaml")
    assert config.geometry is not None, "No geometry in config"
    assert config.mode == "dynamic", "Model must be in dynamic mode"

    model = config.get_model()
    checkpoints = torch.load(base_path / "checkpoints" / "last.ckpt", map_location="cpu")
    model.load_state_dict(checkpoints["model"])
    model = model.to(device)
    model.eval()

    dataset = NeCTDataset(config=config, device="cpu")
    geometry = Geometry.from_cfg(
        config.geometry,
        reconstruction_mode=config.reconstruction_mode,
        sample_outside=config.sample_outside,
    )

    # ── Voxel dimensions ─────────────────────────────────────────────────────
    nVoxels_raw = list(config.geometry.nVoxel)   # [nz, ny, nx]
    dVoxel = list(config.geometry.dVoxel)        # [dz, dy, dx] in mm
    rm = config.sample_outside
    nVoxels = [nVoxels_raw[0], nVoxels_raw[1] + 2 * rm, nVoxels_raw[2] + 2 * rm]

    # Physical size of one (binned) voxel in mm³
    voxel_vol_mm3 = (dVoxel[0] * BINNING) * (dVoxel[1] * BINNING) * (dVoxel[2] * BINNING)
    print(f"Binned voxel volume: {voxel_vol_mm3:.4f} mm³")

    # ── Spatial coordinate ranges ─────────────────────────────────────────────
    def roi_coords(roi, n_full, n_voxels, rm_offset=0):
        """Returns (start, end, n_bins) for one axis."""
        if roi is None:
            return 0.0, 1.0, n_full // BINNING
        n_bins = (roi[1] - roi[0]) // BINNING
        start = (roi[0] - rm_offset) / n_voxels
        end = (roi[1] - rm_offset) / n_voxels
        return start, end, n_bins

    start_z, end_z, z_h = roi_coords(ROI_Z, nVoxels_raw[0], nVoxels[0], rm_offset=0)
    start_y, end_y, y_w = roi_coords(ROI_Y, nVoxels_raw[1], nVoxels[1], rm_offset=rm)
    start_x, end_x, x_w = roi_coords(ROI_X, nVoxels_raw[2], nVoxels[2], rm_offset=rm)

    print(f"Volume shape per timestep: ({z_h}, {y_w}, {x_w})")

    z_lin = torch.linspace(start_z, end_z, steps=z_h, device=device)
    y_lin = torch.linspace(start_y, end_y, steps=y_w, device=device)
    x_lin = torch.linspace(start_x, end_x, steps=x_w, device=device)

    # ── Normalisation constants (same as export_volumes) ─────────────────────
    scale = 1.0 / geometry.max_distance_traveled
    data_min = dataset.minimum.item()
    data_max = dataset.maximum.item()

    def calibrate(raw: np.ndarray) -> np.ndarray:
        return raw * scale * (data_max - data_min) + data_min

    # ── Timestep schedule ─────────────────────────────────────────────────────
    angles = config.geometry.angles
    t_values = np.linspace(0.0, 1.0, N_TIMESTEPS, endpoint=False)

    # ── First volume: threshold + diagnostic images ───────────────────────────
    print("Querying first timestep volume for threshold / neck diagnostics...")
    with torch.no_grad():
        vol0_raw = query_volume(model, float(t_values[0]), z_lin, y_lin, x_lin, device)
    vol0 = calibrate(vol0_raw)

    threshold = THRESHOLD
    if threshold is None:
        threshold = threshold_otsu(vol0)
        print(f"  Otsu threshold = {threshold:.4f}")
    else:
        print(f"  Using manual threshold = {threshold:.4f}")

    # ── Neck split ────────────────────────────────────────────────────────────
    neck_z = NECK_Z_VOXEL if NECK_Z_VOXEL is not None else z_h // 2

    # Save diagnostic slices so you can verify the ROI and neck position
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mid_y = y_w // 2
    mid_x = x_w // 2

    # Percentile-clipped display range so air doesn't crush the contrast
    vmin = float(np.percentile(vol0, 1))
    vmax = float(np.percentile(vol0, 99))
    print(f"  Display range: vmin={vmin:.4f}  vmax={vmax:.4f}  "
          f"(1st–99th percentile of full volume)")

    fig_nc, axes_nc = plt.subplots(2, 2, figsize=(14, 10))

    def show(ax, img, title, xlabel, ylabel, add_neck=False):
        im = ax.imshow(img, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        if add_neck:
            ax.axhline(neck_z, color="red", linewidth=1.5, linestyle="--",
                       label=f"neck z={neck_z}")
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    # Row 0: XZ slice (mid-Y)
    show(axes_nc[0, 0], vol0[:, mid_y, :],
         "XZ slice (mid-Y) — raw attenuation",
         "x voxel (binned)", "z voxel (binned)  [0=top]", add_neck=True)

    axes_nc[0, 1].imshow(vol0[:, mid_y, :] > threshold, cmap="gray", aspect="auto")
    axes_nc[0, 1].axhline(neck_z, color="red", linewidth=1.5, linestyle="--",
                           label=f"neck z={neck_z}")
    axes_nc[0, 1].set_title(f"Sand mask — XZ (mid-Y), threshold={threshold:.4f}")
    axes_nc[0, 1].set_xlabel("x voxel (binned)")
    axes_nc[0, 1].legend(fontsize=8)

    # Row 1: YZ slice (mid-X) and XY slice at neck
    show(axes_nc[1, 0], vol0[:, :, mid_x],
         "YZ slice (mid-X) — raw attenuation",
         "y voxel (binned)", "z voxel (binned)  [0=top]", add_neck=True)

    show(axes_nc[1, 1], vol0[neck_z, :, :],
         f"XY slice at neck z={neck_z} — raw attenuation",
         "x voxel (binned)", "y voxel (binned)", add_neck=False)

    plt.tight_layout()
    nc_path = out_dir / "neck_check.png"
    plt.savefig(nc_path, dpi=150)
    plt.close(fig_nc)
    print(f"  Diagnostic slices saved to {nc_path}")
    print(f"  → Red dashed line marks neck split at z={neck_z} (of {z_h})")
    print(f"  → Adjust NECK_Z_VOXEL if the line doesn't sit at the hourglass neck")
    print(f"Neck split at binned z-index {neck_z} (of {z_h} total z-slices)")

    # ── Main loop ─────────────────────────────────────────────────────────────
    top_vols_mm3 = []
    bot_vols_mm3 = []

    with torch.no_grad():
        for i, t in enumerate(tqdm(t_values, desc="Timesteps")):
            # Reuse already-queried first volume
            if i == 0:
                vol = vol0
            else:
                raw_vol = query_volume(model, float(t), z_lin, y_lin, x_lin, device)
                vol = calibrate(raw_vol)

            sand_mask = vol > threshold  # True where sand

            top_voxels = sand_mask[:neck_z].sum()
            bot_voxels = sand_mask[neck_z:].sum()

            top_vols_mm3.append(top_voxels * voxel_vol_mm3)
            bot_vols_mm3.append(bot_voxels * voxel_vol_mm3)

    top_vols_mm3 = np.array(top_vols_mm3)
    bot_vols_mm3 = np.array(bot_vols_mm3)
    total_mm3 = top_vols_mm3 + bot_vols_mm3

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Convert t in [0,1] to acquisition index for the x-axis
    t_axis = t_values * len(angles)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(t_axis, top_vols_mm3, label="Top chamber", color="steelblue")
    ax.plot(t_axis, bot_vols_mm3, label="Bottom chamber", color="firebrick")
    ax.plot(t_axis, total_mm3, label="Total", color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel("Sand volume (mm³)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Hourglass sand volume over time")

    ax2 = axes[1]
    ax2.plot(t_axis, top_vols_mm3 / (total_mm3 + 1e-9) * 100, color="steelblue", label="Top %")
    ax2.plot(t_axis, bot_vols_mm3 / (total_mm3 + 1e-9) * 100, color="firebrick", label="Bottom %")
    ax2.set_ylabel("Fraction of sand (%)")
    ax2.set_xlabel("Projection index")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plot_path = out_dir / "sand_volume.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

    # Also save the raw numbers as .npz for later use / 3D visualisation
    npz_path = out_dir / "sand_volume.npz"
    np.savez(
        npz_path,
        t_values=t_values,
        projection_indices=t_axis,
        top_volume_mm3=top_vols_mm3,
        bottom_volume_mm3=bot_vols_mm3,
        threshold=threshold,
        neck_z_voxel=neck_z,
        binning=BINNING,
        voxel_vol_mm3=voxel_vol_mm3,
    )
    print(f"Raw data saved to {npz_path}")
    plt.show()


if __name__ == "__main__":
    main()
