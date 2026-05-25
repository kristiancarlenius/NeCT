#!/usr/bin/env python3
"""
Forward-project the neural model through the acquisition geometry.

Produces a Digitally Reconstructed Radiograph (DRR) by ray-marching through
the model at a chosen acquisition angle — the same viewpoint as the raw
X-ray projections.  Optionally places the raw projection alongside for
direct comparison.

Edit the CONFIG section and run on a GPU node.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.filters import threshold_otsu

import nect.sampling.ct_sampling as ct_sampling
from nect.config import get_cfg
from nect.data import NeCTDataset
from nect.sampling import Geometry

# ─────────────────────────────────── CONFIG ───────────────────────────────────

# Path to the run directory that contains config.yaml and model/checkpoints/
MODEL_DIR = Path("/cluster/home/kristiac/NeCT/outputs/dynamic_continious/quadcubes_22_4_22_16_2_4_128_L1/4fps_2750_ac4/model")

# Which projection angle to forward-project (0-based index into geometry.angles)
ANGLE_IDX = 0

# For dynamic models: set to a float in [0, 1] to override the timestep,
# or leave as None to use the angle's own acquisition timestep.
TIMESTEP_OVERRIDE = None

# Integration quality — more points → sharper / less noise, slower
POINTS_PER_RAY = 512

# Rays processed per GPU batch (reduce if you hit OOM)
BATCH_RAYS = 8192

# "attenuation" : line integral (looks like the raw X-ray projection)
# "binary"      : threshold each point → lit if any sample along the ray is
#                 above threshold (binary MIP / ray hit test — shows structure)
PROJECTION_MODE = "binary"

# Threshold applied to the raw model output (before calibration, so in [0, 1]).
# None → Otsu computed from a mid-volume sample before the main pass.
# float → manual value, e.g. 0.02
THRESHOLD = None

# Optional: path to raw projections .npy file for a side-by-side comparison.
# Set to None to show only the DRR.
RAW_PROJ_NPY = None

# Where to save the output (None = show interactively)
SAVE_PATH = MODEL_DIR  # e.g. Path("drr_angle0.png")

# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: Path):
    config = get_cfg(model_dir / "config.yaml")
    model = config.get_model()
    ckpt_path = model_dir / "checkpoints" / "last.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return config, model


def get_timestep(config, geometry: Geometry, angle_idx: int) -> float:
    n = len(geometry.angles)
    if config.geometry.timesteps is not None:
        ts = np.asarray(config.geometry.timesteps, dtype=np.float32)
        ts = ts / ts.max()
    else:
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
    return float(ts[angle_idx])


def _eval_batch(model, pts: torch.Tensor, config, t: float | None) -> torch.Tensor:
    """Evaluate model on a flat [N, 3] point tensor, returning [N, 1] float32."""
    SUB = 5_000_000
    chunks = []
    for s in range(0, pts.shape[0], SUB):
        chunk = pts[s : s + SUB]
        if config.mode == "dynamic":
            out = model(chunk, float(t)).float()
        else:
            out = model(chunk).float()
        chunks.append(out)
    return torch.cat(chunks)


def _estimate_threshold(model, config, geometry: Geometry, t: float | None) -> float:
    """Sample the mid-axial slice of the volume and return Otsu threshold."""
    print("  Computing Otsu threshold from mid-volume sample ...")
    nV = config.geometry.nVoxel
    n_pts = nV[1] * nV[2]
    y_coords = torch.linspace(0.0, 1.0, nV[1], device=device)
    x_coords = torch.linspace(0.0, 1.0, nV[2], device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    z_mid = torch.full((n_pts,), 0.5, device=device)
    pts = torch.stack([z_mid, yy.flatten(), xx.flatten()], dim=1)
    with torch.no_grad():
        vals = _eval_batch(model, pts, config, t).squeeze(1).cpu().numpy()
    thresh = float(threshold_otsu(vals))
    print(f"  Otsu threshold (raw model output): {thresh:.5f}")
    return thresh


def forward_project(model, config, geometry: Geometry, angle_rad: float, t: float | None, thresh: float | None) -> np.ndarray:
    """Ray-march through the model at `angle_rad`, return [nDet_h, nDet_w] image.

    PROJECTION_MODE == "attenuation": line integral (looks like raw X-ray)
    PROJECTION_MODE == "binary"     : 1 if any sample along the ray is above
                                      thresh, 0 otherwise (binary hit test)
    """
    c_geom = geometry.get_c_geometry()
    nDet_h, nDet_w = geometry.nDetector
    total_pixels = nDet_h * nDet_w
    max_dist = float(geometry.max_distance_traveled)
    dist_per_point = max_dist / POINTS_PER_RAY

    all_indices = torch.arange(total_pixels, dtype=torch.int64, device=device)

    # ray_points tensor is [n_rays, POINTS_PER_RAY, 3] — clamp batch size so
    # the total element count stays within int32 (ct_sampling uses 32-bit indexing)
    max_safe_rays = max(1, (2**31 - 1) // (POINTS_PER_RAY * 3))
    batch_rays = min(BATCH_RAYS, max_safe_rays)
    if batch_rays < BATCH_RAYS:
        print(f"  [info] BATCH_RAYS clamped {BATCH_RAYS} → {batch_rays} to stay within int32")

    projection = torch.zeros(total_pixels, dtype=torch.float32)

    for start in range(0, total_pixels, batch_rays):
        end = min(start + batch_rays, total_pixels)
        n_rays = end - start

        ray_points, distances = ct_sampling.sample(
            random_ray_index=all_indices,
            geometry=c_geom,
            angle_rad=angle_rad,
            num_points_per_ray=POINTS_PER_RAY,
            num_rays=n_rays,
            starting_ray_index=start,
            max_ray_distance_per_point=dist_per_point,
            uniform_ray_spacing=True,
            random_detector_offset=0.0,
            device=device.index if device.index is not None else 0,
        )

        with torch.no_grad():
            pts = ray_points.view(-1, 3)
            pts.clamp_(0.0, 1.0)

            zero_mask = torch.all(pts == 0.0, dim=-1)
            atten = torch.zeros(pts.shape[0], 1, device=device, dtype=torch.float32)
            valid_pts = pts[~zero_mask]
            if valid_pts.shape[0] > 0:
                atten[~zero_mask] = _eval_batch(model, valid_pts, config, t)

            atten = atten.view(n_rays, POINTS_PER_RAY)  # [rays, pts]

            if PROJECTION_MODE == "binary":
                # 1.0 if any point along the ray is above threshold, else 0.0
                y_pred = (atten > thresh).any(dim=1).float()
            else:
                y_pred = atten.sum(dim=1) * (distances / max_dist)

        projection[start:end] = y_pred.cpu()

    return projection.numpy().reshape(nDet_h, nDet_w)


def main():
    print(f"Loading model from {MODEL_DIR} ...")
    config, model = load_model(MODEL_DIR)

    geometry = Geometry.from_cfg(
        config.geometry,
        reconstruction_mode=config.reconstruction_mode,
        sample_outside=config.sample_outside,
    )

    angle_rad = float(geometry.angles[ANGLE_IDX])
    angle_deg = float(np.degrees(angle_rad))

    if config.mode == "dynamic":
        t = TIMESTEP_OVERRIDE if TIMESTEP_OVERRIDE is not None else get_timestep(config, geometry, ANGLE_IDX)
        print(f"Angle {ANGLE_IDX}  ({angle_deg:.1f}°)  |  t = {t:.4f}")
    else:
        t = None
        print(f"Angle {ANGLE_IDX}  ({angle_deg:.1f}°)  |  static model")

    thresh = None
    if PROJECTION_MODE == "binary":
        thresh = THRESHOLD if THRESHOLD is not None else _estimate_threshold(model, config, geometry, t)

    mode_str = f"{PROJECTION_MODE}" + (f"  thresh={thresh:.5f}" if thresh is not None else "")
    print(f"Forward-projecting  ({geometry.nDetector[0]} × {geometry.nDetector[1]} px,  {POINTS_PER_RAY} pts/ray  [{mode_str}]) ...")
    drr = forward_project(model, config, geometry, angle_rad, t, thresh)
    print(f"  Output range: [{drr.min():.4f}, {drr.max():.4f}]")

    # ── Plot ──────────────────────────────────────────────────────────────────
    has_raw = RAW_PROJ_NPY is not None
    ncols = 2 if has_raw else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    if PROJECTION_MODE == "binary":
        im0 = axes[0].imshow(drr, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    else:
        vmin, vmax = float(np.percentile(drr, 1)), float(np.percentile(drr, 99))
        im0 = axes[0].imshow(drr, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")

    title = f"{'Binary hit' if PROJECTION_MODE == 'binary' else 'DRR'} — angle {ANGLE_IDX} ({angle_deg:.1f}°)"
    if t is not None:
        title += f"  t={t:.3f}"
    axes[0].set_title(title)
    axes[0].set_xlabel("Detector u (px)")
    axes[0].set_ylabel("Detector v (px)")
    if PROJECTION_MODE != "binary":
        plt.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.04)

    if has_raw:
        raw = np.load(RAW_PROJ_NPY)
        raw_proj = raw[ANGLE_IDX].astype(np.float32)
        rmin, rmax = float(np.percentile(raw_proj, 1)), float(np.percentile(raw_proj, 99))
        im1 = axes[1].imshow(raw_proj, cmap="gray", vmin=rmin, vmax=rmax, aspect="auto")
        axes[1].set_title(f"Raw projection — angle {ANGLE_IDX} ({angle_deg:.1f}°)")
        axes[1].set_xlabel("Detector u (px)")
        axes[1].set_ylabel("Detector v (px)")
        plt.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.04)

    plt.tight_layout()

    if SAVE_PATH is not None:
        plt.savefig(SAVE_PATH, dpi=150)
        print(f"Saved to {SAVE_PATH}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
