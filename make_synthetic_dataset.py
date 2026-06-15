#!/usr/bin/env python3
"""
Generate a synthetic 4D CT dataset: two reference spheres + thin rod + sliding bead.

Phantom (all coordinates in [0,1]³, zyx order):
  - Left sphere   (fixed):  center (0.5, 0.5, 0.2), r=0.10, att=0.08
  - Right sphere  (fixed):  center (0.5, 0.5, 0.8), r=0.10, att=0.08
  - Rod           (fixed):  thin cylinder along x at (z=0.5, y=0.5), r=0.012, att=0.04
  - Bead          (moving): center (0.5, 0.5, bx(t)), r=0.05, att=0.15
                             bx(t) = 0.2 + 0.6*t  (slides left→right over one scan)

The two spheres are the stable anchor; the bead is the dragged feature.
A good 4D reconstruction pins the bead to its correct x-position at each timestep.
A static/blurred reconstruction smears the bead across the full rod length.

Outputs (in OUT_DIR):
  projections.npy   — float32 [N_ANGLES, nDetH, nDetW], ready for NeCT reconstruct()
  geometry.yaml     — NeCT-compatible geometry file
  gt_volumes.npy    — float32 [N_GT, Z, Y, X], ground truth volumes at N_GT timesteps
  gt_timesteps.npy  — float64 [N_GT], corresponding timestep values in [0, 1]

Run on a GPU node:
  python make_synthetic_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import nect.sampling.ct_sampling as ct_sampling
from nect.sampling.geometry import Geometry

# ── CONFIG ────────────────────────────────────────────────────────────────────

OUT_DIR = Path("dataset/synthetic_bead")

N_ANGLES = 360          # one projection per degree
POINTS_PER_RAY = 256    # ray integration steps (increase for less noise)
BATCH_RAYS = 8192       # rays per GPU batch (reduce if OOM)
N_GT_VOLUMES = 11       # number of ground-truth volume slices to export

# Cone-beam geometry (small, fast — matches the real scanner's source/detector ratio)
DSD = 1500.0            # mm, source to detector
DSO = 1000.0            # mm, source to object
N_DET = [128, 128]      # detector pixels [height, width]
D_DET = [1.0, 1.0]     # mm per pixel [height, width]
N_VOX = [128, 128, 128] # reconstruction volume [z, y, x]
D_VOX = [0.67, 0.67, 0.67]  # mm per voxel [z, y, x]

# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def phantom(z: np.ndarray, y: np.ndarray, x: np.ndarray, t: float) -> np.ndarray:
    left  = (z - 0.5)**2 + (y - 0.5)**2 + (x - 0.2)**2 < 0.10**2
    right = (z - 0.5)**2 + (y - 0.5)**2 + (x - 0.8)**2 < 0.10**2
    rod   = ((z - 0.5)**2 + (y - 0.5)**2 < 0.012**2) & (x > 0.2) & (x < 0.8)
    bx    = 0.2 + 0.6 * t
    bead  = (z - 0.5)**2 + (y - 0.5)**2 + (x - bx)**2 < 0.05**2

    out = np.zeros(z.shape, dtype=np.float32)
    out[left]  += 0.08
    out[right] += 0.08
    out[rod]   += 0.04
    out[bead]  += 0.15
    return np.clip(out, 0.0, 1.0)


def forward_project_phantom(geometry: Geometry, angle_rad: float, t: float) -> np.ndarray:
    c_geom = geometry.get_c_geometry()
    nH, nW = geometry.nDetector
    total_pix = nH * nW
    dist_per_pt = geometry.max_distance_traveled / POINTS_PER_RAY
    all_idx = torch.arange(total_pix, dtype=torch.int64, device=DEVICE)
    dev_idx = DEVICE.index if DEVICE.index is not None else 0
    projection = np.zeros(total_pix, dtype=np.float32)

    for start in range(0, total_pix, BATCH_RAYS):
        end = min(start + BATCH_RAYS, total_pix)
        n_rays = end - start

        ray_pts, distances = ct_sampling.sample(
            random_ray_index=all_idx,
            geometry=c_geom,
            angle_rad=float(angle_rad),
            num_points_per_ray=POINTS_PER_RAY,
            num_rays=n_rays,
            starting_ray_index=start,
            max_ray_distance_per_point=dist_per_pt,
            uniform_ray_spacing=True,
            random_detector_offset=0.0,
            device=dev_idx,
        )

        pts = ray_pts.view(-1, 3).cpu().numpy()  # [n_rays * PPR, 3] in zyx [0,1]
        pts = np.clip(pts, 0.0, 1.0)

        vals = phantom(pts[:, 0], pts[:, 1], pts[:, 2], t)
        vals = vals.reshape(n_rays, POINTS_PER_RAY)
        step = (distances / geometry.max_distance_traveled).cpu().numpy()
        projection[start:end] = vals.sum(axis=1) * step

    return projection.reshape(nH, nW)


def make_geometry(angles: np.ndarray, timesteps: np.ndarray) -> Geometry:
    return Geometry(
        nDetector=N_DET,
        dDetector=D_DET,
        mode="cone",
        DSD=DSD,
        DSO=DSO,
        nVoxel=N_VOX,
        dVoxel=D_VOX,
        angles=angles.tolist(),
        radians=True,
        timesteps=timesteps.tolist(),
    )


def save_geometry_yaml(geometry: Geometry, path: Path) -> None:
    d = geometry.to_dict()
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, tuple):
            d[k] = list(v)
    d["radians"] = True
    d.pop("sDetector", None)
    d.pop("sVoxel", None)
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False)


def save_gt_volumes(out_dir: Path) -> None:
    Z, Y, X = N_VOX
    zz, yy, xx = np.meshgrid(
        np.linspace(0, 1, Z, dtype=np.float32),
        np.linspace(0, 1, Y, dtype=np.float32),
        np.linspace(0, 1, X, dtype=np.float32),
        indexing="ij",
    )
    ts = np.linspace(0.0, 1.0, N_GT_VOLUMES)
    volumes = np.stack([phantom(zz, yy, xx, t) for t in ts])
    np.save(out_dir / "gt_volumes.npy", volumes)
    np.save(out_dir / "gt_timesteps.npy", ts)
    print(f"  gt_volumes.npy   shape={volumes.shape}  dtype={volumes.dtype}")

    # save mid-axial slices of t=0, t=0.5, t=1 for a quick visual check
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, ti, label in zip(axes, [0, N_GT_VOLUMES // 2, -1], ["t=0.0", f"t=0.5", "t=1.0"]):
        ax.imshow(volumes[ti, Z // 2], cmap="gray", vmin=0, vmax=0.3)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Ground truth mid-axial slice (bead slides left→right)")
    fig.tight_layout()
    fig.savefig(out_dir / "gt_preview.png", dpi=120)
    plt.close(fig)


def save_projection_preview(projections: np.ndarray, angles: np.ndarray, out_dir: Path) -> None:
    idxs = [0, N_ANGLES // 4, N_ANGLES // 2, 3 * N_ANGLES // 4]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    vmax = float(np.percentile(projections, 99))
    for ax, i in zip(axes, idxs):
        ax.imshow(projections[i], cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(f"angle {np.degrees(angles[i]):.0f}°  t={i/N_ANGLES:.2f}")
        ax.axis("off")
    fig.suptitle("Synthetic projections at 0°, 90°, 180°, 270°")
    fig.tight_layout()
    fig.savefig(out_dir / "proj_preview.png", dpi=120)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR.resolve()}")

    angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    timesteps = np.linspace(0.0, 1.0, N_ANGLES)

    geometry = make_geometry(angles, timesteps)
    print(f"Geometry: {N_DET} detector, {N_VOX} volume, DSD={DSD}, DSO={DSO}")
    print(f"max_distance_traveled = {geometry.max_distance_traveled:.3f} mm")

    print(f"\nForward projecting {N_ANGLES} angles ...")
    projections = np.zeros((N_ANGLES, N_DET[0], N_DET[1]), dtype=np.float32)
    for i, (angle, t) in enumerate(zip(angles, timesteps)):
        if i % 60 == 0:
            print(f"  [{i:3d}/{N_ANGLES}]  angle={np.degrees(angle):6.1f}°  t={t:.3f}"
                  f"  bead_x={0.2 + 0.6*t:.3f}")
        projections[i] = forward_project_phantom(geometry, angle, t)

    np.save(OUT_DIR / "projections.npy", projections)
    print(f"\n  projections.npy  shape={projections.shape}"
          f"  range=[{projections.min():.4f}, {projections.max():.4f}]")

    save_geometry_yaml(geometry, OUT_DIR / "geometry.yaml")
    print(f"  geometry.yaml    {N_ANGLES} angles, timesteps 0→1")

    print("\nSaving ground truth volumes ...")
    save_gt_volumes(OUT_DIR)

    print("\nSaving previews ...")
    save_projection_preview(projections, angles, OUT_DIR)

    print(f"\nDone. Dataset at: {OUT_DIR.resolve()}")
    print("  Use in NeCT:")
    print(f"    geometry = nect.Geometry.from_yaml('{OUT_DIR}/geometry.yaml')")
    print(f"    nect.reconstruct_continious_scan(geometry, '{OUT_DIR}/projections.npy', mode='dynamic', ...)")


if __name__ == "__main__":
    main()
