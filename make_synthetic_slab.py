#!/usr/bin/env python3
"""
Synthetic 4D CT dataset for testing the continuous-scan angular-blur correction.

Why this dataset exists
───────────────────────
The synthetic_bead_ga dataset uses step-and-shoot projections with temporal
mashing. That tests temporal resolution, not angular blur. The hourglass and
synthetic-bead results all showed null results because those phantoms are
angularly symmetric (projections carry equivalent information from any angle).

This dataset generates TRUE continuous-scan projections (angular averages) and
adds a thin flat slab whose projection changes sharply over ~1.4° near the
edge-on angle. For an angular step Δφ=4°:
  - K=1 evaluates at the midpoint (2°), which is 2° away from the spike → misses it
  - K=4 evaluates at 0.5°, 1.5°, 2.5°, 3.5° → the 0.5° sub-step hits the spike

The expected result: K=1 reconstructions show a blurry or missing slab near the
edge-on projections; higher K progressively recovers the slab's sharpness.

Phantom
───────
  - Two reference spheres at x=0.2 and x=0.8 (static, for orientation)
  - Thin horizontal slab, normal along Y (static, angularly sensitive)
    - Thickness ≈ 3 voxels → spike half-width ≈ arctan(0.012/0.8) ≈ 0.86°
  - Moving bead that oscillates along X (dynamic)

Acquisition
───────────
For each angular step Δφ ∈ ANGULAR_STEPS_DEG:
  - N_proj = ceil(360 / Δφ) per rotation × N_ROTATIONS rotations
  - Angles: sequential, wrapping every rotation
  - Each projection = angular average over [φ_j, φ_j + Δφ] with K_GEN=32 sub-steps
  - This is a true continuous-scan simulation

Outputs: dataset/synthetic_slab/
  deg{dd}/
    projections.npy     [N_proj, H, W]   true angular-average projections
    geometry.yaml                         angles at interval starts + timesteps
  gt_volumes.npy        [N_GT, Z, Y, X]  step-and-shoot ground truth
  gt_timesteps.npy      [N_GT]
  phantom_preview.png
  sinogram_preview.png
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import nect.sampling.ct_sampling as ct_sampling
from nect.sampling.geometry import Geometry

# ── CONFIG ────────────────────────────────────────────────────────────────────

OUT_DIR = Path("dataset/synthetic_slab")

ANGULAR_STEPS_DEG = [4, 8, 12, 24]   # degrees per projection (continuous-scan interval)
N_ROTATIONS       = 3                  # full rotations per acquisition
K_GEN             = 32                 # sub-steps for generating accurate angular averages
N_GT_VOLUMES      = 11                 # ground-truth timesteps
POINTS_PER_RAY    = 256
BATCH_RAYS        = 8192

# Bead oscillates N_OSC × N_ROTATIONS times over the full scan
N_OSC_PER_ROTATION = 3

# Slab: thin horizontal plane, normal along Y, extending in XZ
# Thickness = 0.024 in [0,1] coords → 0.024 × 128 ≈ 3.1 voxels
# Spike half-width ≈ arctan(0.012 / 0.8) ≈ 0.86° → full width ≈ 1.72°
# For Δφ=4°: K=1 midpoint at 2° → completely outside spike
# For Δφ=4°, K=4: sub-steps at 0.5°, 1.5°, 2.5°, 3.5° → 0.5° is inside spike
SLAB_THICKNESS = 0.024
SLAB_ATT       = 0.12

BEAD_R   = 0.05
BEAD_ATT = 0.15

# Cone-beam geometry (matches synthetic_bead_ga)
DSD   = 1500.0
DSO   = 1000.0
N_DET = [128, 128]
D_DET = [1.0, 1.0]
N_VOX = [128, 128, 128]
D_VOX = [0.67, 0.67, 0.67]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Phantom ───────────────────────────────────────────────────────────────────

def bead_x_at_t(t: float | np.ndarray, n_total_osc: int) -> float | np.ndarray:
    return 0.5 + 0.3 * np.sin(2.0 * np.pi * n_total_osc * t)


def phantom(z: np.ndarray, y: np.ndarray, x: np.ndarray, t: float,
            n_total_osc: int) -> np.ndarray:
    # Static reference spheres
    left  = (z-0.5)**2 + (y-0.5)**2 + (x-0.2)**2 < 0.10**2
    right = (z-0.5)**2 + (y-0.5)**2 + (x-0.8)**2 < 0.10**2

    # Thin slab: normal along Y, extends in XZ
    # Normal to slab is Y. Edge-on angle is where beam travels along X.
    # Spike is near φ=0° / 180° in the XY rotation plane.
    slab = (
        (np.abs(y - 0.5) < SLAB_THICKNESS / 2)
        & (x > 0.1) & (x < 0.9)
        & (z > 0.1) & (z < 0.9)
    )

    # Moving bead
    bx   = bead_x_at_t(t, n_total_osc)
    bead = (z-0.5)**2 + (y-0.5)**2 + (x-bx)**2 < BEAD_R**2

    out = np.zeros(z.shape, dtype=np.float32)
    out[left]  += 0.08
    out[right] += 0.08
    out[slab]  += SLAB_ATT
    out[bead]  += BEAD_ATT
    return np.clip(out, 0.0, 1.0)


# ── Forward projection ────────────────────────────────────────────────────────

def _make_geometry(angles: np.ndarray, timesteps: np.ndarray) -> Geometry:
    return Geometry(
        nDetector=N_DET, dDetector=D_DET,
        mode="cone", DSD=DSD, DSO=DSO,
        nVoxel=N_VOX, dVoxel=D_VOX,
        angles=angles.tolist(), radians=True,
        timesteps=timesteps.tolist(),
    )


def forward_project_step(
    geometry: Geometry, angle_rad: float, t: float, n_total_osc: int,
) -> np.ndarray:
    """Single-angle step-and-shoot projection."""
    c_geom  = geometry.get_c_geometry()
    nH, nW  = geometry.nDetector
    total   = nH * nW
    dpp     = geometry.max_distance_traveled / POINTS_PER_RAY
    all_idx = torch.arange(total, dtype=torch.int64, device=DEVICE)
    dev_idx = DEVICE.index if DEVICE.index is not None else 0
    proj    = np.zeros(total, dtype=np.float32)

    for start in range(0, total, BATCH_RAYS):
        end    = min(start + BATCH_RAYS, total)
        n_rays = end - start
        ray_pts, distances = ct_sampling.sample(
            random_ray_index           = all_idx,
            geometry                   = c_geom,
            angle_rad                  = float(angle_rad),
            num_points_per_ray         = POINTS_PER_RAY,
            num_rays                   = n_rays,
            starting_ray_index         = start,
            max_ray_distance_per_point = dpp,
            uniform_ray_spacing        = True,
            random_detector_offset     = 0.0,
            device                     = dev_idx,
        )
        pts  = np.clip(ray_pts.view(-1, 3).cpu().numpy(), 0.0, 1.0)
        vals = phantom(pts[:,0], pts[:,1], pts[:,2], t, n_total_osc).reshape(
            n_rays, POINTS_PER_RAY
        )
        step = (distances / geometry.max_distance_traveled).cpu().numpy()
        proj[start:end] = vals.sum(axis=1) * step

    return proj.reshape(nH, nW)


def forward_project_continuous(
    geometry: Geometry,
    phi_start_rad: float,
    phi_end_rad: float,
    t: float,
    n_total_osc: int,
    k_gen: int = K_GEN,
) -> np.ndarray:
    """
    True continuous-scan projection: angular average over [phi_start, phi_end].
    Uses midpoint rule with k_gen sub-steps.
    """
    dphi  = (phi_end_rad - phi_start_rad) / k_gen
    accum = np.zeros((N_DET[0], N_DET[1]), dtype=np.float64)
    for k in range(k_gen):
        phi_k = phi_start_rad + (k + 0.5) * dphi
        accum += forward_project_step(geometry, phi_k, t, n_total_osc)
    return (accum / k_gen).astype(np.float32)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def save_geometry_yaml(geometry: Geometry, path: Path) -> None:
    d = {k: (v.tolist() if isinstance(v, np.ndarray) else
             list(v)    if isinstance(v, tuple)     else v)
         for k, v in geometry.to_dict().items()}
    d["radians"] = True
    d.pop("sDetector", None)
    d.pop("sVoxel", None)
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False)


# ── Dataset generation per Δφ ─────────────────────────────────────────────────

def generate_for_step(delta_deg: int, n_total_osc: int) -> None:
    delta_rad   = np.radians(delta_deg)
    n_per_rot   = 360 // delta_deg
    n_proj      = n_per_rot * N_ROTATIONS
    timesteps   = np.linspace(0.0, 1.0, n_proj, dtype=np.float32)

    # Angles at the START of each projection's interval (sequential, wrapping each rotation)
    angles = np.array(
        [(i % n_per_rot) * delta_rad for i in range(n_proj)],
        dtype=np.float64,
    )

    # Geometry uses start-of-interval angles; angle[i+1] - angle[i] = delta_rad
    # so the trainer derives the interval from consecutive entries.
    geometry = _make_geometry(angles, timesteps)

    out_dir = OUT_DIR / f"deg{delta_deg:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W   = N_DET
    projs  = np.zeros((n_proj, H, W), dtype=np.float32)

    print(f"  Δφ={delta_deg:2d}°  n_proj={n_proj}  n_per_rot={n_per_rot}")
    for i, (phi_s, t) in enumerate(zip(angles, timesteps)):
        phi_e = phi_s + delta_rad
        projs[i] = forward_project_continuous(geometry, phi_s, phi_e, float(t), n_total_osc)
        if i % max(1, n_proj // 8) == 0:
            rot = i // n_per_rot + 1
            print(f"    [{i:4d}/{n_proj}]  rot={rot}/{N_ROTATIONS}  "
                  f"φ={np.degrees(phi_s):5.1f}°–{np.degrees(phi_e):5.1f}°  t={t:.3f}")

    np.save(out_dir / "projections.npy", projs)
    save_geometry_yaml(geometry, out_dir / "geometry.yaml")
    print(f"    → saved {out_dir}/projections.npy  shape={projs.shape}  "
          f"range=[{projs.min():.4f}, {projs.max():.4f}]")


# ── Ground truth ──────────────────────────────────────────────────────────────

def generate_gt(n_total_osc: int) -> None:
    zz, yy, xx = np.meshgrid(
        np.linspace(0, 1, N_VOX[0], dtype=np.float32),
        np.linspace(0, 1, N_VOX[1], dtype=np.float32),
        np.linspace(0, 1, N_VOX[2], dtype=np.float32),
        indexing="ij",
    )
    ts   = np.linspace(0.0, 1.0, N_GT_VOLUMES, dtype=np.float32)
    vols = np.stack([phantom(zz, yy, xx, float(t), n_total_osc) for t in ts])
    np.save(OUT_DIR / "gt_volumes.npy",   vols)
    np.save(OUT_DIR / "gt_timesteps.npy", ts)
    print(f"  gt_volumes.npy  shape={vols.shape}")
    print(f"  gt_timesteps = {ts.tolist()}")
    return vols, ts


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_phantom_preview(vols: np.ndarray, ts: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    t_idxs = [0, len(ts)//2, -1]

    # Row 0: XY slice (z=centre) — shows slab as thin horizontal strip
    for ax, idx in zip(axes[0], t_idxs):
        ax.imshow(vols[idx, N_VOX[0]//2, :, :], cmap="gray", vmin=0, vmax=0.25,
                  origin="lower")
        bx = bead_x_at_t(ts[idx], N_ROTATIONS * N_OSC_PER_ROTATION)
        ax.set_title(f"XY mid-slice  t={ts[idx]:.1f}\nbead_x={bx:.2f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")

    # Row 1: XZ slice (y=centre) — shows slab as solid horizontal band
    for ax, idx in zip(axes[1], t_idxs):
        ax.imshow(vols[idx, :, N_VOX[1]//2, :], cmap="gray", vmin=0, vmax=0.25,
                  origin="lower")
        ax.set_title(f"XZ mid-slice  t={ts[idx]:.1f}")
        ax.set_xlabel("x"); ax.set_ylabel("z")

    fig.suptitle(
        f"Synthetic slab phantom\n"
        f"Slab thickness={SLAB_THICKNESS} (≈{SLAB_THICKNESS*128:.1f} vx)  "
        f"→ spike half-width ≈ {np.degrees(np.arctan(SLAB_THICKNESS/2/0.8)):.2f}°"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phantom_preview.png", dpi=120)
    plt.close(fig)
    print("  phantom_preview.png saved")


def plot_sinogram_preview() -> None:
    """
    Show how the slab's projection varies with angle.
    Compare step-and-shoot at 1° resolution vs true continuous-scan at 4°.
    Uses a single-rotation phantom at t=0.
    """
    angles_1deg = np.radians(np.arange(0, 360, 1.0))
    t0 = 0.0
    n_total_osc = N_ROTATIONS * N_OSC_PER_ROTATION

    tmp_geom = _make_geometry(angles_1deg, np.zeros_like(angles_1deg))

    # Step-and-shoot sinogram (centre row of detector)
    print("  Computing sinogram preview (360 step-and-shoot projections)...")
    centre_row = N_DET[0] // 2
    sino_ss = np.zeros(360)
    for i, phi in enumerate(angles_1deg):
        p = forward_project_step(tmp_geom, float(phi), t0, n_total_osc)
        sino_ss[i] = float(p[centre_row].max())  # max over detector width

    # True continuous-scan at 4°
    delta_rad = np.radians(4)
    n4 = 360 // 4
    angles_4deg = np.radians(np.arange(0, 360, 4.0))
    sino_cont4 = np.zeros(n4)
    for i, phi_s in enumerate(angles_4deg):
        p = forward_project_continuous(tmp_geom, phi_s, phi_s + delta_rad,
                                       t0, n_total_osc, k_gen=K_GEN)
        sino_cont4[i] = float(p[centre_row].max())

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    ax = axes[0]
    ax.plot(np.arange(360), sino_ss, lw=0.8, color="steelblue", label="step-and-shoot 1°")
    ax.set_title("Step-and-shoot: max detector signal vs angle (centre row)")
    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("max projection value")
    ax.legend()
    # Mark expected spike positions
    for phi_edge in [0, 180]:
        ax.axvline(phi_edge, color="red", ls="--", alpha=0.5)
    ax.text(2, ax.get_ylim()[1]*0.9, "← slab edge-on", color="red", fontsize=9)

    ax = axes[1]
    ax.bar(np.degrees(angles_4deg), sino_cont4, width=3.5, color="coral",
           alpha=0.8, label="continuous-scan 4° (averaged)")
    ax.set_title("True continuous-scan 4°: max detector signal vs projection start angle")
    ax.set_xlabel("projection start angle (degrees)")
    ax.set_ylabel("max projection value (averaged)")
    ax.legend()

    fig.suptitle(
        f"Slab angular response — spike width ≈ "
        f"{2*np.degrees(np.arctan(SLAB_THICKNESS/2/0.8)):.1f}° "
        f"(much narrower than 4° step)"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sinogram_preview.png", dpi=120)
    plt.close(fig)
    print("  sinogram_preview.png saved")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_total_osc = N_ROTATIONS * N_OSC_PER_ROTATION

    print(f"Device     : {DEVICE}")
    print(f"Output     : {OUT_DIR.resolve()}")
    print(f"Phantom    : slab thickness={SLAB_THICKNESS} ({SLAB_THICKNESS*128:.1f} vx), "
          f"spike width≈{2*np.degrees(np.arctan(SLAB_THICKNESS/2/0.8)):.2f}°")
    print(f"Angular steps: {ANGULAR_STEPS_DEG}°  |  N_ROTATIONS={N_ROTATIONS}  "
          f"|  N_OSC={n_total_osc}\n")

    print("=== Ground truth volumes ===")
    vols, ts = generate_gt(n_total_osc)
    plot_phantom_preview(vols, ts)

    print("\n=== Continuous-scan projections per angular step ===")
    for delta_deg in ANGULAR_STEPS_DEG:
        generate_for_step(delta_deg, n_total_osc)

    print("\n=== Sinogram preview ===")
    plot_sinogram_preview()

    print(f"\n{'='*60}")
    print(f"Done.  Dataset at: {OUT_DIR.resolve()}")
    spike_hw = np.degrees(np.arctan(SLAB_THICKNESS / 2 / 0.8))
    print(f"\nSlab spike half-width ≈ {spike_hw:.2f}°")
    for d in ANGULAR_STEPS_DEG:
        midpoint = d / 2
        caught = midpoint < spike_hw
        print(f"  Δφ={d:2d}°: K=1 midpoint at {midpoint:.1f}° → "
              f"{'INSIDE' if caught else 'OUTSIDE'} spike (need K≥{int(np.ceil(d / spike_hw / 2))})")


if __name__ == "__main__":
    main()
