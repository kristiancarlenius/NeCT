#!/usr/bin/env python3
"""
Synthetic 4D CT dataset for testing continuous-scan obstruction-ghost correction.

Why this dataset exists
───────────────────────
The bead experiment showed null results: a sphere is angularly symmetric, so its
projection looks identical from every direction and K-step vs K=1 give nearly the
same forward projection regardless of motion speed.

The slab experiment also showed null results: the slab is static, so no temporal
variation exists for K to exploit; the thin-spike signal is simply too weak to
recover from the averaged projection.

This dataset uses the obstruction-ghost mechanism that matches the thesis scenario:
a dense sphere (obstructor) oscillates in X and periodically passes in front of a
static soft-tissue target sphere, as seen from the gantry at φ ≈ 90°.

Ghost artifact mechanism
────────────────────────
At gantry φ ≈ 90° (source at +Y in TIGRE convention), rays travel in the −Y
direction.  The obstructor lives at y = 0.65 and the target at y = 0.35, so the
obstructor is between source and target whenever the two are X-aligned.

Obstruction zone: |x_obs(t) − 0.5| < R_obs + R_tgt ≈ 0.15

Within one angular window Δφ at this gantry angle the obstructor may transition
into or out of the blocking zone while the gantry also sweeps past 90°:

  K = 1  → evaluates at midpoint time: binary block / no-block → wrong
  K ≥ 4  → evaluates multiple sub-times: correctly weights the mix

Expected result: K = 1 reconstructions show the target at the wrong intensity
(ghost remnant or ghost absence) near the obstruction angles; increasing K
progressively resolves the target correctly.

Phantom
───────
  - Two reference spheres at (x=0.2, y=0.5) and (x=0.8, y=0.5) — static
  - Static target sphere at (x=0.5, y=0.35, z=0.5), att=0.10
  - Moving dense obstructor at (x(t), y=0.65, z=0.5), att=0.45
      x(t) = 0.5 + OBSTRUCTOR_AMP × sin(2π × N_total_osc × t)

Acquisition
───────────
For each angular step Δφ ∈ ANGULAR_STEPS_DEG:
  - N_proj = ceil(360 / Δφ) per rotation × N_ROTATIONS rotations
  - Angles monotonically increasing (no 2π wrap) so trainer can infer intervals
  - Each projection = angular average over [φ_j, φ_j + Δφ] with K_GEN=32 sub-steps
  - Sub-step time tracks the actual t_j at each sub-angle, so the obstructor
    position and gantry angle are jointly correct at every sub-step
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

OUT_DIR = Path("Datasets/synthetic_ghost")

ANGULAR_STEPS_DEG = [4, 8, 12, 24]
N_ROTATIONS        = 3
K_GEN              = 32
N_GT_VOLUMES       = 11
POINTS_PER_RAY     = 256
BATCH_RAYS         = 8192

N_OSC_PER_ROTATION = 4   # oscillations per gantry rotation

# Moving dense obstructor — at y=0.65, oscillates in X
# At φ=90° (source at +Y), obstructor is in front of target (y=0.35)
OBSTRUCTOR_ATT = 0.45
OBSTRUCTOR_R   = 0.09
OBSTRUCTOR_Y   = 0.65
OBSTRUCTOR_AMP = 0.30

# Static soft-tissue target — what gets blocked
TARGET_ATT = 0.10
TARGET_R   = 0.06
TARGET_X   = 0.50
TARGET_Y   = 0.35
TARGET_Z   = 0.50

# Reference spheres for orientation
REF_ATT = 0.08
REF_R   = 0.06

# Geometry (matches synthetic_slab for comparability)
DSD   = 1500.0
DSO   = 1000.0
N_DET = [128, 128]
D_DET = [1.0, 1.0]
N_VOX = [128, 128, 128]
D_VOX = [0.67, 0.67, 0.67]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Phantom (ZYX coordinate convention) ──────────────────────────────────────

def obstructor_x_at_t(t: float | np.ndarray, n_total_osc: int) -> float | np.ndarray:
    return 0.5 + OBSTRUCTOR_AMP * np.sin(2.0 * np.pi * n_total_osc * t)


def phantom(z: np.ndarray, y: np.ndarray, x: np.ndarray, t: float,
            n_total_osc: int) -> np.ndarray:
    # Static reference spheres
    ref_left  = (z - 0.5)**2 + (y - 0.5)**2 + (x - 0.2)**2 < REF_R**2
    ref_right = (z - 0.5)**2 + (y - 0.5)**2 + (x - 0.8)**2 < REF_R**2

    # Static target
    target = (
        (z - TARGET_Z)**2 + (y - TARGET_Y)**2 + (x - TARGET_X)**2 < TARGET_R**2
    )

    # Moving obstructor
    obs_x = obstructor_x_at_t(t, n_total_osc)
    obstructor = (
        (z - 0.5)**2 + (y - OBSTRUCTOR_Y)**2 + (x - obs_x)**2 < OBSTRUCTOR_R**2
    )

    out = np.zeros(z.shape, dtype=np.float32)
    out[ref_left]   += REF_ATT
    out[ref_right]  += REF_ATT
    out[target]     += TARGET_ATT
    out[obstructor] += OBSTRUCTOR_ATT
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
        vals = phantom(pts[:, 0], pts[:, 1], pts[:, 2], t, n_total_osc).reshape(
            n_rays, POINTS_PER_RAY
        )
        step = (distances / geometry.max_distance_traveled).cpu().numpy()
        proj[start:end] = vals.sum(axis=1) * step

    return proj.reshape(nH, nW)


def forward_project_continuous(
    geometry: Geometry,
    phi_start_rad: float,
    phi_end_rad: float,
    t_start: float,
    t_end: float,
    n_total_osc: int,
    k_gen: int = K_GEN,
) -> np.ndarray:
    """
    True continuous-scan projection: angular + temporal average over the window.
    Each sub-step uses the actual sub-angle AND sub-time, so obstructor position
    and gantry angle are jointly correct at every sub-step.
    """
    dphi = (phi_end_rad - phi_start_rad) / k_gen
    dt   = (t_end - t_start) / k_gen
    accum = np.zeros((N_DET[0], N_DET[1]), dtype=np.float64)
    for k in range(k_gen):
        phi_k = phi_start_rad + (k + 0.5) * dphi
        t_k   = t_start       + (k + 0.5) * dt
        accum += forward_project_step(geometry, phi_k, float(t_k), n_total_osc)
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
    delta_rad  = np.radians(delta_deg)
    n_per_rot  = 360 // delta_deg
    n_proj     = n_per_rot * N_ROTATIONS
    timesteps  = np.linspace(0.0, 1.0, n_proj, dtype=np.float32)

    # Monotonically increasing angles — no 2π wrap so trainer can infer intervals
    angles = np.array([i * delta_rad for i in range(n_proj)], dtype=np.float64)

    geometry = _make_geometry(angles, timesteps)

    out_dir = OUT_DIR / f"deg{delta_deg:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W  = N_DET
    projs = np.zeros((n_proj, H, W), dtype=np.float32)

    # Compute per-projection time step width (uniform)
    dt = 1.0 / n_proj

    print(f"  Δφ={delta_deg:2d}°  n_proj={n_proj}  n_per_rot={n_per_rot}")
    for i, (phi_s, t_s) in enumerate(zip(angles, timesteps)):
        phi_e = phi_s + delta_rad
        t_e   = t_s   + dt
        projs[i] = forward_project_continuous(
            geometry, phi_s, phi_e, float(t_s), float(t_e), n_total_osc
        )
        if i % max(1, n_proj // 8) == 0:
            rot = i // n_per_rot + 1
            obs_x = obstructor_x_at_t(float(t_s), n_total_osc)
            print(f"    [{i:4d}/{n_proj}]  rot={rot}/{N_ROTATIONS}  "
                  f"φ={np.degrees(phi_s):5.1f}°–{np.degrees(phi_e):5.1f}°  "
                  f"t={t_s:.3f}  obs_x={obs_x:.3f}")

    np.save(out_dir / "projections.npy", projs)
    save_geometry_yaml(geometry, out_dir / "geometry.yaml")
    print(f"    → saved {out_dir}/projections.npy  shape={projs.shape}  "
          f"range=[{projs.min():.4f}, {projs.max():.4f}]")


# ── Ground truth ──────────────────────────────────────────────────────────────

def generate_gt(n_total_osc: int):
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
    print(f"  gt_timesteps    = {ts.tolist()}")
    return vols, ts


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_phantom_preview(vols: np.ndarray, ts: np.ndarray, n_total_osc: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    t_idxs = [0, len(ts)//4, len(ts)//2, -1]

    for col, idx in enumerate(t_idxs):
        obs_x = obstructor_x_at_t(ts[idx], n_total_osc)
        # Top row: XY slice at z=centre — shows obstructor and target positions
        ax = axes[0, col]
        ax.imshow(vols[idx, N_VOX[0]//2, :, :], cmap="gray", vmin=0, vmax=0.5,
                  origin="lower")
        ax.set_title(f"XY  t={ts[idx]:.2f}\nobs_x={obs_x:.2f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        # Target marker
        ax.axhline(int(TARGET_Y * N_VOX[1]), color="cyan", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(int(OBSTRUCTOR_Y * N_VOX[1]), color="red", ls="--", lw=0.8, alpha=0.6)

        # Bottom row: XZ slice at y=obstructor — shows obstructor moving
        ax = axes[1, col]
        ax.imshow(vols[idx, :, int(OBSTRUCTOR_Y * N_VOX[1]), :],
                  cmap="gray", vmin=0, vmax=0.5, origin="lower")
        ax.set_title(f"XZ (y=obs)  t={ts[idx]:.2f}")
        ax.set_xlabel("x"); ax.set_ylabel("z")

    fig.suptitle(
        f"Synthetic ghost phantom  —  obstructor (red, y={OBSTRUCTOR_Y}, att={OBSTRUCTOR_ATT})"
        f"  /  target (cyan, y={TARGET_Y}, att={TARGET_ATT})\n"
        f"N_osc={n_total_osc}  amp={OBSTRUCTOR_AMP}"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phantom_preview.png", dpi=120)
    plt.close(fig)
    print("  phantom_preview.png saved")


def plot_obstruction_preview(n_total_osc: int) -> None:
    """
    Show how the target's projection signal changes as obstructor position varies,
    and compare step-and-shoot vs continuous-scan projections near φ=90°.
    """
    # Scan t from 0 to 1 at very fine resolution to show signal vs obs_x
    t_fine   = np.linspace(0, 1, 1000)
    obs_xs   = obstructor_x_at_t(t_fine, n_total_osc)
    in_block = np.abs(obs_xs - TARGET_X) < (OBSTRUCTOR_R + TARGET_R)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    ax = axes[0]
    ax.plot(t_fine, obs_xs, color="red", lw=1, label="obstructor x(t)")
    ax.axhline(TARGET_X, color="cyan", ls="--", lw=1, label=f"target x={TARGET_X}")
    ax.fill_between(t_fine,
                    TARGET_X - (OBSTRUCTOR_R + TARGET_R),
                    TARGET_X + (OBSTRUCTOR_R + TARGET_R),
                    alpha=0.15, color="cyan", label="blocking zone")
    ax.set_xlabel("normalised time t")
    ax.set_ylabel("obstructor X position")
    ax.set_title("Obstructor trajectory and blocking zone (target is blocked when curves overlap)")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.fill_between(t_fine, 0, in_block.astype(float),
                    alpha=0.5, color="orange", label="obstruction active (φ≈90°)")
    ax.set_xlabel("normalised time t")
    ax.set_ylabel("blocking (1=yes)")
    ax.set_title("Timesteps where target is potentially blocked (need coincident gantry φ≈90°)")
    ax.legend()

    fig.suptitle(
        f"Ghost obstruction preview\n"
        f"Blocking fraction per oscillation: "
        f"{np.mean(in_block):.1%}   "
        f"(R_obs={OBSTRUCTOR_R}, R_tgt={TARGET_R}, amp={OBSTRUCTOR_AMP})"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "obstruction_preview.png", dpi=120)
    plt.close(fig)
    print("  obstruction_preview.png saved")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_total_osc = N_ROTATIONS * N_OSC_PER_ROTATION

    print(f"Device       : {DEVICE}")
    print(f"Output       : {OUT_DIR.resolve()}")
    print(f"Obstructor   : att={OBSTRUCTOR_ATT}  R={OBSTRUCTOR_R}  y={OBSTRUCTOR_Y}  amp={OBSTRUCTOR_AMP}")
    print(f"Target       : att={TARGET_ATT}  R={TARGET_R}  x={TARGET_X}  y={TARGET_Y}")
    print(f"N_osc_total  : {n_total_osc}  ({N_OSC_PER_ROTATION}/rotation × {N_ROTATIONS} rotations)")
    blocking_frac = np.mean(
        np.abs(obstructor_x_at_t(np.linspace(0, 1, 10000), n_total_osc) - TARGET_X)
        < (OBSTRUCTOR_R + TARGET_R)
    )
    print(f"Blocking zone: x ∈ [{TARGET_X-(OBSTRUCTOR_R+TARGET_R):.2f}, "
          f"{TARGET_X+(OBSTRUCTOR_R+TARGET_R):.2f}]  "
          f"fraction of time in zone: {blocking_frac:.1%}")
    print()

    for delta_deg in ANGULAR_STEPS_DEG:
        n_proj  = (360 // delta_deg) * N_ROTATIONS
        dx_max  = 2 * np.pi * n_total_osc * OBSTRUCTOR_AMP / n_proj
        print(f"  Δφ={delta_deg:2d}°  n_proj={n_proj:4d}  "
              f"max Δx per step={dx_max:.4f}  "
              f"(R_obs={OBSTRUCTOR_R}, crossing ~{int(np.ceil(OBSTRUCTOR_R/dx_max))} steps)")
    print()

    print("=== Ground truth volumes ===")
    vols, ts = generate_gt(n_total_osc)
    plot_phantom_preview(vols, ts, n_total_osc)
    plot_obstruction_preview(n_total_osc)

    print("\n=== Continuous-scan projections per angular step ===")
    for delta_deg in ANGULAR_STEPS_DEG:
        generate_for_step(delta_deg, n_total_osc)

    print(f"\n{'='*60}")
    print(f"Done.  Dataset at: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
