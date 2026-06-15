#!/usr/bin/env python3
"""
Synthetic 4D CT dataset with hybrid golden-angle acquisition.

Phantom: two fixed reference spheres + thin rod + sliding bead.
  bead_x(t) = 0.2 + 0.6*t  (slides left→right over the full scan, t in [0,1])

Golden-angle ordering (Δθ ≈ 111.246°) means any N consecutive projections
span ~360° — unlike linear ordering where N projections cover only a narrow
wedge. This lets the continuous reconstruction use very small temporal windows
while still having well-conditioned angular coverage ("going around" many times
conceptually even within a short time slice).

Blur quantification
───────────────────
For each frame size N_F (projections per temporal window):
  - dynamic projections:   each at its actual (angle_i, t_i)         [ground truth input]
  - reference projections: same angles but bead frozen at t_center    [what static recon uses]
  - diff = dynamic - reference  →  motion artifact in sinogram space
  - RMSE(diff) = "sinogram blur" — directly measurable without reconstruction

A perfect continuous reconstruction recovers the dynamic signal from the dynamic
projections alone. A static reconstruction uses the reference projections and
sees a bead smeared over bead_x ∈ [x_start, x_end].

Outputs in OUT_DIR:
  full/
    projections.npy       [N_TOTAL, H, W]    full golden-angle scan
    geometry.yaml                             golden-angle angles + timesteps

  frames_N{nf}/              (one per entry in FRAME_SIZES)
    projections.npy       [N_TOTAL, H, W]    same as full (golden-angle order)
    geometry.yaml                             same as full
    ref_projections.npy   [N_TOTAL, H, W]    static reference (bead at t_center per frame)
    diff_projections.npy  [N_TOTAL, H, W]    dynamic - reference  (motion artifact)
    frame_geometries/
      frame_000.yaml, ...                     per-frame geometry for the continuous trainer
    metrics.npz                               RMSE, bead_blur, angular coverage per frame

  gt_volumes.npy            [N_GT, Z, Y, X]  ground truth volumes at N_GT timesteps
  gt_timesteps.npy          [N_GT]

  blur_analysis.png         RMSE + bead blur width vs frame size
  angular_coverage.png      polar plots: angle distribution inside one frame per N_F
  proj_preview.png          4 sample projections (dynamic)
  gt_preview.png            mid-axial GT slices at t=0, 0.5, 1.0
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

OUT_DIR = Path("dataset/synthetic_bead_ga")

N_TOTAL = 360           # total projections in the full scan
GA_DEG  = 111.246       # golden angle (degrees) — standard for CT
POINTS_PER_RAY = 256    # ray integration steps
BATCH_RAYS     = 8192   # rays per GPU batch (reduce if OOM)
N_GT_VOLUMES   = 11     # ground-truth volumes to export

# Bead oscillates N_OSCILLATIONS times across the rod during one full scan.
# 3 cycles makes blur visible even at 4 projections (~ 4/360 of a cycle).
N_OSCILLATIONS = 3

# Frame sizes in degrees = number of projections (one per degree with N_TOTAL=360)
FRAME_SIZES = [1, 4, 8, 12]

# Reference angle for the example comparison images (degrees).
# 0° = x-ray along y-axis → bead moves left-right in detector → best for showing drag.
EXAMPLE_ANGLE_DEG = 0.0

# Cone-beam geometry
DSD   = 1500.0
DSO   = 1000.0
N_DET = [128, 128]
D_DET = [1.0, 1.0]
N_VOX = [128, 128, 128]
D_VOX = [0.67, 0.67, 0.67]

# ─────────────────────────────────────────────────────────────────────────────

GA_RAD = np.radians(GA_DEG)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEAD_R   = 0.05   # bead radius in [0,1] space
BEAD_ATT = 0.15   # bead linear attenuation coefficient


# ── Phantom ───────────────────────────────────────────────────────────────────

def bead_x_at_t(t: float | np.ndarray) -> float | np.ndarray:
    """Bead x-position at timestep t (sinusoidal, centre=0.5, amplitude=0.3)."""
    return 0.5 + 0.3 * np.sin(2.0 * np.pi * N_OSCILLATIONS * t)


def bead_blur_width_over(t_start: float, t_end: float, n_samples: int = 200) -> float:
    """Peak-to-peak bead displacement over a time window."""
    ts = np.linspace(t_start, t_end, n_samples)
    bx = bead_x_at_t(ts)
    return float(bx.max() - bx.min())


def phantom(z: np.ndarray, y: np.ndarray, x: np.ndarray, t: float) -> np.ndarray:
    """Evaluate phantom at coordinate arrays z,y,x ∈ [0,1] and timestep t ∈ [0,1]."""
    left  = (z-0.5)**2 + (y-0.5)**2 + (x-0.2)**2 < 0.10**2
    right = (z-0.5)**2 + (y-0.5)**2 + (x-0.8)**2 < 0.10**2
    rod   = ((z-0.5)**2 + (y-0.5)**2 < 0.012**2) & (x > 0.2) & (x < 0.8)
    bx    = bead_x_at_t(t)
    bead  = (z-0.5)**2 + (y-0.5)**2 + (x-bx)**2 < BEAD_R**2
    out   = np.zeros(z.shape, dtype=np.float32)
    out[left]  += 0.08
    out[right] += 0.08
    out[rod]   += 0.04
    out[bead]  += BEAD_ATT
    return np.clip(out, 0.0, 1.0)


# ── Forward projection ────────────────────────────────────────────────────────

def _make_c_geometry(geometry: Geometry):
    return geometry.get_c_geometry()


def forward_project_phantom(geometry: Geometry, angle_rad: float, t: float) -> np.ndarray:
    """Ray-march through the phantom at (angle_rad, t), return [H, W] projection."""
    c_geom   = _make_c_geometry(geometry)
    nH, nW   = geometry.nDetector
    total    = nH * nW
    dpp      = geometry.max_distance_traveled / POINTS_PER_RAY
    all_idx  = torch.arange(total, dtype=torch.int64, device=DEVICE)
    dev_idx  = DEVICE.index if DEVICE.index is not None else 0
    proj     = np.zeros(total, dtype=np.float32)

    for start in range(0, total, BATCH_RAYS):
        end    = min(start + BATCH_RAYS, total)
        n_rays = end - start
        ray_pts, distances = ct_sampling.sample(
            random_ray_index      = all_idx,
            geometry              = c_geom,
            angle_rad             = float(angle_rad),
            num_points_per_ray    = POINTS_PER_RAY,
            num_rays              = n_rays,
            starting_ray_index    = start,
            max_ray_distance_per_point = dpp,
            uniform_ray_spacing   = True,
            random_detector_offset= 0.0,
            device                = dev_idx,
        )
        pts  = np.clip(ray_pts.view(-1, 3).cpu().numpy(), 0.0, 1.0)
        vals = phantom(pts[:,0], pts[:,1], pts[:,2], t).reshape(n_rays, POINTS_PER_RAY)
        step = (distances / geometry.max_distance_traveled).cpu().numpy()
        proj[start:end] = vals.sum(axis=1) * step

    return proj.reshape(nH, nW)


def forward_project_time_averaged(
    geometry: Geometry, angle_rad: float, t_start: float, t_end: float, n_samples: int = 32
) -> np.ndarray:
    """
    Simulate a long-exposure projection: average the phantom forward-projected at
    `angle_rad` over `n_samples` evenly-spaced timesteps in [t_start, t_end].
    This is what the detector would record if it were open for the whole time window.
    """
    ts   = np.linspace(t_start, t_end, n_samples)
    proj = np.zeros((geometry.nDetector[0], geometry.nDetector[1]), dtype=np.float64)
    for t in ts:
        proj += forward_project_phantom(geometry, angle_rad, float(t))
    return (proj / n_samples).astype(np.float32)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def make_geometry(angles: np.ndarray, timesteps: np.ndarray) -> Geometry:
    return Geometry(
        nDetector=N_DET, dDetector=D_DET,
        mode="cone", DSD=DSD, DSO=DSO,
        nVoxel=N_VOX, dVoxel=D_VOX,
        angles=angles.tolist(), radians=True,
        timesteps=timesteps.tolist(),
    )


def save_geometry_yaml(geometry: Geometry, path: Path) -> None:
    d = {k: (v.tolist() if isinstance(v, np.ndarray) else
             list(v)    if isinstance(v, tuple)     else v)
         for k, v in geometry.to_dict().items()}
    d["radians"] = True
    d.pop("sDetector", None)
    d.pop("sVoxel", None)
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False)


# ── Golden angle ──────────────────────────────────────────────────────────────

def golden_angle_sequence(n: int) -> np.ndarray:
    """N projection angles in golden-angle order, in [0, 2π)."""
    return np.array([(i * GA_RAD) % (2.0 * np.pi) for i in range(n)])


def max_angular_gap(angles_rad: np.ndarray) -> float:
    """Largest gap (radians) between adjacent sorted angles, including wrap-around."""
    s    = np.sort(angles_rad % (2 * np.pi))
    gaps = np.append(np.diff(s), (2 * np.pi - s[-1]) + s[0])
    return float(gaps.max())


# ── Full scan ─────────────────────────────────────────────────────────────────

def project_full_scan(geometry: Geometry, angles: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    n    = len(angles)
    H, W = N_DET
    out  = np.zeros((n, H, W), dtype=np.float32)
    for i, (a, t) in enumerate(zip(angles, timesteps)):
        if i % 60 == 0:
            print(f"  [{i:3d}/{n}]  angle={np.degrees(a):6.1f}°  "
                  f"t={t:.3f}  bead_x={bead_x_at_t(t):.3f}")
        out[i] = forward_project_phantom(geometry, a, t)
    return out


# ── Frame blur analysis ───────────────────────────────────────────────────────

def analyse_frames(
    full_projs: np.ndarray,
    geometry:   Geometry,
    angles:     np.ndarray,
    timesteps:  np.ndarray,
    frame_size: int,
    out_dir:    Path,
) -> tuple[float, float]:
    """
    For each non-overlapping frame of `frame_size` consecutive projections:
      1. Extract dynamic projections (already computed, bead at each t_i).
      2. Forward-project the same angles but with the bead frozen at t_center
         → "reference projections" = what static reconstruction uses.
      3. diff = dynamic - reference → motion artifact in sinogram space.
      4. Compute per-frame RMSE, bead blur width, and angular coverage.

    Saves projections, reference, diff, per-frame geometry YAMLs, and metrics.
    Returns (mean_rmse, mean_bead_blur_width).
    """
    n_frames = N_TOTAL // frame_size
    H, W     = N_DET

    ref_projs  = np.zeros_like(full_projs)
    diff_projs = np.zeros_like(full_projs)

    rmse_arr         = np.zeros(n_frames)
    t_centers        = np.zeros(n_frames)
    bead_x_ranges    = np.zeros((n_frames, 2))
    angular_max_gaps = np.zeros(n_frames)

    geom_dir = out_dir / "frame_geometries"
    geom_dir.mkdir(parents=True, exist_ok=True)

    for k in range(n_frames):
        s, e = k * frame_size, (k + 1) * frame_size
        frame_angles = angles[s:e]
        frame_ts     = timesteps[s:e]
        t_center     = float(frame_ts.mean())
        t_centers[k] = t_center

        if k % max(1, n_frames // 5) == 0:
            print(f"    frame {k:2d}/{n_frames}  "
                  f"t=[{frame_ts[0]:.3f},{frame_ts[-1]:.3f}]  t_center={t_center:.3f}  "
                  f"bead_x=[{bead_x_at_t(frame_ts[0]):.3f},{bead_x_at_t(frame_ts[-1]):.3f}]")

        # Reference: same angles, bead frozen at t_center
        for j, a in enumerate(frame_angles):
            ref_projs[s + j] = forward_project_phantom(geometry, a, t_center)

        diff_projs[s:e] = full_projs[s:e] - ref_projs[s:e]
        rmse_arr[k]     = float(np.sqrt(np.mean(diff_projs[s:e]**2)))
        bead_x_ranges[k] = [bead_x_at_t(frame_ts[0]), bead_x_at_t(frame_ts[-1])]
        angular_max_gaps[k] = max_angular_gap(frame_angles)

        # Per-frame geometry: used to run continuous trainer on a single frame
        save_geometry_yaml(
            make_geometry(frame_angles, frame_ts),
            geom_dir / f"frame_{k:03d}.yaml",
        )

    bead_blur_widths = np.array([
        bead_blur_width_over(timesteps[k*frame_size], timesteps[min((k+1)*frame_size-1, N_TOTAL-1)])
        for k in range(n_frames)
    ])

    # Save
    np.save(out_dir / "projections.npy",      full_projs)
    np.save(out_dir / "ref_projections.npy",  ref_projs)
    np.save(out_dir / "diff_projections.npy", diff_projs)
    save_geometry_yaml(geometry, out_dir / "geometry.yaml")
    np.savez(
        out_dir / "metrics.npz",
        rmse             = rmse_arr,
        t_centers        = t_centers,
        bead_x_ranges    = bead_x_ranges,
        bead_blur_widths = bead_blur_widths,
        angular_max_gaps = angular_max_gaps,
        frame_size       = np.array(frame_size),
        bead_diameter    = np.array(2 * BEAD_R),
    )

    mean_rmse       = float(rmse_arr.mean())
    mean_blur_width = float(bead_blur_widths.mean())
    ratio           = mean_blur_width / (2 * BEAD_R)
    print(f"    → mean RMSE={mean_rmse:.5f}  "
          f"peak-peak bead displacement={mean_blur_width:.3f}  "
          f"bead_diam={2*BEAD_R:.3f}  blur/diam={ratio:.2f}x")

    return mean_rmse, mean_blur_width


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_gt_preview(out_dir: Path) -> None:
    zz, yy, xx = np.meshgrid(
        np.linspace(0,1,N_VOX[0], dtype=np.float32),
        np.linspace(0,1,N_VOX[1], dtype=np.float32),
        np.linspace(0,1,N_VOX[2], dtype=np.float32),
        indexing="ij",
    )
    ts = np.linspace(0.0, 1.0, N_GT_VOLUMES)
    vols = np.stack([phantom(zz, yy, xx, t) for t in ts])
    np.save(out_dir / "gt_volumes.npy", vols)
    np.save(out_dir / "gt_timesteps.npy", ts)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, idx, label in zip(axes, [0, N_GT_VOLUMES//2, -1], ["t=0.0", "t=0.5", "t=1.0"]):
        ax.imshow(vols[idx, N_VOX[0]//2], cmap="gray", vmin=0, vmax=0.3)
        bx = bead_x_at_t(ts[idx])
        ax.set_title(f"{label}   bead_x={bx:.2f}")
        ax.axis("off")
    fig.suptitle("Ground truth: mid-axial slice (bead slides left→right)")
    fig.tight_layout()
    fig.savefig(out_dir / "gt_preview.png", dpi=120)
    plt.close(fig)
    print(f"  gt_volumes.npy  shape={vols.shape}")


def plot_proj_preview(full_projs: np.ndarray, angles: np.ndarray, timesteps: np.ndarray, out_dir: Path) -> None:
    idxs = [0, N_TOTAL//4, N_TOTAL//2, 3*N_TOTAL//4]
    vmax = float(np.percentile(full_projs, 99))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, i in zip(axes, idxs):
        ax.imshow(full_projs[i], cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(f"proj {i}\n{np.degrees(angles[i]):.1f}°  t={timesteps[i]:.2f}")
        ax.axis("off")
    fig.suptitle("Sample golden-angle projections (dynamic: bead at t_i)")
    fig.tight_layout()
    fig.savefig(out_dir / "proj_preview.png", dpi=120)
    plt.close(fig)


def plot_angular_coverage(angles: np.ndarray, out_dir: Path) -> None:
    """Polar plots showing how N_F consecutive golden-angle projections cover 360°."""
    fig, axes = plt.subplots(
        1, len(FRAME_SIZES), figsize=(5 * len(FRAME_SIZES), 5),
        subplot_kw={"projection": "polar"},
    )
    # Use frame starting at 1/4 of the scan (avoids t=0 edge)
    start_idx = N_TOTAL // 4
    for ax, nf in zip(axes, FRAME_SIZES):
        frame_angles = angles[start_idx : start_idx + nf]
        ax.scatter(frame_angles, np.ones(nf), s=25, alpha=0.8)
        gap = np.degrees(max_angular_gap(frame_angles))
        ax.set_title(f"N_F={nf}\nmax gap={gap:.1f}°", pad=15)
        ax.set_yticklabels([])
        ax.set_theta_zero_location("N")
    fig.suptitle("Angular coverage within one temporal frame (golden-angle ordering)")
    fig.tight_layout()
    fig.savefig(out_dir / "angular_coverage.png", dpi=120)
    plt.close(fig)


def plot_example_images(
    geometry:   Geometry,
    timesteps:  np.ndarray,
    out_dir:    Path,
) -> None:
    """
    For each frame size, generate two projections at EXAMPLE_ANGLE_DEG:
      - non-combined:  sharp projection at t_center of the first frame (1 exposure)
      - time-averaged: long-exposure over the full frame window (motion blur)

    Saves a single comparison figure: rows = frame sizes, cols = [sharp, blurred, diff].
    """
    angle_rad = np.radians(EXAMPLE_ANGLE_DEG)
    n_rows = len(FRAME_SIZES)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))

    for row, nf in enumerate(FRAME_SIZES):
        # Use the middle frame so the bead is near maximum velocity
        mid_frame = (N_TOTAL // nf) // 2
        s = mid_frame * nf
        e = s + nf
        frame_ts  = timesteps[s:e]
        t_center  = float(frame_ts.mean())
        t_start   = float(frame_ts[0])
        t_end     = float(frame_ts[-1])

        print(f"  Example N_F={nf:2d}: t=[{t_start:.3f},{t_end:.3f}]  "
              f"bead_x(t_center)={bead_x_at_t(t_center):.3f}")

        sharp   = forward_project_phantom(geometry, angle_rad, t_center)
        blurred = forward_project_time_averaged(geometry, angle_rad, t_start, t_end)
        diff    = blurred - sharp

        vmax = max(float(sharp.max()), float(blurred.max())) * 1.05
        dmax = float(np.abs(diff).max()) or 1e-6

        ax_s, ax_b, ax_d = axes[row]

        ax_s.imshow(sharp,   cmap="gray", vmin=0, vmax=vmax)
        ax_s.set_title(f"N_F={nf}  non-combined\n(t={t_center:.3f}, bead_x={bead_x_at_t(t_center):.3f})")

        ax_b.imshow(blurred, cmap="gray", vmin=0, vmax=vmax)
        blur_w = bead_blur_width_over(t_start, t_end)
        ax_b.set_title(f"N_F={nf}  {nf}-angle mashed\n"
                       f"(bead blur Δx={blur_w:.3f} = {blur_w/(2*BEAD_R):.1f}× diam)")

        ax_d.imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        ax_d.set_title(f"Difference (blurred − sharp)\nRMSE={np.sqrt(np.mean(diff**2)):.5f}")

        for ax in (ax_s, ax_b, ax_d):
            ax.axis("off")

    fig.suptitle(
        f"Non-combined vs mashed projections at {EXAMPLE_ANGLE_DEG:.0f}°\n"
        f"Bead: sinusoidal, {N_OSCILLATIONS} cycles/scan  |  "
        f"Bead diameter = {2*BEAD_R:.2f} = {int(2*BEAD_R*N_DET[1])} px",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "example_images.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  example_images.png  saved → {path}")


def plot_blur_analysis(
    frame_sizes:        list[int],
    mean_rmse_list:     list[float],
    mean_blur_list:     list[float],
    out_dir:            Path,
) -> None:
    """
    Two-axis plot:
      left  axis — sinogram RMSE (motion artifact in projection space)
      right axis — bead blur width in [0,1] units and as multiple of bead diameter
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    color_rmse = "#e74c3c"
    color_blur = "#3498db"

    ax1.plot(frame_sizes, mean_rmse_list, "o-", color=color_rmse, lw=2, label="Sinogram RMSE")
    ax1.set_xlabel("Frame size  N_F  (projections per temporal window)")
    ax1.set_ylabel("Mean sinogram RMSE  (dynamic − static reference)", color=color_rmse)
    ax1.tick_params(axis="y", labelcolor=color_rmse)

    ax2.plot(frame_sizes, mean_blur_list, "s--", color=color_blur, lw=2, label="Bead blur width")
    ax2.axhline(2 * BEAD_R, color=color_blur, lw=1, ls=":", alpha=0.6, label="Bead diameter")
    ax2.set_ylabel("Bead position spread  Δx  (in [0,1] units)", color=color_blur)
    ax2.tick_params(axis="y", labelcolor=color_blur)

    # annotate blur/diameter ratio
    for nf, bw in zip(frame_sizes, mean_blur_list):
        ax2.annotate(f"{bw/(2*BEAD_R):.1f}×", (nf, bw), textcoords="offset points",
                     xytext=(4, 4), fontsize=9, color=color_blur)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Sub-angling blur vs temporal frame size\n"
                  "(blur/diameter > 1 = bead smeared larger than itself)")
    fig.tight_layout()
    fig.savefig(out_dir / "blur_analysis.png", dpi=120)
    plt.close(fig)
    print(f"  blur_analysis.png saved")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"Output : {OUT_DIR.resolve()}")
    print(f"Golden angle : {GA_DEG}°  ({GA_RAD:.4f} rad)")
    print(f"N_TOTAL={N_TOTAL}  FRAME_SIZES={FRAME_SIZES}")

    # Golden-angle sequence
    angles    = golden_angle_sequence(N_TOTAL)
    timesteps = np.linspace(0.0, 1.0, N_TOTAL)

    # One geometry covers the whole scan; per-frame geometries are written per frame size
    full_geometry = make_geometry(angles, timesteps)
    print(f"max_distance_traveled = {full_geometry.max_distance_traveled:.3f} mm\n")

    # ── 1. Full golden-angle scan ────────────────────────────────────────────
    print("=== Full scan (dynamic projections) ===")
    full_projs = project_full_scan(full_geometry, angles, timesteps)

    full_dir = OUT_DIR / "full"
    full_dir.mkdir(exist_ok=True)
    np.save(full_dir / "projections.npy", full_projs)
    save_geometry_yaml(full_geometry, full_dir / "geometry.yaml")
    print(f"  projections.npy  shape={full_projs.shape}  "
          f"range=[{full_projs.min():.4f}, {full_projs.max():.4f}]\n")

    # ── 2. Ground truth volumes + example images ─────────────────────────────
    print("=== Ground truth volumes ===")
    plot_gt_preview(OUT_DIR)
    plot_proj_preview(full_projs, angles, timesteps, OUT_DIR)
    plot_angular_coverage(angles, OUT_DIR)

    print("\n=== Example comparison images (non-combined vs mashed) ===")
    plot_example_images(full_geometry, timesteps, OUT_DIR)

    # ── 3. Per-frame blur analysis ───────────────────────────────────────────
    mean_rmse_list = []
    mean_blur_list = []

    for nf in FRAME_SIZES:
        if N_TOTAL % nf != 0:
            print(f"  [skip] N_TOTAL={N_TOTAL} is not divisible by frame_size={nf}")
            continue
        frame_dir = OUT_DIR / f"frames_N{nf}"
        frame_dir.mkdir(exist_ok=True)
        print(f"\n=== Frame size N_F={nf} ({N_TOTAL//nf} frames) ===")
        mr, mb = analyse_frames(full_projs, full_geometry, angles, timesteps, nf, frame_dir)
        mean_rmse_list.append(mr)
        mean_blur_list.append(mb)

    # ── 4. Blur summary plot ─────────────────────────────────────────────────
    valid_frame_sizes = [nf for nf in FRAME_SIZES if N_TOTAL % nf == 0]
    if len(valid_frame_sizes) >= 2:
        plot_blur_analysis(valid_frame_sizes, mean_rmse_list, mean_blur_list, OUT_DIR)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done.  Dataset at: {OUT_DIR.resolve()}")
    print()
    print("Full scan (feed to continuous trainer):")
    print(f"  geometry  = nect.Geometry.from_yaml('{full_dir}/geometry.yaml')")
    print(f"  nect.reconstruct_continious_scan(geometry, '{full_dir}/projections.npy', mode='dynamic', ...)")
    print()
    print("Per-frame comparison (static vs dynamic for one frame):")
    if valid_frame_sizes:
        nf = valid_frame_sizes[0]
        frame_dir = OUT_DIR / f"frames_N{nf}"
        print(f"  # Frame 0 — static:  use ref_projections[0:N_F] with frame_geometries/frame_000.yaml")
        print(f"  # Frame 0 — dynamic: use projections[0:N_F] with frame_geometries/frame_000.yaml")
    print()
    print("Sub-angling blur metrics (per frame size):")
    for nf, mr, mb in zip(valid_frame_sizes, mean_rmse_list, mean_blur_list):
        print(f"  N_F={nf:3d}:  sinogram RMSE={mr:.5f}  "
              f"bead_blur={mb:.3f}  ({mb/(2*BEAD_R):.1f}× bead diameter)")


if __name__ == "__main__":
    main()
