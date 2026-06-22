#!/usr/bin/env python3
"""
Synthetic 4D CT dataset: sequential multi-rotation acquisition.

Acquisition model
─────────────────
The gantry completes N_ROTATIONS full rotations, one projection per degree,
so consecutive projections are exactly 1° apart. This means K consecutive
projections span exactly K° of one rotation — making the "degree" mashing
concept physically unambiguous.

  N_TOTAL = N_ROTATIONS × N_PER_ROTATION  (e.g. 5 × 360 = 1800 projections)
  angles  = [0°, 1°, …, 359°, 0°, 1°, …]  (wraps every rotation)

Bead motion
───────────
The bead oscillates sinusoidally N_OSCILLATIONS times over the full scan.
With N_OSC = N_ROTATIONS × 3 = 15 oscillations in 1800 projections, the bead
moves ≈ 0.63× its own diameter within a 4° window — clearly visible blur.

Blur quantification
───────────────────
For each frame size N_F (= K degrees, K consecutive projections):
  - dynamic:   each projection at its own (angle_i, t_i)
  - reference: same angles, bead frozen at frame t_center
  - diff = dynamic − reference  →  motion artifact magnitude
  - RMSE(diff) = "sinogram blur"

Outputs in OUT_DIR:
  full/
    projections.npy     [N_TOTAL, H, W]    multi-rotation sequential scan
    geometry.yaml                           sequential angles + timesteps

  frames_N{nf}/              (one per entry in FRAME_SIZES)
    projections.npy     [N_TOTAL, H, W]    same projections
    geometry.yaml                           same geometry
    ref_projections.npy [N_TOTAL, H, W]    bead frozen at t_center per frame
    diff_projections.npy[N_TOTAL, H, W]    dynamic − reference
    frame_geometries/
      frame_000.yaml, …                    per-frame geometry for the trainer
    metrics.npz                             RMSE, bead_blur, angular span

  gt_volumes.npy            [N_GT, Z, Y, X]
  gt_timesteps.npy          [N_GT]
  blur_analysis.png
  angular_coverage.png      shows narrow-wedge coverage per frame (sequential)
  proj_preview.png
  gt_preview.png
  example_images.png        non-combined vs mashed at EXAMPLE_ANGLE_DEG
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

OUT_DIR = Path("Datasets/synthetic_bead_ga")

N_ROTATIONS    = 5      # full gantry rotations
N_PER_ROTATION = 360    # projections per rotation (1 per degree)
N_TOTAL        = N_ROTATIONS * N_PER_ROTATION   # 1800

# Bead completes N_OSCILLATIONS cycles over the full scan.
# 3 per rotation = 15 total → bead moves ~0.63× diameter within a 4° window.
N_OSCILLATIONS = N_ROTATIONS * 3

POINTS_PER_RAY = 256
BATCH_RAYS     = 8192
N_GT_VOLUMES   = 11

# Frame sizes in degrees (= number of consecutive projections, since 1°/proj)
FRAME_SIZES = [1, 4, 8, 12]

# Reference angle for example comparison images
EXAMPLE_ANGLE_DEG = 0.0

# Cone-beam geometry
DSD   = 1500.0
DSO   = 1000.0
N_DET = [128, 128]
D_DET = [1.0, 1.0]
N_VOX = [128, 128, 128]
D_VOX = [0.67, 0.67, 0.67]

# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEAD_R   = 0.05
BEAD_ATT = 0.15


# ── Phantom ───────────────────────────────────────────────────────────────────

def bead_x_at_t(t: float | np.ndarray) -> float | np.ndarray:
    return 0.5 + 0.3 * np.sin(2.0 * np.pi * N_OSCILLATIONS * t)


def bead_blur_width_over(t_start: float, t_end: float, n_samples: int = 200) -> float:
    ts = np.linspace(t_start, t_end, n_samples)
    bx = bead_x_at_t(ts)
    return float(bx.max() - bx.min())


def phantom(z: np.ndarray, y: np.ndarray, x: np.ndarray, t: float) -> np.ndarray:
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

def forward_project_phantom(geometry: Geometry, angle_rad: float, t: float) -> np.ndarray:
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
        vals = phantom(pts[:,0], pts[:,1], pts[:,2], t).reshape(n_rays, POINTS_PER_RAY)
        step = (distances / geometry.max_distance_traveled).cpu().numpy()
        proj[start:end] = vals.sum(axis=1) * step

    return proj.reshape(nH, nW)


def forward_project_time_averaged(
    geometry: Geometry, angle_rad: float, t_start: float, t_end: float, n_samples: int = 32,
) -> np.ndarray:
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


# ── Sequential angle sequence ─────────────────────────────────────────────────

def sequential_angle_sequence(n_rotations: int, n_per_rotation: int) -> np.ndarray:
    """
    N_ROTATIONS full rotations at N_PER_ROTATION projections each (1°/step).
    Consecutive projections are exactly 1° apart — so K consecutive projections
    = K degrees of one rotation.
    """
    step = 2.0 * np.pi / n_per_rotation
    return np.array([(i % n_per_rotation) * step for i in range(n_rotations * n_per_rotation)])


def angular_span_deg(angles_rad: np.ndarray) -> float:
    """Angular span (degrees) covered by a set of projection angles."""
    s = np.sort(angles_rad % (2 * np.pi))
    gaps = np.append(np.diff(s), (2 * np.pi - s[-1]) + s[0])
    return float(np.degrees(2 * np.pi - gaps.max()))


# ── Full scan ─────────────────────────────────────────────────────────────────

def project_full_scan(geometry: Geometry, angles: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    n    = len(angles)
    H, W = N_DET
    out  = np.zeros((n, H, W), dtype=np.float32)
    for i, (a, t) in enumerate(zip(angles, timesteps)):
        if i % 180 == 0:
            rot = i // N_PER_ROTATION + 1
            deg = int(np.degrees(a))
            print(f"  [{i:4d}/{n}]  rot={rot}/{N_ROTATIONS}  "
                  f"{deg:3d}°  t={t:.3f}  bead_x={bead_x_at_t(t):.3f}")
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
    n_frames = N_TOTAL // frame_size
    H, W     = N_DET

    ref_projs  = np.zeros_like(full_projs)
    diff_projs = np.zeros_like(full_projs)

    rmse_arr         = np.zeros(n_frames)
    t_centers        = np.zeros(n_frames)
    bead_blur_widths = np.zeros(n_frames)
    angular_spans    = np.zeros(n_frames)

    geom_dir = out_dir / "frame_geometries"
    geom_dir.mkdir(parents=True, exist_ok=True)

    for k in range(n_frames):
        s, e = k * frame_size, (k + 1) * frame_size
        frame_angles = angles[s:e]
        frame_ts     = timesteps[s:e]
        t_center     = float(frame_ts.mean())
        t_centers[k] = t_center

        if k % max(1, n_frames // 5) == 0:
            rot = s // N_PER_ROTATION + 1
            deg_start = int(np.degrees(frame_angles[0]))
            print(f"    frame {k:3d}/{n_frames}  rot={rot}  "
                  f"deg={deg_start}°–{deg_start+frame_size-1}°  "
                  f"t=[{frame_ts[0]:.3f},{frame_ts[-1]:.3f}]")

        for j, a in enumerate(frame_angles):
            ref_projs[s + j] = forward_project_phantom(geometry, a, t_center)

        diff_projs[s:e]   = full_projs[s:e] - ref_projs[s:e]
        rmse_arr[k]       = float(np.sqrt(np.mean(diff_projs[s:e]**2)))
        bead_blur_widths[k] = bead_blur_width_over(float(frame_ts[0]), float(frame_ts[-1]))
        angular_spans[k]  = angular_span_deg(frame_angles)

        save_geometry_yaml(
            make_geometry(frame_angles, frame_ts),
            geom_dir / f"frame_{k:03d}.yaml",
        )

    np.save(out_dir / "projections.npy",      full_projs)
    np.save(out_dir / "ref_projections.npy",  ref_projs)
    np.save(out_dir / "diff_projections.npy", diff_projs)
    save_geometry_yaml(geometry, out_dir / "geometry.yaml")
    np.savez(
        out_dir / "metrics.npz",
        rmse             = rmse_arr,
        t_centers        = t_centers,
        bead_blur_widths = bead_blur_widths,
        angular_spans    = angular_spans,
        frame_size       = np.array(frame_size),
        bead_diameter    = np.array(2 * BEAD_R),
    )

    mean_rmse = float(rmse_arr.mean())
    mean_blur = float(bead_blur_widths.mean())
    print(f"    → mean RMSE={mean_rmse:.5f}  "
          f"bead_blur={mean_blur:.3f}  bead_diam={2*BEAD_R:.3f}  "
          f"blur/diam={mean_blur/(2*BEAD_R):.2f}x  "
          f"angular_span={float(angular_spans.mean()):.1f}°")
    return mean_rmse, mean_blur


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_gt_preview(out_dir: Path) -> None:
    zz, yy, xx = np.meshgrid(
        np.linspace(0, 1, N_VOX[0], dtype=np.float32),
        np.linspace(0, 1, N_VOX[1], dtype=np.float32),
        np.linspace(0, 1, N_VOX[2], dtype=np.float32),
        indexing="ij",
    )
    ts   = np.linspace(0.0, 1.0, N_GT_VOLUMES)
    vols = np.stack([phantom(zz, yy, xx, t) for t in ts])
    np.save(out_dir / "gt_volumes.npy", vols)
    np.save(out_dir / "gt_timesteps.npy", ts)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, idx, label in zip(axes, [0, N_GT_VOLUMES//2, -1], ["t=0.0", "t=0.5", "t=1.0"]):
        ax.imshow(vols[idx, N_VOX[0]//2], cmap="gray", vmin=0, vmax=0.3)
        ax.set_title(f"{label}   bead_x={bead_x_at_t(ts[idx]):.2f}")
        ax.axis("off")
    fig.suptitle("Ground truth mid-axial slice (bead oscillates 3×/rotation)")
    fig.tight_layout()
    fig.savefig(out_dir / "gt_preview.png", dpi=120)
    plt.close(fig)
    print(f"  gt_volumes.npy  shape={vols.shape}")


def plot_proj_preview(full_projs: np.ndarray, angles: np.ndarray, timesteps: np.ndarray, out_dir: Path) -> None:
    idxs = [0, N_TOTAL//4, N_TOTAL//2, 3*N_TOTAL//4]
    vmax = float(np.percentile(full_projs, 99))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, i in zip(axes, idxs):
        rot = i // N_PER_ROTATION + 1
        ax.imshow(full_projs[i], cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(f"proj {i}  rot {rot}\n{np.degrees(angles[i]):.0f}°  t={timesteps[i]:.2f}")
        ax.axis("off")
    fig.suptitle("Sample projections from the sequential multi-rotation scan")
    fig.tight_layout()
    fig.savefig(out_dir / "proj_preview.png", dpi=120)
    plt.close(fig)


def plot_angular_coverage(angles: np.ndarray, out_dir: Path) -> None:
    """
    Polar plots showing the angular coverage of one frame of each size.
    With sequential 1°/step ordering, K projections cover exactly K° (a narrow wedge).
    This is the trade-off vs golden angle: coarser temporal resolution needed
    for good angular coverage.
    """
    fig, axes = plt.subplots(
        1, len(FRAME_SIZES), figsize=(5 * len(FRAME_SIZES), 5),
        subplot_kw={"projection": "polar"},
    )
    start_idx = N_TOTAL // 4
    for ax, nf in zip(axes, FRAME_SIZES):
        frame_angles = angles[start_idx : start_idx + nf]
        ax.scatter(frame_angles, np.ones(nf), s=25, alpha=0.8)
        span = angular_span_deg(frame_angles)
        ax.set_title(f"N_F={nf} ({nf}°)\nspan={span:.1f}°", pad=15)
        ax.set_yticklabels([])
        ax.set_theta_zero_location("N")
    fig.suptitle(
        "Angular coverage per frame (sequential ordering)\n"
        "K projections = K° wedge — needs multiple rotations for full coverage"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "angular_coverage.png", dpi=120)
    plt.close(fig)


def plot_example_images(geometry: Geometry, timesteps: np.ndarray, out_dir: Path) -> None:
    angle_rad = np.radians(EXAMPLE_ANGLE_DEG)
    n_rows = len(FRAME_SIZES)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))

    # Pick the frame that starts at EXAMPLE_ANGLE_DEG in the middle rotation
    mid_rot_start = (N_ROTATIONS // 2) * N_PER_ROTATION
    example_start = mid_rot_start + int(EXAMPLE_ANGLE_DEG)

    for row, nf in enumerate(FRAME_SIZES):
        s = example_start
        e = s + nf
        frame_ts  = timesteps[s:e]
        t_center  = float(frame_ts.mean())
        t_start   = float(frame_ts[0])
        t_end     = float(frame_ts[-1])
        print(f"  Example N_F={nf:2d}: {nf}° window  "
              f"t=[{t_start:.3f},{t_end:.3f}]  bead_x={bead_x_at_t(t_center):.3f}")

        sharp   = forward_project_phantom(geometry, angle_rad, t_center)
        blurred = forward_project_time_averaged(geometry, angle_rad, t_start, t_end)
        diff    = blurred - sharp

        vmax = max(float(sharp.max()), float(blurred.max())) * 1.05
        dmax = float(np.abs(diff).max()) or 1e-6
        blur_w = bead_blur_width_over(t_start, t_end)

        ax_s, ax_b, ax_d = axes[row]
        ax_s.imshow(sharp,   cmap="gray", vmin=0, vmax=vmax)
        ax_s.set_title(f"N_F={nf} ({nf}°)  non-combined\nbead_x={bead_x_at_t(t_center):.3f}")

        ax_b.imshow(blurred, cmap="gray", vmin=0, vmax=vmax)
        ax_b.set_title(f"N_F={nf} ({nf}°)  {nf}°-mashed\n"
                       f"Δx={blur_w:.3f} = {blur_w/(2*BEAD_R):.1f}× diam")

        ax_d.imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        ax_d.set_title(f"Difference\nRMSE={np.sqrt(np.mean(diff**2)):.5f}")

        for ax in (ax_s, ax_b, ax_d):
            ax.axis("off")

    fig.suptitle(
        f"Non-combined vs mashed at {EXAMPLE_ANGLE_DEG:.0f}°\n"
        f"Bead: {N_OSCILLATIONS} oscillations over {N_ROTATIONS} rotations  |  "
        f"Bead diameter = {2*BEAD_R:.2f} = {int(2*BEAD_R*N_DET[1])} px",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "example_images.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  example_images.png  saved → {path}")


def plot_blur_analysis(
    frame_sizes:    list[int],
    mean_rmse_list: list[float],
    mean_blur_list: list[float],
    out_dir:        Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    color_rmse, color_blur = "#e74c3c", "#3498db"

    ax1.plot(frame_sizes, mean_rmse_list, "o-", color=color_rmse, lw=2, label="Sinogram RMSE")
    ax1.set_xlabel("Frame size N_F = degrees (1°/projection)")
    ax1.set_ylabel("Mean sinogram RMSE  (dynamic − static reference)", color=color_rmse)
    ax1.tick_params(axis="y", labelcolor=color_rmse)

    ax2.plot(frame_sizes, mean_blur_list, "s--", color=color_blur, lw=2, label="Bead blur width")
    ax2.axhline(2 * BEAD_R, color=color_blur, lw=1, ls=":", alpha=0.6, label="Bead diameter")
    ax2.set_ylabel("Peak-peak bead displacement  Δx  ([0,1] units)", color=color_blur)
    ax2.tick_params(axis="y", labelcolor=color_blur)

    for nf, bw in zip(frame_sizes, mean_blur_list):
        ax2.annotate(f"{bw/(2*BEAD_R):.1f}×", (nf, bw),
                     textcoords="offset points", xytext=(4, 4), fontsize=9, color=color_blur)

    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title(
        f"Sub-angle blur vs frame size\n"
        f"({N_ROTATIONS} rotations × {N_PER_ROTATION} proj/rot,  {N_OSCILLATIONS} bead oscillations)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "blur_analysis.png", dpi=120)
    plt.close(fig)
    print("  blur_analysis.png saved")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device  : {DEVICE}")
    print(f"Output  : {OUT_DIR.resolve()}")
    print(f"N_ROTATIONS={N_ROTATIONS}  N_PER_ROTATION={N_PER_ROTATION}  "
          f"N_TOTAL={N_TOTAL}  N_OSCILLATIONS={N_OSCILLATIONS}")
    print(f"FRAME_SIZES={FRAME_SIZES}  (degrees = consecutive projections)")

    angles    = sequential_angle_sequence(N_ROTATIONS, N_PER_ROTATION)
    timesteps = np.linspace(0.0, 1.0, N_TOTAL)

    full_geometry = make_geometry(angles, timesteps)
    print(f"max_distance_traveled = {full_geometry.max_distance_traveled:.3f} mm\n")

    # ── 1. Full scan ─────────────────────────────────────────────────────────
    print(f"=== Full scan ({N_TOTAL} projections, {N_ROTATIONS} rotations) ===")
    full_projs = project_full_scan(full_geometry, angles, timesteps)

    full_dir = OUT_DIR / "full"
    full_dir.mkdir(exist_ok=True)
    np.save(full_dir / "projections.npy", full_projs)
    save_geometry_yaml(full_geometry, full_dir / "geometry.yaml")
    print(f"  projections.npy  shape={full_projs.shape}  "
          f"range=[{full_projs.min():.4f}, {full_projs.max():.4f}]\n")

    # ── 2. Ground truth + previews ───────────────────────────────────────────
    print("=== Ground truth volumes ===")
    plot_gt_preview(OUT_DIR)
    plot_proj_preview(full_projs, angles, timesteps, OUT_DIR)
    plot_angular_coverage(angles, OUT_DIR)

    print("\n=== Example comparison images ===")
    plot_example_images(full_geometry, timesteps, OUT_DIR)

    # ── 3. Per-frame blur analysis ───────────────────────────────────────────
    mean_rmse_list, mean_blur_list, valid_sizes = [], [], []
    for nf in FRAME_SIZES:
        if N_TOTAL % nf != 0:
            print(f"  [skip] N_TOTAL={N_TOTAL} not divisible by {nf}")
            continue
        frame_dir = OUT_DIR / f"frames_N{nf}"
        frame_dir.mkdir(exist_ok=True)
        print(f"\n=== Frame size N_F={nf} ({nf}°, {N_TOTAL//nf} frames) ===")
        mr, mb = analyse_frames(full_projs, full_geometry, angles, timesteps, nf, frame_dir)
        mean_rmse_list.append(mr)
        mean_blur_list.append(mb)
        valid_sizes.append(nf)

    if len(valid_sizes) >= 2:
        plot_blur_analysis(valid_sizes, mean_rmse_list, mean_blur_list, OUT_DIR)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done.  Dataset at: {OUT_DIR.resolve()}")
    print(f"\nFull scan → continuous trainer:")
    print(f"  geometry = nect.Geometry.from_yaml('{full_dir}/geometry.yaml')")
    print(f"  nect.reconstruct_continious_scan(geometry, '{full_dir}/projections.npy', mode='dynamic', ...)")
    print(f"\nSub-angle blur metrics:")
    for nf, mr, mb in zip(valid_sizes, mean_rmse_list, mean_blur_list):
        print(f"  {nf:2d}°:  RMSE={mr:.5f}  bead_blur={mb:.3f}  ({mb/(2*BEAD_R):.2f}× diam)")


if __name__ == "__main__":
    main()
