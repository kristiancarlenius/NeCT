"""
Render a side-by-side XZ / YZ slice video of the hourglass reconstruction
over time.

For each timestep the dynamic model is queried and two orthogonal slices
are rendered next to each other.  The result is saved as an MP4 (requires
ffmpeg) with a GIF fallback via imageio.

CONFIG mirrors 60_hourglass_sand_volume.py — set the same ROI / BINNING.
"""

import shutil
import subprocess
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from nect.config import get_cfg
from nect.data import NeCTDataset
from nect.sampling import Geometry

# ─────────────────────────── CONFIG ──────────────────────────────────────────

MODEL_PATH = "/cluster/home/kristiac/NeCT/outputs/dynamic_continious/quadcubes_21_4_21_16_2_4_128_L1/2026-04-16T02-30-34/model/"

N_TIMESTEPS = 10        # number of frames in the video
BINNING     = 8         # match what you used in script 60

ROI_Z = [240, 1056]    # full-res voxel coords  (binned 30–132)
ROI_Y = [136, 560]     # full-res voxel coords  (binned 17–70)
ROI_X = [184, 600]     # full-res voxel coords  (binned 23–75)

# Fixed slice positions *in the binned, ROI-cropped volume*.
# None → mid-point of that axis.
SLICE_Y = None   # XZ view: which Y slice to show
SLICE_X = None   # YZ view: which X slice to show

FPS        = 1          # frames per second in the output video
OUTPUT_DIR = Path(MODEL_PATH).parent
VIDEO_NAME = "hourglass_slices"   # .mp4 written; .gif fallback

# ─────────────────────────────────────────────────────────────────────────────


def query_volume(model, t, z_lin, y_lin, x_lin, device):
    z_h, y_w, x_w = len(z_lin), len(y_lin), len(x_lin)
    output = torch.zeros((z_h, y_w, x_w), device="cpu", dtype=torch.float32)
    t_tensor = torch.tensor(t, device=device)
    with torch.no_grad():
        for ii, z_ in enumerate(z_lin):
            z, y, x = torch.meshgrid(
                [z_.unsqueeze(0), y_lin, x_lin], indexing="ij"
            )
            grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t().to(device)
            output[ii] = model(grid, t_tensor).view(y_w, x_w).cpu()
    return output.numpy()


def main():
    device = torch.device(0)
    base_path = Path(MODEL_PATH)

    print("Loading config and model...")
    config = get_cfg(base_path / "config.yaml")
    assert config.mode == "dynamic", "Model must be dynamic"

    model = config.get_model()
    ckpt = torch.load(base_path / "checkpoints" / "last.ckpt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    dataset  = NeCTDataset(config=config, device="cpu")
    geometry = Geometry.from_cfg(
        config.geometry,
        reconstruction_mode=config.reconstruction_mode,
        sample_outside=config.sample_outside,
    )

    nVoxels_raw = list(config.geometry.nVoxel)
    dVoxel      = list(config.geometry.dVoxel)
    rm          = config.sample_outside
    nVoxels     = [nVoxels_raw[0], nVoxels_raw[1] + 2 * rm, nVoxels_raw[2] + 2 * rm]

    scale    = 1.0 / geometry.max_distance_traveled
    data_min = dataset.minimum.item()
    data_max = dataset.maximum.item()

    def calibrate(raw):
        return raw * scale * (data_max - data_min) + data_min

    def roi_coords(roi, n_full, n_voxels, rm_offset=0):
        if roi is None:
            return 0.0, 1.0, n_full // BINNING
        n_bins = (roi[1] - roi[0]) // BINNING
        start  = (roi[0] - rm_offset) / n_voxels
        end    = (roi[1] - rm_offset) / n_voxels
        return start, end, n_bins

    start_z, end_z, z_h = roi_coords(ROI_Z, nVoxels_raw[0], nVoxels[0])
    start_y, end_y, y_w = roi_coords(ROI_Y, nVoxels_raw[1], nVoxels[1], rm_offset=rm)
    start_x, end_x, x_w = roi_coords(ROI_X, nVoxels_raw[2], nVoxels[2], rm_offset=rm)

    print(f"Cropped volume shape: ({z_h}, {y_w}, {x_w})")

    z_lin = torch.linspace(start_z, end_z, steps=z_h, device=device)
    y_lin = torch.linspace(start_y, end_y, steps=y_w, device=device)
    x_lin = torch.linspace(start_x, end_x, steps=x_w, device=device)

    slice_y = SLICE_Y if SLICE_Y is not None else y_w // 2
    slice_x = SLICE_X if SLICE_X is not None else x_w // 2
    print(f"XZ slice at y={slice_y}, YZ slice at x={slice_x}")

    t_values = np.linspace(0.0, 1.0, N_TIMESTEPS, endpoint=False)

    # ── Establish display range from first timestep ───────────────────────────
    print("Querying first timestep to set display range...")
    vol0 = calibrate(query_volume(model, float(t_values[0]), z_lin, y_lin, x_lin, device))
    vmin = float(np.percentile(vol0, 1))
    vmax = float(np.percentile(vol0, 99))
    print(f"Display range: [{vmin:.4f}, {vmax:.4f}]")

    # ── Render frames ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    angles  = config.geometry.angles
    n_proj  = len(angles)

    for i, t in enumerate(tqdm(t_values, desc="Rendering frames")):
        if i == 0:
            vol = vol0
        else:
            vol = calibrate(query_volume(model, float(t), z_lin, y_lin, x_lin, device))

        proj_idx = int(t * n_proj)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"t={t:.3f}  (projection ≈ {proj_idx}/{n_proj})", fontsize=11)

        axes[0].imshow(vol[:, slice_y, :], cmap="gray",
                       aspect="auto", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"XZ  (y={slice_y})")
        axes[0].set_xlabel("x voxel (binned)")
        axes[0].set_ylabel("z voxel (binned)  [0=top]")

        axes[1].imshow(vol[:, :, slice_x], cmap="gray",
                       aspect="auto", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"YZ  (x={slice_x})")
        axes[1].set_xlabel("y voxel (binned)")
        axes[1].set_ylabel("z voxel (binned)  [0=top]")

        plt.tight_layout()

        # Render figure to numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        frame = buf[:, :, 1:]   # ARGB → RGB
        frames.append(frame)
        plt.close(fig)

    # ── Write video ───────────────────────────────────────────────────────────
    mp4_path = OUTPUT_DIR / f"{VIDEO_NAME}.mp4"
    gif_path = OUTPUT_DIR / f"{VIDEO_NAME}.gif"
    frames_dir = OUTPUT_DIR / "_frames_tmp"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("Saving frames...")
    for i, frame in enumerate(frames):
        imageio.imwrite(str(frames_dir / f"frame_{i:04d}.png"), frame)

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(mp4_path),
        ], check=True)
        print(f"Video saved → {mp4_path}")
    except Exception as e:
        print(f"ffmpeg failed ({e}), falling back to GIF...")
        imageio.mimwrite(str(gif_path), frames, fps=FPS)
        print(f"GIF saved → {gif_path}")
    finally:
        shutil.rmtree(frames_dir)


if __name__ == "__main__":
    main()
