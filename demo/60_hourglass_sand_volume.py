"""
Hourglass sand volume analysis — batch over all runs.

Iterates every immediate subdirectory of BASE_DIR that contains a model/
folder, runs the full sand-volume analysis, and writes sand_volume.npz,
sand_volume.png, neck_check.png, and mae.txt into each run's directory.

Threshold is chosen per fps tier:
  4fps → THRESHOLD_4FPS = 0.16
  8fps → THRESHOLD_8FPS = -0.045

Geometry for this dataset:
  nVoxel = [1148, 748, 748]   — z is the tall/vertical axis of the hourglass
  dVoxel = ~0.1377 mm (isotropic)
  With BINNING=4: output shape is (287, 187, 187), neck auto-midpoint at z=143

Usage:
    Edit the CONFIG section below and run on a GPU node.
    Set SKIP_EXISTING = True to resume an interrupted batch run.
"""

import re
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

BASE_DIR = Path(
    "/cluster/home/kristiac/NeCT/outputs/dynamic_continious"
    "/quadcubes_22_4_22_16_2_4_128_L1"
)

# Per-fps attenuation thresholds for sand segmentation
THRESHOLD_4FPS = 0.16
THRESHOLD_8FPS = -0.045

# How many evenly-spaced timesteps to sample across the full acquisition
N_TIMESTEPS = 800

# Spatial binning factor (4 = 4× faster/lower-res; 1 = full resolution)
BINNING = 1

# Optional ROI in *full-resolution* voxel coordinates [start, end].
# Set to None to use the full volume.
ROI_Z = [240, 1056]
ROI_Y = [136, 560]
ROI_X = [184, 600]

# Z-voxel index (in the *binned, ROI-cropped* output) of the hourglass neck.
NECK_Z_VOXEL = 370

# ── Glitch filtering ──────────────────────────────────────────────────────────
FILTER_GLITCHES = True
FILTER_SIGMA    = 60

# Skip runs that already have a mae.txt (useful for resuming)
SKIP_EXISTING = True #

# ── Plot-only mode ────────────────────────────────────────────────────────────
# Reload volumes from a previous run's sand_volume.npz instead of re-querying.
PLOT_ONLY = False

# ─────────────────────────────────────────────────────────────────────────────

FPS_RE = re.compile(r"^(\d+)fps")


def threshold_for(run_name: str) -> float:
    m = FPS_RE.match(run_name)
    if not m:
        raise ValueError(f"Cannot determine fps from run name '{run_name}'")
    fps = int(m.group(1))
    if fps == 4:
        return THRESHOLD_4FPS
    if fps == 8:
        return THRESHOLD_8FPS
    raise ValueError(f"No threshold configured for {fps}fps (run: '{run_name}')")


def find_runs(base: Path) -> list[Path]:
    runs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "model").is_dir())
    if not runs:
        print(f"No subdirectories with a model/ folder found under {base}")
    return runs


# ─────────────────────────────────────────────────────────────────────────────

def filter_glitches(arr: np.ndarray, t_axis: np.ndarray, sigma: float, label: str):
    K = 5
    v0 = float(np.median(arr[:K]))
    v1 = float(np.median(arr[-K:]))
    t0, t1 = float(t_axis[0]), float(t_axis[-1])
    truth = v0 + (v1 - v0) * (t_axis - t0) / (t1 - t0)
    residuals = arr - truth

    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad == 0:
        print(f"  {label}: MAD=0, nothing filtered")
        return arr.copy(), np.zeros(len(arr), dtype=bool), truth

    threshold = sigma * mad * 1.4826
    bad = np.abs(residuals) > threshold

    clean = arr.copy()
    clean[bad] = truth[bad]

    bad_proj = t_axis[bad]
    print(f"\n  {label}: {bad.sum()} point(s) flagged for removal")
    if bad.any():
        print(f"    Projection indices: {bad_proj.astype(int).tolist()}")
        if len(bad_proj) >= 2:
            ns = np.arange(len(bad_proj))
            fit_slope, fit_intercept = np.polyfit(ns, bad_proj, 1)
            predicted = fit_intercept + fit_slope * ns
            ss_res = np.sum((bad_proj - predicted) ** 2)
            ss_tot = np.sum((bad_proj - bad_proj.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
            print(f"    Pattern fit: index(n) = {fit_intercept:.1f} + {fit_slope:.1f}·n  "
                  f"(R²={r2:.4f})")
            if r2 > 0.999:
                print(f"    → Nearly perfect linear spacing: glitches every "
                      f"~{fit_slope:.1f} projections")

    return clean, bad, truth


def query_volume(
    model,
    t: float,
    z_lin: torch.Tensor,
    y_lin: torch.Tensor,
    x_lin: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    z_h, y_w, x_w = len(z_lin), len(y_lin), len(x_lin)
    output = torch.zeros((z_h, y_w, x_w), device="cpu", dtype=torch.float32)
    for ii, z_ in enumerate(z_lin):
        z, y, x = torch.meshgrid(
            [z_.unsqueeze(0), y_lin, x_lin],
            indexing="ij",
        )
        grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t().to(device)
        output[ii] = model(grid, float(t)).view(y_w, x_w).cpu()
    return output.numpy()


# ─────────────────────────────────────────────────────────────────────────────

def process_run(run_dir: Path) -> None:
    out_dir = run_dir
    npz_path = out_dir / "sand_volume.npz"
    threshold = threshold_for(run_dir.name)

    print(f"\n{'='*60}")
    print(f"Run: {run_dir.name}  |  threshold={threshold}")
    print(f"{'='*60}")

    if PLOT_ONLY:
        print(f"PLOT_ONLY: loading volumes from {npz_path}")
        data = np.load(npz_path)
        top_vols_mm3 = data["top_volume_mm3"]
        bot_vols_mm3 = data["bottom_volume_mm3"]
        t_axis = data["projection_indices"]
    else:
        model_path = run_dir / "model"
        device = torch.device(0)

        print("Loading config and model...")
        config = get_cfg(model_path / "config.yaml")
        assert config.geometry is not None, "No geometry in config"
        assert config.mode == "dynamic", "Model must be in dynamic mode"

        model = config.get_model()
        checkpoints = torch.load(model_path / "checkpoints" / "last.ckpt", map_location="cpu")
        model.load_state_dict(checkpoints["model"])
        model = model.to(device)
        model.eval()

        dataset = NeCTDataset(config=config, device="cpu")
        geometry = Geometry.from_cfg(
            config.geometry,
            reconstruction_mode=config.reconstruction_mode,
            sample_outside=config.sample_outside,
        )

        nVoxels_raw = list(config.geometry.nVoxel)
        dVoxel = list(config.geometry.dVoxel)
        rm = config.sample_outside
        nVoxels = [nVoxels_raw[0], nVoxels_raw[1] + 2 * rm, nVoxels_raw[2] + 2 * rm]

        voxel_vol_mm3 = (dVoxel[0] * BINNING) * (dVoxel[1] * BINNING) * (dVoxel[2] * BINNING)
        print(f"Binned voxel volume: {voxel_vol_mm3:.4f} mm³")

        def roi_coords(roi, n_full, n_voxels, rm_offset=0):
            if roi is None:
                return 0.0, 1.0, n_full // BINNING
            n_bins = (roi[1] - roi[0]) // BINNING
            start = (roi[0] - rm_offset) / n_voxels
            end   = (roi[1] - rm_offset) / n_voxels
            return start, end, n_bins

        start_z, end_z, z_h = roi_coords(ROI_Z, nVoxels_raw[0], nVoxels[0], rm_offset=0)
        start_y, end_y, y_w = roi_coords(ROI_Y, nVoxels_raw[1], nVoxels[1], rm_offset=rm)
        start_x, end_x, x_w = roi_coords(ROI_X, nVoxels_raw[2], nVoxels[2], rm_offset=rm)
        print(f"Volume shape per timestep: ({z_h}, {y_w}, {x_w})")

        z_lin = torch.linspace(start_z, end_z, steps=z_h, device=device)
        y_lin = torch.linspace(start_y, end_y, steps=y_w, device=device)
        x_lin = torch.linspace(start_x, end_x, steps=x_w, device=device)

        scale    = 1.0 / geometry.max_distance_traveled
        data_min = dataset.minimum.item()
        data_max = dataset.maximum.item()

        def calibrate(raw: np.ndarray) -> np.ndarray:
            return raw * scale * (data_max - data_min) + data_min

        angles   = config.geometry.angles
        t_values = np.linspace(0.0, 1.0, N_TIMESTEPS, endpoint=False)

        print("Querying first timestep volume for threshold / neck diagnostics...")
        with torch.no_grad():
            vol0_raw = query_volume(model, float(t_values[0]), z_lin, y_lin, x_lin, device)
        vol0 = calibrate(vol0_raw)

        print(f"  Using threshold = {threshold:.4f}")

        neck_z = NECK_Z_VOXEL if NECK_Z_VOXEL is not None else z_h // 2
        mid_y  = y_w // 2
        mid_x  = x_w // 2

        vmin = float(np.percentile(vol0, 1))
        vmax = float(np.percentile(vol0, 99))

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

        show(axes_nc[0, 0], vol0[:, mid_y, :],
             "XZ slice (mid-Y) — raw attenuation",
             "x voxel (binned)", "z voxel (binned)  [0=top]", add_neck=True)

        axes_nc[0, 1].imshow(vol0[:, mid_y, :] > threshold, cmap="gray", aspect="auto")
        axes_nc[0, 1].axhline(neck_z, color="red", linewidth=1.5, linestyle="--",
                               label=f"neck z={neck_z}")
        axes_nc[0, 1].set_title(f"Sand mask — XZ (mid-Y), threshold={threshold:.4f}")
        axes_nc[0, 1].set_xlabel("x voxel (binned)")
        axes_nc[0, 1].legend(fontsize=8)

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

        top_vols_mm3 = []
        bot_vols_mm3 = []

        with torch.no_grad():
            for i, t in enumerate(tqdm(t_values, desc=run_dir.name)):
                vol = vol0 if i == 0 else calibrate(
                    query_volume(model, float(t), z_lin, y_lin, x_lin, device)
                )
                sand_mask = vol > threshold
                top_vols_mm3.append(sand_mask[:neck_z].sum() * voxel_vol_mm3)
                bot_vols_mm3.append(sand_mask[neck_z:].sum() * voxel_vol_mm3)

        top_vols_mm3 = np.array(top_vols_mm3)
        bot_vols_mm3 = np.array(bot_vols_mm3)
        t_axis = t_values * len(angles)

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

    # ── Glitch filtering ──────────────────────────────────────────────────────
    total_vols_mm3 = top_vols_mm3 + bot_vols_mm3
    top_vols_clean = top_vols_mm3.copy()
    bot_vols_clean = bot_vols_mm3.copy()
    top_linear = bot_linear = total_linear = None

    if FILTER_GLITCHES:
        print(f"Glitch filter (sigma={FILTER_SIGMA}):")
        _, top_bad,  top_linear   = filter_glitches(top_vols_mm3,   t_axis, FILTER_SIGMA, "Top chamber")
        _, bot_bad,  bot_linear   = filter_glitches(bot_vols_mm3,   t_axis, FILTER_SIGMA, "Bottom chamber")
        _,       _, total_linear  = filter_glitches(total_vols_mm3, t_axis, FILTER_SIGMA, "Total")

        bad  = top_bad | bot_bad
        keep = ~bad
        if bad.any():
            print(f"  Dropping {bad.sum()} timestep(s) (combined top+bot mask)")
        t_axis         = t_axis[keep]
        top_vols_clean = top_vols_mm3[keep]
        bot_vols_clean = bot_vols_mm3[keep]
        top_linear     = top_linear[keep]
        bot_linear     = bot_linear[keep]
        total_linear   = total_linear[keep]

    total_clean = top_vols_clean + bot_vols_clean

    # ── Error metrics ─────────────────────────────────────────────────────────
    total_truth = float(total_clean.mean())

    mae_total = float(np.mean(np.abs(total_clean - total_truth)))
    mse_total = float(np.mean((total_clean - total_truth) ** 2))

    delta_top      = top_vols_clean - top_vols_clean[0]
    delta_bot      = bot_vols_clean - bot_vols_clean[0]
    residual_1to1  = delta_top + delta_bot
    mae_1to1 = float(np.mean(np.abs(residual_1to1)))
    mse_1to1 = float(np.mean(residual_1to1 ** 2))

    # Deviation of each chamber from its own linear trend.
    # (The conservation-derived truth top_truth = total_truth - bot reduces to
    #  mae_top == mae_bot == mae_total by algebra, so it carries no extra info.)
    t_idx = np.arange(len(top_vols_clean), dtype=float)
    top_trend = np.polyval(np.polyfit(t_idx, top_vols_clean, 1), t_idx)
    bot_trend = np.polyval(np.polyfit(t_idx, bot_vols_clean, 1), t_idx)

    mae_top = float(np.mean(np.abs(top_vols_clean - top_trend)))
    mse_top = float(np.mean((top_vols_clean - top_trend) ** 2))
    mae_bot = float(np.mean(np.abs(bot_vols_clean - bot_trend)))
    mse_bot = float(np.mean((bot_vols_clean - bot_trend) ** 2))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(t_axis, top_vols_clean, label="Top chamber",    color="steelblue")
    ax.plot(t_axis, bot_vols_clean, label="Bottom chamber", color="firebrick")
    ax.plot(t_axis, total_clean,    label="Total",          color="mediumpurple")
    if top_linear is not None:
        ax.plot(t_axis, top_linear,   color="steelblue",   linewidth=1, alpha=0.4, linestyle="--", label="Top filter line")
        ax.plot(t_axis, bot_linear,   color="firebrick",   linewidth=1, alpha=0.4, linestyle="--", label="Bot filter line")
        ax.plot(t_axis, total_linear, color="mediumpurple", linewidth=1, alpha=0.4, linestyle="--", label="Total filter line")
    ax.set_ylabel("Sand volume (mm³)")
    ax.set_title(f"Hourglass sand volume — {run_dir.name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t_axis, top_vols_clean / (total_clean + 1e-9) * 100, color="steelblue",  label="Top %")
    ax2.plot(t_axis, bot_vols_clean / (total_clean + 1e-9) * 100, color="firebrick",  label="Bottom %")
    ax2.set_ylabel("Fraction of sand (%)")
    ax2.set_xlabel("Timesteps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plot_path = out_dir / "sand_volume.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

    mae_path = out_dir / "mae.txt"
    with open(mae_path, "w") as f:
        f.write("Sand volume conservation metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"N_TIMESTEPS={N_TIMESTEPS}  BINNING={BINNING}\n")
        f.write(f"Truth total (mean): {total_truth:.2f} mm³\n\n")

        f.write("Total conservation (total should be constant):\n")
        f.write(f"  MAE : {mae_total:.4f} mm³\n")
        f.write(f"  MSE : {mse_total:.4f} mm⁶\n\n")

        f.write("1:1 tracking (Δtop = −Δbot, residual = Δtop + Δbot):\n")
        f.write(f"  MAE : {mae_1to1:.4f} mm³\n")
        f.write(f"  MSE : {mse_1to1:.4f} mm⁶\n\n")

        f.write("Top chamber vs. linear trend:\n")
        f.write(f"  MAE : {mae_top:.4f} mm³\n")
        f.write(f"  MSE : {mse_top:.4f} mm⁶\n\n")

        f.write("Bottom chamber vs. linear trend:\n")
        f.write(f"  MAE : {mae_bot:.4f} mm³\n")
        f.write(f"  MSE : {mse_bot:.4f} mm⁶\n")
    print(f"Conservation metrics saved to {mae_path}")


def main():
    runs = find_runs(BASE_DIR)
    print(f"Found {len(runs)} run(s) under {BASE_DIR}")

    for run_dir in runs:
        if SKIP_EXISTING and not PLOT_ONLY and (run_dir / "mae.txt").exists():
            print(f"Skipping {run_dir.name} (mae.txt already exists)")
            continue
        try:
            process_run(run_dir)
        except Exception as e:
            print(f"ERROR processing {run_dir.name}: {e}")
            continue


if __name__ == "__main__":
    main()
