"""
Compare static reconstructions across projection counts and acquisitions.

All models are queried on a single canonical grid (taken from the ground-truth
geometry) so that every slice index maps to the same physical location regardless
of each model's own nVoxel or dVoxel settings.

Directory layout expected on the cluster:
  BASE_DIR/
    100_ac1/model/config.yaml  + checkpoints/last.ckpt
    100_ac2/model/config.yaml  + checkpoints/last.ckpt
    360_ac1/...
    1400_ac1/...   ← ground truth; provides the canonical grid

Usage:
    Edit the CONFIG block and run on a GPU node.
    Outputs:
      comparison.png      — orthogonal slice grid (rows=models, cols=planes)
      metrics.png         — PSNR / SSIM / MAE bar charts vs ground truth
      metrics.npz         — raw metric arrays for further analysis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from nect.config import get_cfg
from nect.data import NeCTDataset
from nect.sampling import Geometry

# ───────────────────────────── CONFIG ────────────────────────────────────────

BASE_DIR = Path(
    "/cluster/home/kristiac/NeCT/outputs/static_continious"
    "/hash_grid_23_4_23_16_2_4_128_L1"
)

# Ground-truth folder — defines the canonical grid
GT_NAME = "1400_ac1"

# Which folders to compare (order = rows in the figure).
# Use None to auto-discover every folder in BASE_DIR that contains model/.
COMPARE_NAMES: list[str] | None = [
    "100_ac1", "100_ac2", "100_ac3", "100_ac4", "100_ac6",
    "360_ac1", "360_ac2", "360_ac3", "360_ac4", "360_ac6",
    "1400_ac1",
]

# Spatial downsampling factor (1 = full res, 2 = 2× faster / lower mem).
BINNING = 1

# Crop fractions: only query this sub-region of the full volume.
# (start, end) as fractions of each axis — cuts empty air around the sample.
CROP_Z = (0.10, 0.90)   # top / bottom
CROP_Y = (0.10, 0.75)   # front / back
CROP_X = (0.25, 0.75)   # left / right

# Slice fractions within the cropped volume in [0, 1].
SLICE_Z = 0.5
SLICE_Y = 0.5
SLICE_X = 0.5

OUTPUT_PNG  = BASE_DIR / "comparison.png"
OUTPUT_PSNR = BASE_DIR / "psnr.png"
OUTPUT_SSIM = BASE_DIR / "ssim.png"
OUTPUT_MAE  = BASE_DIR / "mae.png"
OUTPUT_NPZ  = BASE_DIR / "metrics.npz"
OUTPUT_TXT  = BASE_DIR / "metrics.txt"

# ─────────────────────────────────────────────────────────────────────────────


def load_model(model_dir: Path, device: torch.device):
    """Load model and calibration constants from a model/ directory."""
    config = get_cfg(model_dir / "config.yaml")
    model = config.get_model()
    ckpt = torch.load(model_dir / "checkpoints" / "last.ckpt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    dataset = NeCTDataset(config=config, device="cpu")
    geometry = Geometry.from_cfg(
        config.geometry,
        reconstruction_mode=config.reconstruction_mode,
        sample_outside=config.sample_outside,
    )
    scale = 1.0 / geometry.max_distance_traveled
    data_min = dataset.minimum.item()
    data_max = dataset.maximum.item()
    return model, scale, data_min, data_max


def build_canonical_grid(gt_model_dir: Path, binning: int):
    """Return (z_lin, y_lin, x_lin) spanning only the cropped sample region."""
    config = get_cfg(gt_model_dir / "config.yaml")
    nVoxel = list(config.geometry.nVoxel)  # [nz, ny, nx]
    rm = config.sample_outside

    # Inner coordinate range after removing rm padding (y and x only).
    rm_frac_y = rm / (nVoxel[1] + 2 * rm)
    rm_frac_x = rm / (nVoxel[2] + 2 * rm)
    y_inner = (rm_frac_y, 1.0 - rm_frac_y)
    x_inner = (rm_frac_x, 1.0 - rm_frac_x)

    # Apply crop fractions to each axis's usable range.
    def crop_range(lo, hi, c0, c1):
        span = hi - lo
        return lo + c0 * span, lo + c1 * span

    z_lo, z_hi = crop_range(0.0, 1.0,         *CROP_Z)
    y_lo, y_hi = crop_range(*y_inner,           *CROP_Y)
    x_lo, x_hi = crop_range(*x_inner,           *CROP_X)

    nz = max(1, int((CROP_Z[1] - CROP_Z[0]) * nVoxel[0] / binning))
    ny = max(1, int((CROP_Y[1] - CROP_Y[0]) * nVoxel[1] / binning))
    nx = max(1, int((CROP_X[1] - CROP_X[0]) * nVoxel[2] / binning))

    z_lin = torch.linspace(z_lo, z_hi, steps=nz)
    y_lin = torch.linspace(y_lo, y_hi, steps=ny)
    x_lin = torch.linspace(x_lo, x_hi, steps=nx)
    return z_lin, y_lin, x_lin


@torch.no_grad()
def reconstruct_volume(
    model,
    scale: float,
    data_min: float,
    data_max: float,
    z_lin: torch.Tensor,
    y_lin: torch.Tensor,
    x_lin: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Query a static model on the canonical grid and calibrate."""
    nz, ny, nx = len(z_lin), len(y_lin), len(x_lin)
    vol = torch.zeros((nz, ny, nx), dtype=torch.float32)
    z_lin_d = z_lin.to(device)
    y_lin_d = y_lin.to(device)
    x_lin_d = x_lin.to(device)
    for i, z_ in enumerate(z_lin_d):
        z, y, x = torch.meshgrid(
            [z_.unsqueeze(0), y_lin_d, x_lin_d], indexing="ij"
        )
        grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t()
        raw = model(grid).reshape(ny, nx)
        calibrated = raw * scale * (data_max - data_min) + data_min
        vol[i] = calibrated.cpu()
    return vol.numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gt_model_dir = BASE_DIR / GT_NAME / "model"
    print("Building canonical grid from ground truth ...")
    z_lin, y_lin, x_lin = build_canonical_grid(gt_model_dir, BINNING)
    nz, ny, nx = len(z_lin), len(y_lin), len(x_lin)
    print(f"  Canonical grid (cropped): ({nz}, {ny}, {nx})")

    if COMPARE_NAMES is None:
        names = sorted(
            d.name for d in BASE_DIR.iterdir()
            if d.is_dir() and (d / "model" / "config.yaml").exists()
        )
    else:
        names = COMPARE_NAMES

    zi = int(SLICE_Z * nz)
    yi = int(SLICE_Y * ny)
    xi = int(SLICE_X * nx)

    def norm01_inplace(vol: np.ndarray) -> np.ndarray:
        """Normalise to [0,1] in-place — no temporary copy allocated."""
        lo = float(np.percentile(vol, 1))
        hi = float(np.percentile(vol, 99))
        if hi == lo:
            vol[:] = 0.0
            return vol
        vol -= lo
        vol /= (hi - lo)
        np.clip(vol, 0.0, 1.0, out=vol)
        return vol

    def process(vol: np.ndarray):
        """Normalise in-place and extract the three display slices."""
        norm01_inplace(vol)
        return vol, vol[zi].copy(), vol[:, yi, :].copy(), vol[:, :, xi].copy()

    # ── Reconstruct GT first, keep in RAM for metrics ─────────────────────────
    print(f"Reconstructing {GT_NAME} (ground truth) ...")
    gt_model_dir = BASE_DIR / GT_NAME / "model"
    if not (gt_model_dir / "config.yaml").exists():
        print("Ground truth config not found — aborting.")
        return
    model, scale, data_min, data_max = load_model(gt_model_dir, device)
    gt_raw = reconstruct_volume(model, scale, data_min, data_max, z_lin, y_lin, x_lin, device)
    del model; torch.cuda.empty_cache()
    gt_vol, gt_xy, gt_xz, gt_yz = process(gt_raw)
    del gt_raw

    # slices stored for plotting: {name: (xy, xz, yz)}
    slices: dict[str, tuple] = {GT_NAME: (gt_xy, gt_xz, gt_yz)}

    metric_names: list[str] = []
    psnr_vals:    list[float] = []
    ssim_vals:    list[float] = []
    mae_vals:     list[float] = []

    # ── Reconstruct each comparison model, free volume after metrics/slices ───
    valid_names = [GT_NAME]
    for name in names:
        if name == GT_NAME:
            continue
        model_dir = BASE_DIR / name / "model"
        if not (model_dir / "config.yaml").exists():
            print(f"  Skipping {name}: config.yaml not found")
            continue
        print(f"Reconstructing {name} ...")
        model, scale, data_min, data_max = load_model(model_dir, device)
        raw = reconstruct_volume(model, scale, data_min, data_max, z_lin, y_lin, x_lin, device)
        del model; torch.cuda.empty_cache()

        vol, xy, xz, yz = process(raw)
        del raw

        slices[name] = (xy, xz, yz)
        metric_names.append(name)
        psnr_vals.append(float(peak_signal_noise_ratio(gt_vol, vol, data_range=1.0)))
        ssim_vals.append(float(structural_similarity(gt_vol, vol, data_range=1.0)))
        mae_vals.append(float(np.mean(np.abs(vol - gt_vol))))
        valid_names.append(name)
        del vol

    n_models = len(slices)
    if n_models == 0:
        print("No models found — check BASE_DIR and COMPARE_NAMES.")
        return

    # ── Plot: rows = models, cols = {XY slice, XZ slice, YZ slice} ───────────
    n_cols = 3
    fig, axes = plt.subplots(n_models, n_cols, figsize=(4 * n_cols, 3 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    col_titles = [f"XY  (z={zi})", f"XZ  (y={yi})", f"YZ  (x={xi})"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9)

    for i, (name, (xy, xz, yz)) in enumerate(slices.items()):
        kw = dict(cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        axes[i, 0].imshow(xy, **kw)
        axes[i, 1].imshow(xz, **kw)
        axes[i, 2].imshow(yz, **kw)
        axes[i, 0].set_ylabel(name, fontsize=7, rotation=0, labelpad=60, va="center")
        for j in range(n_cols):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.suptitle("Static reconstructions — canonical grid comparison", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison to {OUTPUT_PNG}")

    if not metric_names:
        print("No comparison models found — skipping metrics.")
        return

    np.savez(
        OUTPUT_NPZ,
        names=metric_names,
        psnr=psnr_vals,
        ssim=ssim_vals,
        mae=mae_vals,
    )
    print(f"Saved raw metrics to {OUTPUT_NPZ}")

    col_w = max(len(n) for n in metric_names)
    with open(OUTPUT_TXT, "w") as f:
        f.write(f"Reconstruction quality vs ground truth ({GT_NAME})\n")
        f.write(f"BINNING={BINNING}  canonical grid from {GT_NAME}\n")
        f.write("=" * (col_w + 42) + "\n")
        f.write(f"{'Model':<{col_w}}   {'PSNR (dB)':>10}   {'SSIM':>8}   {'MAE':>10}\n")
        f.write("-" * (col_w + 42) + "\n")
        for name, psnr, ssim, mae in zip(metric_names, psnr_vals, ssim_vals, mae_vals):
            f.write(f"{name:<{col_w}}   {psnr:>10.4f}   {ssim:>8.4f}   {mae:>10.6f}\n")
    print(f"Saved scores to {OUTPUT_TXT}")

    # ── Colour each bar by projection count ──────────────────────────────────
    def proj_count(name: str) -> str:
        return name.split("_")[0]

    from matplotlib.patches import Patch

    proj_groups = sorted({proj_count(n) for n in metric_names}, key=int)
    palette = plt.cm.tab10.colors
    colour_map = {g: palette[i % len(palette)] for i, g in enumerate(proj_groups)}
    bar_colours = [colour_map[proj_count(n)] for n in metric_names]
    legend_handles = [Patch(color=colour_map[g], label=f"{g} proj") for g in proj_groups]

    x = np.arange(len(metric_names))
    figw = max(8, len(metric_names) * 0.8)

    def _bar_plot(vals, ylabel, title, path, fmt, pad):
        fig, ax = plt.subplots(figsize=(figw, 4), constrained_layout=True)
        bars = ax.bar(x, vals, color=bar_colours, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — vs ground truth ({GT_NAME})")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=8)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                    fmt.format(v), ha="center", va="bottom", fontsize=7)
        ax.legend(handles=legend_handles, title="Projection count", fontsize=9, framealpha=0.9)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {path.name} to {path}")

    _bar_plot(psnr_vals, "PSNR (dB)",            "PSNR (higher is better)", OUTPUT_PSNR, "{:.1f}",  0.2)
    _bar_plot(ssim_vals, "SSIM",                  "SSIM (higher is better)", OUTPUT_SSIM, "{:.3f}",  0.005)
    _bar_plot(mae_vals,  "MAE (attenuation units)", "MAE (lower is better)", OUTPUT_MAE,  "{:.4f}",  0.0)


if __name__ == "__main__":
    main()
