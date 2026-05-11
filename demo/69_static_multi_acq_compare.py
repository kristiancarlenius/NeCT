from __future__ import annotations

import atexit
import gc
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_erosion, sobel as sobel2d, laplace as laplace2d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from nect.config import get_cfg
from nect.data import NeCTDataset
from nect.sampling import Geometry

BASE_DIR = Path(
    "/cluster/home/kristiac/NeCT/outputs/static_continious"
    "/hash_grid_23_4_23_16_2_4_128_L1"
)

GT_NAME = "1400_ac1"

COMPARE_NAMES: list[str] | None = [
    "100_ac1", "100_ac2", "100_ac3", "100_ac4", "100_ac6",
    "360_ac1", "360_ac2", "360_ac3", "360_ac4", "360_ac6",
    "1400_ac1",
]

BINNING = 1

CROP_Z = (0.10, 0.90)
CROP_Y = (0.10, 0.75)
CROP_X = (0.25, 0.75)

SLICE_Z = 0.5
SLICE_Y = 0.5
SLICE_X = 0.5

MASK_RADIUS_FRAC = 0.45

SCRATCH_DIR: Path | None = BASE_DIR / ".tmp"

OUTPUT_PNG      = BASE_DIR / "comparison.png"
OUTPUT_PSNR     = BASE_DIR / "psnr.png"
OUTPUT_SSIM     = BASE_DIR / "ssim.png"
OUTPUT_MAE      = BASE_DIR / "mae.png"
OUTPUT_GRAD     = BASE_DIR / "grad_magnitude.png"
OUTPUT_LAPVAR   = BASE_DIR / "lap_variance.png"
OUTPUT_COMBINED = BASE_DIR / "combined_score.png"
OUTPUT_NPZ      = BASE_DIR / "metrics.npz"
OUTPUT_TXT      = BASE_DIR / "metrics.txt"


def load_model(model_dir: Path, device: torch.device):
    config = get_cfg(model_dir / "config.yaml")
    model = config.get_model()

    ckpt_path = model_dir / "checkpoints" / "last.ckpt"
    inf_path  = model_dir / "checkpoints" / "inference.pt"

    if inf_path.exists():
        sd = torch.load(inf_path, map_location="cpu")
        model.load_state_dict(sd)
        del sd
    else:
        print(f"    Extracting model-only weights (one-time, saving inference.pt) ...")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        torch.save(ckpt["model"], inf_path)
        del ckpt
        gc.collect()

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
    config = get_cfg(gt_model_dir / "config.yaml")
    nVoxel = list(config.geometry.nVoxel)
    rm = config.sample_outside

    rm_frac_y = rm / (nVoxel[1] + 2 * rm)
    rm_frac_x = rm / (nVoxel[2] + 2 * rm)
    y_inner = (rm_frac_y, 1.0 - rm_frac_y)
    x_inner = (rm_frac_x, 1.0 - rm_frac_x)

    def crop_range(lo, hi, c0, c1):
        span = hi - lo
        return lo + c0 * span, lo + c1 * span

    z_lo, z_hi = crop_range(0.0, 1.0,   *CROP_Z)
    y_lo, y_hi = crop_range(*y_inner,    *CROP_Y)
    x_lo, x_hi = crop_range(*x_inner,    *CROP_X)

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
    path: str | None = None,
) -> np.ndarray:
    nz, ny, nx = len(z_lin), len(y_lin), len(x_lin)
    if path is not None:
        vol: np.ndarray = np.memmap(path, dtype=np.float32, mode="w+", shape=(nz, ny, nx))
    else:
        vol = np.zeros((nz, ny, nx), dtype=np.float32)
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
        vol[i] = calibrated.cpu().numpy()
    return vol


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

    if MASK_RADIUS_FRAC > 0.0:
        cy_c, cx_c = ny / 2.0, nx / 2.0
        radius = MASK_RADIUS_FRAC * min(ny, nx)
        yy, xx = np.ogrid[:ny, :nx]
        mask_2d = ((yy - cy_c) ** 2 + (xx - cx_c) ** 2) <= radius ** 2
    else:
        mask_2d = np.ones((ny, nx), dtype=bool)

    def process(vol: np.ndarray):
        step = max(1, nz // 150)
        sample = np.concatenate([np.array(vol[z])[mask_2d] for z in range(0, nz, step)])
        lo = float(np.percentile(sample, 1))
        hi = float(np.percentile(sample, 99))
        del sample
        chunk = 64
        if hi > lo:
            for z0 in range(0, nz, chunk):
                z1 = min(z0 + chunk, nz)
                s = np.array(vol[z0:z1])
                s -= lo
                s /= (hi - lo)
                np.clip(s, 0.0, 1.0, out=s)
                s[:, ~mask_2d] = 0.0
                vol[z0:z1] = s
        else:
            vol[:] = 0.0
        return vol, vol[zi].copy(), vol[:, yi, :].copy(), vol[:, :, xi].copy()

    def sharpness_metrics(xy: np.ndarray, xz: np.ndarray, yz: np.ndarray) -> tuple[float, float]:
        def _interior(s: np.ndarray) -> np.ndarray:
            return binary_erosion(s > 0, iterations=2)

        def _grad(s: np.ndarray) -> float:
            s = s.astype(np.float32)
            gx = sobel2d(s, axis=1)
            gy = sobel2d(s, axis=0)
            m = _interior(s)
            return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)[m])) if m.any() else 0.0

        def _lapvar(s: np.ndarray) -> float:
            s = s.astype(np.float32)
            m = _interior(s)
            return float(np.var(laplace2d(s)[m])) if m.any() else 0.0

        mean_grad = (_grad(xy) + _grad(xz) + _grad(yz)) / 3
        lap_var   = (_lapvar(xy) + _lapvar(xz) + _lapvar(yz)) / 3
        return mean_grad, lap_var

    def compute_metrics_streamed(gt_vol, vol):
        chunk = 64
        mse_sum, mae_sum, n_px, ssim_sum = 0.0, 0.0, 0, 0.0
        for z0 in range(0, nz, chunk):
            z1 = min(z0 + chunk, nz)
            gt_c = np.array(gt_vol[z0:z1])
            v_c  = np.array(vol[z0:z1])
            gt_m = gt_c[:, mask_2d]
            v_m  = v_c[:, mask_2d]
            d    = v_m - gt_m
            mse_sum += float(np.sum(d ** 2))
            mae_sum += float(np.sum(np.abs(d)))
            n_px    += d.size
            for k in range(z1 - z0):
                ssim_sum += float(structural_similarity(gt_c[k], v_c[k], data_range=1.0))
        mse  = mse_sum / max(n_px, 1)
        mae  = mae_sum / max(n_px, 1)
        psnr = float(10.0 * np.log10(1.0 / (mse + 1e-12)))
        ssim = ssim_sum / nz
        return psnr, ssim, mae

    if SCRATCH_DIR is not None:
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        tmp_dir: Path | None = SCRATCH_DIR
        atexit.register(shutil.rmtree, str(SCRATCH_DIR), ignore_errors=True)
    else:
        tmp_dir = None

    def _mmap_path(name: str) -> str | None:
        return str(tmp_dir / name) if tmp_dir is not None else None

    print(f"Reconstructing {GT_NAME} (ground truth) ...")
    gt_model_dir = BASE_DIR / GT_NAME / "model"
    if not (gt_model_dir / "config.yaml").exists():
        print("Ground truth config not found — aborting.")
        return
    model, scale, data_min, data_max = load_model(gt_model_dir, device)
    gt_vol = reconstruct_volume(
        model, scale, data_min, data_max, z_lin, y_lin, x_lin, device,
        path=_mmap_path("gt.mmap"),
    )
    del model; torch.cuda.empty_cache()
    gt_vol, gt_xy, gt_xz, gt_yz = process(gt_vol)
    gt_grad, gt_lapvar = sharpness_metrics(gt_xy, gt_xz, gt_yz)
    print(f"  GT sharpness — grad: {gt_grad:.4f}  lap_var: {gt_lapvar:.6f}")

    slices: dict[str, tuple] = {GT_NAME: (gt_xy, gt_xz, gt_yz)}

    metric_names: list[str] = []
    psnr_vals:    list[float] = []
    ssim_vals:    list[float] = []
    mae_vals:     list[float] = []
    grad_vals:    list[float] = []
    lapvar_vals:  list[float] = []

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
        raw = reconstruct_volume(
            model, scale, data_min, data_max, z_lin, y_lin, x_lin, device,
            path=_mmap_path("cmp.mmap"),
        )
        del model; torch.cuda.empty_cache()

        vol, xy, xz, yz = process(raw)
        del raw

        slices[name] = (xy, xz, yz)
        metric_names.append(name)
        psnr, ssim, mae = compute_metrics_streamed(gt_vol, vol)
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
        mae_vals.append(mae)
        mg, lv = sharpness_metrics(xy, xz, yz)
        grad_vals.append(mg)
        lapvar_vals.append(lv)
        valid_names.append(name)
        del vol

    n_models = len(slices)
    if n_models == 0:
        print("No models found — check BASE_DIR and COMPARE_NAMES.")
        return

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
        grad=grad_vals,
        lapvar=lapvar_vals,
    )
    print(f"Saved raw metrics to {OUTPUT_NPZ}")

    col_w = max(len(n) for n in metric_names)
    sep = col_w + 66
    with open(OUTPUT_TXT, "w") as f:
        f.write(f"Reconstruction quality vs ground truth ({GT_NAME})\n")
        f.write(f"BINNING={BINNING}  canonical grid from {GT_NAME}\n")
        f.write(f"GT sharpness — grad_mag: {gt_grad:.4f}  lap_var: {gt_lapvar:.6f}\n")
        f.write("=" * sep + "\n")
        f.write(f"{'Model':<{col_w}}   {'PSNR (dB)':>10}   {'SSIM':>8}   {'MAE':>10}"
                f"   {'Grad Mag':>10}   {'Lap Var':>12}\n")
        f.write("-" * sep + "\n")
        for name, psnr, ssim, mae, gm, lv in zip(
                metric_names, psnr_vals, ssim_vals, mae_vals, grad_vals, lapvar_vals):
            f.write(f"{name:<{col_w}}   {psnr:>10.4f}   {ssim:>8.4f}   {mae:>10.6f}"
                    f"   {gm:>10.4f}   {lv:>12.6f}\n")
    print(f"Saved scores to {OUTPUT_TXT}")

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

    _bar_plot(psnr_vals,   "PSNR (dB)",              "PSNR (higher is better, ref-dependent)",                    OUTPUT_PSNR,   "{:.1f}",   0.2)
    _bar_plot(ssim_vals,   "SSIM",                   "SSIM (higher is better, ref-dependent)",                    OUTPUT_SSIM,   "{:.3f}",   0.005)
    _bar_plot(mae_vals,    "MAE",                     "MAE (lower is better, ref-dependent)",                      OUTPUT_MAE,    "{:.4f}",   0.0)
    _bar_plot(grad_vals,   "Mean gradient magnitude", "Sharpness — gradient magnitude (no ref, higher = sharper)", OUTPUT_GRAD,   "{:.4f}",   0.0)
    _bar_plot(lapvar_vals, "Laplacian variance",      "Sharpness — Laplacian variance (no ref, higher = sharper)", OUTPUT_LAPVAR, "{:.6f}",   0.0)

    # ── Combined score: geometric mean of normalised sharpness and accuracy ────
    # grad_norm  ∈ [0,1], higher = sharper
    # acc_norm   ∈ [0,1], higher = more accurate (1 - normalised MAE)
    # combined   = sqrt(grad_norm * acc_norm)  — geometric mean penalises extremes
    g = np.array(grad_vals, dtype=np.float64)
    m = np.array(mae_vals,  dtype=np.float64)

    g_range = g.max() - g.min()
    m_range = m.max() - m.min()
    grad_norm = (g - g.min()) / g_range if g_range > 0 else np.full_like(g, 0.5)
    acc_norm  = 1.0 - ((m - m.min()) / m_range if m_range > 0 else np.full_like(m, 0.5))
    combined  = np.sqrt(grad_norm * acc_norm)

    fig, ax = plt.subplots(figsize=(figw, 4), constrained_layout=True)
    bars = ax.bar(x, combined, color=bar_colours, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Combined score (higher is better)")
    ax.set_title(
        f"Combined score — sharpness × accuracy vs {GT_NAME}\n"
        f"geometric mean of normalised grad magnitude and normalised (1−MAE)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for bar, v, gn, an in zip(bars, combined, grad_norm, acc_norm):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.legend(handles=legend_handles, title="Projection count", fontsize=9, framealpha=0.9)
    plt.savefig(OUTPUT_COMBINED, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_COMBINED.name} to {OUTPUT_COMBINED}")

    with open(OUTPUT_TXT, "a") as f:
        f.write("\nCombined score (geometric mean of norm. grad and norm. accuracy vs GT)\n")
        f.write("-" * sep + "\n")
        for name, cs, gn, an in zip(metric_names, combined, grad_norm, acc_norm):
            f.write(f"{name:<{col_w}}   combined={cs:.4f}   grad_norm={gn:.4f}   acc_norm={an:.4f}\n")


if __name__ == "__main__":
    main()
