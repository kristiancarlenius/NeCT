from __future__ import annotations

import atexit
import gc
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion, rotate as nd_rotate, shift as nd_shift, sobel as sobel2d, laplace as laplace2d
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

N_SLICES = 10  # number of evenly spaced XY slices along Z

# Counter-clockwise rotation applied to display slices only (not metrics).
# Keys are prefixes matched against the run name.
SLICE_ROTATION: dict[str, float] = {
    "100_": 7,
    "360_": 1.6,
}

# Pixel shift applied to display slices only (not metrics): (rows_down, cols_right).
# Negative = up/left.
SLICE_SHIFT: dict[str, tuple[float, float]] = {
    "100_": (5, -16),
}

MASK_RADIUS_FRAC = 0.45

SCRATCH_DIR: Path | None = BASE_DIR / ".tmp"

OUTPUT_PNG      = BASE_DIR / "comparison.png"
OUTPUT_SLICES   = BASE_DIR / "slices"
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

    if MASK_RADIUS_FRAC > 0.0:
        cy_c, cx_c = ny / 2.0, nx / 2.0
        radius = MASK_RADIUS_FRAC * min(ny, nx)
        yy, xx = np.ogrid[:ny, :nx]
        mask_2d = ((yy - cy_c) ** 2 + (xx - cx_c) ** 2) <= radius ** 2
    else:
        mask_2d = np.ones((ny, nx), dtype=bool)

    # Evenly spaced Z indices for the N_SLICES XY slices
    z_indices = [int(round((i + 1) / (N_SLICES + 1) * nz)) for i in range(N_SLICES)]
    z_indices = [min(max(z, 0), nz - 1) for z in z_indices]
    z_fracs   = [(zi_val / nz) * (CROP_Z[1] - CROP_Z[0]) + CROP_Z[0] for zi_val in z_indices]

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
        xy_slices = [vol[z].copy() for z in z_indices]
        return vol, xy_slices

    def sharpness_metrics(xy_slices: list[np.ndarray]) -> tuple[float, float]:
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

        mean_grad = float(np.mean([_grad(s) for s in xy_slices]))
        lap_var   = float(np.mean([_lapvar(s) for s in xy_slices]))
        return mean_grad, lap_var

    def _rotate_for(name: str, xy_slices: list[np.ndarray]) -> list[np.ndarray]:
        angle = next(
            (deg for prefix, deg in SLICE_ROTATION.items() if name.startswith(prefix)),
            0.0,
        )
        if angle == 0.0:
            return xy_slices
        return [nd_rotate(sl, angle, reshape=False, order=1, mode="constant", cval=0.0)
                for sl in xy_slices]

    def _shift_for(name: str, xy_slices: list[np.ndarray]) -> list[np.ndarray]:
        shift = next(
            (s for prefix, s in SLICE_SHIFT.items() if name.startswith(prefix)),
            None,
        )
        if shift is None:
            return xy_slices
        return [nd_shift(sl, shift, order=1, mode="constant", cval=0.0)
                for sl in xy_slices]

    def compute_metrics_from_slices(
        gt_slices: list[np.ndarray], model_slices: list[np.ndarray]
    ) -> tuple[float, float, float]:
        mse_sum, mae_sum, n_px, ssim_sum = 0.0, 0.0, 0, 0.0
        for gt_sl, v_sl in zip(gt_slices, model_slices):
            gt_m = gt_sl[mask_2d]
            v_m  = v_sl[mask_2d]
            d    = v_m - gt_m
            mse_sum  += float(np.sum(d ** 2))
            mae_sum  += float(np.sum(np.abs(d)))
            n_px     += d.size
            ssim_sum += float(structural_similarity(gt_sl, v_sl, data_range=1.0))
        mse  = mse_sum / max(n_px, 1)
        mae  = mae_sum / max(n_px, 1)
        psnr = float(10.0 * np.log10(1.0 / (mse + 1e-12)))
        ssim = ssim_sum / len(gt_slices)
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
    gt_vol, gt_xy_slices = process(gt_vol)
    gt_xy_slices = _shift_for(GT_NAME, _rotate_for(GT_NAME, gt_xy_slices))
    gt_grad, gt_lapvar = sharpness_metrics(gt_xy_slices)
    print(f"  GT sharpness — grad: {gt_grad:.4f}  lap_var: {gt_lapvar:.6f}")

    slices: dict[str, list[np.ndarray]] = {GT_NAME: gt_xy_slices}

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

        vol, xy_slices = process(raw)
        del raw, vol

        xy_slices = _shift_for(name, _rotate_for(name, xy_slices))
        slices[name] = xy_slices
        metric_names.append(name)
        psnr, ssim, mae = compute_metrics_from_slices(gt_xy_slices, xy_slices)
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
        mae_vals.append(mae)
        mg, lv = sharpness_metrics(xy_slices)
        grad_vals.append(mg)
        lapvar_vals.append(lv)
        valid_names.append(name)

    n_models = len(slices)
    if n_models == 0:
        print("No models found — check BASE_DIR and COMPARE_NAMES.")
        return

    n_cols = N_SLICES
    fig, axes = plt.subplots(n_models, n_cols, figsize=(2 * n_cols, 2.5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    col_titles = [f"XY  z={zf:.2f}" for zf in z_fracs]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9)

    for i, (name, xy_slices) in enumerate(slices.items()):
        kw = dict(cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for j, sl in enumerate(xy_slices):
            axes[i, j].imshow(sl, **kw)
        axes[i, 0].set_ylabel(name, fontsize=7, rotation=0, labelpad=60, va="center")
        for j in range(n_cols):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.suptitle("Static reconstructions — canonical grid comparison", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison to {OUTPUT_PNG}")

    OUTPUT_SLICES.mkdir(parents=True, exist_ok=True)
    for name, xy_slices in slices.items():
        safe_name = name.replace("/", "_")
        for j, sl in enumerate(xy_slices):
            zf = z_fracs[j]
            out = OUTPUT_SLICES / f"{safe_name}_z{zf:.2f}.png"
            img_u8 = (np.clip(sl, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(img_u8, mode="L").save(out)
    print(f"Saved individual slices to {OUTPUT_SLICES}/")

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

    # ── Combined score: arithmetic mean of normalised sharpness and accuracy ────
    # grad_norm  ∈ [0,1], higher = sharper
    # acc_norm   ∈ [0,1], higher = more accurate (1 - normalised MAE)
    # combined   = (grad_norm + acc_norm) / 2  — arithmetic mean avoids zeroing out
    # models that are worst on one metric (geometric mean collapses to 0 for those).
    g = np.array(grad_vals, dtype=np.float64)
    m = np.array(mae_vals,  dtype=np.float64)

    g_range = g.max() - g.min()
    m_range = m.max() - m.min()
    grad_norm = (g - g.min()) / g_range if g_range > 0 else np.full_like(g, 0.5)
    acc_norm  = 1.0 - ((m - m.min()) / m_range if m_range > 0 else np.full_like(m, 0.5))
    combined  = (grad_norm*0.4 + acc_norm*0.6)

    fig, ax = plt.subplots(figsize=(figw, 4), constrained_layout=True)
    bars = ax.bar(x, combined, color=bar_colours, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Combined score (higher is better)")
    ax.set_title(
        f"Combined score — sharpness + accuracy vs {GT_NAME}\n"
        f"arithmetic mean of normalised grad magnitude and normalised (1−MAE)"
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
        f.write("\nCombined score (arithmetic mean of norm. grad and norm. accuracy vs GT)\n")
        f.write("-" * sep + "\n")
        for name, cs, gn, an in zip(metric_names, combined, grad_norm, acc_norm):
            f.write(f"{name:<{col_w}}   combined={cs:.4f}   grad_norm={gn:.4f}   acc_norm={an:.4f}\n")


if __name__ == "__main__":
    main()
