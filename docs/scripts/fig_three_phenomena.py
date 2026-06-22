#!/usr/bin/env python3
"""
Generate Figure 3: Three governing phenomena.

Three vertically stacked panels:
  A  Hash Collision Rate
  B  Frequency Coverage
  C  Spatial-Temporal Asymmetry

Output:
  docs/images/fig_three_phenomena.pdf
  docs/images/fig_three_phenomena.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "images"
OUT_DIR.mkdir(exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY     = "#1E2A3A"
BLUE     = "#2980B9"
ORANGE   = "#E67E22"
GREEN    = "#27AE60"
RED      = "#C0392B"
LGRAY    = "#ECF0F1"
MGRAY    = "#95A5A6"
DGRAY    = "#5D6D7E"
BG       = "#F7F8FA"
WHITE    = "#FFFFFF"

# ── Figure / GridSpec ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 24), facecolor=BG)
gs  = GridSpec(3, 1, left=0.02, right=0.99, bottom=0.01, top=0.97,
               hspace=0.0, figure=fig)
axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

for ax in axes:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(WHITE)
    # Subtle panel border
    ax.add_patch(Rectangle((0, 0), 10, 10,
                            facecolor=WHITE, edgecolor="#D5DBDB",
                            linewidth=1.2, zorder=0, clip_on=False))




# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def panel_label(ax, letter, title):
    ax.text(0.4, 9.65, letter, fontsize=16, fontweight="bold",
            color=NAVY, va="top", zorder=10)
    ax.text(5.0, 9.65, title, fontsize=11.5, fontweight="bold",
            ha="center", va="top", color=NAVY, zorder=10)


def box(ax, x, y, w, h, fc, ec=DGRAY, lw=1.3, zorder=2, label=None,
        label_size=9, label_color=NAVY):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                            linewidth=lw, zorder=zorder))
    if label:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=label_size, color=label_color, fontweight="bold",
                zorder=zorder + 1)


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=2.0, ms=12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=ms),
                zorder=8)


# ─────────────────────────────────────────────────────────────────────────────
# Panel A — Hash Collision Rate
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]
panel_label(ax, "A", "Hash Collision Rate")

# ── Hash table: 8 cells ───────────────────────────────────────────────────────
N_CELLS       = 8
CELL_W, CELL_H = 1.0, 1.3
TABLE_X0       = 1.0
TABLE_Y0       = 5.5
COLL_IDX       = 3         # cell that gets the collisions

for i in range(N_CELLS):
    x  = TABLE_X0 + i * CELL_W
    fc = "#FDE8CC" if i == COLL_IDX else LGRAY
    ec = ORANGE    if i == COLL_IDX else "#AAB7B8"
    box(ax, x, TABLE_Y0, CELL_W, CELL_H, fc, ec, lw=2.0 if i == COLL_IDX else 1.2,
        label=str(i), label_size=8)

ax.text(5.0, TABLE_Y0 - 0.35, "Hash table  (T = 8 cells shown)",
        ha="center", va="top", fontsize=8, color=MGRAY, style="italic")

# ── Three arrows → same cell ──────────────────────────────────────────────────
cell_cx  = TABLE_X0 + COLL_IDX * CELL_W + CELL_W / 2   # 4.5
cell_top = TABLE_Y0 + CELL_H                             # 6.8

srcs = [
    (cell_cx - 1.9, 8.9, "vertex A", BLUE),
    (cell_cx,       9.1, "vertex B", ORANGE),
    (cell_cx + 1.9, 8.9, "vertex C", GREEN),
]
for sx, sy, lbl, c in srcs:
    arrow(ax, sx, sy, cell_cx, cell_top, color=c, lw=2.0)
    ax.text(sx, sy + 0.15, lbl, ha="center", va="bottom",
            fontsize=9, color=c, fontweight="bold")

# ── Bottom: two mini collision diagrams ──────────────────────────────────────

def _mini_collision(ax, cx, n_arrows, fc_box, label1, label2, lc):
    bw, bh = 1.6, 0.9
    bx, by = cx - bw / 2, 2.2
    box(ax, bx, by, bw, bh, fc_box, lw=1.5, label="hash\ncell", label_size=7.5)
    target_y = by + bh
    # Sources spread linearly above the box
    if n_arrows == 1:
        src_xs = [cx]
    else:
        src_xs = np.linspace(cx - 1.5, cx + 1.5, n_arrows)
    src_y = by + bh + 1.6
    for sx in src_xs:
        arrow(ax, sx, src_y, cx, target_y, color="#555555", lw=1.3, ms=9)
    ax.text(cx, by - 0.25, label1, ha="center", va="top",
            fontsize=8.5, fontweight="bold", color=NAVY)
    ax.text(cx, by - 0.72, label2, ha="center", va="top",
            fontsize=8.5, color=lc, fontweight="bold")


_mini_collision(ax, cx=2.5, n_arrows=1,
                fc_box="#D5F5E3",
                label1="d = 2,  T = 2²⁰",
                label2="collision rate ≈ 1",
                lc="#1A6F00")

ax.text(5, 2.7, "vs", ha="center", va="center",
        fontsize=12, color=MGRAY, fontweight="bold")

_mini_collision(ax, cx=7.5, n_arrows=7,
                fc_box="#FADBD8",
                label1="d = 3,  T = 2²⁰",
                label2="collision rate = 2¹⁰",
                lc=RED)

# Thin horizontal divider
ax.plot([0.4, 9.6], [4.9, 4.9], "--", color="#CCCCCC", lw=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Panel B — Frequency Coverage
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]
panel_label(ax, "B", "Frequency Coverage")


def make_pore_img(n_levels):
    """Synthetic 64×64 pore image, sharpness proportional to n_levels."""
    sz = 64
    img = np.zeros((sz, sz), dtype=float)
    for px, py in [(16, 16), (48, 16), (16, 48), (48, 48), (32, 32)]:
        y, x = np.ogrid[:sz, :sz]
        img += ((x - px)**2 + (y - py)**2 < 7**2).astype(float)
    img = np.clip(img, 0, 1)
    # Simulate resolution: downsample then upsample
    step = max(1, sz // (2 ** (n_levels // 2 + 1)))
    lo   = img[::step, ::step]
    up   = np.repeat(np.repeat(lo, step, axis=0), step, axis=1)[:sz, :sz]
    return up


def _draw_level_stack(ax, x0, x1, y_top, n_levels, col_label):
    w       = x1 - x0
    stack_h = 5.8
    lh      = stack_h / n_levels     # height of one level strip
    cmap    = plt.cm.Blues           # coarse=light, fine=dark

    for i in range(n_levels):
        y     = y_top - (i + 1) * lh
        alpha = 0.25 + 0.75 * i / max(n_levels - 1, 1)
        fc    = cmap(alpha)

        ax.add_patch(Rectangle((x0, y), w, lh - 0.04,
                               facecolor=fc, edgecolor=WHITE,
                               linewidth=0.8, zorder=2))

        # Vertical grid lines (frequency markers)
        n_lines = min(2 ** (i + 1), 28)
        for j in range(1, n_lines):
            lx = x0 + j * w / n_lines
            ax.plot([lx, lx], [y, y + lh - 0.04],
                    "-", color=WHITE, lw=0.6, alpha=0.7, zorder=3)

        # Level label on the left (first and last only)
        if i == 0 or i == n_levels - 1:
            lbl = ("Level 0\n(coarsest)" if i == 0
                   else f"Level {n_levels - 1}\n(finest)")
            ax.text(x0 - 0.15, y + lh / 2, lbl,
                    ha="right", va="center", fontsize=6.5, color=NAVY,
                    multialignment="center")

    # Column header
    ax.text((x0 + x1) / 2, y_top + 0.22, col_label,
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=NAVY)

    # Reconstruction thumbnail below stack
    thumb_y0  = y_top - stack_h - 0.15
    thumb_h   = 2.2
    thumb_y1  = thumb_y0 - thumb_h
    pore_img  = make_pore_img(n_levels)
    ax.imshow(pore_img, extent=[x0, x1, thumb_y1, thumb_y0],
              cmap="gray", origin="upper", aspect="auto",
              vmin=0, vmax=1, zorder=2)
    ax.add_patch(Rectangle((x0, thumb_y1), w, thumb_h,
                            facecolor="none", edgecolor=DGRAY,
                            linewidth=1.0, zorder=3))
    sharpness = "blurry pores" if n_levels <= 4 else "sharper pores"
    ax.text((x0 + x1) / 2, thumb_y1 - 0.15, sharpness,
            ha="center", va="top", fontsize=7.5, color=DGRAY, style="italic")


_draw_level_stack(ax, x0=0.5, x1=4.2,  y_top=8.8, n_levels=4, col_label="L = 4")
_draw_level_stack(ax, x0=5.8, x1=9.5,  y_top=8.8, n_levels=8, col_label="L = 8")

ax.text(5.0, 0.05,
        "More levels  →  finer frequency coverage  →  slower training",
        ha="center", va="bottom", fontsize=8.5, color=NAVY,
        style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Panel C — Spatial-Temporal Asymmetry
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[2]
panel_label(ax, "C", "Spatial-Temporal Asymmetry")

# ── Left: QuadCubes baseline — four equal boxes ───────────────────────────────
BW, BH  = 1.8, 1.5    # box width / height
GAP     = 0.25
CX_L    = 2.5          # column x-centre
CX_R    = 7.5
TOP_Y   = 8.2

BOX_COLOR_STATIC   = "#2980B9"   # blue = spatial
BOX_COLOR_TEMPORAL = "#7FB3D3"   # lighter blue = temporal

box_defs_L = [
    ("xyz", BOX_COLOR_STATIC),
    ("xyt", BOX_COLOR_TEMPORAL),
    ("xzt", BOX_COLOR_TEMPORAL),
    ("yzt", BOX_COLOR_TEMPORAL),
]
for i, (lbl, fc) in enumerate(box_defs_L):
    row, col = divmod(i, 2)
    bx = CX_L - BW - GAP / 2 + col * (BW + GAP)
    by = TOP_Y - BH - row * (BH + GAP)
    box(ax, bx, by, BW, BH, fc, lw=1.5, label=lbl,
        label_color=WHITE, label_size=10)

ax.text(CX_L, TOP_Y + 0.25, "QuadCubes baseline",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=NAVY)
ax.text(CX_L, TOP_Y - 2 * BH - GAP - 0.35,
        "uniform capacity\nfor all sub-spaces",
        ha="center", va="top", fontsize=8, color=MGRAY,
        multialignment="center")

# ── Right: CombinedCubes — xyz larger/darker, 3 temporal smaller/lighter ──────
XYZ_W, XYZ_H   = 3.8, 2.2    # large xyz box
TMP_W, TMP_H   = 1.15, 1.4   # small temporal box
TMP_GAP        = 0.18

xyz_x = CX_R - XYZ_W / 2
xyz_y = TOP_Y - XYZ_H
box(ax, xyz_x, xyz_y, XYZ_W, XYZ_H, "#1A5276", lw=2.0,
    label="xyz", label_color=WHITE, label_size=13)

# Three temporal boxes below xyz
tmp_labels = ["xyt", "xzt", "yzt"]
tmp_total  = 3 * TMP_W + 2 * TMP_GAP
tmp_x0     = CX_R - tmp_total / 2
TMP_Y      = xyz_y - TMP_H - 0.45
for j, lbl in enumerate(tmp_labels):
    tx = tmp_x0 + j * (TMP_W + TMP_GAP)
    box(ax, tx, TMP_Y, TMP_W, TMP_H, "#AED6F1", lw=1.3,
        label=lbl, label_color=NAVY, label_size=9)

ax.text(CX_R, TOP_Y + 0.25, "CombinedCubes",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=NAVY)

# Motivation text below each right element
ax.text(CX_R, xyz_y - 0.05,
        "sharp static\npore boundaries",
        ha="center", va="top", fontsize=7.5, color="#1A5276",
        multialignment="center", style="italic")
ax.text(CX_R, TMP_Y - 0.18,
        "slowly evolving\nfluid front",
        ha="center", va="top", fontsize=7.5, color="#2471A3",
        multialignment="center", style="italic")

# Overall caption below left column
ax.text(CX_L, 1.35,
        "xyz needs more capacity:\nsharp, static boundaries",
        ha="center", va="center", fontsize=8, color=DGRAY,
        multialignment="center", bbox=dict(boxstyle="round,pad=0.3",
                                            facecolor=LGRAY, edgecolor="none"))
ax.text(CX_R, 1.35,
        "temporal encoders need less:\nslowly evolving fluid front",
        ha="center", va="center", fontsize=8, color=DGRAY,
        multialignment="center", bbox=dict(boxstyle="round,pad=0.3",
                                            facecolor=LGRAY, edgecolor="none"))

# Vertical divider between left / right in panel C
ax.plot([5, 5], [0.5, 9.3], "--", color="#CCCCCC", lw=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = OUT_DIR / f"fig_three_phenomena.{ext}"
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"Saved → {path}")

plt.close(fig)
