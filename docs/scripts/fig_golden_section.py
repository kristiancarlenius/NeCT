#!/usr/bin/env python3
"""
Generate Figure 2: Hybrid golden-section angle sampling.

Four-panel circle diagram showing how golden-angle revolution offsets
distribute projections across 3 rotations, plus a timing comparison
strip (step-and-shoot vs continuous) and a time axis.

Output:
  docs/images/fig_golden_section.pdf
  docs/images/fig_golden_section.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "images"
OUT_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PHI          = (1 + 5**0.5) / 2
GOLDEN_DEG   = 360.0 / (PHI ** 2)   # ≈ 137.508°
N_PER_REV    = 10
STEP_DEG     = 360.0 / N_PER_REV    # 36°

# Colours
C1       = "#2980B9"   # blue   — revolution 1
C2       = "#E67E22"   # orange — revolution 2
C3       = "#27AE60"   # green  — revolution 3
C_FAINT  = "#9BAAB4"   # muted grey for previous-revolution dots
C_SAMPLE = "#8D6E63"   # brown sample square
C_RING   = "#6E7E8A"   # orbit circle
C_ARROW  = "#555555"   # rotation arrow
C_DEAD   = "#BFC9CA"   # dead-time (motor settling) bars
BG       = "#F7F8FA"

# ── Angle sets (0° = top/12 o'clock, clockwise) ───────────────────────────────

def rev_angles(start_deg: float) -> list[float]:
    return [(start_deg + i * STEP_DEG) % 360 for i in range(N_PER_REV)]

REV1 = rev_angles(0.0)
REV2 = rev_angles(GOLDEN_DEG)           # ≈ 137.5°
REV3 = rev_angles(2.0 * GOLDEN_DEG)     # ≈ 275°

S2 = f"{GOLDEN_DEG:.1f}°"
S3 = f"{(2.0 * GOLDEN_DEG) % 360:.0f}°"


def to_xy(angles_deg: list[float]):
    """0° = top (12 o'clock), clockwise → (x, y) on unit circle."""
    a = np.radians(angles_deg)
    return np.sin(a), np.cos(a)


# ── Figure / GridSpec ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = GridSpec(
    3, 4,
    height_ratios=[1.5, 5.6, 0.95],
    hspace=0.08, wspace=0.05,
    left=0.06, right=0.98,
    top=0.93, bottom=0.05,
    figure=fig,
)

panel_ax  = [fig.add_subplot(gs[1, c]) for c in range(4)]
timing_ax = fig.add_subplot(gs[0, :])
time_ax   = fig.add_subplot(gs[2, :])

fig.suptitle("Hybrid golden-section angle sampling",
             fontsize=17, fontweight="bold", color="#1C2833", y=0.98)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Circle panels
# ─────────────────────────────────────────────────────────────────────────────

def draw_panel(ax, dot_sets, title, title_color="#222222"):
    """
    dot_sets: list of (angles_deg, color, alpha, dot_size)
    Drawn back-to-front (faint sets first).
    """
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    # Subtle panel border
    ax.add_patch(Rectangle((-1.45, -1.45), 2.9, 2.9,
                            facecolor="white", edgecolor="#D5DBDB",
                            linewidth=1.2, zorder=0))

    # Orbit circle
    ax.add_patch(Circle((0, 0), 1.0, fill=False,
                         edgecolor=C_RING, linewidth=2.0, zorder=2))

    # Centre sample square
    s = 0.14
    ax.add_patch(Rectangle((-s, -s), 2 * s, 2 * s,
                             facecolor=C_SAMPLE, edgecolor="#4E342E",
                             linewidth=1.5, zorder=5))

    # Rotation arrow (CCW arc in standard math coords, 0=right)
    t_arc = np.linspace(np.radians(60), np.radians(300), 70)
    r_arc = 0.30
    xa, ya = r_arc * np.cos(t_arc), r_arc * np.sin(t_arc)
    ax.plot(xa, ya, "-", color=C_ARROW, lw=1.8, zorder=6,
            solid_capstyle="round")
    ax.annotate(
        "", xy=(xa[-1], ya[-1]), xytext=(xa[-6], ya[-6]),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                        lw=1.8, mutation_scale=12),
        zorder=7,
    )

    # Dots
    for angles, color, alpha, dot_size in dot_sets:
        x, y = to_xy(angles)
        ax.scatter(x, y, s=dot_size, c=color, alpha=alpha, zorder=3,
                   edgecolors="white" if alpha > 0.4 else "none",
                   linewidths=0.7)

    # Panel title (multiline)
    ax.text(0.5, 1.04, title, ha="center", va="bottom",
            transform=ax.transAxes, fontsize=12.5, color=title_color,
            fontweight="bold", multialignment="center")


# ── Panel data shorthand ───────────────────────────────────────────────────────
FA = dict(color=C_FAINT, alpha=0.35, size=55)   # faint (previous revolution)
V1 = dict(color=C1,      alpha=0.95, size=85)   # vivid rev 1
V2 = dict(color=C2,      alpha=0.95, size=85)   # vivid rev 2
V3 = dict(color=C3,      alpha=0.95, size=85)   # vivid rev 3


# Panel 1 — Revolution 1 only
draw_panel(
    panel_ax[0],
    [(REV1, V1["color"], V1["alpha"], V1["size"])],
    f"Revolution 1\nφ₀ = 0°",
    C1,
)

# Panel 2 — Rev 1 faint + Rev 2 vivid
draw_panel(
    panel_ax[1],
    [(REV1, FA["color"], FA["alpha"], FA["size"]),
     (REV2, V2["color"], V2["alpha"], V2["size"])],
    f"Revolution 2\nφ₀ = {S2}",
    C2,
)

# Panel 3 — Rev 1 & 2 faint + Rev 3 vivid
draw_panel(
    panel_ax[2],
    [(REV1, FA["color"], FA["alpha"], FA["size"]),
     (REV2, FA["color"], FA["alpha"], FA["size"]),
     (REV3, V3["color"], V3["alpha"], V3["size"])],
    f"Revolution 3\nφ₀ = {S3}",
    C3,
)

# Panel 4 — All three at full opacity
draw_panel(
    panel_ax[3],
    [(REV1, V1["color"], V1["alpha"], V1["size"]),
     (REV2, V2["color"], V2["alpha"], V2["size"]),
     (REV3, V3["color"], V3["alpha"], V3["size"])],
    "3 revolutions combined:\nnear-uniform coverage",
    "#1C2833",
)

# ── Legend on panel 4 ─────────────────────────────────────────────────────────
leg_handles = [
    mpatches.Patch(facecolor=C1, label="Rev 1"),
    mpatches.Patch(facecolor=C2, label="Rev 2"),
    mpatches.Patch(facecolor=C3, label="Rev 3"),
]
panel_ax[3].legend(handles=leg_handles, loc="lower center",
                   fontsize=11, frameon=True, framealpha=0.85,
                   edgecolor="#CCCCCC", ncol=3,
                   bbox_to_anchor=(0.5, -0.02))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Timing comparison strip (above panels)
# ─────────────────────────────────────────────────────────────────────────────
# xlim 0-4 roughly aligns each unit with one panel column.

timing_ax.set_xlim(-0.85, 4.05)
timing_ax.set_ylim(0.0, 1.0)
timing_ax.axis("off")
timing_ax.set_facecolor(BG)

EXP_FRAC  = 0.62    # fraction of slot devoted to exposure
DEAD_FRAC = 1.0 - EXP_FRAC

SS_Y, SS_H = 0.54, 0.38   # step-and-shoot bar: y-bottom, height
CS_Y, CS_H = 0.08, 0.38   # continuous bar: y-bottom, height

REV_COLORS = [C1, C2, C3, "#7F8C8D"]

for i, c in enumerate(REV_COLORS):
    x0    = float(i)
    ew    = 0.90 * EXP_FRAC     # exposure block width
    dw    = 0.90 * DEAD_FRAC    # dead-time block width

    # Step-and-shoot: [Exposure | Motor settling]
    timing_ax.add_patch(Rectangle((x0 + 0.05, SS_Y), ew, SS_H,
                                   facecolor=c, alpha=0.82,
                                   edgecolor="none", zorder=2))
    timing_ax.add_patch(Rectangle((x0 + 0.05 + ew, SS_Y), dw, SS_H,
                                   facecolor=C_DEAD, alpha=0.82,
                                   edgecolor="none", zorder=2))

    # Continuous scan: [Exposure only]
    timing_ax.add_patch(Rectangle((x0 + 0.05, CS_Y), ew, CS_H,
                                   facecolor=c, alpha=0.82,
                                   edgecolor="none", zorder=2))

# Annotation labels inside the first slot only (so they don't clutter)
cx_exp  = 0.05 + 0.90 * EXP_FRAC / 2
cx_dead = 0.05 + 0.90 * EXP_FRAC + 0.90 * DEAD_FRAC / 2

timing_ax.text(cx_exp,  SS_Y + SS_H / 2, "Exposure",
               ha="center", va="center", fontsize=9.5,
               color="white", fontweight="bold")
timing_ax.text(cx_dead, SS_Y + SS_H / 2, "Motor\nsettling",
               ha="center", va="center", fontsize=8.5,
               color="#555555", multialignment="center")
timing_ax.text(cx_exp,  CS_Y + CS_H / 2, "Exposure only",
               ha="center", va="center", fontsize=9.5,
               color="white", fontweight="bold")

# Row labels
timing_ax.text(-0.05, SS_Y + SS_H / 2, "Step-and-shoot:",
               ha="right", va="center", fontsize=11.5,
               color="#333333", fontweight="bold")
timing_ax.text(-0.05, CS_Y + CS_H / 2, "Continuous scan:",
               ha="right", va="center", fontsize=11.5,
               color="#333333", fontweight="bold")

# Legend
leg2 = [mpatches.Patch(facecolor="#7F8C8D", alpha=0.82, label="Exposure time"),
        mpatches.Patch(facecolor=C_DEAD,    alpha=0.82, label="Motor settling (dead time)")]
timing_ax.legend(handles=leg2, loc="upper right",
                 fontsize=10, frameon=False,
                 handlelength=1.0, handleheight=0.75,
                 bbox_to_anchor=(1.0, 1.35))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Time axis (below panels)
# ─────────────────────────────────────────────────────────────────────────────
# ticks at x = 0, 1, 2, 3  →  t=0, T, 2T, 3T
# (panel i spans x ≈ [i, i+1], so tick i is at the left edge of panel i)

time_ax.set_xlim(-0.1, 4.35)
time_ax.set_ylim(0.0, 1.0)
time_ax.axis("off")
time_ax.set_facecolor(BG)

# Axis arrow
time_ax.annotate(
    "", xy=(4.25, 0.5), xytext=(0.0, 0.5),
    arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=2.2, mutation_scale=14),
    zorder=3,
)

# Ticks + labels
tick_pos    = [0.0, 1.0, 2.0, 3.0]
tick_labels = [r"$t = 0$", r"$t = T$", r"$t = 2T$", r"$t = 3T$"]
for x, lbl in zip(tick_pos, tick_labels):
    time_ax.plot([x, x], [0.33, 0.67], "-", color="#2C3E50", lw=2.0)
    time_ax.text(x, 0.12, lbl, ha="center", va="top",
                 fontsize=13, color="#2C3E50")

time_ax.text(4.30, 0.5, "time", ha="left", va="center",
             fontsize=13, color="#2C3E50", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save
# ─────────────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = OUT_DIR / f"fig_golden_section.{ext}"
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"Saved → {path}")

plt.close(fig)
