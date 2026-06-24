#!/usr/bin/env python3
"""
Generate 4 figures illustrating sub-angle differentiation via overlapping projections.

  fig_subangle_1 — orbit: single projection, K=3 sub-angle radial lines
  fig_subangle_2 — signal: single projection → all sub-angles look identical (uniform)
  fig_subangle_3 — orbit: two projections whose arcs overlap (3rd sub-angle of Proj 1
                           is the same angle as the 1st sub-angle of Proj 2)
  fig_subangle_4 — signal: both projections overlaid → non-uniform profile recoverable

Output (PDF + PNG each):
  docs/images/fig_subangle_{1,2,3,4}.{pdf,png}

Usage:
  python docs/scripts/fig_subangle_differentiation.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "images"
OUT_DIR.mkdir(exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
BG        = "#F7F8FA"
WHITE     = "#FFFFFF"
NAVY      = "#1E2A3A"
BLUE      = "#2980B9"
BLUE_LT   = "#AED6F1"
ORANGE    = "#E67E22"
ORANGE_LT = "#FAD7A0"
MGRAY     = "#95A5A6"
DGRAY     = "#5D6D7E"
LGRAY     = "#ECF0F1"
SHARED_C  = "#8E44AD"   # purple for the shared sub-angle

# ── Geometry ───────────────────────────────────────────────────────────────────
ARC_WIDTH    = 60.0     # degrees each projection integrates over
K            = 3        # accumulation steps (sub-angles) per projection
PROJ_A_START = 0.0
PROJ_B_START = 40.0     # places B's 1st sub-angle == A's 3rd sub-angle


def sub_angles(start: float) -> list[float]:
    step = ARC_WIDTH / K
    return [start + (i + 0.5) * step for i in range(K)]


A_SUBS = sub_angles(PROJ_A_START)   # [10°, 30°, 50°]
B_SUBS = sub_angles(PROJ_B_START)   # [50°, 70°, 90°]  ← B_SUBS[0] == A_SUBS[2]


def sig(theta: float) -> float:
    """Simulated per-sub-angle signal — smooth hump peaking near 50°."""
    return 0.35 + 0.45 * np.sin(np.radians(theta) * np.pi / np.radians(90.0))


SIG_A  = [sig(θ) for θ in A_SUBS]
SIG_B  = [sig(θ) for θ in B_SUBS]
MEAN_A = float(np.mean(SIG_A))
MEAN_B = float(np.mean(SIG_B))


# ── Shared helpers ─────────────────────────────────────────────────────────────

def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"Saved → {path}")
    plt.close(fig)


def to_xy(deg: float, r: float = 1.0):
    """0° = 12-o'clock, clockwise → (x, y)."""
    return r * np.sin(np.radians(deg)), r * np.cos(np.radians(deg))


def make_orbit_ax(suptitle: str):
    fig = plt.figure(figsize=(5, 5), facecolor=BG)
    ax  = fig.add_axes([0.05, 0.05, 0.90, 0.86])
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(WHITE)
    # orbit ring
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.sin(t), np.cos(t), color=MGRAY, lw=2.0, alpha=0.50)
    # sample square at centre
    s = 0.075
    ax.add_patch(plt.Polygon(
        [[-s, -s], [s, -s], [s, s], [-s, s]], closed=True,
        facecolor="#8D6E63", edgecolor="#4E342E", lw=1.4, zorder=8))
    fig.suptitle(suptitle, fontsize=11.5, fontweight="bold", color=NAVY, y=0.99)
    return fig, ax


def draw_sector(ax, start: float, width: float,
                fill_color, line_color, fill_alpha: float = 0.20) -> None:
    theta = np.linspace(start, start + width, 80)
    xs = np.concatenate([[0], np.sin(np.radians(theta)), [0]])
    ys = np.concatenate([[0], np.cos(np.radians(theta)), [0]])
    ax.fill(xs, ys, color=fill_color, alpha=fill_alpha, zorder=2)
    ax.plot(np.sin(np.radians(theta)),
            np.cos(np.radians(theta)), color=line_color, lw=2.2, zorder=3)
    for d in [start, start + width]:
        x, y = to_xy(d)
        ax.plot([0, x], [0, y], color=line_color, lw=0.9, alpha=0.4, zorder=2)


def draw_sub_line(ax, deg: float, color, dot_size: int = 7):
    x, y = to_xy(deg)
    ax.plot([0, x], [0, y], color=color, lw=1.9, ls="--", alpha=0.85, zorder=5)
    ax.plot(x, y, "o", color=color, ms=dot_size, zorder=6,
            markeredgecolor=WHITE, markeredgewidth=0.9)
    return x, y


def signal_ax(fig_kw: dict | None = None):
    fig = plt.figure(figsize=(6, 4.5), facecolor=BG, **(fig_kw or {}))
    ax  = fig.add_axes([0.14, 0.18, 0.80, 0.66])
    ax.set_facecolor(WHITE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DGRAY)
    ax.spines["bottom"].set_color(DGRAY)
    ax.tick_params(colors=DGRAY, labelsize=10)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — orbit: single projection
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = make_orbit_ax("Projection 1  —  K = 3 sub-angles")

draw_sector(ax, PROJ_A_START, ARC_WIDTH, BLUE, BLUE)
for i, deg in enumerate(A_SUBS):
    x, y = draw_sub_line(ax, deg, BLUE)
    ax.text(x * 1.22, y * 1.22,
            f"a{i + 1}\n{deg:.0f}°",
            ha="center", va="center", fontsize=8.5, color=BLUE, fontweight="bold")

mx, my = to_xy(PROJ_A_START + ARC_WIDTH / 2, r=1.58)
ax.text(mx, my, f"Proj 1\n0°–60°",
        ha="center", va="center", fontsize=9.5, color=BLUE, fontweight="bold")
ax.text(0, -1.52,
        f"K = {K} sub-angles together integrate the full arc",
        ha="center", va="center", fontsize=8.5, color=DGRAY, style="italic")

save(fig, "fig_subangle_1")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — signal: single projection → uniform
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = signal_ax()

xs = np.arange(K, dtype=float)
ax.bar(xs, [MEAN_A] * K,
       color=BLUE_LT, edgecolor=BLUE, lw=1.5, width=0.55, zorder=3,
       label=f"Reconstructed (uniform = {MEAN_A:.2f})")
ax.plot(xs, SIG_A, "o--", color=BLUE, lw=1.4, ms=8, alpha=0.28, zorder=2,
        label="True signal (unknown to model)")
ax.axhline(MEAN_A, color=BLUE, lw=1.5, ls="--", alpha=0.50)
ax.text(xs[-1] + 0.38, MEAN_A + 0.025,
        f"avg = {MEAN_A:.2f}", ha="right", va="bottom", fontsize=9, color=BLUE)

ax.set_xticks(xs)
ax.set_xticklabels([f"a{i + 1}  ({θ:.0f}°)" for i, θ in enumerate(A_SUBS)])
ax.set_ylabel("Sub-angle signal contribution", fontsize=10, color=DGRAY)
ax.set_ylim(0, 1.10)
ax.legend(fontsize=9, frameon=True, framealpha=0.90, edgecolor="#CCCCCC",
          loc="upper right")
ax.set_title("One projection  →  sub-angles are indistinguishable",
             fontsize=11, color=NAVY, fontweight="bold", pad=7)
ax.text(0.5, -0.26,
        "Only the arc total is measured — model distributes signal uniformly",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=8.5, color=DGRAY, style="italic")

save(fig, "fig_subangle_2")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — orbit: two overlapping projections
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = make_orbit_ax(
    "Projections 1 & 2  —  overlapping arcs (shared sub-angle)")

# Draw sectors — overlap region gets both fills, resulting in visible blend
draw_sector(ax, PROJ_A_START, ARC_WIDTH, BLUE,   BLUE,   fill_alpha=0.22)
draw_sector(ax, PROJ_B_START, ARC_WIDTH, ORANGE, ORANGE, fill_alpha=0.22)

# Overlap sector highlight on top
OVERLAP_START = PROJ_B_START
OVERLAP_END   = PROJ_A_START + ARC_WIDTH   # = 60°
draw_sector(ax, OVERLAP_START, OVERLAP_END - OVERLAP_START,
            SHARED_C, SHARED_C, fill_alpha=0.30)

# Sub-angles for A
for i, deg in enumerate(A_SUBS):
    shared = deg in B_SUBS
    color  = SHARED_C if shared else BLUE
    size   = 9 if shared else 7
    x, y   = draw_sub_line(ax, deg, color, dot_size=size)
    label  = f"a{i + 1}=b1\n{deg:.0f}°" if shared else f"a{i + 1}\n{deg:.0f}°"
    ax.text(x * 1.22, y * 1.22, label,
            ha="center", va="center", fontsize=8.0, color=color, fontweight="bold")

# Sub-angles for B (skip shared)
for i, deg in enumerate(B_SUBS):
    if deg in A_SUBS:
        continue
    x, y = draw_sub_line(ax, deg, ORANGE)
    ax.text(x * 1.22, y * 1.22,
            f"b{i + 1}\n{deg:.0f}°",
            ha="center", va="center", fontsize=8.0, color=ORANGE, fontweight="bold")

# Sector labels
mx_a, my_a = to_xy(15.0, r=1.56)   # near top of Proj 1 arc
ax.text(mx_a, my_a, "Proj 1\n0°–60°",
        ha="center", va="center", fontsize=9, color=BLUE, fontweight="bold")
mx_b, my_b = to_xy(82.0, r=1.45)   # far right for Proj 2
ax.text(mx_b, my_b, "Proj 2\n40°–100°",
        ha="left", va="center", fontsize=9, color=ORANGE, fontweight="bold")

# Overlap callout — outside circle, above the a3=b1 dot, with arrow
dot_x, dot_y = to_xy(A_SUBS[2])            # shared dot on the orbit ring
box_x, box_y = to_xy(38.0, r=1.54)         # outside circle, above the dot
ax.annotate(
    "shared\n50°",
    xy=(dot_x, dot_y), xytext=(box_x, box_y),
    ha="center", va="center",
    fontsize=8, color=SHARED_C, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.22", facecolor=WHITE,
              edgecolor=SHARED_C, lw=1.2, alpha=0.95),
    arrowprops=dict(arrowstyle="-|>", color=SHARED_C, lw=1.2, mutation_scale=9),
    zorder=9,
)

handles = [
    mpatches.Patch(facecolor=BLUE_LT,   edgecolor=BLUE,   label="Proj 1 only"),
    mpatches.Patch(facecolor=ORANGE_LT, edgecolor=ORANGE, label="Proj 2 only"),
    mpatches.Patch(facecolor="#D7BDE2",  edgecolor=SHARED_C, label="Overlap region"),
    plt.Line2D([0], [0], marker="o", color=SHARED_C, ms=8, lw=0,
               markeredgecolor=WHITE, markeredgewidth=0.8,
               label="Shared sub-angle"),
]
ax.legend(handles=handles, loc="lower left", fontsize=7.5,
          frameon=True, framealpha=0.90, edgecolor="#CCCCCC",
          bbox_to_anchor=(-0.04, -0.06))

save(fig, "fig_subangle_3")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — signal: two projections overlaid → non-uniform
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = signal_ax()

# 5 unique sub-angles: a1, a2, a3=b1, b2, b3
unique_angles  = A_SUBS + [d for d in B_SUBS if d not in A_SUBS]
unique_signals = SIG_A  + [SIG_B[i] for i, d in enumerate(B_SUBS) if d not in A_SUBS]
bar_colors     = [BLUE, BLUE, SHARED_C, ORANGE, ORANGE]
edge_colors    = [BLUE, BLUE, SHARED_C, ORANGE, ORANGE]
xs = np.arange(len(unique_angles), dtype=float)

ax.bar(xs, unique_signals,
       color=[c + "55" for c in bar_colors],  # transparent fill via hex alpha
       edgecolor=edge_colors, lw=1.6, width=0.55, zorder=3)

# Proj 1 line (a1→a2→a3)
ax.plot(xs[:3], SIG_A, "o-", color=BLUE, lw=2.0, ms=8, zorder=5,
        markeredgecolor=WHITE, markeredgewidth=0.9, label="Proj 1 sub-angles")
# Proj 2 line (b1→b2→b3) — b1 is at xs[2]
b_xs = [xs[2]] + list(xs[3:])
b_ys = [SIG_B[0]] + [SIG_B[i] for i, d in enumerate(B_SUBS) if d not in A_SUBS]
ax.plot(b_xs, b_ys, "o-", color=ORANGE, lw=2.0, ms=8, zorder=5,
        markeredgecolor=WHITE, markeredgewidth=0.9, label="Proj 2 sub-angles")

# Shared sub-angle marker (on top)
ax.plot(xs[2], unique_signals[2], "o", color=SHARED_C, ms=12, zorder=7,
        markeredgecolor=WHITE, markeredgewidth=1.2)
ax.annotate(
    "shared\nconstraint",
    xy=(xs[2], unique_signals[2]),
    xytext=(xs[2] - 0.80, unique_signals[2] + 0.13),
    fontsize=8.5, color=SHARED_C, fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=SHARED_C, lw=1.3, mutation_scale=10),
    zorder=8,
)

# Uniform reference lines (what single projections assumed)
ax.axhline(MEAN_A, color=BLUE,   lw=1.3, ls=":",  alpha=0.55)
ax.axhline(MEAN_B, color=ORANGE, lw=1.3, ls=":",  alpha=0.55)
ax.text(-0.45, MEAN_A + 0.02, f"Proj 1 avg", ha="left",  va="bottom",
        fontsize=7.5, color=BLUE,   alpha=0.65)
ax.text(-0.45, MEAN_B - 0.04, f"Proj 2 avg", ha="left",  va="top",
        fontsize=7.5, color=ORANGE, alpha=0.65)

ax.set_xticks(xs)
xlabels = [f"a{i+1}\n({θ:.0f}°)" for i, θ in enumerate(A_SUBS)]
xlabels[2] = f"a3 = b1\n({A_SUBS[2]:.0f}°)"
xlabels += [f"b{i+2}\n({θ:.0f}°)" for i, θ in enumerate(B_SUBS[1:])]
ax.set_xticklabels(xlabels)
ax.set_ylabel("Sub-angle signal contribution", fontsize=10, color=DGRAY)
ax.set_ylim(0, 1.10)
ax.legend(fontsize=9, frameon=True, framealpha=0.90, edgecolor="#CCCCCC",
          loc="upper right")
ax.set_title("Overlapping projections  →  non-uniform profile recoverable",
             fontsize=11, color=NAVY, fontweight="bold", pad=7)
ax.text(0.5, -0.26,
        "Shared sub-angle links the two measurements — "
        "variation across angles becomes visible",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=8.5, color=DGRAY, style="italic")

save(fig, "fig_subangle_4")
