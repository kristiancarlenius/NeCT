#!/usr/bin/env python3
"""
Generate the two-branch thesis overview figure.

Output:
  docs/images/thesis_overview.pdf
  docs/images/thesis_overview.png

Usage:
  python docs/scripts/thesis_overview.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / "images"
OUT_DIR.mkdir(exist_ok=True)

# ── Figure dimensions (data units ≈ inches) ───────────────────────────────────
FW, FH = 14.0, 9.0

# ── Color palette ─────────────────────────────────────────────────────────────
BG      = "#F4F6F8"
NAVY    = "#1E2A3A"     # title bar, root node
BLUE    = "#1A6FA0"     # left branch (encoder optimisation)
BLUE_LT = "#2B87BF"    # slightly lighter for 2nd left box
BLUE_HL = "#0F4E7A"    # darkest for 3rd / highlight box
ORANGE  = "#C05C1B"    # right branch (continuous scanning)
ORANGE2 = "#9E4410"    # darker right box
GREEN   = "#1B6B3A"    # footer
WHITE   = "#FFFFFF"
GRAY    = "#BDC3C7"    # frame border
RED     = "#B03020"    # annotation numbers

# ── Main axes ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Layout constants ──────────────────────────────────────────────────────────
LEFT_X,  RIGHT_X  = 3.5,  10.5
LW,  LH           = 5.0,  0.85   # left box width / height
RW,  RH           = 5.0,  1.05   # right box width / height
ARND = "round,pad=0.12"

# Y positions (measured from bottom of figure)
FOOTER_BOT, FOOTER_TOP = 0.15, 1.2
L_BOX3_CY = 2.3       # left box 3 (MixedCubes)
L_BOX2_CY = 3.75      # left box 2 (CombinedCubes)
L_BOX1_CY = 5.2       # left box 1 (QuadCubes baseline)
R_BOX2_CY = 3.05      # right box 2 (Fly-scan)
R_BOX1_CY = 4.95      # right box 1 (Step-and-shoot)
HEADER_Y  = 5.9       # branch header text
JUNCTION_Y = 6.5      # horizontal split line
ROOT_CY   = 7.3       # root box centre
TITLE_BOT = 8.0       # title strip bottom edge
TITLE_TOP = 8.75      # title strip top edge  (= frame top)
FRAME_BOT = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fancy_box(cx, cy, w, h, color, alpha=1.0, zorder=3):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=ARND,
        facecolor=color, edgecolor=WHITE,
        linewidth=2, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(p)


def box_text(cx, cy, line1, line2="", size1=11, size2=9.5, color=WHITE, zorder=5):
    if line2:
        ax.text(cx, cy + 0.12, line1, ha="center", va="center",
                fontsize=size1, color=color, fontweight="bold", zorder=zorder)
        ax.text(cx, cy - 0.2, line2, ha="center", va="center",
                fontsize=size2, color=color, alpha=0.92, zorder=zorder)
    else:
        ax.text(cx, cy, line1, ha="center", va="center",
                fontsize=size1, color=color, fontweight="bold", zorder=zorder)


def arrow_down(x, y_top, y_bot, color=NAVY, lw=2.2):
    ax.annotate(
        "", xy=(x, y_bot), xytext=(x, y_top),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=18),
        zorder=4,
    )


def annot(x, y, txt, color=RED, size=9, ha="left"):
    ax.text(x, y, txt, ha=ha, va="center",
            fontsize=size, color=color, fontstyle="italic",
            fontweight="bold", zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Outer frame
# ─────────────────────────────────────────────────────────────────────────────
frame = Rectangle((0.15, FRAME_BOT), FW - 0.3, TITLE_TOP - FRAME_BOT,
                  linewidth=2.5, edgecolor=GRAY, facecolor="none", zorder=1)
ax.add_patch(frame)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Title strip
# ─────────────────────────────────────────────────────────────────────────────
title_h = TITLE_TOP - TITLE_BOT
title_strip = FancyBboxPatch(
    (0.15, TITLE_BOT), FW - 0.3, title_h,
    boxstyle="square,pad=0",
    facecolor=NAVY, edgecolor="none", zorder=2,
)
ax.add_patch(title_strip)
ax.text(FW / 2, (TITLE_BOT + TITLE_TOP) / 2,
        "NeCTv2: Optimizing Implicit Neural Representations for Discrete and Continuous 4D-CT",
        ha="center", va="center", fontsize=11.5, color=WHITE,
        fontweight="bold", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Root node
# ─────────────────────────────────────────────────────────────────────────────
fancy_box(FW / 2, ROOT_CY, 4.0, 0.72, NAVY, zorder=4)
ax.text(FW / 2, ROOT_CY, "NeCT  QuadCubes",
        ha="center", va="center", fontsize=13, color=WHITE,
        fontweight="bold", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Branch connectors  (root → horizontal split → branch headers)
# ─────────────────────────────────────────────────────────────────────────────
root_bottom = ROOT_CY - 0.36        # bottom of root box
kw = dict(color=NAVY, lw=2, zorder=3, solid_capstyle="round")

# vertical from root down to horizontal
ax.plot([FW / 2, FW / 2], [root_bottom, JUNCTION_Y], **kw)

# horizontal split
ax.plot([LEFT_X, RIGHT_X], [JUNCTION_Y, JUNCTION_Y], **kw)

# left arm: horizontal end → arrow pointing to branch header
ax.plot([LEFT_X, LEFT_X], [JUNCTION_Y, HEADER_Y + 0.35], **kw)
arrow_down(LEFT_X, HEADER_Y + 0.35, HEADER_Y + 0.1, color=NAVY)

# right arm
ax.plot([RIGHT_X, RIGHT_X], [JUNCTION_Y, HEADER_Y + 0.35], **kw)
arrow_down(RIGHT_X, HEADER_Y + 0.35, HEADER_Y + 0.1, color=NAVY)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Branch headers
# ─────────────────────────────────────────────────────────────────────────────
ax.text(LEFT_X, HEADER_Y, "Part I — Encoder Optimisation",
        ha="center", va="center", fontsize=12, color=BLUE,
        fontweight="bold", zorder=5)

ax.text(RIGHT_X, HEADER_Y, "Part II — Continuous Scanning",
        ha="center", va="center", fontsize=12, color=ORANGE,
        fontweight="bold", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Left branch boxes
# ─────────────────────────────────────────────────────────────────────────────

# Box 1 — QuadCubes baseline
fancy_box(LEFT_X, L_BOX1_CY, LW, LH, BLUE)
box_text(LEFT_X, L_BOX1_CY,
         "QuadCubes baseline",
         "44.8 GB  ·  35.89 dB")

# Arrow 1  (box1 bottom → box2 top)
b1_bot = L_BOX1_CY - LH / 2        # 4.775
b2_top = L_BOX2_CY + LH / 2        # 4.175
arrow_down(LEFT_X, b1_bot, b2_top, color=BLUE)
annot(LEFT_X + 0.25, (b1_bot + b2_top) / 2, "−60% VRAM", color=RED)

# Box 2 — CombinedCubes
fancy_box(LEFT_X, L_BOX2_CY, LW, LH, BLUE_LT)
box_text(LEFT_X, L_BOX2_CY,
         "CombinedCubes",
         "18.3 GB  ·  37.28 dB")

# Arrow 2  (box2 bottom → box3 top)
b2_bot = L_BOX2_CY - LH / 2        # 3.325
b3_top = L_BOX3_CY + LH / 2        # 2.725
arrow_down(LEFT_X, b2_bot, b3_top, color=BLUE_LT)
annot(LEFT_X + 0.25, (b2_bot + b3_top) / 2, "+1.84 dB vs baseline", color=RED)

# Box 3 — MixedCubes (highlight)
fancy_box(LEFT_X, L_BOX3_CY, LW, LH, BLUE_HL)
box_text(LEFT_X, L_BOX3_CY,
         "MixedCubes  ★",
         "22.1 GB  ·  37.73 dB")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Right branch boxes  (with scan-mode icons)
# ─────────────────────────────────────────────────────────────────────────────

ICON_X = RIGHT_X - RW / 2 + 0.55   # icon centre X (near left edge of box)
TEXT_X = RIGHT_X + 0.35             # text centre X (shifted right)

# ── Box 1: Step-and-shoot ────────────────────────────────────────────────────
fancy_box(RIGHT_X, R_BOX1_CY, RW, RH, ORANGE)

# Step-and-shoot icon: 5 discrete radial lines
r_ico = 0.30
for angle in [25, 65, 105, 145, 185]:
    dx = r_ico * np.cos(np.radians(angle))
    dy = r_ico * np.sin(np.radians(angle))
    ax.plot([ICON_X, ICON_X + dx], [R_BOX1_CY, R_BOX1_CY + dy],
            "-", color=WHITE, lw=2.0, zorder=6, solid_capstyle="round")
    ax.plot(ICON_X + dx, R_BOX1_CY + dy,
            "o", color=WHITE, ms=3.5, zorder=7)
ax.plot(ICON_X, R_BOX1_CY, "o", color=WHITE, ms=5, zorder=7)

ax.text(TEXT_X, R_BOX1_CY + 0.13, "Step-and-shoot",
        ha="center", va="center", fontsize=11, color=WHITE,
        fontweight="bold", zorder=5)
ax.text(TEXT_X, R_BOX1_CY - 0.22, "discrete angle acquisition",
        ha="center", va="center", fontsize=9, color=WHITE,
        alpha=0.88, zorder=5)

# Arrow (box1 bottom → box2 top)
rb1_bot = R_BOX1_CY - RH / 2       # 4.425
rb2_top = R_BOX2_CY + RH / 2       # 3.575
arrow_down(RIGHT_X, rb1_bot, rb2_top, color=ORANGE)
annot(RIGHT_X + 0.25, (rb1_bot + rb2_top) / 2,
      "corrects angular\nintegration blur", color=RED, size=8.5)

# ── Box 2: Fly-scan + K-step ─────────────────────────────────────────────────
fancy_box(RIGHT_X, R_BOX2_CY, RW, RH, ORANGE2)

# Fly-scan icon: continuous arc with arrowhead
r_ico = 0.30
theta_arc = np.linspace(30, 340, 120)
ax.plot(ICON_X + r_ico * np.cos(np.radians(theta_arc)),
        R_BOX2_CY + r_ico * np.sin(np.radians(theta_arc)),
        "-", color=WHITE, lw=2.0, zorder=6)
# Arrowhead at end of arc
t_end, t_pre = 340, 325
ax.annotate(
    "",
    xy=(ICON_X + r_ico * np.cos(np.radians(t_end)),
        R_BOX2_CY + r_ico * np.sin(np.radians(t_end))),
    xytext=(ICON_X + r_ico * np.cos(np.radians(t_pre)),
            R_BOX2_CY + r_ico * np.sin(np.radians(t_pre))),
    arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=1.5, mutation_scale=12),
    zorder=7,
)

ax.text(TEXT_X, R_BOX2_CY + 0.13, "Fly-scan + K-step correction",
        ha="center", va="center", fontsize=11, color=WHITE,
        fontweight="bold", zorder=5)
ax.text(TEXT_X, R_BOX2_CY - 0.22, "continuous-angle acquisition",
        ha="center", va="center", fontsize=9, color=WHITE,
        alpha=0.88, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Footer strip
# ─────────────────────────────────────────────────────────────────────────────
footer_h = FOOTER_TOP - FOOTER_BOT
footer = FancyBboxPatch(
    (0.15, FOOTER_BOT), FW - 0.3, footer_h,
    boxstyle="square,pad=0",
    facecolor=GREEN, edgecolor="none", zorder=2,
)
ax.add_patch(footer)
ax.text(FW / 2, (FOOTER_BOT + FOOTER_TOP) / 2,
        "Faster reconstructions  ·  Lower hardware requirements  ·  New acquisition mode for 4D-CT",
        ha="center", va="center", fontsize=11, color=WHITE,
        fontweight="bold", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Save
# ─────────────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = OUT_DIR / f"thesis_overview.{ext}"
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"Saved → {path}")

plt.close(fig)
