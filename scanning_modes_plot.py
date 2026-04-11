#!/usr/bin/env python3
"""
Two-panel diagram illustrating stop-and-shoot vs continuous CT scanning.

Stop-and-shoot: three acquisition points marked with X on a circular orbit.
Continuous:     three coloured arcs between the same points on the orbit.

Output: results/scanning_modes.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

COLORS = ["#E84545", "#2E86AB", "#57A773"]   # red, blue, green
LABELS = ["Position 1", "Position 2", "Position 3"]

# Angles (degrees) for the three points — evenly spaced around the circle
POINT_ANGLES_DEG = [90, 210, 330]

RADIUS = 1.0
N_ARC_POINTS = 300   # resolution of each arc segment

# ── Helpers ───────────────────────────────────────────────────────────────────

def point_on_circle(angle_deg: float, r: float = RADIUS):
    a = np.radians(angle_deg)
    return r * np.cos(a), r * np.sin(a)


def arc_xy(start_deg: float, end_deg: float, r: float = RADIUS):
    """Return (x, y) arrays for an arc from start_deg to end_deg (CCW)."""
    # Always go counter-clockwise
    if end_deg <= start_deg:
        end_deg += 360
    angles = np.linspace(np.radians(start_deg), np.radians(end_deg), N_ARC_POINTS)
    return r * np.cos(angles), r * np.sin(angles)


OBJECT_RADIUS = 0.55   # solid black object
GAP           = 0.10   # white space between object and scan orbit


def draw_circle_outline(ax, r: float = RADIUS):
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(r * np.cos(theta), r * np.sin(theta),
            color="lightgrey", linewidth=1.5, zorder=1)
    # White gap ring (drawn over any background, under the object)
    gap_circle = plt.Circle((0, 0), OBJECT_RADIUS + GAP,
                             color="white", zorder=2)
    ax.add_patch(gap_circle)
    # Solid black object circle
    obj_circle = plt.Circle((0, 0), OBJECT_RADIUS,
                             color="black", zorder=3)
    ax.add_patch(obj_circle)


def style_ax(ax, title: str):
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)


# ── Plot ──────────────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    angles = POINT_ANGLES_DEG
    points = [point_on_circle(a) for a in angles]

    # ── Left: Stop-and-shoot ──────────────────────────────────────────────────
    ax = axes[0]
    draw_circle_outline(ax)

    for (x, y), color, label in zip(points, COLORS, LABELS):
        ax.scatter(x, y, marker="x", s=200, color=color,
                   linewidths=2.5, zorder=5, label=label)

    ax.legend(handles=[
        mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)
    ], loc="lower center", bbox_to_anchor=(0.5, -0.06),
       ncol=3, fontsize=9, frameon=False)

    style_ax(ax, "Stop-and-shoot")

    # ── Right: Continuous ─────────────────────────────────────────────────────
    ax = axes[1]
    draw_circle_outline(ax)

    n = len(angles)
    for i in range(n):
        start = angles[i]
        end   = angles[(i + 1) % n]
        x_arc, y_arc = arc_xy(start, end)
        ax.plot(x_arc, y_arc, color=COLORS[i], linewidth=3.5, zorder=4)

    # Draw the same X markers but lighter so the arc colours are the focus
    for (x, y), color in zip(points, COLORS):
        ax.scatter(x, y, marker="x", s=120, color=color,
                   linewidths=2, zorder=5, alpha=0.5)

    ax.legend(handles=[
        mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)
    ], loc="lower center", bbox_to_anchor=(0.5, -0.06),
       ncol=3, fontsize=9, frameon=False)

    style_ax(ax, "Continuous")

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "scanning_modes.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
