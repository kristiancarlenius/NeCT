#!/usr/bin/env python3
"""
Interactive tool for defining crop regions on a composite epoch image.

The composite image contains 6 sub-images arranged in a grid, separated by
white borders.  This tool lets you draw a rectangle for each panel, then
saves the 6 crops to crops.json so that plot scripts can load them without
needing to re-specify coordinates.

Usage:
    python crop_tool.py <path-to-any-epoch-image.png>

Controls:
    - Click and drag to draw a rectangle around the current panel.
    - Press Enter or Space to confirm the current selection and move to the next.
    - Press Backspace to redo the current panel.
    - Press Escape to quit without saving.
    - After all 6 panels are confirmed, crops.json is written automatically.

The output crops.json contains a list of 6 dicts:
    [{"x0": int, "y0": int, "x1": int, "y1": int}, ...]
in the order you defined them.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import numpy as np
from PIL import Image

CROPS_FILE = Path(__file__).parent / "crop_img_comparison.json"
N_PANELS = 1
COLORS = ["#e41a1c"]


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


class CropTool:
    def __init__(self, img: np.ndarray, image_path: str):
        self.img = img
        self.image_path = image_path
        self.crops: list[dict] = []
        self.current: int = 0           # panel index being defined
        self._selection: tuple | None = None   # raw RectangleSelector coords

        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.imshow(img, cmap="gray", aspect="equal", interpolation="nearest")
        self.ax.set_title(self._title(), fontsize=12)

        # Overlay patches for already-confirmed panels
        self._patches: list[patches.Rectangle] = []

        # Status text in the lower left
        self._status = self.ax.text(
            0.01, 0.01, "", transform=self.ax.transAxes,
            fontsize=10, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
            verticalalignment="bottom",
        )
        self._update_status("Draw a box around panel 1, then press Enter.")

        self._selector = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=True,
            props=dict(edgecolor=COLORS[self.current], linewidth=2, fill=False),
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.tight_layout()
        plt.show()

    # ── callbacks ────────────────────────────────────────────────────────────

    def _on_select(self, eclick, erelease):
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        self._selection = (int(round(x0)), int(round(y0)),
                           int(round(x1)), int(round(y1)))
        self._update_status(
            f"Panel {self.current + 1}: ({self._selection[0]},{self._selection[1]}) → "
            f"({self._selection[2]},{self._selection[3]})  |  "
            "Enter=confirm  Backspace=redo  Esc=quit"
        )

    def _on_key(self, event):
        if event.key in ("enter", " "):
            self._confirm()
        elif event.key == "backspace":
            self._redo()
        elif event.key == "escape":
            print("Aborted — nothing saved.")
            plt.close(self.fig)

    # ── actions ──────────────────────────────────────────────────────────────

    def _confirm(self):
        if self._selection is None:
            self._update_status("Draw a box first, then press Enter.")
            return

        x0, y0, x1, y1 = self._selection
        self.crops.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})

        # Draw a permanent colored rectangle
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor=COLORS[self.current], facecolor="none",
        )
        self.ax.add_patch(rect)
        self.ax.text(
            x0 + 4, y0 + 20, str(self.current + 1),
            color=COLORS[self.current], fontsize=14, fontweight="bold",
        )
        self._patches.append(rect)
        self._selection = None
        self.current += 1

        if self.current == N_PANELS:
            self._finish()
        else:
            # Update selector color for next panel
            self._selector.set_props(edgecolor=COLORS[self.current])
            self.ax.set_title(self._title(), fontsize=12)
            self._update_status(
                f"Draw a box around panel {self.current + 1}, then press Enter."
            )
            self.fig.canvas.draw()

    def _redo(self):
        if self.current == 0:
            return
        self.current -= 1
        self.crops.pop()
        p = self._patches.pop()
        p.remove()
        # Remove the number label (last text added)
        if self.ax.texts:
            self.ax.texts[-1].remove()
        self._selector.set_props(edgecolor=COLORS[self.current])
        self.ax.set_title(self._title(), fontsize=12)
        self._update_status(
            f"Redo panel {self.current + 1}: draw a box, then press Enter."
        )
        self.fig.canvas.draw()

    def _finish(self):
        self._selector.set_active(False)
        self.ax.set_title("All 6 panels defined — saving crops.json", fontsize=12)
        self._update_status(f"Saved {N_PANELS} crops to {CROPS_FILE}")
        self.fig.canvas.draw()

        data = {"source_image": self.image_path, "crops": self.crops}
        with open(CROPS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved crops to {CROPS_FILE}")
        for i, c in enumerate(self.crops):
            print(f"  Panel {i+1}: x0={c['x0']} y0={c['y0']} x1={c['x1']} y1={c['y1']}")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _title(self) -> str:
        return (
            f"Panel {self.current + 1} / {N_PANELS}  —  "
            f"{Path(self.image_path).name}"
        )

    def _update_status(self, msg: str):
        self._status.set_text(msg)
        self.fig.canvas.draw_idle()


def main():
    if len(sys.argv) < 2:
        # If crops.json already exists, show a preview and exit
        if CROPS_FILE.exists():
            with open(CROPS_FILE) as f:
                data = json.load(f)
            print(f"crops.json already exists (source: {data.get('source_image', '?')})")
            for i, c in enumerate(data["crops"]):
                print(f"  Panel {i+1}: x0={c['x0']} y0={c['y0']} x1={c['x1']} y1={c['y1']}")
            print("\nTo redefine crops, run:  python crop_tool.py <image.png>")
        else:
            print("Usage: python crop_tool.py <path-to-epoch-image.png>")
        return

    image_path = sys.argv[1]
    img = load_image(image_path)
    print(f"Image size: {img.shape[1]} × {img.shape[0]} (w × h)")
    CropTool(img, image_path)


if __name__ == "__main__":
    main()
