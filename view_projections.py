import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

proj = np.load("/home/user/Documents/NeCT/output/proj_4fps_2750.npy")
print("shape:", proj.shape, "| dtype:", proj.dtype, "| min:", proj.min(), "| max:", proj.max())

n = len(proj)
vmin, vmax = proj.min(), proj.max()

fig, ax = plt.subplots(figsize=(8, 7))
plt.subplots_adjust(bottom=0.15)

im = ax.imshow(proj[0], cmap="gray", vmin=vmin, vmax=vmax)
title = ax.set_title(f"Projection 0 / {n - 1}")
ax.axis("off")
plt.colorbar(im, ax=ax)

ax_slider = plt.axes([0.15, 0.05, 0.7, 0.04])
slider = Slider(ax_slider, "Index", 0, n - 1, valinit=0, valstep=1)

def update(val):
    i = int(slider.val)
    im.set_data(proj[i])
    title.set_text(f"Projection {i} / {n - 1}")
    fig.canvas.draw_idle()

slider.on_changed(update)

def on_key(event):
    i = int(slider.val)
    if event.key == "right":
        slider.set_val(min(i + 1, n - 1))
    elif event.key == "left":
        slider.set_val(max(i - 1, 0))
    elif event.key == "shift+right":
        slider.set_val(min(i + 10, n - 1))
    elif event.key == "shift+left":
        slider.set_val(max(i - 10, 0))

fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()
