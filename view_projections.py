import numpy as np
import matplotlib.pyplot as plt

proj = np.load("/home/user/Documents/NeCT/output/proj_4fps_2750.npy")
print("shape:", proj.shape, "| dtype:", proj.dtype, "| min:", proj.min(), "| max:", proj.max())

# single projection at angle 0
plt.figure()
plt.imshow(proj[0], cmap="gray")
plt.title("Projection angle 0")
plt.colorbar()
plt.show()

# grid of 8 evenly spaced angles
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
step = len(proj) // 8
for i, ax in enumerate(axes.flat):
    ax.imshow(proj[i * step], cmap="gray")
    ax.set_title(f"angle {i * step}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# sinogram: all angles through the middle detector row
mid = proj.shape[1] // 2
plt.figure()
plt.imshow(proj[:, mid, :], cmap="gray", aspect="auto")
plt.title(f"Sinogram (row {mid})")
plt.xlabel("detector column")
plt.ylabel("angle index")
plt.colorbar()
plt.show()
