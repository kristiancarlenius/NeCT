import re
import matplotlib.pyplot as plt

# === Set your log file paths here ===
log_files = [
    "/home/user/Documents/img_comp/pr100_ac2/epoch_losses_norm36.txt",
    "/home/user/Documents/img_comp/pr360_ac6/epoch_losses.txt",
    "/home/user/Documents/img_comp/pr100_ac6/epoch_losses_norm36.txt",
    "/home/user/Documents/img_comp/pr360_ac4/epoch_losses.txt",
    "/home/user/Documents/img_comp/pr100_ac4/epoch_losses_norm36.txt",
    "/home/user/Documents/img_comp/pr360_ac3/epoch_losses.txt",
    "/home/user/Documents/img_comp/pr100_ac3/epoch_losses_norm36.txt"
]

# === Helper function to extract first loss per epoch ===
def parse_log(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    # Extract (epoch, loss)
    matches = re.findall(r"epoch=(\d+),\s*avg_loss=([\d.]+)", content)
    seen_epochs = set()
    x_vals, y_vals = [], []
    for epoch, loss in matches:
        epoch = float(int(epoch)*0.0095)
        if epoch not in seen_epochs:
            seen_epochs.add(epoch)
            x_vals.append(epoch)
            y_vals.append(float(loss))
    return x_vals, y_vals

# === Parse and plot all runs ===
plt.figure(figsize=(9, 6))
for i, path in enumerate(log_files, start=1):
    x, y = parse_log(path)
    plt.plot(x, y, marker='o', linewidth=2, markersize=4, label=log_files[i-1][30:39])

plt.title("Comparison of Average Loss per Epoch by Hour", fontsize=14)
plt.xlabel("Hours", fontsize=12)
plt.ylabel("Average Loss", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
