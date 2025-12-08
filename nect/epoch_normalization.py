import re

# ==== SETTINGS ====
input_path  = "/home/user/Documents/img_comp/pr360_ac12/epoch_losses.txt"         # original log file
output_path = "/home/user/Documents/img_comp/pr360_ac12/epoch_losses_norm140.txt"  # where to save result
factor = 3.89                          # how many losses to average into one

# ==== STEP 1: parse log and keep first loss per epoch ====
def parse_log_first_epoch(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    matches = re.findall(r"epoch=(\d+),\s*avg_loss=([\d.]+)", content)

    seen_epochs = set()
    epochs, losses = [], []

    for epoch, loss in matches:
        epoch = int(epoch)
        if epoch not in seen_epochs:
            seen_epochs.add(epoch)
            epochs.append(epoch)
            losses.append(float(loss))

    return epochs, losses


# ==== STEP 2: downsample by averaging ~factor points ====
def downsample_by_factor(epochs, losses, factor):
    n = len(losses)
    num_groups = int(n / factor)  # drop remainder, OK per your requirement

    new_epochs = []
    new_losses = []

    for g in range(num_groups):
        start = int(round(g * factor))
        end   = int(round((g + 1) * factor))
        if end > n:
            break

        group_epochs = epochs[start:end]
        group_losses = losses[start:end]

        if not group_losses:
            continue

        # use last epoch in each group (simple & consistent)
        new_epochs.append(group_epochs[-1])
        new_losses.append(sum(group_losses) / len(group_losses))

    return new_epochs, new_losses


# ==== RUN ====
epochs, losses = parse_log_first_epoch(input_path)
new_epochs, new_losses = downsample_by_factor(epochs, losses, factor)

# ==== STEP 3: save in the SAME FORMAT ====
with open(output_path, "w") as f:
    for e, l in zip(new_epochs, new_losses):
        f.write(f"epoch={e}, avg_loss={l:.6f}\n")

print(f"Saved compressed log to: {output_path}")
print(f"Original epochs: {len(epochs)}")
print(f"Compressed epochs: {len(new_epochs)}")
