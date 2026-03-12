import os
from datetime import datetime, timedelta

# ─── Add your file pairs here ───────────────────────────────────────────────
file_pairs = [
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_17_16_2_4_128_L1/2026-03-10T15-19-20/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_17_16_2_4_128_L1/2026-03-10T15-19-20/images/0525_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_18_16_2_4_128_L1/2026-03-10T16-13-27/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_18_16_2_4_128_L1/2026-03-10T16-13-27/images/0500_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_19_16_2_4_128_L1/2026-03-10T18-12-22/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_19_16_2_4_128_L1/2026-03-10T18-12-22/images/0500_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_20_16_2_4_128_L1/2026-03-10T18-27-39/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_20_16_2_4_128_L1/2026-03-10T18-27-39/images/0400_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_21_16_2_4_128_L1/2026-03-10T18-45-35/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_21_16_2_4_128_L1/2026-03-10T18-45-35/images/0375_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_22_16_2_4_128_L1/2026-03-10T19-11-03/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_22_16_2_4_128_L1/2026-03-10T19-11-03/images/0275_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_17_16_2_4_128_L1/2026-03-06T01-32-58/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_17_16_2_4_128_L1/2026-03-06T01-32-58/images/0475_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_18_16_2_4_128_L1/2026-02-12T16-24-33/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_18_16_2_4_128_L1/2026-02-12T16-24-33/images/0450_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_19_16_2_4_128_L1/2026-03-06T05-02-36/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_19_16_2_4_128_L1/2026-03-06T05-02-36/images/0400_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_20_16_2_4_128_L1/2026-03-06T06-00-17/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_20_16_2_4_128_L1/2026-03-06T06-00-17/images/0375_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_21_16_2_4_128_L1/2026-03-06T06-35-10/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_21_16_2_4_128_L1/2026-03-06T06-35-10/images/0325_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_22_16_2_4_128_L1/2026-03-11T19-56-36/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_22_16_2_4_128_L1/2026-03-11T19-56-36/images/0275_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_17_16_2_4_128_L1/2026-03-06T01-32-58/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_17_16_2_4_128_L1/2026-03-06T01-32-58/images/0400_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_18_16_2_4_128_L1/2026-01-30T15-46-42/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_18_16_2_4_128_L1/2026-01-30T15-46-42/images/0375_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_19_16_2_4_128_L1/2026-02-12T16-24-33/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_19_16_2_4_128_L1/2026-02-12T16-24-33/images/0350_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_20_16_2_4_128_L1/2026-02-08T15-53-51/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_20_16_2_4_128_L1/2026-02-08T15-53-51/images/0350_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_21_16_2_4_128_L1/2026-03-05T21-00-01/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_21_16_2_4_128_L1/2026-03-05T21-00-01/images/0300_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_22_16_2_4_128_L1/2026-03-05T21-22-07/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_22_16_2_4_128_L1/2026-03-05T21-22-07/images/0200_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_17_16_2_4_128_L1/2026-03-05T20-48-55/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_17_16_2_4_128_L1/2026-03-05T20-48-55/images/0375_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_18_16_2_4_128_L1/2026-01-28T22-34-56/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_18_16_2_4_128_L1/2026-01-28T22-34-56/images/0375_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_19_16_2_4_128_L1/2026-01-28T22-34-27/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_19_16_2_4_128_L1/2026-01-28T22-34-27/images/0325_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_20_16_2_4_128_L1/2026-02-08T15-53-51/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_20_16_2_4_128_L1/2026-02-08T15-53-51/images/0325_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_21_16_2_4_128_L1/2026-02-08T15-53-51/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_21_16_2_4_128_L1/2026-02-08T15-53-51/images/0275_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_22_16_2_4_128_L1/2026-03-03T16-23-40/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_22_16_2_4_128_L1/2026-03-03T16-23-40/images/0175_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_17_16_2_4_128_L1/2026-03-03T16-19-57/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_17_16_2_4_128_L1/2026-03-03T16-19-57/images/0375_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_18_16_2_4_128_L1/2026-03-03T16-21-04/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_18_16_2_4_128_L1/2026-03-03T16-21-04/images/0375_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_19_16_2_4_128_L1/2026-03-03T16-21-04/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_19_16_2_4_128_L1/2026-03-03T16-21-04/images/0350_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_20_16_2_4_128_L1/2026-03-03T16-21-04/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_20_16_2_4_128_L1/2026-03-03T16-21-04/images/0300_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_21_16_2_4_128_L1/2026-03-03T16-19-57/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_21_16_2_4_128_L1/2026-03-03T16-19-57/images/0275_1400.png"),
    ("/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_22_16_2_4_128_L1/2026-02-13T15-57-13/images/0000_0000.png", "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_22_16_2_4_128_L1/2026-02-13T15-57-13/images/0200_1400.png")
]
# ────────────────────────────────────────────────────────────────────────────


TARGET_SECONDS = 13 * 3600 + 47 * 60 + 10   # 13:47:10 → 49630 s
MAX_EPOCHS     = 525
EPOCH_STEP     = 25


def get_creation_time(path: str) -> datetime:
    stat = os.stat(path)
    ts = getattr(stat, "st_birthtime", None) or stat.st_mtime
    return datetime.fromtimestamp(ts)


def format_diff(delta: timedelta) -> str:
    total_seconds = int(abs(delta.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def epoch_from_filename(path: str) -> int:
    """Parse epoch number from filenames like 0525_1400.png."""
    name = os.path.basename(path)
    return int(name[:4])


def best_epoch(s_per_epoch: float, target_s: float, step: int, max_val: int) -> int:
    candidates = range(step, max_val + 1, step)
    return min(candidates, key=lambda e: abs(e * s_per_epoch - target_s))


print(f"\n{'#':<5} {'Model':<30} {'Elapsed':>10}  {'Epochs run':>10}  {'s/epoch':>8}  {'Target epochs':>13}  {'Projected time':>14}")
print("─" * 100)

for i, (f1, f2) in enumerate(file_pairs, start=1):
    try:
        t1 = get_creation_time(f1)
        t2 = get_creation_time(f2)
        elapsed_s = abs((t2 - t1).total_seconds())
        epochs_run = epoch_from_filename(f2)
        s_per_epoch = elapsed_s / epochs_run
        target_epochs = best_epoch(s_per_epoch, TARGET_SECONDS, EPOCH_STEP, MAX_EPOCHS)
        projected = format_diff(timedelta(seconds=target_epochs * s_per_epoch))

        model = f2.split("/")[-4]

        print(
            f"{i:<5} {model:<30} {format_diff(t2 - t1):>10}  "
            f"{epochs_run:>10}  {s_per_epoch:>8.1f}  {target_epochs:>13}  {projected:>14}"
        )
    except FileNotFoundError as e:
        print(f"{i:<5} ERROR - {e}")

print()