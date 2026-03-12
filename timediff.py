import os
from datetime import datetime, timedelta

# ─── One entry per run: path to 0000_0000.png ────────────────────────────────
start_files = [
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_17_16_2_4_128_L1/2026-03-10T15-19-20/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_18_16_2_4_128_L1/2026-03-10T16-13-27/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_19_16_2_4_128_L1/2026-03-10T18-12-22/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_20_16_2_4_128_L1/2026-03-10T18-27-39/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_21_16_2_4_128_L1/2026-03-10T18-45-35/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_17_4_22_16_2_4_128_L1/2026-03-10T19-11-03/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_17_16_2_4_128_L1/2026-03-06T01-32-58/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_18_16_2_4_128_L1/2026-02-12T16-24-33/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_19_16_2_4_128_L1/2026-03-06T05-02-36/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_20_16_2_4_128_L1/2026-03-06T06-00-17/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_21_16_2_4_128_L1/2026-03-06T06-35-10/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_18_4_22_16_2_4_128_L1/2026-03-11T19-56-36/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_17_16_2_4_128_L1/2026-03-06T01-32-58/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_18_16_2_4_128_L1/2026-01-30T15-46-42/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_19_16_2_4_128_L1/2026-02-12T16-24-33/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_20_16_2_4_128_L1/2026-02-08T15-53-51/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_21_16_2_4_128_L1/2026-03-05T21-00-01/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_20_4_22_16_2_4_128_L1/2026-03-05T21-22-07/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_17_16_2_4_128_L1/2026-03-05T20-48-55/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_18_16_2_4_128_L1/2026-01-28T22-34-56/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_19_16_2_4_128_L1/2026-01-28T22-34-27/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_20_16_2_4_128_L1/2026-02-08T15-53-51/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_21_16_2_4_128_L1/2026-02-08T15-53-51/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_21_4_22_16_2_4_128_L1/2026-03-03T16-23-40/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_17_16_2_4_128_L1/2026-03-03T16-19-57/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_18_16_2_4_128_L1/2026-03-03T16-21-04/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_19_16_2_4_128_L1/2026-03-03T16-21-04/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_20_16_2_4_128_L1/2026-03-03T16-21-04/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_21_16_2_4_128_L1/2026-03-03T16-19-57/images/0000_0000.png",
    "/cluster/home/kristiac/NeCT/outputs/sizediff/quadcubes_22_4_22_16_2_4_128_L1/2026-02-13T15-57-13/images/0000_0000.png",
]
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SECONDS = 13 * 3600 + 47 * 60 + 10   # 13:47:10 → 49630 s
EPOCH_STEP     = 25
EPOCH_SUFFIX   = "_1400.png"


def get_mtime(path: str) -> datetime:
    stat = os.stat(path)
    ts = getattr(stat, "st_birthtime", None) or stat.st_mtime
    return datetime.fromtimestamp(ts)


def format_td(seconds: float) -> str:
    total = int(abs(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_best_epoch(images_dir: str, t0: datetime):
    """
    Iterate through epoch checkpoints in steps of EPOCH_STEP.
    Stop when we overshoot TARGET_SECONDS (or run out of files).
    Return (best_epoch, elapsed_seconds) for whichever side is closer.
    """
    prev_epoch, prev_elapsed = None, None

    epoch = EPOCH_STEP
    while True:
        fname = f"{epoch:04d}{EPOCH_SUFFIX}"
        fpath = os.path.join(images_dir, fname)

        if not os.path.exists(fpath):
            # No more checkpoints — best is whatever we had last
            if prev_epoch is None:
                return None
            return prev_epoch, prev_elapsed

        elapsed = (get_mtime(fpath) - t0).total_seconds()

        if elapsed >= TARGET_SECONDS:
            # Overshot — compare this epoch vs previous
            if prev_epoch is None:
                return epoch, elapsed   # first file already over target
            if abs(elapsed - TARGET_SECONDS) <= abs(prev_elapsed - TARGET_SECONDS):
                return epoch, elapsed
            else:
                return prev_epoch, prev_elapsed

        prev_epoch, prev_elapsed = epoch, elapsed
        epoch += EPOCH_STEP


print(f"\n{'#':<5} {'Model':<30} {'Best epoch':>10}  {'Elapsed at best':>15}  {'Projected time':>14}")
print("─" * 85)

for i, start_path in enumerate(start_files, start=1):
    try:
        images_dir = os.path.dirname(start_path)
        t0 = get_mtime(start_path)
        model = start_path.split("/")[-4]

        result = find_best_epoch(images_dir, t0)
        if result is None:
            print(f"{i:<5} {model:<30}  no checkpoints found")
            continue

        best_ep, elapsed_s = result
        print(
            f"{i:<5} {model:<30} {best_ep:>10}  "
            f"{format_td(elapsed_s):>15}  {format_td(TARGET_SECONDS):>14}"
        )
    except FileNotFoundError as e:
        print(f"{i:<5} ERROR - {e}")

print()
