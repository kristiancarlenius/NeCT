"""Split a large dynamic .npy projection dataset into N equal temporal chunks.

Usage:
    python scripts/split_dataset.py <projections.npy> <geometry.yaml> [--parts 4] [--out-dir <dir>]

Each output part is written to <out-dir>/part_1/, part_2/, … containing
projections.npy and geometry.yaml with the matching angle/timestep slice.
The input file is memory-mapped so only one chunk is held in RAM at a time.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def split(proj_path: Path, geo_path: Path, n_parts: int, out_dir: Path) -> None:
    proj = np.load(proj_path, mmap_mode="r")
    n = proj.shape[0]
    print(f"Input shape: {proj.shape}  dtype: {proj.dtype}")

    with open(geo_path) as f:
        geo = yaml.safe_load(f)

    angles = geo.get("angles")
    timesteps = geo.get("timesteps")

    if angles is not None and len(angles) != n:
        sys.exit(f"angles length {len(angles)} does not match projection count {n}")
    if timesteps is not None and len(timesteps) != n:
        sys.exit(f"timesteps length {len(timesteps)} does not match projection count {n}")

    base = n // n_parts
    remainder = n % n_parts
    splits = []
    start = 0
    for i in range(n_parts):
        end = start + base + (1 if i < remainder else 0)
        splits.append((start, end))
        start = end

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (s, e) in enumerate(splits, 1):
        part_dir = out_dir / f"part_{i}"
        part_dir.mkdir(exist_ok=True)

        print(f"Part {i}: projections [{s}:{e}] ({e - s} frames) → {part_dir}")
        np.save(part_dir / "projections.npy", np.array(proj[s:e]))

        geo_part = dict(geo)
        if angles is not None:
            geo_part["angles"] = angles[s:e]
        if timesteps is not None:
            geo_part["timesteps"] = timesteps[s:e]

        with open(part_dir / "geometry.yaml", "w") as f:
            yaml.dump(geo_part, f, default_flow_style=False)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projections", type=Path, help="Path to the large .npy file")
    parser.add_argument("geometry", type=Path, help="Path to the geometry.yaml file")
    parser.add_argument("--parts", type=int, default=4, help="Number of parts (default: 4)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as projections file)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.projections.parent
    split(args.projections, args.geometry, args.parts, out_dir)


if __name__ == "__main__":
    main()
