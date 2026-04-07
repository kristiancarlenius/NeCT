from pathlib import Path
from nect.scivis_data import SciVisDataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--engine", required=False)
    args = parser.parse_args()

    data = SciVisDataset(base_path=Path(args.path), dataset=args.dataset, scale=True)
    engine = "leap"
    if args.engine is not None:
        engine = args.engine
    projections = data.generate_projections(nangles=49, save=True, method=engine)