import argparse
from pathlib import Path

from src.simulation.config import SimConfig
from src.simulation.io import write_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/synthetic")
    ap.add_argument("--n_cows", type=int, default=100)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with_accel_1hz", action="store_true")
    args = ap.parse_args()

    cfg = SimConfig(
        n_cows=args.n_cows,
        days=args.days,
        seed=args.seed,
        generate_accel_1hz=args.with_accel_1hz,
    )
    write_dataset(Path(args.out_dir), cfg)


if __name__ == "__main__":
    main()