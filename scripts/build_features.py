import argparse
from pathlib import Path

from src.features.config import FeatureConfig
from src.features.featurizer import load_measurements, build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, default="data/synthetic/measurements_5min.csv")
    ap.add_argument("--out_csv", type=str, default="data/processed/features_5min.csv")
    ap.add_argument("--baseline_hours", type=int, default=24)
    args = ap.parse_args()

    cfg = FeatureConfig(baseline_hours=args.baseline_hours)

    df = load_measurements(args.in_csv, cfg)
    feat_df, feat_cols = build_features(df, cfg)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(out_path, index=False)

    print(f"[OK] wrote: {out_path.resolve()}")
    print(f"[INFO] engineered feature columns: {len(feat_cols)}")


if __name__ == "__main__":
    main()