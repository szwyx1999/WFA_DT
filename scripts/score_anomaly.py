import argparse
from pathlib import Path
import json
import pandas as pd

from src.models.anomaly.config import AnomalyConfig
from src.models.anomaly.detector import fit_and_score, default_feature_columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, default="data/processed/features_5min.csv")
    ap.add_argument("--out_csv", type=str, default="data/processed/anomaly_scores.csv")
    ap.add_argument("--contamination", type=float, default=0.03)
    ap.add_argument("--train_on_normal", action="store_true")  # only applicable for demo synthetic data
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["cow_id", "timestamp"]).reset_index(drop=True)

    cfg = AnomalyConfig(contamination=args.contamination)
    feat_cols = default_feature_columns(df)

    train_df = None
    if args.train_on_normal and "state_true" in df.columns:
        train_df = df[df["state_true"] == 0].copy()

    scores_df, det = fit_and_score(df, cfg, feature_cols=feat_cols, train_df=train_df)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(out_path, index=False)

    with open(out_path.parent / "anomaly_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(det.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote: {out_path.resolve()}")
    print(f"[INFO] anomaly rate: {scores_df['is_anomaly'].mean():.4f}")


if __name__ == "__main__":
    main()