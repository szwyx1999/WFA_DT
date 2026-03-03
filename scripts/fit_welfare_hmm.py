import argparse
from pathlib import Path
import pandas as pd

from src.models.welfare_hmm.config import WelfareHMMConfig
from src.models.welfare_hmm.pipeline import fit_welfare_hmm, save_outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_features", type=str, default="data/processed/features_5min.csv")
    ap.add_argument("--in_anomaly", type=str, default="data/processed/anomaly_scores.csv")
    ap.add_argument("--out_csv", type=str, default="data/processed/welfare_states.csv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n_iter", type=int, default=10)
    ap.add_argument("--anomaly_weight", type=float, default=0.0)
    args = ap.parse_args()

    feat = pd.read_csv(args.in_features)
    feat["timestamp"] = pd.to_datetime(feat["timestamp"])

    if Path(args.in_anomaly).exists():
        anom = pd.read_csv(args.in_anomaly)
        anom["timestamp"] = pd.to_datetime(anom["timestamp"])
        feat = feat.merge(anom, on=["cow_id", "timestamp"], how="left")
        # if missing, assume not anomaly
        if "is_anomaly" in feat.columns:
            feat["is_anomaly"] = feat["is_anomaly"].fillna(0).astype(int)
    else:
        feat["is_anomaly"] = 0

    cfg = WelfareHMMConfig(k_states=args.k, n_iter=args.n_iter, anomaly_weight=args.anomaly_weight)

    out_df, meta = fit_welfare_hmm(feat, cfg)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.parent / "welfare_hmm_meta.json"
    save_outputs(out_df, meta, str(out_path), str(meta_path))

    print(f"[OK] wrote: {out_path.resolve()}")
    print(f"[OK] wrote: {meta_path.resolve()}")
    print(f"[INFO] final total loglik: {meta['loglik_history'][-1]:.3f}")


if __name__ == "__main__":
    main()