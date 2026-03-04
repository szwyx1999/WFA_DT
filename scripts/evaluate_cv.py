import argparse
from pathlib import Path
import json
import pandas as pd

from src.evaluation.config import EvalConfig
from src.evaluation.evaluator import evaluate_groupkfold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="data/processed/features_5min.csv")
    ap.add_argument("--measurements", type=str, default="data/synthetic/measurements_5min.csv")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--hmm_iter", type=int, default=6)
    args = ap.parse_args()

    feat = pd.read_csv(args.features)
    meas = pd.read_csv(args.measurements)

    cfg = EvalConfig(n_splits=args.splits, hmm_n_iter=args.hmm_iter)

    report, folds_df = evaluate_groupkfold(feat, meas, cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds_path = out_dir / "eval_folds.csv"
    report_path = out_dir / "eval_report.json"

    folds_df.to_csv(folds_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote: {folds_path.resolve()}")
    print(f"[OK] wrote: {report_path.resolve()}")
    print("[SUMMARY]", report["summary"])


if __name__ == "__main__":
    main()