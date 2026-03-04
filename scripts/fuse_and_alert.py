import argparse
from pathlib import Path
import pandas as pd

from src.fusion.config import FusionConfig
from src.fusion.fusion import fuse_and_alert, save_fusion_outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--welfare_csv", type=str, default="data/processed/welfare_states.csv")
    ap.add_argument("--anomaly_csv", type=str, default="data/processed/anomaly_scores.csv")
    ap.add_argument("--out_csv", type=str, default="data/processed/fused_alerts.csv")

    ap.add_argument("--risk_th", type=float, default=0.55)
    ap.add_argument("--run_k", type=int, default=6)
    ap.add_argument("--pause_on_anom", action="store_true")
    ap.add_argument("--no_pause_on_anom", action="store_true")

    args = ap.parse_args()

    pause = True
    if args.no_pause_on_anom:
        pause = False
    if args.pause_on_anom:
        pause = True

    cfg = FusionConfig(
        welfare_risk_threshold=args.risk_th,
        welfare_run_k=args.run_k,
        pause_welfare_when_anomaly=pause,
    )

    w = pd.read_csv(args.welfare_csv)
    a = pd.read_csv(args.anomaly_csv)

    fused, meta = fuse_and_alert(w, a, cfg)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.parent / "fusion_meta.json"
    save_fusion_outputs(fused, meta, str(out_path), str(meta_path))

    print(f"[OK] wrote: {out_path.resolve()}")
    print(f"[OK] wrote: {meta_path.resolve()}")
    print(f"[INFO] anomaly_alarm_rate={meta['summary']['anomaly_alarm_rate']:.4f}, welfare_alarm_rate={meta['summary']['welfare_alarm_rate']:.4f}")


if __name__ == "__main__":
    main()