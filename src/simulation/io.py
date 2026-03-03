from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from .config import SimConfig
from .generator import generate_synthetic_5min, generate_accel_1hz


def write_dataset(out_dir: str | Path, cfg: SimConfig) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    measurements, events, meta = generate_synthetic_5min(cfg)
    measurements.to_csv(out_dir / "measurements_5min.csv", index=False)
    events.to_csv(out_dir / "events.csv", index=False)

    # Optional 1Hz accel
    accel = generate_accel_1hz(cfg)
    if len(accel) > 0:
        # keep dependency minimal: gzip CSV
        accel.to_csv(out_dir / "accel_1hz.csv.gz", index=False, compression="gzip")

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote dataset to: {out_dir.resolve()}")