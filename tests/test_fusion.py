import numpy as np
import pandas as pd

from src.fusion.config import FusionConfig
from src.fusion.fusion import fuse_and_alert


def test_welfare_run_k_requires_consecutive_points():
    t = pd.date_range("2026-01-01", periods=10, freq="5min")

    welfare = pd.DataFrame({
        "cow_id": [0]*10,
        "timestamp": t,
        "welfare_risk": [0.2, 0.6, 0.6, 0.2, 0.6, 0.6, 0.6, 0.2, 0.6, 0.6],
    })
    anomaly = pd.DataFrame({
        "cow_id": [0]*10,
        "timestamp": t,
        "anomaly_score": np.zeros(10),
        "is_anomaly": np.zeros(10, dtype=int),
    })

    cfg = FusionConfig(welfare_risk_threshold=0.55, welfare_run_k=3, pause_welfare_when_anomaly=True)
    fused, meta = fuse_and_alert(welfare, anomaly, cfg)

    # only positions 6 should have runlen>=3 (indices 4,5,6 are three consecutive > th)
    assert fused["welfare_alarm"].sum() >= 1
    assert fused.loc[fused["welfare_alarm"] == 1, "timestamp"].min() == t[6]


def test_pause_welfare_when_anomaly():
    t = pd.date_range("2026-01-01", periods=6, freq="5min")
    welfare = pd.DataFrame({"cow_id":[0]*6, "timestamp":t, "welfare_risk":[0.7]*6})
    anomaly = pd.DataFrame({"cow_id":[0]*6, "timestamp":t, "is_anomaly":[0,1,1,0,0,0], "anomaly_score":[0]*6})

    cfg = FusionConfig(welfare_risk_threshold=0.55, welfare_run_k=3, pause_welfare_when_anomaly=True)
    fused, _ = fuse_and_alert(welfare, anomaly, cfg)

    # since we pause during anomaly at t1,t2, the run gets broken
    # alarm should happen later, not at t2
    alarm_times = fused.loc[fused["welfare_alarm"] == 1, "timestamp"]
    assert (alarm_times.min() >= t[4])  # earliest possible is around t4 here