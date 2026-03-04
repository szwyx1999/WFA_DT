import numpy as np
import pandas as pd

from src.evaluation.metrics import safe_auroc, safe_auprc, time_to_detect_minutes


def test_metrics_defined_on_two_class():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert safe_auroc(y, s) > 0.9
    assert safe_auprc(y, s) > 0.9


def test_time_to_detect():
    events = pd.DataFrame([{"cow_id": 0, "start": "2026-01-01 00:10:00", "end": "2026-01-01 00:40:00"}])
    alerts = pd.DataFrame({
        "cow_id": [0]*10,
        "timestamp": pd.date_range("2026-01-01 00:00:00", periods=10, freq="5min"),
        "welfare_alarm": [0,0,0,1,1,0,0,0,0,0],
    })
    ttd = time_to_detect_minutes(events, alerts, "welfare_alarm")
    # first alarm at 00:15, event start 00:10 => 5 minutes
    assert abs(ttd - 5.0) < 1e-6