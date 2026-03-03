import numpy as np
import pandas as pd

from src.models.anomaly.config import AnomalyConfig
from src.models.anomaly.detector import fit_and_score


def test_missingness_increases_score():
    # Construct two datasets: 
    #   - A: Complete with no missing values; 
    #   - B: Missing values masked with all 1s
    t = pd.date_range("2026-01-01", periods=20, freq="5min")
    base = pd.DataFrame({
        "cow_id": [0]*20,
        "timestamp": t,
        "x_z": np.linspace(0, 1, 20),
        "y_z": np.linspace(1, 0, 20),
        "rumination_min_5min_is_missing": [0]*20,
        "activity_mean_5min_is_missing": [0]*20,
    })

    df = base.copy()
    df2 = base.copy()
    df2["cow_id"] = 1
    df2["rumination_min_5min_is_missing"] = 1
    df2["activity_mean_5min_is_missing"] = 1
    df = pd.concat([df, df2], ignore_index=True)

    cfg = AnomalyConfig(contamination=0.1, beta_missing=5.0, random_state=0)
    scored, det = fit_and_score(df, cfg)

    s0 = scored[scored["cow_id"] == 0]["anomaly_score"].mean()
    s1 = scored[scored["cow_id"] == 1]["anomaly_score"].mean()
    assert s1 > s0  # more missingness => higher anomaly score


def test_predict_outputs_binary():
    t = pd.date_range("2026-01-01", periods=30, freq="5min")
    df = pd.DataFrame({
        "cow_id": [0]*30,
        "timestamp": t,
        "a_z": np.random.randn(30),
        "b_z": np.random.randn(30),
        "a_is_missing": [0]*30,
    })

    cfg = AnomalyConfig(contamination=0.2, random_state=0)
    scored, _ = fit_and_score(df, cfg)
    assert set(scored["is_anomaly"].unique()).issubset({0, 1})