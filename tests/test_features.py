import numpy as np
import pandas as pd

from src.features.config import FeatureConfig
from src.features.featurizer import build_features


def test_output_shape_and_keys():
    cfg = FeatureConfig(windows=("15min",), baseline_hours=0)
    t = pd.date_range("2026-01-01 00:00:00", periods=6, freq="5min")
    df = pd.DataFrame({
        "cow_id": [0]*6,
        "timestamp": t,
        "rumination_min_5min": [0, 1, 2, 3, 4, 5],
        "activity_mean_5min": [0.1]*6,
        "activity_inactive_frac_5min": [0.9]*6,
        "thi": [70]*6,
        "methane_intensity_g_per_kg_milk": [100]*6,
        "milk_yield_kg_session": [np.nan]*6,
        "body_weight_kg": [np.nan]*6,
        "state_true": [0]*6,
    })

    out, feat_cols = build_features(df, cfg)
    assert len(out) == len(df)
    assert out[["cow_id", "timestamp"]].isna().sum().sum() == 0
    assert len(feat_cols) > 0


def test_rolling_mean_simple():
    # window=15min on 5min grid => last 3 points
    cfg = FeatureConfig(windows=("15min",), baseline_hours=0)
    t = pd.date_range("2026-01-01 00:00:00", periods=6, freq="5min")
    df = pd.DataFrame({
        "cow_id": [0]*6,
        "timestamp": t,
        "rumination_min_5min": [0, 1, 2, 3, 4, 5],
        "activity_mean_5min": [0.1]*6,
        "activity_inactive_frac_5min": [0.9]*6,
        "thi": [70]*6,
        "methane_intensity_g_per_kg_milk": [100]*6,
        "milk_yield_kg_session": [np.nan]*6,
        "body_weight_kg": [np.nan]*6,
        "state_true": [0]*6,
    })

    out, _ = build_features(df, cfg)
    col = "rumination_min_5min_mean_15min"
    # expected rolling means:
    # t0: mean([0])=0
    # t1: mean([0,1])=0.5
    # t2: mean([0,1,2])=1
    # t3: mean([1,2,3])=2
    # t4: mean([2,3,4])=3
    # t5: mean([3,4,5])=4
    exp = np.array([0, 0.5, 1, 2, 3, 4], dtype=float)
    assert np.allclose(out[col].to_numpy(), exp, atol=1e-8)


def test_milk_last_and_hours_since():
    cfg = FeatureConfig(windows=("15min",), baseline_hours=0)
    t = pd.date_range("2026-01-01 05:00:00", periods=4, freq="5min")
    df = pd.DataFrame({
        "cow_id": [0]*4,
        "timestamp": t,
        "rumination_min_5min": [1, 1, 1, 1],
        "activity_mean_5min": [0.2, 0.2, 0.2, 0.2],
        "activity_inactive_frac_5min": [0.8, 0.8, 0.8, 0.8],
        "thi": [70, 70, 70, 70],
        "methane_intensity_g_per_kg_milk": [100, 100, 100, 100],
        "milk_yield_kg_session": [10.0, np.nan, np.nan, np.nan],
        "body_weight_kg": [np.nan]*4,
        "state_true": [0]*4,
    })

    out, _ = build_features(df, cfg)
    assert np.allclose(out["milk_last"].to_numpy(), [10, 10, 10, 10], atol=1e-8)
    # 0, 5, 10, 15 minutes => 0, 1/12, 2/12, 3/12 hours
    exp_h = np.array([0, 1/12, 2/12, 3/12], dtype=float)
    assert np.allclose(out["hours_since_milk"].to_numpy(), exp_h, atol=1e-8)


def test_weight_sparse_no_crash():
    cfg = FeatureConfig(windows=("15min",), baseline_hours=0)
    t = pd.date_range("2026-01-01 00:00:00", periods=6, freq="5min")
    df = pd.DataFrame({
        "cow_id": [0]*6,
        "timestamp": t,
        "rumination_min_5min": [1]*6,
        "activity_mean_5min": [0.2]*6,
        "activity_inactive_frac_5min": [0.8]*6,
        "thi": [70]*6,
        "methane_intensity_g_per_kg_milk": [100]*6,
        "milk_yield_kg_session": [np.nan]*6,
        "body_weight_kg": [np.nan]*6,
        "state_true": [0]*6,
    })

    out, _ = build_features(df, cfg)
    assert out["weight_last"].isna().all()
    assert out["hours_since_weight"].isna().all()