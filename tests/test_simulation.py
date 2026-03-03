import numpy as np
import pandas as pd

from src.simulation.config import SimConfig
from src.simulation.generator import generate_synthetic_5min


def test_time_grid_is_5min():
    cfg = SimConfig(n_cows=3, days=2, seed=1)
    meas, events, meta = generate_synthetic_5min(cfg)

    # pick one cow and check delta
    df = meas[meas["cow_id"] == 0].sort_values("timestamp")
    ts = pd.to_datetime(df["timestamp"])
    deltas = ts.diff().dropna().dt.total_seconds().to_numpy()
    assert np.all(deltas == 300.0)  # 5 min = 300 sec


def test_rumination_range():
    cfg = SimConfig(n_cows=5, days=1, seed=2)
    meas, _, _ = generate_synthetic_5min(cfg)
    x = meas["rumination_min_5min"].dropna().to_numpy()
    assert x.min() >= 0.0
    assert x.max() <= 5.0


def test_milk_only_at_milking_times():
    cfg = SimConfig(n_cows=2, days=2, seed=3, milkings_per_day=2, milking_times=("05:00", "17:00"))
    meas, _, _ = generate_synthetic_5min(cfg)

    milk = meas.dropna(subset=["milk_yield_kg_session"]).copy()
    assert len(milk) > 0  # should have some sessions
    hhmm = pd.to_datetime(milk["timestamp"]).dt.strftime("%H:%M")
    assert set(hhmm.unique()).issubset(set(cfg.milking_times))


def test_weight_is_sparse():
    cfg = SimConfig(n_cows=10, days=7, seed=4, weight_measure_prob_per_day=1.0/7.0)
    meas, _, _ = generate_synthetic_5min(cfg)

    # average weight measurements per cow should be small (<= ~2 over 7 days typically)
    counts = meas.groupby("cow_id")["body_weight_kg"].apply(lambda s: s.notna().sum())
    assert counts.mean() <= 3.0


def test_forced_stress_reduces_rumination():
    # force a heat event for cow 0, lasting 12 hours
    cfg = SimConfig(
        n_cows=1,
        days=2,
        seed=5,
        forced_events=[(0, "heat", "2026-01-01 06:00:00", "2026-01-01 18:00:00", 0.9)],
    )
    meas, events, _ = generate_synthetic_5min(cfg)
    assert len(events) >= 1

    df = meas.sort_values("timestamp")
    ts = pd.to_datetime(df["timestamp"])
    during = df[(ts >= pd.Timestamp("2026-01-01 06:00:00")) & (ts < pd.Timestamp("2026-01-01 18:00:00"))]
    before = df[(ts >= pd.Timestamp("2026-01-01 00:00:00")) & (ts < pd.Timestamp("2026-01-01 06:00:00"))]

    # use median to be robust to noise/missing
    r_during = during["rumination_min_5min"].median(skipna=True)
    r_before = before["rumination_min_5min"].median(skipna=True)
    assert r_during < r_before