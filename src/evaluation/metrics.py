from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # AUROC undefined if only one class present
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def false_alarms_per_cow_day(df: pd.DataFrame, alarm_col: str, label_col: str) -> float:
    """
    Count alarm points where label==0, normalize by cow-days.
    """
    # assume 5-min grid => 288 points per day
    points_per_day = 288.0
    false_points = ((df[alarm_col] == 1) & (df[label_col] == 0)).sum()
    n_cows = df["cow_id"].nunique()
    n_days = df.groupby("cow_id")["timestamp"].apply(lambda s: (pd.to_datetime(s).max() - pd.to_datetime(s).min()).total_seconds() / 86400.0).mean()
    cow_days = max(1e-9, float(n_cows * max(n_days, 1e-6)))
    # convert point-rate to event-rate-ish (per day) by dividing by points per day
    return float(false_points / points_per_day / cow_days)


def time_to_detect_minutes(events: pd.DataFrame, alerts: pd.DataFrame, alarm_col: str) -> float:
    """
    For each true welfare interval (cow_id, start, end), find first alarm within [start, end).
    Return mean time-to-detect in minutes over detected events.
    """
    if events.empty:
        return float("nan")

    alerts = alerts.copy()
    alerts["timestamp"] = pd.to_datetime(alerts["timestamp"])

    ttd = []
    for row in events.itertuples(index=False):
        cow_id = int(row.cow_id)
        start = pd.Timestamp(row.start)
        end = pd.Timestamp(row.end)

        a = alerts[(alerts["cow_id"] == cow_id) & (alerts["timestamp"] >= start) & (alerts["timestamp"] < end)]
        a = a[a[alarm_col] == 1]
        if len(a) == 0:
            continue
        first = a["timestamp"].min()
        ttd.append((first - start).total_seconds() / 60.0)

    if len(ttd) == 0:
        return float("nan")
    return float(np.mean(ttd))