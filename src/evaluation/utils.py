from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


def attach_welfare_label(features_df: pd.DataFrame, measurements_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a binary welfare label for evaluation.
    For synthetic data, treat state_true > 0 as "welfare degraded".
    """
    f = features_df.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])

    # If features already has state_true, just use it (no merge needed).
    if "state_true" in f.columns:
        f["label_welfare"] = (f["state_true"].fillna(0).astype(int) > 0).astype(int)
        return f

    # Otherwise, pull state_true from measurements and merge in.
    m = measurements_df[["cow_id", "timestamp", "state_true"]].copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"])

    df = f.merge(m, on=["cow_id", "timestamp"], how="left")

    # In case something else adds suffixes, be defensive.
    if "state_true" not in df.columns:
        if "state_true_y" in df.columns:
            df["state_true"] = df["state_true_y"]
        elif "state_true_x" in df.columns:
            df["state_true"] = df["state_true_x"]
        else:
            df["state_true"] = 0

    df["label_welfare"] = (df["state_true"].fillna(0).astype(int) > 0).astype(int)
    return df


def welfare_event_intervals(df: pd.DataFrame, label_col: str = "label_welfare") -> pd.DataFrame:
    """
    Build contiguous positive intervals (per cow) from a 0/1 label on 5-min grid.
    Output columns: cow_id, start, end
    """
    df = df.sort_values(["cow_id", "timestamp"]).reset_index(drop=True)
    events = []

    for cow_id, g in df.groupby("cow_id"):
        y = g[label_col].to_numpy(dtype=int)
        ts = pd.to_datetime(g["timestamp"]).to_numpy()
        on = False
        start = None

        for i in range(len(y)):
            if (y[i] == 1) and (not on):
                on = True
                start = pd.Timestamp(ts[i])
            if on and (y[i] == 0):
                end = pd.Timestamp(ts[i])  # end is first 0 timestamp
                events.append({"cow_id": int(cow_id), "start": start, "end": end})
                on = False
                start = None

        if on and start is not None:
            # close at last timestamp + 5min
            end = pd.Timestamp(ts[-1]) + pd.Timedelta(minutes=5)
            events.append({"cow_id": int(cow_id), "start": start, "end": end})

    return pd.DataFrame(events)