from __future__ import annotations
from dataclasses import asdict
from typing import Tuple, Dict
import json
import numpy as np
import pandas as pd

from .config import FusionConfig


def _consecutive_run(flag: np.ndarray) -> np.ndarray:
    """
    Return run length up to each position.
    Example: flag=[0,1,1,0,1] -> run=[0,1,2,0,1]
    """
    run = np.zeros_like(flag, dtype=int)
    cur = 0
    for i, v in enumerate(flag):
        if v:
            cur += 1
        else:
            cur = 0
        run[i] = cur
    return run


def _eventize(flag: np.ndarray, timestamps: np.ndarray, min_gap_minutes: int) -> np.ndarray:
    """
    Convert a boolean flag series to event ids.
    We merge events that are closer than min_gap_minutes.
    Returns event_id per row (0 = no event; positive integers for event groups).
    """
    event_id = np.zeros(len(flag), dtype=int)
    cur_id = 0
    last_on_time = None
    gap = pd.Timedelta(minutes=min_gap_minutes)

    for i, on in enumerate(flag):
        if not on:
            continue

        t = pd.Timestamp(timestamps[i])
        if last_on_time is None or (t - last_on_time) > gap:
            cur_id += 1
        event_id[i] = cur_id
        last_on_time = t

    return event_id


def fuse_and_alert(
    welfare_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    cfg: FusionConfig,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Inputs:
      welfare_df: cow_id,timestamp,welfare_risk,(optional p_state_k, state_viterbi)
      anomaly_df: cow_id,timestamp,(anomaly_score,is_anomaly)

    Output:
      fused_df: merged + alarms + event ids
      meta: thresholds and summary stats for reproducibility
    """
    w = welfare_df.copy()
    a = anomaly_df.copy()

    w["timestamp"] = pd.to_datetime(w["timestamp"])
    a["timestamp"] = pd.to_datetime(a["timestamp"])

    df = w.merge(a, on=["cow_id", "timestamp"], how="left")

    # fill missing anomaly info as "not anomaly"
    if "is_anomaly" in df.columns:
        df["is_anomaly"] = df["is_anomaly"].fillna(0).astype(int)
    if "anomaly_score" in df.columns:
        # keep NaN for now; we'll handle thresholding
        pass

    # anomaly alarm
    if cfg.use_is_anomaly_flag and "is_anomaly" in df.columns:
        df["anomaly_alarm"] = (df["is_anomaly"] == 1).astype(int)
        anomaly_threshold = None
    else:
        # quantile-based threshold
        scores = df["anomaly_score"].to_numpy(dtype=float)
        finite = np.isfinite(scores)
        if finite.any():
            anomaly_threshold = float(np.quantile(scores[finite], cfg.anomaly_score_quantile))
            df["anomaly_alarm"] = ((df["anomaly_score"] >= anomaly_threshold) & finite).astype(int)
        else:
            anomaly_threshold = None
            df["anomaly_alarm"] = 0

    # welfare candidate flag
    df["welfare_candidate"] = (df["welfare_risk"] >= cfg.welfare_risk_threshold).astype(int)

    # optionally pause welfare accumulation when anomaly is on
    if cfg.pause_welfare_when_anomaly:
        df["welfare_candidate_gated"] = df["welfare_candidate"].where(df["anomaly_alarm"] == 0, 0).astype(int)
    else:
        df["welfare_candidate_gated"] = df["welfare_candidate"]

    # compute consecutive run within each cow
    df = df.sort_values(["cow_id", "timestamp"]).reset_index(drop=True)
    df["welfare_runlen"] = 0

    for cow_id, idx in df.groupby("cow_id").indices.items():
        idx = np.asarray(idx)
        run = _consecutive_run(df.loc[idx, "welfare_candidate_gated"].to_numpy(dtype=int) == 1)
        df.loc[idx, "welfare_runlen"] = run

    # welfare alarm: only if runlen >= K
    df["welfare_alarm"] = (df["welfare_runlen"] >= int(cfg.welfare_run_k)).astype(int)

    # combined / triage label (simple, interpretable)
    # 0=ok, 1=anomaly_only, 2=welfare_only, 3=both
    df["alarm_code"] = (
        df["anomaly_alarm"].astype(int) + 2 * df["welfare_alarm"].astype(int)
    ).astype(int)

    # event ids (per cow)
    df["anomaly_event_id"] = 0
    df["welfare_event_id"] = 0
    for cow_id, idx in df.groupby("cow_id").indices.items():
        idx = np.asarray(idx)
        ts = df.loc[idx, "timestamp"].to_numpy()
        df.loc[idx, "anomaly_event_id"] = _eventize(
            df.loc[idx, "anomaly_alarm"].to_numpy(dtype=int) == 1,
            ts,
            cfg.min_gap_minutes_between_events,
        )
        df.loc[idx, "welfare_event_id"] = _eventize(
            df.loc[idx, "welfare_alarm"].to_numpy(dtype=int) == 1,
            ts,
            cfg.min_gap_minutes_between_events,
        )

    meta = {
        "cfg": asdict(cfg),
        "derived": {
            "anomaly_threshold": anomaly_threshold,
            "welfare_risk_threshold": cfg.welfare_risk_threshold,
            "welfare_run_k": cfg.welfare_run_k,
        },
        "summary": {
            "anomaly_alarm_rate": float(df["anomaly_alarm"].mean()),
            "welfare_alarm_rate": float(df["welfare_alarm"].mean()),
            "both_alarm_rate": float(((df["anomaly_alarm"] == 1) & (df["welfare_alarm"] == 1)).mean()),
        },
    }

    return df, meta


def save_fusion_outputs(df: pd.DataFrame, meta: dict, out_csv: str, meta_json: str) -> None:
    df.to_csv(out_csv, index=False)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)