from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd

from .config import FeatureConfig


def load_measurements(csv_path: str, cfg: FeatureConfig) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
    df = df.sort_values([cfg.cow_col, cfg.time_col]).reset_index(drop=True)
    return df


def _group_rolling(df: pd.DataFrame, cfg: FeatureConfig, col: str, window: str, stat: str) -> pd.Series:
    # groupby rolling on time index
    g = df.set_index(cfg.time_col).groupby(cfg.cow_col, group_keys=False)[col]
    r = g.rolling(window, min_periods=1)
    if stat == "mean":
        out = r.mean()
    elif stat == "std":
        out = r.std(ddof=0)
    else:
        raise ValueError(f"Unknown stat: {stat}")
    # drop the cow_id level, keep timestamp index aligned with original order
    return out.reset_index(level=0, drop=True).sort_index().reset_index(drop=True)


def _last_value_and_time_since(df: pd.DataFrame, cfg: FeatureConfig, value_col: str, prefix: str) -> pd.DataFrame:
    """
    For sparse measurements (milk/weight):
      - {prefix}_last: forward-filled last observed value
      - {prefix}_ts_last: forward-filled timestamp of last observation
      - hours_since_{prefix}: time since last observation (NaN before first)
    """
    out = df[[cfg.cow_col, cfg.time_col, value_col]].copy()

    # last value
    out[f"{prefix}_last"] = out.groupby(cfg.cow_col)[value_col].ffill()

    # timestamp of last observation
    ts_last = out[cfg.time_col].where(out[value_col].notna(), pd.NaT)
    out[f"{prefix}_ts_last"] = ts_last.groupby(out[cfg.cow_col]).ffill()

    # hours since last
    dt = out[cfg.time_col] - out[f"{prefix}_ts_last"]
    out[f"hours_since_{prefix}"] = dt.dt.total_seconds() / 3600.0
    out.loc[out[f"{prefix}_ts_last"].isna(), f"hours_since_{prefix}"] = np.nan

    return out[[f"{prefix}_last", f"hours_since_{prefix}", f"{prefix}_ts_last"]]


def _milk_rolling_sum(df: pd.DataFrame, cfg: FeatureConfig) -> pd.Series:
    # treat missing milk as 0 for rolling sum
    tmp = df[[cfg.cow_col, cfg.time_col, cfg.milk_col]].copy()
    tmp["milk_filled0"] = tmp[cfg.milk_col].fillna(0.0)

    g = tmp.set_index(cfg.time_col).groupby(cfg.cow_col, group_keys=False)["milk_filled0"]
    s = g.rolling(cfg.milk_sum_window, min_periods=1).sum()
    return s.reset_index(level=0, drop=True).sort_index().reset_index(drop=True)


def _baseline_zscore(df: pd.DataFrame, cfg: FeatureConfig, cols: List[str]) -> pd.DataFrame:
    """
    Per-cow baseline: use first cfg.baseline_hours hours to compute mean/std.
    Add {col}_z for each col in cols.
    """
    if cfg.baseline_hours <= 0:
        return df

    df = df.copy()
    cow = cfg.cow_col
    time = cfg.time_col

    t0 = df.groupby(cow)[time].transform("min")
    baseline_end = t0 + pd.Timedelta(hours=cfg.baseline_hours)
    in_base = df[time] < baseline_end

    # compute stats on baseline portion, then merge back by cow_id
    base_stats = {}
    for c in cols:
        stats = (
            df.loc[in_base, [cow, c]]
            .groupby(cow)[c]
            .agg(["mean", "std"])
            .rename(columns={"mean": f"{c}__base_mean", "std": f"{c}__base_std"})
        )
        base_stats[c] = stats

    # merge all stats once
    merged = df[[cow]].drop_duplicates().set_index(cow)
    for c, stats in base_stats.items():
        merged = merged.join(stats, how="left")

    df = df.merge(merged.reset_index(), on=cow, how="left")

    for c in cols:
        mu = df[f"{c}__base_mean"]
        sd = df[f"{c}__base_std"].replace(0.0, np.nan)
        # if sd is NaN (e.g., all missing), fall back to 1
        sd = sd.fillna(1.0)
        df[f"{c}_z"] = (df[c] - mu) / sd

    # optional: drop helper columns
    for c in cols:
        df.drop(columns=[f"{c}__base_mean", f"{c}__base_std"], inplace=True)

    return df


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns:
      features_df: original columns + engineered features
      feature_cols: list of engineered feature column names (for model training)
    """
    df = df.copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
    df = df.sort_values([cfg.cow_col, cfg.time_col]).reset_index(drop=True)

    feature_cols: List[str] = []

    # 1) missingness masks for key raw columns
    raw_cols = [cfg.rum_col, cfg.act_col, cfg.inact_col, cfg.thi_col, cfg.methane_col, cfg.milk_col, cfg.weight_col]
    for c in raw_cols:
        mcol = f"{c}_is_missing"
        df[mcol] = df[c].isna().astype(int)
        feature_cols.append(mcol)

    # 2) rolling stats for continuous dense-ish signals
    dense_cols = [cfg.rum_col, cfg.act_col, cfg.inact_col, cfg.thi_col, cfg.methane_col]
    for w in cfg.windows:
        for c in dense_cols:
            mean_name = f"{c}_mean_{w}"
            std_name = f"{c}_std_{w}"
            df[mean_name] = _group_rolling(df, cfg, c, w, "mean")
            df[std_name] = _group_rolling(df, cfg, c, w, "std")
            feature_cols.extend([mean_name, std_name])

    # 3) milk sparse features (last value + time since + rolling 24h sum)
    milk_pack = _last_value_and_time_since(df, cfg, cfg.milk_col, "milk")
    df["milk_last"] = milk_pack["milk_last"]
    df["hours_since_milk"] = milk_pack["hours_since_milk"]
    df["milk_24h_sum"] = _milk_rolling_sum(df, cfg)
    feature_cols.extend(["milk_last", "hours_since_milk", "milk_24h_sum"])

    # 4) weight sparse features (last + time since + change when new measurement occurs)
    w_pack = _last_value_and_time_since(df, cfg, cfg.weight_col, "weight")
    df["weight_last"] = w_pack["weight_last"]
    df["hours_since_weight"] = w_pack["hours_since_weight"]

    # weight_change: nonzero only when a new weight measurement appears
    # (detect change in weight_ts_last)
    ts_last = w_pack["weight_ts_last"]
    new_meas = ts_last.notna() & (ts_last != ts_last.shift(1))
    df["weight_change"] = 0.0
    df.loc[new_meas, "weight_change"] = df.loc[new_meas, "weight_last"] - df["weight_last"].shift(1)
    feature_cols.extend(["weight_last", "hours_since_weight", "weight_change"])

    # 5) baseline z-score (optional) for a selected subset
    z_cols = [cfg.rum_col, cfg.act_col, cfg.inact_col, cfg.thi_col, cfg.methane_col, "milk_last", "weight_last"]
    df = _baseline_zscore(df, cfg, z_cols)
    for c in z_cols:
        if f"{c}_z" in df.columns:
            feature_cols.append(f"{c}_z")

    return df, feature_cols