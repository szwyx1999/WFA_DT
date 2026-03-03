from __future__ import annotations
from dataclasses import asdict
from typing import List, Tuple, Dict, Optional
import json
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from .config import WelfareHMMConfig
from .hmm import GaussianHMM


def select_welfare_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Pick a small, interpretable set of features for the HMM.
    Using *_z is nice because per-cow baseline normalization is already done.
    """
    candidates = [
        "rumination_min_5min_z",
        "activity_mean_5min_z",
        "activity_inactive_frac_5min_z",
        "thi_z",
        "methane_intensity_g_per_kg_milk_z",
        "milk_last_z",
        "weight_last_z",
    ]
    cols = [c for c in candidates if c in df.columns]

    # Add a lightweight "data quality" signal so the HMM knows some rows are junky.
    # (still downweight anomalies in training, but having this helps inference.)
    if "rumination_min_5min_is_missing" in df.columns:
        cols.append("rumination_min_5min_is_missing")
    if "activity_mean_5min_is_missing" in df.columns:
        cols.append("activity_mean_5min_is_missing")

    return cols


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: WelfareHMMConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[int, np.ndarray], SimpleImputer]:
    """
    Turn long table into per-cow sequences.
    Also returns a mapping cow_id -> row indices, so we can stitch outputs back.
    """
    df = df.sort_values(["cow_id", "timestamp"]).reset_index(drop=True)

    # Impute NaNs globally 
    # for demo, simply using median imputation. the HMM will learn to treat imputed values as "normal" since they're common.
    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(df[feature_cols].to_numpy(dtype=float))

    sequences: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    idx_map: Dict[int, np.ndarray] = {}

    for cow_id, idx in df.groupby("cow_id").indices.items():
        idx = np.asarray(idx)
        idx_map[int(cow_id)] = idx

        X = X_all[idx]
        sequences.append(X)

        if cfg.use_anomaly_weights and "is_anomaly" in df.columns:
            w = np.where(df.loc[idx, "is_anomaly"].to_numpy(dtype=int) == 1, cfg.anomaly_weight, 1.0)
        else:
            w = np.ones(len(idx), dtype=float)
        weights.append(w)

    return sequences, weights, idx_map, imputer


def fit_welfare_hmm(
    features_df: pd.DataFrame,
    cfg: WelfareHMMConfig,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Fit HMM, then output:
      - posterior probs per state
      - welfare_risk = P(state != normal)
      - viterbi state
    """
    df = features_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if feature_cols is None:
        feature_cols = select_welfare_feature_cols(df)
    assert len(feature_cols) > 0, "No usable feature columns found for welfare HMM."

    seqs, ws, idx_map, imputer = build_sequences(df, feature_cols, cfg)

    D = len(feature_cols)
    hmm = GaussianHMM(K=cfg.k_states, D=D, var_floor=cfg.var_floor, eps=cfg.eps, random_state=cfg.random_state)

    ll_hist = hmm.fit(seqs, weights=ws, n_iter=cfg.n_iter, tol=cfg.tol)

    # reorder states so state 0 ~ "normal" by rumination feature if available
    order = None
    rum_col = "rumination_min_5min_z"
    if rum_col in feature_cols:
        j = feature_cols.index(rum_col)
        order = hmm.reorder_states_by_feature(j, descending=True)

    # inference
    out = df[["cow_id", "timestamp"]].copy()
    prob_cols = [f"p_state_{k}" for k in range(cfg.k_states)]
    for c in prob_cols:
        out[c] = np.nan
    out["welfare_risk"] = np.nan
    out["state_viterbi"] = np.nan

    # need the same imputed X used earlier
    X_all = imputer.transform(df[feature_cols].to_numpy(dtype=float))

    for cow_id, idx in idx_map.items():
        X = X_all[idx]
        gamma = hmm.predict_proba(X)              # (T,K)
        path = hmm.viterbi(X).astype(int)

        out.loc[idx, prob_cols] = gamma
        out.loc[idx, "welfare_risk"] = 1.0 - gamma[:, 0]  # state0=normal after ordering
        out.loc[idx, "state_viterbi"] = path

    meta = {
        "cfg": asdict(cfg),
        "feature_cols": feature_cols,
        "loglik_history": ll_hist,
        "state_order": order.tolist() if order is not None else None,
        "pi": hmm.pi.tolist(),
        "A": hmm.A.tolist(),
        "means": hmm.means.tolist(),
        "vars": hmm.vars_.tolist(),
    }
    return out, meta


def save_outputs(out_df: pd.DataFrame, meta: dict, out_csv: str, meta_json: str) -> None:
    out_df.to_csv(out_csv, index=False)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)