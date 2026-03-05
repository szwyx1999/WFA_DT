from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from src.models.anomaly.config import AnomalyConfig
from src.models.anomaly.detector import AnomalyDetector, default_feature_columns

from src.models.welfare_hmm.config import WelfareHMMConfig
from src.models.welfare_hmm.pipeline import fit_hmm_model, infer_hmm

from src.fusion.config import FusionConfig
from src.fusion.fusion import fuse_and_alert

from .utils import attach_welfare_label, welfare_event_intervals
from .metrics import safe_auroc, safe_auprc, brier, false_alarms_per_cow_day, time_to_detect_minutes
from .config import EvalConfig


def _subset_by_cows(df: pd.DataFrame, cows: np.ndarray) -> pd.DataFrame:
    return df[df["cow_id"].isin(cows)].copy()


def evaluate_groupkfold(
    features_df: pd.DataFrame,
    measurements_df: pd.DataFrame,
    cfg: EvalConfig,
) -> Dict:
    """
    Full CV:
      per fold:
        - fit anomaly on train cows
        - fit HMM on train cows
        - infer on test cows
        - fuse -> alarms
        - compute metrics on test cows
    """
    df = attach_welfare_label(features_df, measurements_df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["cow_id", "timestamp"]).reset_index(drop=True)

    cows = df["cow_id"].unique()
    groups = df["cow_id"].to_numpy()

    gkf = GroupKFold(n_splits=cfg.n_splits)

    fold_rows: List[Dict] = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        # Note: train_idx/test_idx are row indices; we want cow sets
        train_cows = np.unique(df.iloc[train_idx]["cow_id"].to_numpy())
        test_cows = np.unique(df.iloc[test_idx]["cow_id"].to_numpy())

        train_df = _subset_by_cows(df, train_cows)
        test_df = _subset_by_cows(df, test_cows)

        # ---- Anomaly model (fit on train cows only) ----
        anom_cfg = AnomalyConfig(contamination=cfg.contamination)
        det = AnomalyDetector(anom_cfg)

        # avoid leaking labels into anomaly features
        anom_cols = [c for c in default_feature_columns(train_df) if c not in ("label_welfare",)]
        det.fit(train_df, feature_cols=anom_cols)

        test_anom = test_df[["cow_id", "timestamp"]].copy()
        test_anom["anomaly_score"] = det.score(test_df)
        test_anom["is_anomaly"] = det.predict(test_df)

        # ---- HMM (fit on train cows only) ----
        hmm_cfg = WelfareHMMConfig(
            k_states=cfg.hmm_k,
            n_iter=cfg.hmm_n_iter,
            anomaly_weight=cfg.anomaly_weight,
        )

        # provide anomaly flags to train_df for weighting
        train_df = train_df.merge(
            test_anom.iloc[0:0],  # empty just to keep columns stable
            on=["cow_id", "timestamp"], how="left"
        )
        # recompute anomaly flags for train using fitted detector
        train_df["is_anomaly"] = det.predict(train_df)

        hmm, imputer, hmm_cols, hmm_meta = fit_hmm_model(train_df, hmm_cfg)
        welfare_test = infer_hmm(test_df, hmm, imputer, hmm_cols, k_states=cfg.hmm_k)

        # ---- Fusion / alerts on test cows ----
        fusion_cfg = FusionConfig(
            welfare_risk_threshold=cfg.welfare_risk_th,
            welfare_run_k=cfg.welfare_run_k,
            pause_welfare_when_anomaly=cfg.pause_on_anomaly,
        )
        fused_test, fusion_meta = fuse_and_alert(welfare_test, test_anom, fusion_cfg)

        # attach label to fused_test for metric computation
        fused_test = fused_test.merge(
            test_df[["cow_id", "timestamp", "label_welfare"]],
            on=["cow_id", "timestamp"],
            how="left",
        )

        y = fused_test["label_welfare"].to_numpy(dtype=int)
        risk = fused_test["welfare_risk"].to_numpy(dtype=float)

        # events for time-to-detect
        ev = welfare_event_intervals(test_df, label_col="label_welfare")
        ttd = time_to_detect_minutes(ev, fused_test, alarm_col="welfare_alarm")

        pred_out = fused_test[["cow_id","timestamp","label_welfare","welfare_risk","welfare_alarm","anomaly_alarm"]].copy()
        pred_out["fold"] = fold
        pred_out.to_csv(f"data/temp/predictions_fold{fold}.csv", index=False)

        row = {
            "fold": fold,
            "n_train_cows": int(len(train_cows)),
            "n_test_cows": int(len(test_cows)),
            "auroc_risk": safe_auroc(y, risk),
            "auprc_risk": safe_auprc(y, risk),
            "brier_risk": brier(y, np.clip(risk, 0.0, 1.0)),
            "false_alarms_per_cow_day": false_alarms_per_cow_day(fused_test, "welfare_alarm", "label_welfare"),
            "time_to_detect_min": ttd,
            "anomaly_alarm_rate": float(fused_test["anomaly_alarm"].mean()),
            "welfare_alarm_rate": float(fused_test["welfare_alarm"].mean()),
        }
        fold_rows.append(row)

    folds_df = pd.DataFrame(fold_rows)

    report = {
        "config": cfg.__dict__,
        "folds": fold_rows,
        "summary": {
            "auroc_mean": float(folds_df["auroc_risk"].mean(skipna=True)),
            "auroc_std": float(folds_df["auroc_risk"].std(skipna=True)),
            "auprc_mean": float(folds_df["auprc_risk"].mean(skipna=True)),
            "auprc_std": float(folds_df["auprc_risk"].std(skipna=True)),
            "brier_mean": float(folds_df["brier_risk"].mean(skipna=True)),
            "false_alarms_per_cow_day_mean": float(folds_df["false_alarms_per_cow_day"].mean(skipna=True)),
            "ttd_min_mean": float(folds_df["time_to_detect_min"].mean(skipna=True)),
        },
    }
    return report, folds_df