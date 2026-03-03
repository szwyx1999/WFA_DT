from __future__ import annotations
from dataclasses import asdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import AnomalyConfig


EXCLUDE_COLS = {"cow_id", "timestamp", "state_true"}


def default_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    auto select features:
    - numeric columns
    - exclude cow_id/timestamp/state_true
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in num_cols if c not in EXCLUDE_COLS]
    return cols


def missing_rate_from_masks(df: pd.DataFrame) -> np.ndarray:
    """
    compute missing rate (0~1) from *_is_missing mask columns
    """
    mask_cols = [c for c in df.columns if c.endswith("_is_missing")]
    if not mask_cols:
        return np.zeros(len(df), dtype=float)
    m = df[mask_cols].to_numpy(dtype=float)
    return m.mean(axis=1)


class AnomalyDetector:
    """
    Anomaly Detection
    - use IsolationForest to get anomaly score based on feature values
    """

    def __init__(self, cfg: AnomalyConfig):
        self.cfg = cfg
        self.feature_cols: List[str] = []
        self.pipe: Optional[Pipeline] = None
        self.threshold_: Optional[float] = None

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "AnomalyDetector":
        if feature_cols is None:
            feature_cols = default_feature_columns(df)
        self.feature_cols = feature_cols

        X = df[self.feature_cols].to_numpy(dtype=float)

        self.pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("iso", IsolationForest(
                    n_estimators=self.cfg.n_estimators,
                    contamination=self.cfg.contamination,
                    random_state=self.cfg.random_state,
                    n_jobs=-1,
                )),
            ]
        )
        self.pipe.fit(X)

        # set threshold for fused score
        scores = self.score(df)
        q = self.cfg.threshold_quantile
        if q is None:
            # use (1 - contamination) quantile as threshold
            q = 1.0 - float(self.cfg.contamination)
        self.threshold_ = float(np.quantile(scores, q))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        assert self.pipe is not None, "Call fit() first."
        X = df[self.feature_cols].to_numpy(dtype=float)

        # IsolationForest: score_samples 
        # higher score means more normal, lower score means more anomalous
        model_score = -self.pipe.named_steps["iso"].score_samples(
            self.pipe.named_steps["scaler"].transform(
                self.pipe.named_steps["imputer"].transform(X)
            )
        )

        miss_rate = missing_rate_from_masks(df)

        return self.cfg.alpha_model * model_score + self.cfg.beta_missing * miss_rate

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        scores = self.score(df)
        assert self.threshold_ is not None, "Call fit() first."
        return (scores >= self.threshold_).astype(int)

    def to_dict(self) -> dict:
        return {"config": asdict(self.cfg), "feature_cols": self.feature_cols, "threshold": self.threshold_}


def fit_and_score(
    df: pd.DataFrame,
    cfg: AnomalyConfig,
    feature_cols: Optional[List[str]] = None,
    train_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, AnomalyDetector]:
    """
    train and score df:
    - train_df can be used to train on "normal segments only" (useful when demo synthetic data has state_true)
    """
    det = AnomalyDetector(cfg)
    det.fit(train_df if train_df is not None else df, feature_cols=feature_cols)

    out = df[["cow_id", "timestamp"]].copy()
    out["anomaly_score"] = det.score(df)
    out["is_anomaly"] = det.predict(df)
    return out, det