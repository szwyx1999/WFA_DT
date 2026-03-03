from dataclasses import dataclass


@dataclass
class AnomalyConfig:
    # IsolationForest
    contamination: float = 0.03          # expected anomaly ratio (demo 1%~5%)
    n_estimators: int = 200
    random_state: int = 42

    # Score fusion
    alpha_model: float = 1.0            # model anomaly score weight
    beta_missing: float = 2.0           # missing rate/flow interruption feature weight (strongly recommended >0)

    # Thresholding
    threshold_quantile: float | None = None
    # If not set, use the quantile corresponding to contamination as the threshold