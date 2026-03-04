from dataclasses import dataclass


@dataclass
class FusionConfig:
    # thresholds (demo defaults, to be tuned)
    welfare_risk_threshold: float = 0.55     # trigger candidate if risk > this
    welfare_run_k: int = 6                   # consecutive points (6*5min=30min)

    # anomaly thresholding
    use_is_anomaly_flag: bool = True
    anomaly_score_quantile: float = 0.97     # if is_anomaly not used, use score quantile

    # gating: if anomaly is active, optionally pause welfare accumulation
    pause_welfare_when_anomaly: bool = True

    # for report/summary
    min_gap_minutes_between_events: int = 60