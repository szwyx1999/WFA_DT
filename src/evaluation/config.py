from dataclasses import dataclass


@dataclass
class EvalConfig:
    # CV split
    n_splits: int = 3
    random_state: int = 42

    # HMM training inside each fold (small for demo)
    hmm_k: int = 3
    hmm_n_iter: int = 6
    anomaly_weight: float = 0.0

    # anomaly model
    contamination: float = 0.03

    # fusion/alert
    # welfare_risk_th: float = 0.55
    # welfare_risk_th = 0.70
    welfare_risk_th: float = 0.65
    
    welfare_run_k: int = 6
    # welfare_run_k: int = 12

    pause_on_anomaly: bool = True