from dataclasses import dataclass


@dataclass
class WelfareHMMConfig:
    k_states: int = 3              # 0 normal, 1 mild, 2 severe (wll order them after training)
    n_iter: int = 10               # EM iterations
    tol: float = 1e-4              # stop if loglik improvement is tiny
    random_state: int = 42

    # numerical stability
    var_floor: float = 1e-3        # keep variances away from zero
    eps: float = 1e-12

    # anomaly weighting (from anomaly detection model)
    use_anomaly_weights: bool = True
    anomaly_weight: float = 0.0    # weight for rows with is_anomaly==1 during training