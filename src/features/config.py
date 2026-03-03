from dataclasses import dataclass
from typing import Tuple


@dataclass
class FeatureConfig:
    # rolling windows to compute on 5-min grid
    windows: Tuple[str, ...] = ("1h", "6h")

    # milk window aggregation
    milk_sum_window: str = "24h"

    # baseline normalization (per cow, using first baseline_hours hours)
    # set <=0 to disable
    baseline_hours: int = 24

    # minimal columns we expect in measurements_5min.csv
    cow_col: str = "cow_id"
    time_col: str = "timestamp"

    # raw signal columns
    rum_col: str = "rumination_min_5min"
    act_col: str = "activity_mean_5min"
    inact_col: str = "activity_inactive_frac_5min"
    milk_col: str = "milk_yield_kg_session"
    weight_col: str = "body_weight_kg"
    methane_col: str = "methane_intensity_g_per_kg_milk"
    thi_col: str = "thi"  # already computed in component1