from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pandas as pd


@dataclass
class SimConfig:
    # Core
    seed: int = 42
    n_cows: int = 100
    start: str = "2026-01-01 00:00:00"
    days: int = 3
    freq_5min: str = "5min"

    # Milk sessions (per day)
    milkings_per_day: int = 2
    milking_times: Tuple[str, ...] = ("05:00", "17:00")  # local clock

    # Body weight measurement sparsity
    weight_measure_prob_per_day: float = 1.0 / 7.0  # weekly-ish
    weight_irregular_jitter_hours: int = 6

    # Missingness rates (independent)
    p_miss_activity_5min: float = 0.005
    p_miss_rumination: float = 0.01
    p_miss_env: float = 0.005
    # milk is sparse by design; leave missing outside milking sessions

    # Dropout segments (simulate sensor outages)
    dropout_prob_per_cow: float = 0.25
    dropout_min_hours: int = 1
    dropout_max_hours: int = 6

    # Stress events
    event_rate_per_cow_day: float = 0.15  # expected events per cow per day
    event_min_hours: int = 6
    event_max_hours: int = 24
    # Two event types: "heat" and "illness"
    p_heat_event: float = 0.6

    # Environment dynamics
    temp_mean_c: float = 18.0
    temp_daily_amp: float = 7.0
    humidity_mean_pct: float = 60.0
    humidity_daily_amp: float = 15.0
    ammonia_mean_ppm: float = 8.0

    # Optional accel 1Hz generation
    generate_accel_1hz: bool = False
    accel_1hz_cows: Optional[List[int]] = None  # None => default subset
    accel_col: str = "activity_count"

    # Optional forced events for deterministic tests / demos
    forced_events: Optional[List[Tuple[int, str, str, str, float]]] = None
    # (cow_id, event_type, start_ts, end_ts, severity in [0,1])

    def start_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    def end_timestamp(self) -> pd.Timestamp:
        return self.start_timestamp() + pd.Timedelta(days=self.days)

    def time_index_5min(self) -> pd.DatetimeIndex:
        # inclusive left, exclusive right
        return pd.date_range(self.start_timestamp(), self.end_timestamp(), freq=self.freq_5min, inclusive="left")