from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .config import SimConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _clamp(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(a, lo), hi)


def _thi(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """
    Temperature-Humidity Index (one common dairy formula).
    This is a simplified version sufficient for synthetic generation.
    """
    rh = rh_pct
    return temp_c - (0.55 - 0.0055 * rh) * (temp_c - 14.5)


def _sample_events(cfg: SimConfig, rng: np.random.Generator, t_index: pd.DatetimeIndex) -> pd.DataFrame:
    events: List[Dict] = []

    # Forced events (useful for tests)
    if cfg.forced_events:
        for cow_id, etype, s, e, sev in cfg.forced_events:
            events.append(
                dict(
                    cow_id=int(cow_id),
                    event_type=str(etype),
                    start=pd.Timestamp(s),
                    end=pd.Timestamp(e),
                    severity=float(sev),
                )
            )

    # Random events
    n_steps = len(t_index)
    for cow_id in range(cfg.n_cows):
        # number of events ~ Poisson(rate * days)
        lam = cfg.event_rate_per_cow_day * cfg.days
        k = rng.poisson(lam=lam)
        for _ in range(k):
            etype = "heat" if rng.random() < cfg.p_heat_event else "illness"
            dur_h = rng.integers(cfg.event_min_hours, cfg.event_max_hours + 1)
            dur_steps = int((dur_h * 60) / 5)
            if dur_steps < 1:
                dur_steps = 1
            start_idx = int(rng.integers(0, max(1, n_steps - dur_steps)))
            start_ts = t_index[start_idx]
            end_ts = t_index[min(n_steps - 1, start_idx + dur_steps - 1)] + pd.Timedelta(minutes=5)
            sev = float(_clamp(rng.normal(0.6, 0.2, size=1), 0.2, 1.0)[0])
            events.append(dict(cow_id=cow_id, event_type=etype, start=start_ts, end=end_ts, severity=sev))

    if not events:
        return pd.DataFrame(columns=["cow_id", "event_type", "start", "end", "severity"])

    ev = pd.DataFrame(events).sort_values(["cow_id", "start"]).reset_index(drop=True)

    # Clip to time window
    t0, t1 = t_index[0], t_index[-1] + pd.Timedelta(minutes=5)
    ev["start"] = ev["start"].clip(lower=t0, upper=t1)
    ev["end"] = ev["end"].clip(lower=t0, upper=t1)
    ev = ev[ev["end"] > ev["start"]].reset_index(drop=True)
    return ev


def _build_state_true(cfg: SimConfig, t_index: pd.DatetimeIndex, events: pd.DataFrame) -> np.ndarray:
    """
    Returns state_true: shape (n_cows, T) with values:
    0 normal, 1 stress, 2 severe stress (severity-based).
    """
    T = len(t_index)
    state = np.zeros((cfg.n_cows, T), dtype=np.int8)
    if events.empty:
        return state

    # Precompute timestamp to index
    # (5-min grid, we can use searchsorted)
    t_vals = t_index.values

    for row in events.itertuples(index=False):
        cow_id = int(row.cow_id)
        sev = float(row.severity)
        level = 2 if sev >= 0.75 else 1
        s = np.searchsorted(t_vals, pd.Timestamp(row.start).to_datetime64())
        e = np.searchsorted(t_vals, pd.Timestamp(row.end).to_datetime64())
        s = max(0, min(T, s))
        e = max(0, min(T, e))
        if e > s:
            state[cow_id, s:e] = np.maximum(state[cow_id, s:e], level)
    return state


def _environment_series(cfg: SimConfig, t_index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    # daily cycle
    minutes = (t_index.hour * 60 + t_index.minute).to_numpy()
    phase = 2 * np.pi * (minutes / (24 * 60))

    temp = cfg.temp_mean_c + cfg.temp_daily_amp * np.sin(phase - np.pi / 2) + rng.normal(0, 0.8, size=len(t_index))
    rh = cfg.humidity_mean_pct + cfg.humidity_daily_amp * np.sin(phase + np.pi / 3) + rng.normal(0, 2.0, size=len(t_index))
    rh = _clamp(rh, 20.0, 98.0)

    # ammonia loosely correlated with humidity and time-in-barn (proxy)
    ammonia = cfg.ammonia_mean_ppm + 0.05 * (rh - cfg.humidity_mean_pct) + rng.normal(0, 1.0, size=len(t_index))
    ammonia = _clamp(ammonia, 0.0, 40.0)

    df = pd.DataFrame(
        {
            "timestamp": t_index,
            "temp_c": temp,
            "humidity_pct": rh,
            "ammonia_ppm": ammonia,
        }
    )
    df["thi"] = _thi(df["temp_c"].to_numpy(), df["humidity_pct"].to_numpy())
    return df


def _milking_mask(cfg: SimConfig, t_index: pd.DatetimeIndex) -> np.ndarray:
    # mark exact 5-min bins that match milking_times (HH:MM)
    ts_str = pd.Series(t_index).dt.strftime("%H:%M").to_numpy()
    target = set(cfg.milking_times[: cfg.milkings_per_day])
    return np.array([s in target for s in ts_str], dtype=bool)


def _apply_dropout(cfg: SimConfig, rng: np.random.Generator, T: int) -> np.ndarray:
    """
    Returns boolean mask "dropped" of length T to simulate sensor outage segment.
    """
    dropped = np.zeros(T, dtype=bool)
    if rng.random() > cfg.dropout_prob_per_cow:
        return dropped
    dur_h = int(rng.integers(cfg.dropout_min_hours, cfg.dropout_max_hours + 1))
    dur_steps = max(1, int((dur_h * 60) / 5))
    start = int(rng.integers(0, max(1, T - dur_steps)))
    dropped[start : start + dur_steps] = True
    return dropped


def generate_synthetic_5min(cfg: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Returns:
      measurements_5min: long table (cow_id, timestamp, ... + state_true)
      events: (cow_id, event_type, start, end, severity)
      meta: dict with config + some summary
    """
    rng = np.random.default_rng(cfg.seed)
    t_index = cfg.time_index_5min()
    T = len(t_index)

    env = _environment_series(cfg, t_index, rng)

    events = _sample_events(cfg, rng, t_index)
    state_true = _build_state_true(cfg, t_index, events)  # (n_cows, T)

    milking_mask = _milking_mask(cfg, t_index)

    # Cow-level heterogeneity (random effects)
    milk_base = rng.normal(32.0, 5.0, size=cfg.n_cows)          # kg/day baseline
    weight_base = rng.normal(650.0, 45.0, size=cfg.n_cows)      # kg
    rum_base = rng.normal(0.35, 0.06, size=cfg.n_cows)          # fraction of 5-min ruminating (~0-1)
    act_base = rng.normal(0.22, 0.05, size=cfg.n_cows)          # movement prob per second proxy (for aggregated)
    sens_heat = _clamp(rng.normal(1.0, 0.25, size=cfg.n_cows), 0.5, 1.8)
    sens_ill = _clamp(rng.normal(1.0, 0.25, size=cfg.n_cows), 0.5, 1.8)

    # Time covariates
    minutes = (t_index.hour * 60 + t_index.minute).to_numpy()
    circ = np.sin(2 * np.pi * (minutes / (24 * 60)) - np.pi / 2)  # peaks midday
    circ2 = np.cos(2 * np.pi * (minutes / (24 * 60)))            # phase-shifted

    # Pre-alloc arrays
    rum_min = np.zeros((cfg.n_cows, T), dtype=float)
    act_mean = np.zeros((cfg.n_cows, T), dtype=float)
    act_inactive_frac = np.zeros((cfg.n_cows, T), dtype=float)
    milk_session = np.full((cfg.n_cows, T), np.nan, dtype=float)
    body_weight = np.full((cfg.n_cows, T), np.nan, dtype=float)
    methane_intensity = np.zeros((cfg.n_cows, T), dtype=float)

    thi = env["thi"].to_numpy()

    # Determine "heat pressure" from environment (shared) - used as extra stress driver
    heat_pressure = _clamp((thi - 68.0) / 12.0, 0.0, 1.5)  # 0 when cool, higher when hot

    # Generate per cow
    for i in range(cfg.n_cows):
        s = state_true[i]  # 0,1,2

        # stress effect scalar (combine event + environment)
        # heat event amplifies heat_pressure; illness event independent of env
        # We approximate: if state==stress/severe => apply larger negative effect
        sev_scalar = np.where(s == 0, 0.0, np.where(s == 1, 1.0, 1.6))

        # Map events to type-specific effects
        # Create a per-step "event heat vs illness" indicator
        if events.empty:
            ev_heat = np.zeros(T)
            ev_ill = np.zeros(T)
        else:
            ev_heat = np.zeros(T)
            ev_ill = np.zeros(T)
            cow_events = events[events["cow_id"] == i]
            if not cow_events.empty:
                t_vals = t_index.values
                for row in cow_events.itertuples(index=False):
                    s_idx = np.searchsorted(t_vals, pd.Timestamp(row.start).to_datetime64())
                    e_idx = np.searchsorted(t_vals, pd.Timestamp(row.end).to_datetime64())
                    if e_idx <= s_idx:
                        continue
                    if row.event_type == "heat":
                        ev_heat[s_idx:e_idx] = np.maximum(ev_heat[s_idx:e_idx], row.severity)
                    else:
                        ev_ill[s_idx:e_idx] = np.maximum(ev_ill[s_idx:e_idx], row.severity)

        # Rumination (minutes within 5-min window, range 0..5)
        # baseline fraction + circadian + stress reductions
        rum_frac = rum_base[i] + 0.06 * (-circ) + rng.normal(0, 0.03, size=T)
        rum_drop = 0.18 * sens_heat[i] * (ev_heat + 0.7 * heat_pressure) + 0.22 * sens_ill[i] * ev_ill
        rum_frac = rum_frac * (1.0 - _clamp(rum_drop, 0.0, 0.7))
        rum_frac = _clamp(rum_frac, 0.0, 1.0)
        rum_min[i] = 5.0 * rum_frac

        # Activity summaries (5-min)
        # heat may increase restlessness; illness decreases activity
        act_level = act_base[i] + 0.08 * (circ) + rng.normal(0, 0.02, size=T)
        act_level += 0.08 * sens_heat[i] * (ev_heat + 0.5 * heat_pressure)  # restlessness
        act_level -= 0.12 * sens_ill[i] * ev_ill                            # lethargy
        act_level = _clamp(act_level, 0.01, 0.95)
        act_mean[i] = act_level
        act_inactive_frac[i] = _clamp(1.0 - act_level + rng.normal(0, 0.05, size=T), 0.0, 1.0)

        # Milk yield per milking session (sparse)
        # daily milk baseline modulated by stress (with a simple lag approximation)
        daily_milk = milk_base[i] * (1.0 - 0.10 * sens_heat[i] * (ev_heat.mean() + heat_pressure.mean() * 0.3)
                                     - 0.15 * sens_ill[i] * (ev_ill.mean()))
        daily_milk = max(10.0, float(daily_milk))
        per_session = daily_milk / float(cfg.milkings_per_day)

        # Add within-day modulation: morning vs evening slightly different
        for t in range(T):
            if milking_mask[t]:
                # stress at time t reduces session yield
                local_penalty = 0.10 * sens_heat[i] * (ev_heat[t] + 0.5 * heat_pressure[t]) + 0.15 * sens_ill[i] * ev_ill[t]
                y = per_session * (1.0 - _clamp(local_penalty, 0.0, 0.6))
                y = y * (1.0 + 0.05 * circ2[t]) + rng.normal(0, 0.6)
                milk_session[i, t] = max(0.0, float(y))

        # Body weight (weekly / irregular measurements)
        # random walk + occasional stress-related dip
        base_w = weight_base[i] + rng.normal(0, 5.0)
        w = base_w + np.cumsum(rng.normal(0, 0.03, size=T))  # slow drift
        w -= 3.0 * (0.6 * sens_ill[i] * ev_ill + 0.3 * sens_heat[i] * ev_heat)  # small dips
        # decide measurement points
        # per day probability weight_measure_prob_per_day; if selected, measure at random time (jitter)
        measure = np.zeros(T, dtype=bool)
        for d in range(cfg.days):
            if rng.random() < cfg.weight_measure_prob_per_day:
                # pick one index within that day
                day_start = d * (24 * 60 // 5)
                day_end = min(T, (d + 1) * (24 * 60 // 5))
                if day_end > day_start:
                    idx = int(rng.integers(day_start, day_end))
                    # jitter within +/- jitter_hours (converted to steps)
                    jitter_steps = int((cfg.weight_irregular_jitter_hours * 60) / 5)
                    idx = int(_clamp(np.array([idx + rng.integers(-jitter_steps, jitter_steps + 1)]), day_start, day_end - 1)[0])
                    measure[idx] = True
        body_weight[i, measure] = w[measure] + rng.normal(0, 2.0, size=measure.sum())

        # Methane intensity (derived from simple energy-balance proxy)
        # DMI proxy ~ 0.025*BW + 0.10*daily_milk (very rough)
        # CH4_g_day ~ 20 * DMI (rough scaling), intensity = CH4 / milk (g/kg)
        # bw_daily = np.nanmean(body_weight[i]) if np.isfinite(np.nanmean(body_weight[i])) else weight_base[i]
        bw_vals = body_weight[i]
        if np.isfinite(bw_vals).any():
            bw_daily = float(np.nanmean(bw_vals))
        else:
            bw_daily = float(weight_base[i])
        dmi = 0.025 * bw_daily + 0.10 * daily_milk
        ch4_g_day = 20.0 * dmi * (1.0 + 0.05 * heat_pressure.mean() + 0.10 * ev_ill.mean())
        intensity = ch4_g_day / max(1e-6, daily_milk)  # g/kg milk
        # small time variation + noise
        methane_intensity[i] = _clamp(intensity + 2.0 * circ + rng.normal(0, 3.0, size=T), 50.0, 600.0)

        # Sensor dropout segments (set to NaN later per modality)
        # We'll return a dropout mask for activity/rumination/env at once
        dropped = _apply_dropout(cfg, rng, T)
        if dropped.any():
            rum_min[i, dropped] = np.nan
            act_mean[i, dropped] = np.nan
            act_inactive_frac[i, dropped] = np.nan

    # Apply independent missingness
    miss_act = rng.random(size=act_mean.shape) < cfg.p_miss_activity_5min
    miss_rum = rng.random(size=rum_min.shape) < cfg.p_miss_rumination

    rum_min[miss_rum] = np.nan
    act_mean[miss_act] = np.nan
    act_inactive_frac[miss_act] = np.nan

    # Env missingness shared across cows (sensor station)
    miss_env = rng.random(size=T) < cfg.p_miss_env
    env.loc[miss_env, ["temp_c", "humidity_pct", "ammonia_ppm", "thi"]] = np.nan

    # Build long table
    rows = []
    for i in range(cfg.n_cows):
        df_i = pd.DataFrame(
            {
                "cow_id": i,
                "timestamp": t_index,
                "rumination_min_5min": rum_min[i],
                "activity_mean_5min": act_mean[i],
                "activity_inactive_frac_5min": act_inactive_frac[i],
                "milk_yield_kg_session": milk_session[i],
                "body_weight_kg": body_weight[i],
                "methane_intensity_g_per_kg_milk": methane_intensity[i],
                "temp_c": env["temp_c"].to_numpy(),
                "humidity_pct": env["humidity_pct"].to_numpy(),
                "ammonia_ppm": env["ammonia_ppm"].to_numpy(),
                "thi": env["thi"].to_numpy(),
                "state_true": state_true[i].astype(int),
            }
        )
        rows.append(df_i)

    measurements = pd.concat(rows, ignore_index=True)

    meta = {
        "config": asdict(cfg),
        "notes": {
            "time_grid": "5-min aligned table; milk/weight are sparse by design; optional 1Hz accel stream available.",
            "state_true": "0 normal, 1 stress, 2 severe (synthetic ground truth for evaluation only).",
        },
        "shape": {"measurements_rows": int(len(measurements)), "n_events": int(len(events))},
    }

    return measurements, events, meta


def generate_accel_1hz(cfg: SimConfig) -> pd.DataFrame:
    """
    Optional: generate a 1Hz accelerometer-like activity stream.
    For demo size, generate for a subset of cows by default.
    """
    if not cfg.generate_accel_1hz:
        return pd.DataFrame(columns=["cow_id", "timestamp", cfg.accel_col])

    rng = np.random.default_rng(cfg.seed + 999)
    start = cfg.start_timestamp()
    end = cfg.end_timestamp()

    # Choose cows
    if cfg.accel_1hz_cows is None:
        # default subset: min(5, n_cows)
        cows = list(range(min(5, cfg.n_cows)))
    else:
        cows = cfg.accel_1hz_cows

    t_1hz = pd.date_range(start, end, freq="1s", inclusive="left")
    seconds = (t_1hz.hour * 3600 + t_1hz.minute * 60 + t_1hz.second).to_numpy()
    circ = np.sin(2 * np.pi * (seconds / (24 * 3600)) - np.pi / 2)

    out = []
    for cow_id in cows:
        # per-second probability-ish level -> convert to nonnegative counts
        base = float(_clamp(rng.normal(0.25, 0.06), 0.05, 0.6))
        level = base + 0.08 * circ + rng.normal(0, 0.03, size=len(t_1hz))
        level = _clamp(level, 0.0, 1.0)

        # Activity count as Poisson with mean proportional to level
        lam = 2.0 + 10.0 * level
        counts = rng.poisson(lam=lam).astype(int)

        df = pd.DataFrame({"cow_id": cow_id, "timestamp": t_1hz, cfg.accel_col: counts})
        out.append(df)

    return pd.concat(out, ignore_index=True)