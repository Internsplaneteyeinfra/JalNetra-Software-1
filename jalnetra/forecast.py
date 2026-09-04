"""
jalnetra.forecast — C4: 7-day projection + verification.

Physics backbone (transport prior integrated forward under forecast discharge,
dilution from forecast rainfall, first-flush bump after dry spells) with an
ensemble over forcing and model-error draws → P10/P50/P90 per reach per day.
Skill is verified against later fused truth vs a persistence baseline — the
blueprint's rule: beat persistence or say so.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config
from .config import N_ENSEMBLE, SEED
from .fusion import DECAY_PER_KM

def _project(state0: dict, discharge0: float, rain: np.ndarray,
             dry_days: int, rng) -> np.ndarray:
    """Physics backbone: same structural form the twin/hindcast learns —
    downstream transport with decay + local source diluted by discharge +
    first-flush after dry spells. In production these terms come from the
    learned transport prior + ML residual; in the demo they are set to the
    (known) simulator structure, standing in for a converged hindcast fit."""
    reaches = config.active_reaches()
    forecast_days = config.FORECAST_DAYS
    order = [r.reach_id for r in reaches]
    lengths = {r.reach_id: r.km_end - r.km_start for r in reaches}
    upstream = {r.reach_id: r.upstream for r in reaches}
    source = {r.reach_id: 2.2 + 9.5 * r.outfall_density for r in reaches}

    x = dict(state0)
    traj = np.zeros((forecast_days, len(order)))
    q = discharge0
    for t in range(forecast_days):
        runoff = rain[t] / 30.0
        q = max(0.35, q + 0.55 * runoff - 0.05 * (q - 1.0) + rng.normal(0, 0.04))
        flush_on = dry_days > 7 and rain[t] > 15
        dry_days = 0 if rain[t] > 5 else dry_days + 1
        newx = {}
        for j, rid in enumerate(order):
            up = upstream[rid]
            inflow = newx.get(up, x.get(up, x[rid])) if up else 0.0
            decay = np.exp(-0.06 * lengths[rid] / max(q, 0.5))
            m = 0.55 * inflow * decay + source[rid] / q
            if flush_on:
                m += (3.0 + 6.0 * (source[rid] - 2.2) / 9.5) * min(rain[t] / 40.0, 1.2)
            # blend toward current state so day-1 stays anchored to today
            w = min(0.20 + 0.09 * t, 0.85)
            m = w * m + (1 - w) * x[rid]
            m *= np.exp(rng.normal(0, 0.07 + 0.03 * t))    # growing model error
            newx[rid] = max(m, 0.2)
            traj[t, j] = newx[rid]
        x = newx
    return traj

def make_forecast(fused_today: pd.DataFrame, discharge0: float,
                  rain_fcst: np.ndarray, dry_days: int,
                  issued: pd.Timestamp, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    forecast_days = config.FORECAST_DAYS
    order = [r.reach_id for r in config.active_reaches()]
    base = fused_today.set_index("reach_id")
    ens = []
    for k in range(N_ENSEMBLE):
        s0 = {rid: float(rng.normal(base.loc[rid, "fused_p50"],
                                    1.6 * base.loc[rid, "fused_std"] + 0.3)) for rid in order}
        s0 = {r: max(v, 0.2) for r, v in s0.items()}
        rain_k = rain_fcst * rng.lognormal(0, 0.35, forecast_days)
        ens.append(_project(s0, discharge0, rain_k, dry_days, rng))
    ens = np.stack(ens)                                   # (K, days, reaches)
    rows = []
    for t in range(forecast_days):
        for j, rid in enumerate(order):
            p10, p50, p90 = np.percentile(ens[:, t, j], [10, 50, 90])
            rows.append(dict(issued=issued, day=t + 1,
                             date=issued + pd.Timedelta(days=t + 1),
                             reach_id=rid, bod_p10=p10, bod_p50=p50, bod_p90=p90))
    return pd.DataFrame(rows)

def hindcast_skill(fused: pd.DataFrame, daily: pd.DataFrame,
                   rain_hist: pd.Series, truth_df: pd.DataFrame,
                   n_issues: int = 24, seed: int = SEED) -> pd.DataFrame:
    """Issue forecasts from past dates and verify each horizon day against
    later observations (demo: simulator truth stands in for verification
    observations). The persistence baseline starts from the SAME fused state
    the model starts from — the operationally fair comparison."""
    forecast_days = config.FORECAST_DAYS
    dates = sorted(fused.date.unique())
    usable = dates[60:-forecast_days]
    issues = usable[:: max(1, len(usable) // n_issues)][:n_issues]
    truth = truth_df.set_index(["date", "reach_id"]).true_bod
    start = fused.set_index(["date", "reach_id"]).fused_p50
    day = daily.drop_duplicates("date").set_index("date")
    err_m, err_p = {t: [] for t in range(1, forecast_days + 1)}, \
                   {t: [] for t in range(1, forecast_days + 1)}
    cover = {t: [] for t in range(1, forecast_days + 1)}
    for d0 in issues:
        f0 = fused[fused.date == d0]
        rain = rain_hist.reindex(pd.date_range(d0, periods=forecast_days + 1)[1:]) \
                        .fillna(rain_hist.mean()).values
        fc = make_forecast(f0, float(day.loc[d0].discharge), rain,
                           dry_days=5, issued=pd.Timestamp(d0), seed=seed)
        for _, r in fc.iterrows():
            key = (r.date, r.reach_id)
            if key in truth.index:
                y = float(truth.loc[key])
                y0 = float(start.loc[(pd.Timestamp(d0), r.reach_id)])
                err_m[r.day].append((r.bod_p50 - y) ** 2)
                err_p[r.day].append((y0 - y) ** 2)
                cover[r.day].append(r.bod_p10 <= y <= r.bod_p90)
    rows = []
    for t in range(1, forecast_days + 1):
        rm = np.sqrt(np.mean(err_m[t])); rp = np.sqrt(np.mean(err_p[t]))
        rows.append(dict(day=t, rmse=rm, rmse_persistence=rp,
                         skill_vs_persistence=1 - rm / rp,
                         interval_coverage=float(np.mean(cover[t]))))
    return pd.DataFrame(rows)
