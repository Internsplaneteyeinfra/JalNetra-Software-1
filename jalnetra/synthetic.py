"""
jalnetra.synthetic — SYNTHETIC DEMO DATA GENERATOR.

===============================================================================
HONESTY NOTE (accuracy charter, do not remove):
Everything produced by this module is SIMULATED so the pipeline can run
plug-and-play without credentials or data agreements. Model skill measured on
this data demonstrates that the SOFTWARE works end to end; it says NOTHING
about real-world accuracy, which can only be established against real
satellite scenes and laboratory co-samples (see ingestion.SceneSource /
LabSource plug points).
===============================================================================

The simulator encodes the physics the blueprint describes, so the pipeline is
exercised realistically:
  * seasonal discharge cycle (monsoon Jun–Sep) with dilution,
  * first-flush spikes at monsoon onset,
  * outfall-driven baseline pollution per reach, decaying downstream,
  * optical proxies (turbidity, chl-a, CDOM, TSS) causally linked to COD/BOD
    plus noise — so "satellite" sees pollution only indirectly,
  * monsoon cloud cover hiding satellite observations,
  * random discharge events (anomalies) on high-outfall reaches.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import active_reaches, SEED, DEMO_TRAIN_YEARS

def _season(doy: np.ndarray) -> dict[str, np.ndarray]:
    monsoon = ((doy >= 152) & (doy <= 273)).astype(float)         # Jun–Sep approx
    onset = ((doy >= 152) & (doy <= 181)).astype(float)           # June
    discharge = 1.0 + 2.6 * np.sin(np.clip((doy - 140) / 150, 0, 1) * np.pi) ** 2 * monsoon \
                + 0.15 * np.sin(doy / 365 * 2 * np.pi)
    return {"monsoon": monsoon, "onset": onset, "discharge": np.maximum(discharge, 0.35)}

def generate_history(days: int = 365 * DEMO_TRAIN_YEARS,
                     end: str = "2026-08-11",
                     seed: int = SEED) -> pd.DataFrame:
    """Daily truth + observations per reach. Returns long dataframe."""
    rng = np.random.default_rng(seed)
    reaches = active_reaches()
    anchor_set = set(sorted(range(len(reaches)),
                          key=lambda i: reaches[i].outfall_density,
                          reverse=True)[:min(6, len(reaches))])
    dates = pd.date_range(end=end, periods=days, freq="D")
    doy = dates.dayofyear.values.astype(float)
    s = _season(doy)
    rows = []
    # random anomaly events: (reach index, start day, duration, magnitude)
    hot = [i for i, r in enumerate(reaches) if r.outfall_density > 0.5]
    events = [(rng.choice(hot), int(rng.integers(30, days - 10)),
               int(rng.integers(2, 6)), rng.uniform(1.6, 3.2))
              for _ in range(max(6, days // 120))]

    upstream_bod = np.zeros(days)  # boundary condition entering R01
    for i, r in enumerate(reaches):
        base = 2.2 + 9.5 * r.outfall_density                     # dry-season local BOD source
        flush = s["onset"] * rng.uniform(0.8, 1.2) * (3.0 + 6.0 * r.outfall_density)
        dilution = 1.0 / s["discharge"]
        decay = np.exp(-0.06 * (r.km_end - r.km_start) / max(s["discharge"].mean(), 1e-6))
        bod = (0.55 * upstream_bod * decay + base * dilution + flush
               + rng.normal(0, 0.5, days)).clip(0.3)
        for ridx, t0, dur, mag in events:
            if ridx == i:
                bod[t0:t0 + dur] *= mag
        cod = bod * rng.uniform(2.3, 2.9) + rng.normal(0, 2.0, days)
        cod = np.maximum(cod, bod * 1.5)
        upstream_bod = bod

        # optical proxies: causal but noisy (the honest indirection)
        turb = 8 + 4.5 * bod + 35 * s["monsoon"] * s["discharge"] + rng.normal(0, 2.5, days)
        cdom = 0.4 + 0.11 * bod + rng.normal(0, 0.10, days)
        chla = 3 + 0.9 * bod * (1 - 0.6 * s["monsoon"]) + rng.normal(0, 0.8, days)
        tss  = 12 + 2.2 * bod + 55 * s["monsoon"] + rng.normal(0, 4, days)

        # satellite visibility: ~5-day revisit, monsoon clouds
        overpass = (np.arange(days) % 5 == (i % 5))
        p_clear = np.where(s["monsoon"] > 0, 0.25, 0.85)
        seen = overpass & (rng.random(days) < p_clear)

        # anchors: six highest-outfall reaches instrumented (continuous)
        anchored = i in anchor_set

        for d in range(days):
            rows.append(dict(
                date=dates[d], reach_id=r.reach_id, doy=doy[d],
                monsoon=s["monsoon"][d], discharge=s["discharge"][d],
                width_m=r.mean_width_m, outfall_density=r.outfall_density,
                km_mid=(r.km_start + r.km_end) / 2,
                true_bod=bod[d], true_cod=cod[d],
                turb=max(turb[d], 0.5), cdom=max(cdom[d], 0.05),
                chla=max(chla[d], 0.2), tss=max(tss[d], 1.0),
                sat_seen=bool(seen[d]), anchored=anchored,
            ))
    df = pd.DataFrame(rows)
    df.attrs["synthetic"] = True
    return df

def rainfall_forecast(dates: pd.DatetimeIndex, seed: int = SEED + 7) -> np.ndarray:
    """Demo NWP rainfall forecast (mm/day) for the forecast horizon."""
    rng = np.random.default_rng(seed)
    doy = dates.dayofyear.values
    monsoon = ((doy >= 152) & (doy <= 273))
    return np.where(monsoon, rng.gamma(2.0, 9.0, len(dates)), rng.gamma(1.1, 1.2, len(dates)))
