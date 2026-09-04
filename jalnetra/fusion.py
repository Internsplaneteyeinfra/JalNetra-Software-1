"""
jalnetra.fusion — C3: the fused daily river state ("the twin").

A deliberately simple, honest 1-D estimator per the blueprint (13.5):
 predict:  advection-decay prior moves yesterday's state downstream under the
           day's discharge; process noise inflates variance (uncertainty GROWS
           through observation gaps — the honest behaviour),
 update:   scalar Kalman updates from anchors (always) and satellite
           retrievals (when a cloud-free overpass exists), each weighted by
           its own stated variance.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import active_reaches

DECAY_PER_KM = 0.004         # first-order decay (demo value; learn per reach in prod)
PROCESS_STD = 0.6            # mg/L/day BOD-scale process noise (demo)
ANCHOR_STD = 0.6             # anchor pseudo-observation std (mg/L, demo)

def run_twin(daily: pd.DataFrame, sat_retrievals: pd.DataFrame,
             target: str = "bod") -> pd.DataFrame:
    """daily: one row per reach per date with discharge + anchored flag and
    (demo) anchor pseudo-obs `anchor_val`; sat_retrievals: p10/p50/p90 rows
    for observed reach-days. Returns fused state per reach-day."""
    reaches = active_reaches()
    order = [r.reach_id for r in reaches]
    lengths = {r.reach_id: r.km_end - r.km_start for r in reaches}
    upstream = {r.reach_id: r.upstream for r in reaches}

    sat = sat_retrievals.set_index(["date", "reach_id"]) if len(sat_retrievals) else None
    dates = sorted(daily.date.unique())
    day = daily.set_index(["date", "reach_id"])

    mean = {rid: 5.0 for rid in order}
    var = {rid: 25.0 for rid in order}
    rows = []
    for d in dates:
        new_mean, new_var = {}, {}
        for rid in order:                       # upstream → downstream
            row = day.loc[(d, rid)]
            q = float(row.discharge)
            up = upstream[rid]
            inflow = new_mean.get(up, mean.get(up, mean[rid])) if up else mean[rid]
            v_in = new_var.get(up, var.get(up, var[rid])) if up else var[rid]
            decay = np.exp(-DECAY_PER_KM * lengths[rid] / max(q, 0.3))
            m = 0.6 * inflow * decay + 0.4 * mean[rid]          # transport prior
            # confidence flows downstream with the water: inflow variance
            # contributes with its transport weight; process noise inflates
            v = 0.36 * v_in + 0.16 * var[rid] + PROCESS_STD ** 2
            support = "prior"
            # anchor update
            if bool(row.anchored):
                z, R = float(row.anchor_val), ANCHOR_STD ** 2
                K = v / (v + R); m, v = m + K * (z - m), (1 - K) * v
                support = "anchor"
            # satellite update
            if sat is not None and (d, rid) in sat.index:
                s = sat.loc[(d, rid)]
                z = float(s[f"{target}_p50"])
                R = max(((s[f"{target}_p90"] - s[f"{target}_p10"]) / 3.29) ** 2, 0.05)
                K = v / (v + R); m, v = m + K * (z - m), (1 - K) * v
                support = "sat+anchor" if support == "anchor" else "sat"
            new_mean[rid], new_var[rid] = max(m, 0.2), v
            rows.append(dict(date=d, reach_id=rid, fused_p50=new_mean[rid],
                             fused_std=np.sqrt(v), obs_support=support))
        mean, var = new_mean, new_var
    out = pd.DataFrame(rows)
    out["fused_p10"] = (out.fused_p50 - 1.645 * out.fused_std).clip(0.1)
    out["fused_p90"] = out.fused_p50 + 1.645 * out.fused_std
    return out
