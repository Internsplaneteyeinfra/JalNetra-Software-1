"""
jalnetra.trust — C5: blind validation metrics, lineage, alerts.

Metrics follow the normative definitions in blueprint Appendix P. Lineage
records make every published value reproducible; on synthetic data the record
carries an explicit "SYNTHETIC-DEMO" provenance so it can never be mistaken
for a real measurement chain.
"""
from __future__ import annotations
import hashlib, json
import numpy as np
import pandas as pd
from .config import BOD_CLASS_EDGES, CLASS_NAMES, ABSTAIN
from .retrieval import classify_bod

def _band(v: float) -> int:
    for i, e in enumerate(BOD_CLASS_EDGES):
        if v <= e:
            return i
    return len(BOD_CLASS_EDGES)

def blind_metrics(blind: pd.DataFrame, preds: pd.DataFrame,
                  target: str) -> dict:
    """R², RMSE, 90% interval coverage, class accuracy + abstention rate —
    computed ONLY on the blind holdout (Appendix P definitions)."""
    y = blind[f"lab_{target}"].values
    p50 = preds[f"{target}_p50"].values
    p10, p90 = preds[f"{target}_p10"].values, preds[f"{target}_p90"].values
    ss_res = float(np.sum((y - p50) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    m = dict(
        n=int(len(y)),
        r2=1 - ss_res / ss_tot,
        rmse=float(np.sqrt(np.mean((y - p50) ** 2))),
        interval_coverage=float(np.mean((y >= p10) & (y <= p90))),
    )
    if target == "bod":
        cls = [classify_bod(a, b, c, d) for a, b, c, d in
               zip(p10, p50, p90, preds[f"{target}_in_dist"])]
        keep = [i for i, c in enumerate(cls) if c != ABSTAIN]
        m["abstention_rate"] = 1 - len(keep) / len(cls)
        m["class_accuracy"] = float(np.mean(
            [CLASS_NAMES[_band(y[i])] == cls[i] for i in keep])) if keep else None
    return m

def lineage(product: str, refs: dict) -> dict:
    payload = dict(product=product, provenance="SYNTHETIC-DEMO", **refs)
    h = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                  default=str).encode()).hexdigest()[:16]
    return {"lineage_id": h, **payload}

def build_alerts(fused_today: pd.DataFrame, forecast: pd.DataFrame,
                 anomalies: pd.DataFrame, today: pd.Timestamp) -> list[dict]:
    """Tiered alerts with the claim-ladder basis field (blueprint schema G)."""
    alerts = []
    recent = anomalies[anomalies.date >= today - pd.Timedelta(days=3)] \
             if len(anomalies) else anomalies
    for _, a in (recent.iterrows() if len(recent) else []):
        alerts.append(dict(tier="warning", basis="Detected",
                           reach_id=a.reach_id, date=str(pd.Timestamp(a.date).date()),
                           text=f"Proxy anomaly ({a.signal}, z={a.zscore:.1f}) — "
                                f"co-sampling recommended"))
    ft = fused_today.set_index("reach_id")
    for rid, r in ft.iterrows():
        if r.fused_p10 > BOD_CLASS_EDGES[1]:      # even P10 above bathing class
            alerts.append(dict(tier="likely-violation", basis="Estimated",
                               reach_id=rid, date=str(pd.Timestamp(today).date()),
                               text=f"Fused BOD P10 {r.fused_p10:.1f} mg/L exceeds "
                                    f"bathing-class limit (approximate criteria — verify)"))
    worst = (forecast[forecast.bod_p50 > BOD_CLASS_EDGES[2]]
             .sort_values("bod_p50", ascending=False).head(3))
    for _, f in worst.iterrows():
        alerts.append(dict(tier="advisory", basis="Estimated",
                           reach_id=f.reach_id, date=str(pd.Timestamp(f.date).date()),
                           text=f"Forecast day {int(f.day)}: BOD P50 "
                                f"{f.bod_p50:.1f} (P90 {f.bod_p90:.1f}) mg/L"))
    order = {"likely-violation": 0, "warning": 1, "advisory": 2}
    return sorted(alerts, key=lambda a: order[a["tier"]])[:12]
