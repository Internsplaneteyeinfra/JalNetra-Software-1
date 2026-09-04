#!/usr/bin/env python3
"""
run_demo.py — JalNetra plug-and-play demonstration.

Runs the FULL pipeline end to end on synthetic data:
  C1 ingest → match-ups → C2 train + blind-validate → retrieve → C3 fuse →
  C4 forecast + hindcast skill → C5 alerts/metrics → C6 export + serve.

Usage:
    python run_demo.py            # run pipeline, write dashboard data, serve
    python run_demo.py --no-serve # run pipeline only

HONESTY NOTE: all inputs are simulated (see jalnetra/synthetic.py). Metrics
printed below validate the software pipeline, not real-world accuracy.
"""
import sys, os, shutil
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jalnetra.config import TARGETS
from jalnetra.ingestion import DemoSource, build_matchups, FEATURES
from jalnetra.retrieval import train_all, detect_anomalies
from jalnetra.fusion import run_twin
from jalnetra.config import FORECAST_DAYS
from jalnetra.forecast import make_forecast, hindcast_skill
from jalnetra.trust import blind_metrics, build_alerts, lineage
from jalnetra.dashboard import export_json, serve
from jalnetra.synthetic import rainfall_forecast

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.join(HERE, "dashboard")

def main(serve_after=True):
    print("=" * 64)
    print("JalNetra demo pipeline — SYNTHETIC DATA (see honesty note)")
    print("=" * 64)

    # ---- C1: ingest ---------------------------------------------------
    src = DemoSource()
    scenes, anchors, labs = src.observations(), src.streams(), src.labs()
    today = pd.Timestamp(src.df.date.max())
    print(f"[C1] scenes(observed reach-days)={len(scenes)}  "
          f"anchor-days={len(anchors)}  lab records={len(labs)}")

    matchups = build_matchups(scenes, labs)
    print(f"[C1] match-ups built: {len(matchups)} (±2-day co-location)")

    # ---- C2: train with honest splits, blind-validate ------------------
    holdout_reaches = ["R08"]                    # entire station withheld
    holdout_after = today - pd.Timedelta(days=180)   # out-of-time window
    models, blind, ntr, ncal = train_all(matchups, holdout_reaches, holdout_after)
    print(f"[C2] trained on {ntr} pairs, conformal-calibrated on {ncal}, "
          f"blind holdout {len(blind)} pairs (reach {holdout_reaches[0]} + last 180 d)")

    metrics = {}
    for t in TARGETS:
        preds = models[t].predict(blind)
        metrics[t] = blind_metrics(blind, preds, t)
        m = metrics[t]
        extra = (f"  class_acc={m['class_accuracy']:.2f} "
                 f"abstain={m['abstention_rate']:.2f}" if t == "bod" else "")
        print(f"[C2] BLIND {t.upper()}: R2={m['r2']:.3f} RMSE={m['rmse']:.2f} "
              f"coverage(90%)={m['interval_coverage']:.2f}{extra}")
    print("     ^ synthetic-data figures: they validate the pipeline, "
          "not real-world skill.")

    # retrieve over all observed scene-days for the twin
    sat_all = scenes.copy()
    for t in TARGETS:
        sat_all = sat_all.join(models[t].predict(scenes))

    # anomaly radar (Detected tier)
    anomalies = detect_anomalies(scenes)
    print(f"[C2] anomaly radar: {len(anomalies)} detections over history")

    # ---- C3: fuse -------------------------------------------------------
    import numpy as np
    daily = src.df[["date", "reach_id", "discharge", "anchored"]].copy()
    # demo anchor pseudo-observation = noisy truth (stands in for a real
    # anchor's locally calibrated sensor→BOD model)
    rng = np.random.default_rng(1)
    daily["anchor_val"] = (src.df.true_bod +
                           rng.normal(0, 0.5, len(src.df))).clip(0.2)
    fused = run_twin(daily, sat_all)
    print(f"[C3] twin fused {fused.date.nunique()} days × "
          f"{fused.reach_id.nunique()} reaches")

    # ---- C4: forecast + hindcast skill ---------------------------------
    rain_hist = (src.df.groupby("date").monsoon.first() * 12 + 1)
    truth_df = src.df[["date", "reach_id", "true_bod"]]
    skill = hindcast_skill(fused, daily, rain_hist, truth_df, n_issues=16)
    print("[C4] hindcast skill vs persistence (day: skill, coverage):")
    for _, r in skill.iterrows():
        print(f"      day {int(r.day)}: {r.skill_vs_persistence:+.2f}, "
              f"cov {r.interval_coverage:.2f}")

    fdates = pd.date_range(today, periods=FORECAST_DAYS + 1)[1:]
    rain_fc = rainfall_forecast(fdates)
    f0 = fused[fused.date == today]
    disch0 = float(daily[daily.date == today].discharge.iloc[0])
    forecast = make_forecast(f0, disch0, rain_fc, dry_days=6, issued=today)
    print(f"[C4] 7-day forecast issued {today.date()} for "
          f"{forecast.reach_id.nunique()} reaches")

    # ---- C5: alerts + lineage ------------------------------------------
    alerts = build_alerts(f0, forecast, anomalies, today)
    lin = lineage("dashboard_export", dict(models={t: "gbm+conformal" for t in TARGETS},
                                           issued=str(today.date())))
    print(f"[C5] alerts={len(alerts)}  lineage_id={lin['lineage_id']}")

    # ---- C6: export + serve ---------------------------------------------
    out = os.path.join(DASH, "dashboard_data.json")
    export_json(out, today=today, fused=fused, forecast=forecast,
                skill=skill, metrics=metrics, alerts=alerts,
                anomalies=anomalies)
    print(f"[C6] dashboard data → {out}")

    if serve_after:
        serve(DASH, port=8000)

if __name__ == "__main__":
    main(serve_after="--no-serve" not in sys.argv)
