#!/usr/bin/env python3
"""
run_mithi.py — JalNetra demo for Mithi river (Mumbai).

Runs the full pipeline on synthetic data shaped for the Mumbai stretch,
including villages along the river and the user KML (reach R03, BKC).

Usage:
    python run_mithi.py                          # pipeline + serve on port 8002
    python run_mithi.py --no-serve               # export JSON only
    python run_mithi.py --refresh-minutes 30     # serve + re-export every 30 min

Output:
    dashboard/mithi/dashboard.html
    dashboard/mithi/dashboard_data.json
    dashboard/mithi/river_patch.kml
"""
import argparse
import os
import shutil
import sys
import threading
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import jalnetra.config as cfg

cfg.REACHES_PROVIDER = cfg.mithi_reaches
cfg.FORECAST_DAYS = 14

from jalnetra.config import (  # noqa: E402
    FORECAST_DAYS,
    MITHI_KML_REACH,
    MITHI_LANDMARKS,
    TARGETS,
)
from jalnetra.ingestion import DemoSource, build_matchups, FEATURES
from jalnetra.retrieval import train_all, detect_anomalies
from jalnetra.fusion import run_twin
from jalnetra.forecast import make_forecast, hindcast_skill
from jalnetra.trust import blind_metrics, build_alerts, lineage
from jalnetra.dashboard import export_json, serve
from jalnetra.synthetic import rainfall_forecast

DASH = os.path.join(HERE, "dashboard", "mithi")
KML_SRC = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "3875ff6842ca495e90921823311b1b6f.kml",
)
TODAY_END = "2026-08-18"
HISTORY_DAYS = 90


def run_pipeline(*, quiet=False):
    """Run the full pipeline and write dashboard_data.json."""
    os.makedirs(DASH, exist_ok=True)
    if os.path.isfile(KML_SRC):
        shutil.copy2(KML_SRC, os.path.join(DASH, "river_patch.kml"))
        if not quiet:
            print(f"[setup] KML copied -> {DASH}/river_patch.kml")
    elif not quiet:
        print(f"[setup] KML not found at {KML_SRC} — dashboard still runs")

    if not quiet:
        print("=" * 64)
        print("JalNetra — Mithi (Mumbai) · SYNTHETIC DEMO")
        print("=" * 64)

    src = DemoSource(seed=202, end=TODAY_END)
    scenes, anchors, labs = src.observations(), src.streams(), src.labs()
    today = pd.Timestamp(src.df.date.max())
    if not quiet:
        print(f"[C1] today={today.date()}  scenes={len(scenes)}  "
              f"anchors={len(anchors)}  labs={len(labs)}")

    matchups = build_matchups(scenes, labs)
    holdout_reaches = ["R05"]
    holdout_after = today - pd.Timedelta(days=180)
    models, blind, ntr, ncal = train_all(matchups, holdout_reaches, holdout_after)
    if not quiet:
        print(f"[C2] trained {ntr} pairs, blind holdout {len(blind)}")

    metrics = {}
    for t in TARGETS:
        preds = models[t].predict(blind)
        metrics[t] = blind_metrics(blind, preds, t)

    sat_all = scenes.copy()
    for t in TARGETS:
        sat_all = sat_all.join(models[t].predict(scenes))
    anomalies = detect_anomalies(scenes)

    import numpy as np

    daily = src.df[["date", "reach_id", "discharge", "anchored"]].copy()
    rng = np.random.default_rng(1)
    daily["anchor_val"] = (src.df.true_bod + rng.normal(0, 0.5, len(src.df))).clip(0.2)
    fused = run_twin(daily, sat_all)
    if not quiet:
        print(f"[C3] fused {fused.date.nunique()} days × {fused.reach_id.nunique()} reaches")

    rain_hist = src.df.groupby("date").monsoon.first() * 12 + 1
    truth_df = src.df[["date", "reach_id", "true_bod"]]
    skill = hindcast_skill(fused, daily, rain_hist, truth_df, n_issues=16)

    fdates = pd.date_range(today, periods=FORECAST_DAYS + 1)[1:]
    rain_fc = rainfall_forecast(fdates, seed=209)
    f0 = fused[fused.date == today]
    disch0 = float(daily[daily.date == today].discharge.iloc[0])
    forecast = make_forecast(f0, disch0, rain_fc, dry_days=6, issued=today)
    if not quiet:
        print(f"[C4] {FORECAST_DAYS}-day forecast through {fdates[-1].date()}")

    alerts = build_alerts(f0, forecast, anomalies, today)
    lin = lineage("mithi_export", dict(models={t: "gbm+conformal" for t in TARGETS},
                                       issued=str(today.date())))
    if not quiet:
        print(f"[C5] alerts={len(alerts)}  lineage={lin['lineage_id']}")

    hist = fused[fused.date >= today - pd.Timedelta(days=HISTORY_DAYS)]
    hist_dates = sorted(hist.date.unique())
    if not quiet:
        print(f"[data] history {hist_dates[0].date()} -> {hist_dates[-1].date()} "
              f"({len(hist_dates)} days, exported hourly)")
        print(f"[data] forecast {forecast.date.min().date()} -> "
              f"{forecast.date.max().date()} ({FORECAST_DAYS} days, exported hourly)")

    meta = dict(
        tagline="River organic-pollution twin · Mithi · Kalina → Mahim Creek",
        river_name="Mithi",
        landmarks=[[n, km] for n, km in MITHI_LANDMARKS],
        kml_reach_id=MITHI_KML_REACH,
        history_days=HISTORY_DAYS,
        forecast_days=FORECAST_DAYS,
        demo_notice=(
            "MITHI SYNTHETIC DEMO — Kalina–Vakola to Mahim Creek "
            "(6 reaches). KML patch = reach R03 (BKC–Kurla). "
            "Figures validate software, not real-world accuracy."
        ),
    )
    out = os.path.join(DASH, "dashboard_data.json")
    export_json(out, today=today, fused=fused, forecast=forecast,
                skill=skill, metrics=metrics, alerts=alerts,
                anomalies=anomalies, history_days=HISTORY_DAYS, meta=meta,
                truth_df=src.df[["date", "reach_id", "true_bod", "true_cod"]],
                hourly=True)
    if not quiet:
        print(f"[C6] dashboard data -> {out}")
    return out


def _refresh_loop(minutes: int):
    while True:
        time.sleep(minutes * 60)
        try:
            print(f"\n[refresh] regenerating dashboard_data.json …")
            run_pipeline(quiet=True)
            print("[refresh] done — dashboard will pick up new JSON within 30 min "
                  "or when the tab becomes visible")
        except Exception as exc:
            print(f"[refresh] failed: {exc}")


def main(*, serve_after=True, refresh_minutes=0, port=8002):
    run_pipeline()
    if refresh_minutes > 0:
        threading.Thread(target=_refresh_loop, args=(refresh_minutes,),
                         daemon=True).start()
        print(f"[refresh] background export every {refresh_minutes} min")
    if serve_after:
        serve(DASH, port=port)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mithi JalNetra dashboard pipeline")
    ap.add_argument("--no-serve", action="store_true",
                    help="export JSON only, do not start HTTP server")
    ap.add_argument("--refresh-minutes", type=int, default=0, metavar="N",
                    help="re-run pipeline every N minutes (use with server)")
    ap.add_argument("--port", type=int, default=8002, help="HTTP port (default 8002)")
    args = ap.parse_args()
    main(serve_after=not args.no_serve,
         refresh_minutes=args.refresh_minutes,
         port=args.port)
