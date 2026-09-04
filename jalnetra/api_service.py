"""
jalnetra.api_service — BOD/COD pipeline wrapper for KML-upload API.

Reuses the existing demo pipeline without modifying core modules.
"""
from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import jalnetra.config as cfg
from jalnetra.config import (
    MITHI_LANDMARKS,
    MULA_MUTHA_LANDMARKS,
    TARGETS,
)
from jalnetra.dashboard import _expand_hourly, _r, _scale_series, export_json
from jalnetra.forecast import hindcast_skill, make_forecast
from jalnetra.fusion import run_twin
from jalnetra.ingestion import DemoSource, build_matchups
from jalnetra.kml_locator import locate_kml
from jalnetra.retrieval import classify_bod, detect_anomalies, train_all
from jalnetra.synthetic import rainfall_forecast
from jalnetra.trust import blind_metrics, build_alerts

HISTORY_DAYS = 90
FORECAST_DAYS_API = 10


def _api_today() -> pd.Timestamp:
    """Calendar today (local) — forecast is issued from this date."""
    return pd.Timestamp(date.today())

RIVER_DASHBOARD_META: Dict[str, Dict[str, Any]] = {
    "mithi": {
        "tagline": "River organic-pollution twin · Mithi · Kalina → Mahim Creek",
        "river_name": "Mithi",
        "landmarks": [[n, km] for n, km in MITHI_LANDMARKS],
        "demo_notice": (
            "MITHI SYNTHETIC DEMO — Kalina–Vakola to Mahim Creek (6 reaches). "
            "KML patch auto-matched to uploaded polygon. "
            "Figures validate software, not real-world accuracy."
        ),
    },
    "mula_mutha": {
        "tagline": "River organic-pollution twin · Mula–Mutha · Sangam → Manjari",
        "river_name": "Mula–Mutha",
        "landmarks": [[n, km] for n, km in MULA_MUTHA_LANDMARKS],
        "demo_notice": (
            "MULA–MUTHA SYNTHETIC DEMO — Sangam–Bund Garden to Hadapsar–Manjari "
            "(7 reaches). KML patch auto-matched to uploaded polygon. "
            "Figures validate software, not real-world accuracy."
        ),
    },
    "ganga_pilot": {
        "tagline": "River organic-pollution twin · Ganga pilot · Kanpur → Varanasi",
        "river_name": "Ganga",
        "landmarks": [],
        "demo_notice": (
            "GANGA PILOT SYNTHETIC DEMO — Kanpur to Varanasi (14 reaches). "
            "KML patch auto-matched to uploaded polygon. "
            "Figures validate software, not real-world accuracy."
        ),
    },
    "custom": {
        "tagline": None,  # filled from KML river name at runtime
        "river_name": None,
        "landmarks": [],
        "demo_notice": (
            "CUSTOM RIVER SYNTHETIC DEMO — reaches and chainage from KML "
            "bridge / placemark names. Figures validate software, not "
            "real-world accuracy."
        ),
    },
}


def _ratio_lookup(truth_df: pd.DataFrame) -> dict:
    if truth_df is None or not {"date", "reach_id", "true_bod", "true_cod"}.issubset(
        truth_df.columns
    ):
        return {}
    tr = truth_df.groupby(["date", "reach_id"], as_index=False).agg(
        true_bod=("true_bod", "first"), true_cod=("true_cod", "first")
    )
    tr["ratio"] = (tr.true_cod / tr.true_bod.clip(lower=0.1)).clip(1.2, 5.0)
    return {
        (pd.Timestamp(r.date), r.reach_id): float(r.ratio) for r in tr.itertuples()
    }


def _reach_series(
    fused: pd.DataFrame,
    forecast: pd.DataFrame,
    reach_id: str,
    today: pd.Timestamp,
    ratio_lookup: dict,
    *,
    hourly: bool = True,
) -> Dict[str, Any]:
    hist_start = today - pd.Timedelta(days=HISTORY_DAYS)
    g = fused[(fused.reach_id == reach_id) & (fused.date >= hist_start)].sort_values(
        "date"
    )
    if g.empty:
        raise ValueError(f"No fused history for reach {reach_id}.")

    t = g[g.date == today]
    if t.empty:
        t = g.iloc[[-1]]
    t = t.iloc[0]

    fc = forecast[forecast.reach_id == reach_id].sort_values("day")
    ratios = [ratio_lookup.get((pd.Timestamp(d), reach_id), 2.6) for d in g.date]
    fc_ratio = ratio_lookup.get((pd.Timestamp(today), reach_id), 2.6)

    bod_dates = [str(d.date()) for d in g.date]
    bod_p50 = [_r(v) for v in g.fused_p50]
    bod_p10 = [_r(v) for v in g.fused_p10]
    bod_p90 = [_r(v) for v in g.fused_p90]
    cod_p50, cod_p10, cod_p90 = _scale_series(bod_p50, bod_p10, bod_p90, ratios)

    fc_dates = [str(d.date()) for d in fc.date]
    fc_bod_p50 = [_r(v) for v in fc.bod_p50]
    fc_bod_p10 = [_r(v) for v in fc.bod_p10]
    fc_bod_p90 = [_r(v) for v in fc.bod_p90]
    fc_ratios = [fc_ratio] * len(fc_dates)
    fc_cod_p50, fc_cod_p10, fc_cod_p90 = _scale_series(
        fc_bod_p50, fc_bod_p10, fc_bod_p90, fc_ratios
    )

    cls = classify_bod(t.fused_p10, t.fused_p50, t.fused_p90, True)

    if hourly:
        h_dates, h_p50, h_p10, h_p90, h_sup = _expand_hourly(
            bod_dates, bod_p50, bod_p10, bod_p90, list(g.obs_support)
        )
        h_ratios = []
        for ratio in ratios:
            h_ratios.extend([ratio] * 24)
        h_cod_p50, h_cod_p10, h_cod_p90 = _scale_series(
            h_p50, h_p10, h_p90, h_ratios
        )
        fh_dates, fh_p50, fh_p10, fh_p90, _ = _expand_hourly(
            fc_dates,
            fc_bod_p50,
            fc_bod_p10,
            fc_bod_p90,
            ["forecast"] * len(fc_dates),
        )
        fh_ratios = [fc_ratio] * len(fh_dates)
        fh_cod_p50, fh_cod_p10, fh_cod_p90 = _scale_series(
            fh_p50, fh_p10, fh_p90, fh_ratios
        )
        live = dict(
            dates=h_dates,
            bod_p50=h_p50,
            bod_p10=h_p10,
            bod_p90=h_p90,
            cod_p50=h_cod_p50,
            cod_p10=h_cod_p10,
            cod_p90=h_cod_p90,
            support=h_sup,
            resolution="hourly",
        )
        forecast_block = dict(
            dates=fh_dates,
            bod_p50=fh_p50,
            bod_p10=fh_p10,
            bod_p90=fh_p90,
            cod_p50=fh_cod_p50,
            cod_p10=fh_cod_p10,
            cod_p90=fh_cod_p90,
            resolution="hourly",
        )
    else:
        live = dict(
            dates=bod_dates,
            bod_p50=bod_p50,
            bod_p10=bod_p10,
            bod_p90=bod_p90,
            cod_p50=cod_p50,
            cod_p10=cod_p10,
            cod_p90=cod_p90,
            support=list(g.obs_support),
            resolution="daily",
        )
        forecast_block = dict(
            dates=fc_dates,
            bod_p50=fc_bod_p50,
            bod_p10=fc_bod_p10,
            bod_p90=fc_bod_p90,
            cod_p50=fc_cod_p50,
            cod_p10=fc_cod_p10,
            cod_p90=fc_cod_p90,
            resolution="daily",
        )

    return dict(
        today=dict(
            date=str(pd.Timestamp(t.date).date()),
            bod_p10=_r(t.fused_p10),
            bod_p50=_r(t.fused_p50),
            bod_p90=_r(t.fused_p90),
            cod_p10=_r(float(t.fused_p10) * fc_ratio),
            cod_p50=_r(float(t.fused_p50) * fc_ratio),
            cod_p90=_r(float(t.fused_p90) * fc_ratio),
            class_band=cls,
            support=t.obs_support,
            tier="Estimated",
        ),
        live=live,
        forecast=forecast_block,
    )


def bod_cod_pipeline(kml_bytes: bytes) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Run JalNetra pipeline for KML-detected river.

    Returns (api_payload, dashboard_json, river_key).
    """
    location = locate_kml(kml_bytes)
    profile = location.profile
    river_key = profile.key

    cfg.REACHES_PROVIDER = profile.reaches_fn
    cfg.FORECAST_DAYS = FORECAST_DAYS_API

    today = _api_today()
    today_str = today.strftime("%Y-%m-%d")

    src = DemoSource(seed=profile.demo_seed, end=today_str)
    scenes, _anchors, labs = src.observations(), src.streams(), src.labs()

    matchups = build_matchups(scenes, labs)
    if matchups is None or len(matchups) == 0:
        raise ValueError(
            "No satellite–lab matchups for this KML. "
            "Add bridge Point placemarks or a polygon stretch and retry."
        )
    holdout_after = today - pd.Timedelta(days=180)
    n_reaches = len(profile.reaches_fn())
    # Station holdout needs ≥2 reaches; otherwise time holdout alone is used.
    holdout_reaches = (
        [profile.holdout_reach]
        if n_reaches >= 2 and profile.holdout_reach
        else []
    )
    models, blind, _ntr, _ncal = train_all(
        matchups, holdout_reaches, holdout_after
    )

    metrics = {}
    for t in TARGETS:
        if len(blind) == 0:
            metrics[t] = {"n": 0, "r2": None, "rmse": None, "interval_coverage": None}
            continue
        preds = models[t].predict(blind)
        metrics[t] = blind_metrics(blind, preds, t)

    sat_all = scenes.copy()
    for t in TARGETS:
        sat_all = sat_all.join(models[t].predict(scenes))
    anomalies = detect_anomalies(scenes)

    daily = src.df[["date", "reach_id", "discharge", "anchored"]].copy()
    rng = np.random.default_rng(1)
    daily["anchor_val"] = (src.df.true_bod + rng.normal(0, 0.5, len(src.df))).clip(
        0.2
    )
    fused = run_twin(daily, sat_all)

    rain_hist = src.df.groupby("date").monsoon.first() * 12 + 1
    truth_df = src.df[["date", "reach_id", "true_bod"]]
    skill = hindcast_skill(fused, daily, rain_hist, truth_df, n_issues=16)

    fdates = pd.date_range(today, periods=FORECAST_DAYS_API + 1)[1:]
    rain_fc = rainfall_forecast(fdates, seed=profile.rain_seed)
    f0 = fused[fused.date == today]
    if f0.empty:
        last = fused.date.max()
        f0 = fused[fused.date == last]
    disch_today = daily[daily.date == today]
    if disch_today.empty:
        disch_today = daily[daily.date == daily.date.max()]
    disch0 = float(disch_today.discharge.iloc[0])
    forecast = make_forecast(f0, disch0, rain_fc, dry_days=6, issued=today)
    alerts = build_alerts(f0, forecast, anomalies, today)

    ratios = _ratio_lookup(src.df[["date", "reach_id", "true_bod", "true_cod"]])
    series = _reach_series(
        fused, forecast, location.reach_id, today, ratios, hourly=True
    )

    reach = next(r for r in profile.reaches_fn() if r.reach_id == location.reach_id)

    river_meta = RIVER_DASHBOARD_META.get(river_key, {})
    landmarks = river_meta.get("landmarks") or []
    if location.bridges:
        landmarks = [[n, km] for n, km in location.bridges]
    elif getattr(profile, "landmarks", None):
        landmarks = [[n, km] for n, km in profile.landmarks]  # type: ignore[misc]

    meta = {
        "tagline": river_meta.get("tagline")
        or f"River organic-pollution twin · {location.river_name}",
        "river_name": river_meta.get("river_name") or location.river_name,
        "landmarks": landmarks,
        "kml_reach_id": location.reach_id,
        "history_days": HISTORY_DAYS,
        "forecast_days": FORECAST_DAYS_API,
        "demo_notice": river_meta.get(
            "demo_notice",
            "SYNTHETIC DEMO — figures validate software, not real-world accuracy.",
        ),
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
    try:
        dashboard_data = export_json(
            tmp_path,
            today=today,
            fused=fused,
            forecast=forecast,
            skill=skill,
            metrics=metrics,
            alerts=alerts,
            anomalies=anomalies,
            history_days=HISTORY_DAYS,
            meta=meta,
            truth_df=src.df[["date", "reach_id", "true_bod", "true_cod"]],
            hourly=True,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    api_payload = {
        "generated": today_str,
        "data_refreshed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "demo_notice": (
            "SYNTHETIC DEMO DATA — pipeline validates software logic, "
            "not real-world water-quality accuracy."
        ),
        "forecast_from": str(fdates[0].date()),
        "forecast_to": str(fdates[-1].date()),
        "forecast_days": FORECAST_DAYS_API,
        "history_days": HISTORY_DAYS,
        "location": {
            "river": location.river_name,
            "river_key": location.river_key,
            "reach_id": location.reach_id,
            "reach_name": location.reach_name,
            "area_name": location.area_name,
            "chainage_km": location.chainage_km,
            "km_range": list(location.km_range),
            "villages": location.villages,
            "bridges": [
                {"name": n, "chainage_km": km}
                for n, km in (location.bridges or [])
            ],
            "latitude": location.centroid[1],
            "longitude": location.centroid[0],
            "centroid": {"lon": location.centroid[0], "lat": location.centroid[1]},
            "mean_width_m": reach.mean_width_m,
        },
        **series,
    }
    return api_payload, dashboard_data, river_key


def bod_cod_from_kml(kml_bytes: bytes) -> Dict[str, Any]:
    """Run pipeline and return API payload only (backward compatible)."""
    api_payload, _, _ = bod_cod_pipeline(kml_bytes)
    return api_payload
