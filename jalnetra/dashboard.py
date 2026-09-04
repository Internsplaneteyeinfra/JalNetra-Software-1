"""
jalnetra.dashboard — C6: export dashboard_data.json + serve the dashboard.
Stdlib server only, so the whole product stays plug-and-play.
"""
from __future__ import annotations
import json, os, functools, http.server, socketserver
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .config import active_reaches, CLASS_COLORS, CLASS_LABELS
from .retrieval import classify_bod

def _r(x, n=2):
    return None if x is None else round(float(x), n)

def _diurnal(hour: int) -> float:
    """Hourly variation around daily mean (peak ~14:00)."""
    return 1.0 + 0.10 * np.sin(2 * np.pi * (hour - 14) / 24)

def _expand_hourly(dates, p50, p10, p90, support):
    """Expand daily arrays to hourly timestamps with diurnal modulation."""
    h_dates, h_p50, h_p10, h_p90, h_sup = [], [], [], [], []
    for di, d in enumerate(dates):
        f_base = float(p50[di])
        f_lo = float(p10[di])
        f_hi = float(p90[di])
        sup = support[di]
        day = pd.Timestamp(d).normalize()
        for hour in range(24):
            f = _diurnal(hour)
            ts = day + pd.Timedelta(hours=hour)
            h_dates.append(ts.strftime("%Y-%m-%dT%H:%M"))
            h_p50.append(_r(f_base * f))
            h_p10.append(_r(f_lo * f))
            h_p90.append(_r(f_hi * f))
            h_sup.append(sup)
    return h_dates, h_p50, h_p10, h_p90, h_sup

def _scale_series(p50, p10, p90, ratio):
    """Apply per-day COD/BOD ratio to uncertainty bands."""
    r = np.asarray(ratio, dtype=float)
    return ([_r(v * rv) for v, rv in zip(p50, r)],
            [_r(v * rv) for v, rv in zip(p10, r)],
            [_r(v * rv) for v, rv in zip(p90, r)])

def export_json(path: str, *, today: pd.Timestamp, fused: pd.DataFrame,
                forecast: pd.DataFrame, skill: pd.DataFrame,
                metrics: dict, alerts: list, anomalies: pd.DataFrame,
                history_days: int = 90, meta: dict | None = None,
                truth_df: pd.DataFrame | None = None,
                hourly: bool = True) -> dict:
    reaches = active_reaches()
    hist_start = today - pd.Timedelta(days=history_days)
    f = fused[fused.date >= hist_start].copy()

    ratio_lookup = {}
    if truth_df is not None and {"date", "reach_id", "true_bod", "true_cod"}.issubset(truth_df.columns):
        tr = truth_df.groupby(["date", "reach_id"], as_index=False).agg(
            true_bod=("true_bod", "first"), true_cod=("true_cod", "first"))
        tr["ratio"] = (tr.true_cod / tr.true_bod.clip(lower=0.1)).clip(1.2, 5.0)
        ratio_lookup = {(pd.Timestamp(r.date), r.reach_id): float(r.ratio)
                        for r in tr.itertuples()}

    reach_objs = []
    for r in reaches:
        g = f[f.reach_id == r.reach_id].sort_values("date")
        t = g[g.date == today].iloc[0]
        cls = classify_bod(t.fused_p10, t.fused_p50, t.fused_p90, True)
        fc = forecast[forecast.reach_id == r.reach_id].sort_values("day")

        ratios = [ratio_lookup.get((pd.Timestamp(d), r.reach_id), 2.6) for d in g.date]
        fc_ratio = ratio_lookup.get((pd.Timestamp(today), r.reach_id), 2.6)

        bod_dates = [str(d.date()) for d in g.date]
        bod_p50 = [_r(v) for v in g.fused_p50]
        bod_p10 = [_r(v) for v in g.fused_p10]
        bod_p90 = [_r(v) for v in g.fused_p90]
        bod_sup = list(g.obs_support)

        cod_p50, cod_p10, cod_p90 = _scale_series(bod_p50, bod_p10, bod_p90, ratios)

        fc_dates = [str(d.date()) for d in fc.date]
        fc_bod_p50 = [_r(v) for v in fc.bod_p50]
        fc_bod_p10 = [_r(v) for v in fc.bod_p10]
        fc_bod_p90 = [_r(v) for v in fc.bod_p90]
        fc_ratios = [fc_ratio] * len(fc_dates)
        fc_cod_p50, fc_cod_p10, fc_cod_p90 = _scale_series(
            fc_bod_p50, fc_bod_p10, fc_bod_p90, fc_ratios)

        if hourly:
            h_dates, h_p50, h_p10, h_p90, h_sup = _expand_hourly(
                bod_dates, bod_p50, bod_p10, bod_p90, bod_sup)
            h_ratios = []
            for ratio in ratios:
                h_ratios.extend([ratio] * 24)
            h_cod_p50, h_cod_p10, h_cod_p90 = _scale_series(
                h_p50, h_p10, h_p90, h_ratios)

            fh_dates, fh_p50, fh_p10, fh_p90, _fh_sup = _expand_hourly(
                fc_dates, fc_bod_p50, fc_bod_p10, fc_bod_p90,
                ["forecast"] * len(fc_dates))
            fh_ratios = [fc_ratio] * len(fh_dates)
            fh_cod_p50, fh_cod_p10, fh_cod_p90 = _scale_series(
                fh_p50, fh_p10, fh_p90, fh_ratios)

            hist_block = dict(
                dates=h_dates, p50=h_p50, p10=h_p10, p90=h_p90, support=h_sup,
                cod_p50=h_cod_p50, cod_p10=h_cod_p10, cod_p90=h_cod_p90)
            fc_block = dict(
                dates=fh_dates, p50=fh_p50, p10=fh_p10, p90=fh_p90,
                cod_p50=fh_cod_p50, cod_p10=fh_cod_p10, cod_p90=fh_cod_p90)
            today_cod_p50 = _r(float(t.fused_p50) * fc_ratio)
            today_cod_p10 = _r(float(t.fused_p10) * fc_ratio)
            today_cod_p90 = _r(float(t.fused_p90) * fc_ratio)
        else:
            hist_block = dict(
                dates=bod_dates, p50=bod_p50, p10=bod_p10, p90=bod_p90, support=bod_sup,
                cod_p50=cod_p50, cod_p10=cod_p10, cod_p90=cod_p90)
            fc_block = dict(
                dates=fc_dates, p50=fc_bod_p50, p10=fc_bod_p10, p90=fc_bod_p90,
                cod_p50=fc_cod_p50, cod_p10=fc_cod_p10, cod_p90=fc_cod_p90)
            today_cod_p50 = _r(float(t.fused_p50) * fc_ratio)
            today_cod_p10 = _r(float(t.fused_p10) * fc_ratio)
            today_cod_p90 = _r(float(t.fused_p90) * fc_ratio)

        reach_objs.append(dict(
            id=r.reach_id, name=r.name, km=[r.km_start, r.km_end],
            width_m=r.mean_width_m,
            villages=getattr(r, "villages", None),
            today=dict(
                p10=_r(t.fused_p10), p50=_r(t.fused_p50), p90=_r(t.fused_p90),
                cod_p10=today_cod_p10, cod_p50=today_cod_p50, cod_p90=today_cod_p90,
                cls=cls, support=t.obs_support, tier="Estimated"),
            history=hist_block,
            forecast=fc_block,
        ))

    hist_len = len(reach_objs[0]["history"]["dates"])
    fc_len = len(reach_objs[0]["forecast"]["dates"])

    data = dict(
        generated=str(today.date()),
        data_refreshed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        demo_notice=("SYNTHETIC DEMO DATA — this instance runs the full JalNetra "
                     "pipeline on simulated inputs. Skill figures validate the "
                     "software, not real-world accuracy."),
        class_colors=CLASS_COLORS,
        class_labels=CLASS_LABELS,
        bod_edges=[2.0, 3.0, 6.0, 10.0],
        timeline_resolution="hourly" if hourly else "daily",
        reaches=reach_objs,
        alerts=alerts,
        skill=[{k: _r(v, 3) if isinstance(v, float) else v
                for k, v in row.items()} for row in skill.to_dict("records")],
        blind_metrics={k: ({kk: _r(vv, 3) for kk, vv in v.items()}
                           if isinstance(v, dict) else v)
                       for k, v in metrics.items()},
        anomaly_count_30d=int(len(anomalies[anomalies.date >= today -
                              pd.Timedelta(days=30)])) if len(anomalies) else 0,
    )
    if hourly:
        data["history_hours"] = hist_len
        data["forecast_hours"] = fc_len
    else:
        data["history_days"] = hist_len
        data["forecast_days"] = fc_len

    if meta:
        data.update(meta)
        if hourly:
            meta_hours = dict(history_hours=hist_len, forecast_hours=fc_len,
                              timeline_resolution="hourly")
            data.update({k: v for k, v in meta_hours.items() if k not in data})
    with open(path, "w") as fh:
        json.dump(data, fh)
    return data

def serve(directory: str, port: int = 8000):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=directory)
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"\nJalNetra dashboard →  http://localhost:{port}/dashboard.html")
        print("Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
