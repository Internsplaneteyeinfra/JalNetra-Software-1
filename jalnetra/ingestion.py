"""
jalnetra.ingestion — C1: data sources and the match-up factory.

PLUG-AND-PLAY DESIGN: real deployments implement SceneSource / AnchorSource /
LabSource against actual feeds (Sentinel-2 via a STAC client + aquatic
atmospheric correction, sensor telemetry, laboratory records under agreement).
The demo runs on DemoSource, which wraps the synthetic simulator and exposes
the SAME interface — swapping in real data changes no downstream code.
"""
from __future__ import annotations
from typing import Protocol
import pandas as pd
from . import synthetic

class SceneSource(Protocol):
    def observations(self) -> pd.DataFrame:
        """Satellite-derived proxy observations.
        Columns: date, reach_id, turb, cdom, chla, tss + context columns."""
        ...

class AnchorSource(Protocol):
    def streams(self) -> pd.DataFrame:
        """Continuous ground-sensor proxy streams (daily aggregate in demo)."""
        ...

class LabSource(Protocol):
    def records(self) -> pd.DataFrame:
        """Laboratory COD/BOD records for training/validation match-ups."""
        ...

FEATURES = ["turb", "cdom", "chla", "tss", "doy", "monsoon", "discharge",
            "width_m", "outfall_density", "km_mid"]

class DemoSource:
    """Synthetic implementation of all three sources. DEMO ONLY — see
    synthetic.py honesty note."""
    def __init__(self, seed: int | None = None, days: int | None = None,
                 end: str | None = None):
        kw = {}
        if seed is not None: kw["seed"] = seed
        if days is not None: kw["days"] = days
        if end is not None: kw["end"] = end
        self.df = synthetic.generate_history(**kw)

    def observations(self) -> pd.DataFrame:
        return self.df[self.df.sat_seen].copy()

    def streams(self) -> pd.DataFrame:
        return self.df[self.df.anchored].copy()

    def labs(self) -> pd.DataFrame:
        # monthly lab sampling at 6 station reaches (mimics CPCB cadence)
        d = self.df[self.df.anchored & (self.df.date.dt.day.isin((7, 22)))]
        return d[["date", "reach_id", "true_bod", "true_cod"]].rename(
            columns={"true_bod": "lab_bod", "true_cod": "lab_cod"})

def build_matchups(scenes: pd.DataFrame, labs: pd.DataFrame,
                   max_offset_days: int = 2) -> pd.DataFrame:
    """Co-locate lab records with satellite observations (±N days), keeping
    the time offset as a feature — the blueprint's match-up factory."""
    s = scenes.copy(); s["sdate"] = s.date
    out = []
    for _, lab in labs.iterrows():
        win = s[(s.reach_id == lab.reach_id) &
                (abs((s.sdate - lab.date).dt.days) <= max_offset_days)]
        if len(win):
            m = win.iloc[(win.sdate - lab.date).abs().dt.days.argmin()].copy()
            m["dt_offset"] = abs((m.sdate - lab.date).days)
            m["lab_bod"], m["lab_cod"] = lab.lab_bod, lab.lab_cod
            out.append(m)
    mu = pd.DataFrame(out).reset_index(drop=True)
    return mu
