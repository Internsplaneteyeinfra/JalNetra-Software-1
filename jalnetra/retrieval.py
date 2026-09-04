"""
jalnetra.retrieval — C2: proxies → COD/BOD with honest uncertainty.

Implements the blueprint's claim ladder:
  Detected   — anomaly events (highest confidence),
  Classified — CPCB-style band with abstention,
  Estimated  — mg/L with split-conformal 90% intervals, OOD-gated.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from .config import (BOD_CLASS_EDGES, CLASS_NAMES, ABSTAIN, CONFORMAL_ALPHA,
                     OOD_CONTAMINATION, SEED, TARGETS)
from .ingestion import FEATURES

class RetrievalModel:
    """One target (bod|cod): GBM in log space + split conformal + OOD gate."""
    def __init__(self, target: str, seed: int = SEED):
        self.target = target
        self.model = GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=seed)
        self.ood = IsolationForest(contamination=OOD_CONTAMINATION,
                                   random_state=seed)
        self.q_lo = self.q_hi = None

    # -- training uses spatially/temporally honest splits upstream (see demo)
    def fit(self, train: pd.DataFrame, calib: pd.DataFrame):
        X, y = train[FEATURES].values, np.log1p(train[f"lab_{self.target}"].values)
        self.model.fit(X, y)
        self.ood.fit(X)
        # split-conformal on the calibration fold (absolute residuals, log space)
        Xc = calib[FEATURES].values
        res = np.abs(np.log1p(calib[f"lab_{self.target}"].values)
                     - self.model.predict(Xc))
        k = int(np.ceil((1 - CONFORMAL_ALPHA) * (len(res) + 1))) - 1
        self.qhat = np.sort(res)[min(k, len(res) - 1)]
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].values
        mu = self.model.predict(X)
        in_dist = self.ood.predict(X) == 1
        out = pd.DataFrame(index=df.index)
        out[f"{self.target}_p50"] = np.expm1(mu)
        out[f"{self.target}_p10"] = np.expm1(mu - self.qhat)
        out[f"{self.target}_p90"] = np.expm1(mu + self.qhat)
        out[f"{self.target}_in_dist"] = in_dist
        return out

def classify_bod(p10: float, p50: float, p90: float, in_dist: bool) -> str:
    """Interval-aware class with abstention (blueprint C2.4 rule): publish the
    P50 class only when the P10-P90 interval lies fully or MAJORLY (>=50% of
    its length) within that class band; abstain otherwise or when the input is
    out-of-distribution."""
    if not in_dist:
        return ABSTAIN
    def band(v):
        for i, e in enumerate(BOD_CLASS_EDGES):
            if v <= e:
                return i
        return len(BOD_CLASS_EDGES)
    b = band(p50)
    lo_edge = 0.0 if b == 0 else BOD_CLASS_EDGES[b - 1]
    hi_edge = BOD_CLASS_EDGES[b] if b < len(BOD_CLASS_EDGES) else float("inf")
    width = max(p90 - p10, 1e-9)
    overlap = max(0.0, min(p90, hi_edge) - max(p10, lo_edge))
    return CLASS_NAMES[b] if overlap / width >= 0.5 else ABSTAIN

def detect_anomalies(series: pd.DataFrame, z_thresh: float = 2.8,
                     window: int = 21) -> pd.DataFrame:
    """Detected tier: robust z-score change detection on the sewage-linked
    proxies (turb, cdom) per reach. Works on proxies directly, so it stays
    live even when quantification abstains."""
    ev = []
    for rid, g in series.sort_values("date").groupby("reach_id"):
        for col in ("turb", "cdom"):
            x = g[col].values
            med = pd.Series(x).rolling(window, min_periods=7).median().values
            mad = pd.Series(np.abs(x - np.where(np.isnan(med), np.nanmedian(x), med))) \
                    .rolling(window, min_periods=7).median().values
            z = (x - med) / np.maximum(mad * 1.4826, 1e-6)
            idx = np.where(z > z_thresh)[0]
            for i in idx:
                ev.append(dict(date=g.date.iloc[i], reach_id=rid,
                               signal=col, zscore=float(z[i])))
    ev = pd.DataFrame(ev)
    if len(ev):
        ev = (ev.sort_values("zscore", ascending=False)
                .drop_duplicates(subset=["date", "reach_id"]))
    return ev

def train_all(matchups: pd.DataFrame, holdout_reaches: list[str],
              holdout_after: pd.Timestamp):
    """Honest split protocol (blueprint 13.3.1, demo-scale):
    station-level holdout (entire reaches withheld) + out-of-time test window.
    Returns models, blind test frame."""
    mu = matchups.copy()
    if len(mu) == 0:
        raise ValueError(
            "No satellite–lab matchups available to train BOD/COD models."
        )

    blind = mu[mu.reach_id.isin(holdout_reaches) | (mu.date >= holdout_after)]
    dev = mu.drop(blind.index)
    # Single-reach / small catalogues can put every row in blind — fall back
    # to time-only holdout so GradientBoosting always has train samples.
    if len(dev) < 5:
        blind = mu[mu.date >= holdout_after]
        dev = mu.drop(blind.index)
    if len(dev) < 5:
        mu_sorted = mu.sort_values("date")
        cut = max(2, int(len(mu_sorted) * 0.7))
        train_pool = mu_sorted.iloc[:cut]
        blind = mu_sorted.iloc[cut:]
        if len(blind) == 0:
            blind = mu_sorted.iloc[-max(1, len(mu_sorted) // 5) :]
            train_pool = mu_sorted.drop(blind.index)
        calib = train_pool.sample(
            frac=min(0.3, max(0.2, 2 / max(len(train_pool), 1))),
            random_state=SEED,
        )
        if len(calib) == 0:
            calib = train_pool.iloc[[-1]]
        train = train_pool.drop(calib.index)
        if len(train) == 0:
            train = calib.copy()
    else:
        calib = dev.sample(frac=0.3, random_state=SEED)
        if len(calib) == 0:
            calib = dev.iloc[[-1]]
        train = dev.drop(calib.index)
        if len(train) == 0:
            train = calib.copy()

    models = {t: RetrievalModel(t).fit(train, calib) for t in TARGETS}
    return models, blind, len(train), len(calib)
