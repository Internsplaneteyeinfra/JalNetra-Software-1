# JalNetra — plug-and-play reference implementation

Satellite + AI estimation of river COD/BOD with honest uncertainty, a fused
daily river state, a 7-day forecast, and a ready-to-use dashboard.
Implements the architecture of the *JalNetra Technical Blueprint*
(components C1–C6) at demonstration scale.

## Quickstart (60 seconds)

```bash
pip install numpy pandas scikit-learn   # the only dependencies
python run_demo.py                      # runs pipeline + serves dashboard
# → open http://localhost:8000/dashboard.html
```

No credentials, no downloads, no configuration. `python run_demo.py
--no-serve` runs the pipeline only. The dashboard also works standalone:
just open `dashboard/dashboard.html` in a browser (it carries embedded
demo data and auto-upgrades to fresh `dashboard_data.json` when served).

## Honesty notice — read before demoing

**All inputs are synthetic.** `jalnetra/synthetic.py` simulates a
Kanpur→Varanasi stretch (14 reaches, 3 years) with realistic structure:
monsoon discharge and clouds, first-flush spikes, outfall-driven pollution,
optical proxies causally (and noisily) linked to COD/BOD, and random
discharge events. Every metric the pipeline prints or the dashboard shows
therefore validates the **software**, not real-world accuracy. Real skill
can only be established against real scenes and laboratory co-samples.
CPCB-style class bands in `config.py` are approximate — verify current
criteria before any regulatory use.

## What actually runs (pipeline → blueprint map)

| Stage | Module | Blueprint |
|---|---|---|
| Ingestion + match-up factory | `ingestion.py` | C1 (§13.1) |
| Proxies → COD/BOD, conformal 90% intervals, OOD abstention, class mapper, anomaly radar | `retrieval.py` | C2 (§13.2–13.4) |
| Fused daily state (transport prior + Kalman updates, variance grows through cloud gaps) | `fusion.py` | C3 (§13.5) |
| 7-day ensemble forecast + hindcast skill vs persistence | `forecast.py` | C4 (§14.4–14.5) |
| Blind metrics (Appendix P definitions), lineage records, tiered alerts | `trust.py` | C5 |
| JSON export + stdlib server + HTML dashboard | `dashboard.py`, `dashboard/` | C6 |

Honest-split training is enforced in `retrieval.train_all`: an entire
station reach **and** the last 180 days are quarantined as blind holdout;
printed metrics come only from that blind set. The claim ladder
(Detected / Classified / Estimated, with NA abstention) is carried through
the models, the alerts, the JSON and the UI.

## Plugging in real data (production path)

The demo source implements three interfaces in `ingestion.py`; production
swaps implementations without touching downstream code:

- `SceneSource.observations()` → per-scene reach proxies from real
  Sentinel-2 via a STAC client + an aquatic atmospheric-correction
  processor (benchmark per blueprint Phase 0 before choosing one).
- `AnchorSource.streams()` → telemetry from real multi-parameter sondes.
- `LabSource.records()` → CPCB/SPCB laboratory records under data
  agreements (the match-up factory in `build_matchups` is already real).

Then: learn transport/decay parameters per reach from hindcasts
(`fusion.py` constants are demo values), feed real NWP + gauge data to
`forecast.py`, and put the server behind real auth. The module docstrings
mark every demo shortcut explicitly.

## Files

```
run_demo.py                end-to-end pipeline + server
jalnetra/config.py         reaches, class bands (approximate — verify), constants
jalnetra/synthetic.py      SYNTHETIC demo data generator (honesty note inside)
jalnetra/ingestion.py      C1: source interfaces + match-up factory
jalnetra/retrieval.py      C2: models, conformal intervals, OOD, classes, anomalies
jalnetra/fusion.py         C3: the twin (1-D assimilation)
jalnetra/forecast.py       C4: ensemble forecast + verification
jalnetra/trust.py          C5: blind metrics, lineage, alerts
jalnetra/dashboard.py      C6: JSON export + server
dashboard/dashboard.html   the dashboard (standalone-capable)
```

License note: reference implementation for the blueprint's owner; no
warranty; not fit for regulatory use without the validation programme the
blueprint specifies.
