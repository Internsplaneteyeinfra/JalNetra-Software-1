"""
jalnetra.config — shared constants and configuration.

HONESTY NOTE (accuracy charter): class band thresholds below follow CPCB-style
designated-best-use logic but are APPROXIMATE and must be verified against the
current CPCB criteria before any regulatory use.
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------- class bands
# BOD-keyed water classes (mg/L). Approximate CPCB-style bands — VERIFY.
BOD_CLASS_EDGES = [2.0, 3.0, 6.0, 10.0]          # A | B | C | D | E
CLASS_NAMES = ["A", "B", "C", "D", "E"]
CLASS_LABELS = {
    "A": "Drinking source (conventional treatment not required) — verify criteria",
    "B": "Outdoor bathing",
    "C": "Drinking source with treatment",
    "D": "Wildlife / fisheries",
    "E": "Irrigation / industrial cooling only",
}
CLASS_COLORS = {"A": "#2E7DD1", "B": "#3AA76D", "C": "#E3C51E",
                "D": "#E08A2E", "E": "#D14B3A", "NA": "#6B7A7F"}
ABSTAIN = "NA"  # published when evidence is insufficient (claim-ladder abstention)

# claim ladder — the three-verb vocabulary from the blueprint
TIER_DETECTED, TIER_CLASSIFIED, TIER_ESTIMATED = "Detected", "Classified", "Estimated"

# ------------------------------------------------------------------ pilot reaches
@dataclass
class Reach:
    reach_id: str
    name: str
    km_start: float            # chainage from Kanpur barrage (demo geometry)
    km_end: float
    mean_width_m: float
    outfall_density: float     # relative 0..1 (drives synthetic demo pollution)
    upstream: str | None = None

# Swappable reach catalogue — run scripts set REACHES_PROVIDER before pipeline.
REACHES_PROVIDER = None  # None → pilot_reaches()


def active_reaches() -> list[Reach]:
    """Return the currently selected reach catalogue."""
    fn = REACHES_PROVIDER or pilot_reaches
    return fn()


def pilot_reaches() -> list[Reach]:
    """Kanpur→Varanasi demo stretch, ~350 km, 14 reaches (~25 km each).
    Geometry and outfall densities are ILLUSTRATIVE for the demo."""
    spec = [
        ("R01", "Kanpur Barrage–Jajmau",       0,   25, 520, 0.95),
        ("R02", "Jajmau–Wazidpur",            25,   50, 560, 0.80),
        ("R03", "Wazidpur–Fatehpur N",        50,   75, 600, 0.35),
        ("R04", "Fatehpur N–Fatehpur S",      75,  100, 640, 0.30),
        ("R05", "Fatehpur S–Khaga",          100,  125, 640, 0.25),
        ("R06", "Khaga–Sirathu",             125,  150, 700, 0.22),
        ("R07", "Sirathu–Prayagraj W",       150,  175, 740, 0.40),
        ("R08", "Prayagraj W–Sangam",        175,  200, 820, 0.70),
        ("R09", "Sangam–Jhusi",              200,  225, 900, 0.55),
        ("R10", "Jhusi–Sirsa",               225,  250, 880, 0.30),
        ("R11", "Sirsa–Mirzapur",            250,  275, 860, 0.35),
        ("R12", "Mirzapur–Chunar",           275,  300, 840, 0.30),
        ("R13", "Chunar–Ramnagar",           300,  325, 860, 0.45),
        ("R14", "Ramnagar–Varanasi ghats",   325,  350, 900, 0.75),
    ]
    out, prev = [], None
    for rid, nm, a, b, w, od in spec:
        out.append(Reach(rid, nm, a, b, w, od, upstream=prev))
        prev = rid
    return out


def mula_mutha_reaches() -> list[Reach]:
    """Mula–Mutha urban stretch: Sangam–Bund Garden → Hadapsar–Manjari.
    ~42 km, 7 reaches. R06 is the user KML patch (~73.86°E)."""
    spec = [
        ("R05", "Sangam–Bund Garden",      0,   6,  68, 0.90, "Shivajinagar, Koregaon Park"),
        ("R06", "Bund Garden–Dhanori",     6,  12,  72, 0.88, "Kalyani Nagar, Dhanori, Yerawada"),
        ("R07", "Dhanori–Wadgaon Sheri",  12,  18,  75, 0.82, "Wadgaon Sheri, Lohegaon"),
        ("R08", "Wadgaon Sheri–Kharadi",  18,  24,  78, 0.78, "Kharadi, Chandan Nagar, Vimannagar"),
        ("R09", "Kharadi–Mundhwa",        24,  30,  80, 0.72, "Mundhwa, Keshav Nagar, Magarpatta"),
        ("R10", "Mundhwa–Hadapsar",       30,  36,  85, 0.65, "Hadapsar, Sadesatranali"),
        ("R11", "Hadapsar–Manjari",       36,  42,  90, 0.45, "Manjari, Shewalwadi"),
    ]
    out, prev = [], None
    for rid, nm, a, b, w, od, villages in spec:
        r = Reach(rid, nm, a, b, w, od, upstream=prev)
        r.villages = villages  # type: ignore[attr-defined]
        out.append(r)
        prev = rid
    return out


MULA_MUTHA_LANDMARKS = [
    ("Sangam", 0),
    ("Dhanori (KML)", 9),
    ("Kharadi", 21),
    ("Hadapsar", 33),
    ("Manjari", 42),
]

MULA_MUTHA_KML_REACH = "R06"


def mithi_reaches() -> list[Reach]:
    """Mithi urban stretch: Kalina → Mahim Creek (Mumbai).
    ~9 km, 6 reaches. User KML is the full river polygon (Mahim to Kalina)."""
    spec = [
        ("R01", "Kalina–Vakola",       0.0, 1.5, 32, 0.72, "Kalina, CST, Vidyanagari"),
        ("R02", "Vakola–BKC",          1.5, 3.0, 38, 0.80, "Vakola, Santa Cruz East"),
        ("R03", "BKC–Kurla",           3.0, 4.5, 42, 0.95, "Bandra-Kurla Complex, Kurla West"),
        ("R04", "Kurla–Dharavi N",     4.5, 6.0, 40, 0.92, "Kurla, Sion Koliwada"),
        ("R05", "Dharavi–Mahim",       6.0, 7.5, 45, 0.90, "Dharavi, Mahim East"),
        ("R06", "Mahim Creek mouth",   7.5, 9.0, 75, 0.78, "Mahim, Bandra West, Mahim Creek"),
    ]
    out, prev = [], None
    for rid, nm, a, b, w, od, villages in spec:
        r = Reach(rid, nm, a, b, w, od, upstream=prev)
        r.villages = villages  # type: ignore[attr-defined]
        out.append(r)
        prev = rid
    return out


MITHI_LANDMARKS = [
    ("Kalina", 0),
    ("BKC", 3.5),
    ("Dharavi", 6.2),
    ("Mahim Creek", 9),
]

MITHI_KML_REACH = "R03"

# --------------------------------------------------------------- model targets
TARGETS = ["bod", "cod"]
CONFORMAL_ALPHA = 0.10        # 90% intervals
OOD_CONTAMINATION = 0.02
FORECAST_DAYS = 7
N_ENSEMBLE = 60

# demo period
DEMO_TRAIN_YEARS = 3          # synthetic history length
SEED = 42
