"""
jalnetra.kml_locator — parse uploaded KML and build river reach / chainage.

Chainage and reach names come from Point placemarks (bridges) in the KML when
present. Each uploaded river gets its own synthetic demo seed from the centroid
so responses differ by KML (e.g. Godavari ≠ Mithi).
"""
from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .config import Reach


@dataclass
class RiverProfile:
    key: str
    name: str
    reaches_fn: Callable[[], List[Reach]]
    # (west, south, east, north) approximate bounding box
    bbox: Tuple[float, float, float, float]
    # river axis: upstream (lon, lat) → downstream (lon, lat)
    axis_start: Tuple[float, float]
    axis_end: Tuple[float, float]
    total_km: float
    demo_seed: int
    holdout_reach: str
    rain_seed: int
    landmarks: Optional[List[Tuple[str, float]]] = None


def _local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _parse_coordinate_rings(kml_bytes: bytes) -> List[List[List[float]]]:
    try:
        root = ET.fromstring(kml_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid KML file: {exc}") from exc

    rings: List[List[List[float]]] = []
    for elem in root.iter():
        if _local_name(elem.tag) != "coordinates":
            continue
        if not elem.text or not elem.text.strip():
            continue
        tokens = elem.text.strip().split()
        ring: List[List[float]] = []
        for token in tokens:
            parts = token.split(",")
            if len(parts) >= 2:
                ring.append([float(parts[0]), float(parts[1])])
        # Polygon / LineString rings need ≥2 points; closed polygons ≥3
        if len(ring) >= 3:
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            rings.append(ring)
        elif len(ring) == 2:
            rings.append(ring)
    return rings


def _parse_document_name(root: ET.Element) -> Optional[str]:
    for elem in root.iter():
        if _local_name(elem.tag) not in ("Document", "Folder"):
            continue
        for child in list(elem):
            if _local_name(child.tag) == "name" and child.text and child.text.strip():
                return child.text.strip()
    return None


def _parse_bridge_points(root: ET.Element) -> List[Tuple[str, float, float]]:
    """Named Point placemarks → bridges for chainage."""
    bridges: List[Tuple[str, float, float]] = []
    for pm in root.iter():
        if _local_name(pm.tag) != "Placemark":
            continue
        name = None
        lon = lat = None
        has_point = False
        for child in pm.iter():
            tag = _local_name(child.tag)
            if tag == "name" and child.text and name is None:
                # first name under this placemark subtree may be style; prefer direct
                pass
            if tag == "Point":
                has_point = True
            if tag == "coordinates" and has_point and child.text:
                token = child.text.strip().split()[0]
                parts = token.split(",")
                if len(parts) >= 2:
                    lon, lat = float(parts[0]), float(parts[1])
        # Direct child name is more reliable
        for child in list(pm):
            if _local_name(child.tag) == "name" and child.text and child.text.strip():
                name = child.text.strip()
                break
        if has_point and lon is not None and lat is not None:
            bridges.append((name or f"Bridge {len(bridges) + 1}", lon, lat))
    return bridges


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    )
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _axis_from_points(
    points: List[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Longest-span axis through point cloud (lon, lat)."""
    if not points:
        raise ValueError("No coordinates found in KML.")
    if len(points) == 1:
        lon, lat = points[0]
        return (lon, lat), (lon + 0.01, lat), 1.0

    best_i, best_j, best_d = 0, 1, -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = _haversine_km(points[i][0], points[i][1], points[j][0], points[j][1])
            if d > best_d:
                best_d = d
                best_i, best_j = i, j
    start = points[best_i]
    end = points[best_j]
    # Prefer south/west → north/east as "upstream" heuristic
    if (start[1], start[0]) > (end[1], end[0]):
        start, end = end, start
    total = max(best_d, 0.5)
    return start, end, round(total, 2)


def _project_km_on_axis(
    lon: float,
    lat: float,
    start: Tuple[float, float],
    end: Tuple[float, float],
    total_km: float,
) -> float:
    """Project (lon, lat) onto river axis; return chainage in km."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-12:
        return 0.0
    t = max(0.0, min(1.0, ((lon - x0) * dx + (lat - y0) * dy) / seg_len2))
    return round(t * total_km, 2)


def _seed_from_centroid(lon: float, lat: float) -> int:
    raw = f"{round(lon, 4)}:{round(lat, 4)}".encode("utf-8")
    return int(hashlib.md5(raw).hexdigest()[:8], 16) % 90000 + 1000


def _build_reaches_from_bridges(
    bridges: List[Tuple[str, float, float]],
    axis_start: Tuple[float, float],
    axis_end: Tuple[float, float],
    total_km: float,
) -> Tuple[List[Reach], List[Tuple[str, float]]]:
    """Order bridges along axis; one reach between consecutive bridges."""
    ordered = sorted(
        bridges,
        key=lambda b: _project_km_on_axis(b[1], b[2], axis_start, axis_end, total_km),
    )
    landmarks: List[Tuple[str, float]] = []
    for name, lon, lat in ordered:
        km = _project_km_on_axis(lon, lat, axis_start, axis_end, total_km)
        landmarks.append((name, km))

    reaches: List[Reach] = []
    prev: Optional[str] = None

    # Need ≥2 reaches so station holdout never empties the GBM training set.
    if len(ordered) == 0:
        mid = round(total_km / 2.0, 2) or 0.5
        hi = max(total_km, mid + 0.5)
        r1 = Reach("R01", "Upstream", 0.0, mid, 80.0, 0.55, upstream=None)
        r2 = Reach("R02", "Downstream", mid, hi, 85.0, 0.70, upstream="R01")
        return [r1, r2], landmarks

    if len(ordered) == 1:
        name = ordered[0][0]
        km = landmarks[0][1] if landmarks else round(total_km / 2.0, 2)
        mid = km if 0.0 < km < total_km else round(max(total_km, 1.0) / 2.0, 2)
        if mid <= 0:
            mid = 0.5
        hi = max(total_km, mid + 0.5)
        r1 = Reach(
            "R01", f"{name} (upstream)", 0.0, mid, 80.0, 0.55, upstream=None
        )
        r1.villages = name  # type: ignore[attr-defined]
        r2 = Reach(
            "R02",
            f"{name} (downstream)",
            mid,
            hi,
            85.0,
            0.70,
            upstream="R01",
        )
        r2.villages = name  # type: ignore[attr-defined]
        return [r1, r2], landmarks

    # Reaches span consecutive bridge chainages; first from 0 / last to end
    kms = [lm[1] for lm in landmarks]
    bounds = [0.0] + [(kms[i] + kms[i + 1]) / 2.0 for i in range(len(kms) - 1)] + [
        total_km
    ]
    for i, (name, _lon, _lat) in enumerate(ordered):
        rid = f"R{i + 1:02d}"
        a, b = round(bounds[i], 2), round(bounds[i + 1], 2)
        if b <= a:
            b = a + 0.5
        od = 0.35 + 0.5 * (i / max(1, len(ordered) - 1))
        r = Reach(rid, name, a, b, 70.0 + 5 * i, round(od, 2), upstream=prev)
        r.villages = name  # type: ignore[attr-defined]
        reaches.append(r)
        prev = rid
    return reaches, landmarks


def kml_centroid(kml_bytes: bytes) -> Tuple[float, float]:
    """Return (lon, lat) centroid of all polygon/line coordinates in the KML."""
    rings = _parse_coordinate_rings(kml_bytes)
    xs, ys = [], []
    for ring in rings:
        for lon, lat in ring:
            xs.append(lon)
            ys.append(lat)
    if not xs:
        # Fall back to bridge points only
        try:
            root = ET.fromstring(kml_bytes)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid KML file: {exc}") from exc
        bridges = _parse_bridge_points(root)
        if not bridges:
            raise ValueError("No polygon or point coordinates found in KML.")
        xs = [b[1] for b in bridges]
        ys = [b[2] for b in bridges]
    return sum(xs) / len(xs), sum(ys) / len(ys)


@dataclass
class KmlLocation:
    river_key: str
    river_name: str
    reach_id: str
    reach_name: str
    chainage_km: float
    km_range: Tuple[float, float]
    area_name: str
    villages: Optional[str]
    centroid: Tuple[float, float]
    profile: RiverProfile
    bridges: Optional[List[Tuple[str, float]]] = None


def locate_kml(kml_bytes: bytes) -> KmlLocation:
    """
    Build river profile from uploaded KML.

    Uses Document/Folder name as river name and Point placemark names as
    bridges / chainage landmarks. Synthetic demo seed is unique per centroid.
    """
    try:
        root = ET.fromstring(kml_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid KML file: {exc}") from exc

    lon, lat = kml_centroid(kml_bytes)
    river_name = _parse_document_name(root) or "Uploaded river"
    bridges = _parse_bridge_points(root)

    rings = _parse_coordinate_rings(kml_bytes)
    axis_pts: List[Tuple[float, float]] = []
    for ring in rings:
        for x, y in ring:
            axis_pts.append((x, y))
    for _n, x, y in bridges:
        axis_pts.append((x, y))
    if not axis_pts:
        axis_pts = [(lon, lat)]

    axis_start, axis_end, total_km = _axis_from_points(axis_pts)
    if bridges:
        # Recompute axis from ordered bridges when available
        ordered = sorted(
            bridges,
            key=lambda b: _project_km_on_axis(
                b[1], b[2], axis_start, axis_end, total_km
            ),
        )
        axis_start = (ordered[0][1], ordered[0][2])
        axis_end = (ordered[-1][1], ordered[-1][2])
        total_km = max(
            _haversine_km(axis_start[0], axis_start[1], axis_end[0], axis_end[1]),
            0.5,
        )
        total_km = round(total_km, 2)

    reaches, landmarks = _build_reaches_from_bridges(
        bridges, axis_start, axis_end, total_km
    )
    seed = _seed_from_centroid(lon, lat)
    holdout = reaches[len(reaches) // 2].reach_id

    def _reaches_fn() -> List[Reach]:
        return list(reaches)

    west = min(p[0] for p in axis_pts) - 0.05
    south = min(p[1] for p in axis_pts) - 0.05
    east = max(p[0] for p in axis_pts) + 0.05
    north = max(p[1] for p in axis_pts) + 0.05

    profile = RiverProfile(
        key="custom",
        name=river_name,
        reaches_fn=_reaches_fn,
        bbox=(west, south, east, north),
        axis_start=axis_start,
        axis_end=axis_end,
        total_km=total_km,
        demo_seed=seed,
        holdout_reach=holdout,
        rain_seed=seed + 7,
        landmarks=landmarks,
    )

    chainage = _project_km_on_axis(lon, lat, axis_start, axis_end, total_km)
    reach = next(
        (r for r in reaches if r.km_start <= chainage <= r.km_end),
        min(
            reaches,
            key=lambda r: min(abs(chainage - r.km_start), abs(chainage - r.km_end)),
        ),
    )
    villages = getattr(reach, "villages", None)
    area_name = reach.name
    if villages:
        area_name = f"{reach.name} ({villages})"

    return KmlLocation(
        river_key=profile.key,
        river_name=profile.name,
        reach_id=reach.reach_id,
        reach_name=reach.name,
        chainage_km=chainage,
        km_range=(reach.km_start, reach.km_end),
        area_name=area_name,
        villages=villages,
        centroid=(round(lon, 6), round(lat, 6)),
        profile=profile,
        bridges=landmarks,
    )
