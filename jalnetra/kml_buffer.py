"""
jalnetra.kml_buffer — build a rectangular analysis box around an input KML.

Expands the KML bounding box by a fixed 3 km on every side (UTM metres).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import ee
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.ops import transform, unary_union

BUFFER_KM = 2.0
BUFFER_M = BUFFER_KM * 1000.0


def _local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _parse_coords(coord_text: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for token in coord_text.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def load_geometry_from_kml_bytes(kml_bytes: bytes):
    """Load polygon(s) and/or lines from KML bytes."""
    root = ET.fromstring(kml_bytes)
    polygons: List[Polygon] = []
    lines: List[LineString] = []

    for placemark in root.iter():
        if _local_name(placemark.tag) != "Placemark":
            continue

        for outer in placemark.iter():
            if _local_name(outer.tag) != "outerBoundaryIs":
                continue
            for coords_elem in outer.iter():
                if _local_name(coords_elem.tag) != "coordinates":
                    continue
                if not coords_elem.text or not coords_elem.text.strip():
                    continue
                ring = _parse_coords(coords_elem.text)
                if len(ring) >= 3:
                    if ring[0] != ring[-1]:
                        ring = ring + [ring[0]]
                    poly = Polygon(ring)
                    if poly.is_valid and not poly.is_empty:
                        polygons.append(poly)

        for elem in placemark.iter():
            if _local_name(elem.tag) != "LineString":
                continue
            for coords_elem in elem.iter():
                if _local_name(coords_elem.tag) != "coordinates":
                    continue
                if not coords_elem.text or not coords_elem.text.strip():
                    continue
                pts = _parse_coords(coords_elem.text)
                if len(pts) >= 2:
                    lines.append(LineString(pts))

    parts = []
    if polygons:
        parts.append(unary_union(polygons))
    if lines:
        parts.append(unary_union(lines))

    if not parts:
        raise ValueError("No polygons or lines found in KML.")

    return unary_union(parts)


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _geom_to_ee(geom) -> ee.Geometry:
    if geom.is_empty:
        raise ValueError("Empty geometry.")
    if isinstance(geom, Polygon):
        return ee.Geometry.Polygon([list(geom.exterior.coords)])
    if isinstance(geom, MultiPolygon):
        return ee.Geometry.MultiPolygon(
            [[list(p.exterior.coords)] for p in geom.geoms]
        )
    if isinstance(geom, LineString):
        return ee.Geometry.LineString(list(geom.coords))
    if isinstance(geom, MultiLineString):
        return ee.Geometry.MultiLineString([list(ls.coords) for ls in geom.geoms])
    raise ValueError(f"Unsupported geometry type: {geom.geom_type}")


def _polygon_coords_for_kml(geom) -> List[List[List[float]]]:
    out: List[List[List[float]]] = []
    if isinstance(geom, Polygon):
        out.append([[float(x), float(y)] for x, y in geom.exterior.coords])
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            out.append([[float(x), float(y)] for x, y in p.exterior.coords])
    return out


def _line_coords_for_kml(geom) -> List[List[List[float]]]:
    out: List[List[List[float]]] = []
    if isinstance(geom, LineString):
        out.append([[float(x), float(y)] for x, y in geom.coords])
    elif isinstance(geom, MultiLineString):
        for ls in geom.geoms:
            out.append([[float(x), float(y)] for x, y in ls.coords])
    elif isinstance(geom, Polygon):
        out.append([[float(x), float(y)] for x, y in geom.exterior.coords])
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            out.append([[float(x), float(y)] for x, y in p.exterior.coords])
    return out


def buffered_rectangle_geometry(
    kml_bytes: bytes,
) -> Tuple[ee.Geometry, ee.Geometry, Dict[str, Any]]:
    """
    Build a rectangular analysis region: KML bounds + 2 km on each side.
    """
    geom = load_geometry_from_kml_bytes(kml_bytes)
    clon, clat = float(geom.centroid.x), float(geom.centroid.y)

    epsg = _utm_epsg(clon, clat)
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform
    metric_geom = transform(fwd, geom)

    minx, miny, maxx, maxy = metric_geom.bounds
    rect = box(
        minx - BUFFER_M,
        miny - BUFFER_M,
        maxx + BUFFER_M,
        maxy + BUFFER_M,
    )
    rect_wgs = transform(inv, rect)

    input_ee = _geom_to_ee(geom)
    analysis_ee = _geom_to_ee(rect_wgs)

    ring = [[float(x), float(y)] for x, y in rect_wgs.exterior.coords]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])

    width_km = round((maxx - minx + 2 * BUFFER_M) / 1000.0, 2)
    height_km = round((maxy - miny + 2 * BUFFER_M) / 1000.0, 2)

    meta = {
        "buffer_km": BUFFER_KM,
        "shape": "rectangle",
        "centroid": {"latitude": round(clat, 6), "longitude": round(clon, 6)},
        "rectangle_width_km": width_km,
        "rectangle_height_km": height_km,
        "analysis_box_ring": ring,
        "input_line_coords": _line_coords_for_kml(geom),
        "input_polygon_coords": _polygon_coords_for_kml(geom),
    }
    return analysis_ee, input_ee, meta


def buffered_analysis_geometry(
    kml_bytes: bytes,
    buffer_km: float | None = None,
) -> Tuple[ee.Geometry, ee.Geometry, Dict[str, Any]]:
    """Fixed 2 km rectangular buffer (buffer_km arg ignored — always 2 km)."""
    return buffered_rectangle_geometry(kml_bytes)
