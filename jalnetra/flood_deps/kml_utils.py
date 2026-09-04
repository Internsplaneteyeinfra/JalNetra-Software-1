from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ee
from fastapi import HTTPException


@dataclass
class KmlPlot:
    plot_id: str
    village_name: Optional[str]
    geometry: ee.Geometry
    geojson: Dict[str, Any]


def _local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _parse_kml_coordinate_ring(coord_text: str) -> List[List[float]]:
    ring: List[List[float]] = []
    for token in coord_text.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            ring.append([float(parts[0]), float(parts[1])])
    if len(ring) >= 3 and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _placemark_plot_id(placemark: ET.Element) -> str:
    for child in placemark:
        if _local_name(child.tag) == "name" and child.text and child.text.strip():
            return child.text.strip()
    placemark_id = placemark.get("id")
    if placemark_id:
        return placemark_id
    return "plot"


def _placemark_village_name(placemark: ET.Element) -> Optional[str]:
    for elem in placemark.iter():
        if _local_name(elem.tag) != "SimpleData":
            continue
        field_name = (elem.get("name") or "").strip()
        if field_name == "village" and elem.text and elem.text.strip():
            return elem.text.strip()
    return None


def _placemark_rings(placemark: ET.Element) -> List[List[List[float]]]:
    rings: List[List[List[float]]] = []
    for outer in placemark.iter():
        if _local_name(outer.tag) != "outerBoundaryIs":
            continue
        for coords_elem in outer.iter():
            if _local_name(coords_elem.tag) != "coordinates":
                continue
            if not coords_elem.text or not coords_elem.text.strip():
                continue
            ring = _parse_kml_coordinate_ring(coords_elem.text)
            if len(ring) >= 4:
                rings.append(ring)
    return rings


def parse_kml_plots(kml_bytes: bytes) -> List[KmlPlot]:
    try:
        root = ET.fromstring(kml_bytes)
    except ET.ParseError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid KML file: {exc}") from exc

    plots: List[KmlPlot] = []
    seen_keys: set[str] = set()

    for placemark in root.iter():
        if _local_name(placemark.tag) != "Placemark":
            continue

        rings = _placemark_rings(placemark)
        if not rings:
            continue

        plot_id = _placemark_plot_id(placemark)
        village_name = _placemark_village_name(placemark)
        dedupe_key = f"{plot_id}:{village_name or ''}:{rings[0][0]}"
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        try:
            if len(rings) == 1:
                geometry = ee.Geometry.Polygon(rings[0])
            else:
                geometry = ee.Geometry.MultiPolygon([[ring] for ring in rings])
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not build geometry for plot {plot_id}: {exc}",
            ) from exc

        plots.append(
            KmlPlot(
                plot_id=plot_id,
                village_name=village_name,
                geometry=geometry,
                geojson=geometry.getInfo(),
            )
        )

    if not plots:
        raise HTTPException(status_code=400, detail="No village polygons found in KML file.")

    return plots


def plots_to_combined_geometry(plots: List[KmlPlot]) -> ee.Geometry:
    if len(plots) == 1:
        return plots[0].geometry

    polygons: List[List[List[List[float]]]] = []
    for plot in plots:
        info = plot.geojson
        geom_type = info.get("type")
        coords = info.get("coordinates")
        if geom_type == "Polygon" and coords:
            polygons.append(coords)
        elif geom_type == "MultiPolygon" and coords:
            polygons.extend(coords)

    if not polygons:
        raise HTTPException(status_code=400, detail="Could not merge KML polygons.")

    if len(polygons) == 1:
        return ee.Geometry.Polygon(polygons[0])

    return ee.Geometry.MultiPolygon(polygons)


def kml_bytes_to_ee_geometry(kml_bytes: bytes) -> ee.Geometry:
    plots = parse_kml_plots(kml_bytes)
    return plots_to_combined_geometry(plots)


def geometry_to_geojson(geometry: ee.Geometry) -> dict:
    return geometry.getInfo()


def plots_to_feature_collection(plots: List[KmlPlot]) -> Dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "plot_id": plot.plot_id,
                    "village_name": plot.village_name,
                },
                "geometry": plot.geojson,
            }
            for plot in plots
        ],
    }
