"""Minimal Sentinel-1 scene helpers for flood-water (no matplotlib/plotly)."""
from __future__ import annotations

from typing import Any, Dict, List

import ee

from jalnetra.flood_deps.flood_service import _filtered_collection, _next_day


def scene_records(
    collection: ee.ImageCollection, geometry: ee.Geometry, start_date: str, end_date: str
) -> List[Dict[str, Any]]:
    ee_end = _next_day(end_date)
    filtered = (
        _filtered_collection(collection, start_date, ee_end, geometry)
        .sort("system:time_start")
    )
    count = int(filtered.size().getInfo() or 0)
    if count == 0:
        return []

    def _feature(img: ee.Image) -> ee.Feature:
        millis = img.get("system:time_start")
        return ee.Feature(
            None,
            {
                "system:time_start": millis,
                "date": ee.Date(millis).format("YYYY-MM-dd"),
            },
        )

    features = filtered.map(_feature).getInfo().get("features", [])
    records: List[Dict[str, Any]] = []
    for feat in features:
        props = feat.get("properties") or {}
        millis = props.get("system:time_start")
        date_str = props.get("date")
        if millis is None or not date_str:
            continue
        records.append(
            {
                "date": date_str,
                "millis": int(millis),
                "month": date_str[:7],
            }
        )
    return records


def image_for_millis(
    collection: ee.ImageCollection, geometry: ee.Geometry, millis: int
) -> ee.Image:
    return (
        collection.filter(ee.Filter.eq("system:time_start", millis))
        .first()
        .select("VH")
        .clip(geometry)
    )
