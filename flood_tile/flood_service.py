from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import ee

from speckle_filters import refined_lee, to_db, to_natural

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="flood_tile_ee_")
_SCALE = 10
_VH_THRESHOLD_DB = -20


def _next_day(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt + timedelta(days=1)).strftime("%Y-%m-%d")


def _shift_date(date_str: str, days: int) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt + timedelta(days=days)).strftime("%Y-%m-%d")


def _s1_collection() -> ee.ImageCollection:
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(
            ee.Filter.Or(
                ee.Filter.eq("orbitProperties_pass", "DESCENDING"),
                ee.Filter.eq("orbitProperties_pass", "ASCENDING"),
            )
        )
    )


def _filtered_collection(
    collection: ee.ImageCollection,
    start_date: str,
    end_date: str,
    geometry: ee.Geometry,
) -> ee.ImageCollection:
    return collection.filter(ee.Filter.date(start_date, end_date)).filterBounds(geometry)


def _collection_size(collection: ee.ImageCollection) -> int:
    return int(collection.size().getInfo() or 0)


def _mosaic_for_date_range(
    collection: ee.ImageCollection,
    start_date: str,
    end_date: str,
    geometry: ee.Geometry,
    *,
    label: str,
) -> tuple[ee.Image, Dict[str, Any]]:
    """
    Build a VH mosaic from all Sentinel-1 scenes in [start_date, end_date] (inclusive).

    Earth Engine filterDate end is exclusive, so end_date + 1 day is used internally.
    """
    ee_end = _next_day(end_date)
    if datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(end_date, "%Y-%m-%d"):
        raise ValueError(
            f"Invalid {label} period: start date {start_date} is after end date {end_date}."
        )

    filtered = _filtered_collection(collection, start_date, ee_end, geometry)
    count = _collection_size(filtered)
    if count == 0:
        raise ValueError(
            f"No Sentinel-1 VH images found for the {label} period "
            f"({start_date} to {end_date}) over your KML area. "
            "Widen the date range or choose months with SAR coverage."
        )

    image = filtered.select("VH").mosaic().clip(geometry)
    band_names = image.bandNames().getInfo() or []
    if not band_names:
        raise ValueError(
            f"Sentinel-1 mosaic for the {label} period ({start_date} to {end_date}) "
            "has no VH band. Try different dates with confirmed SAR coverage."
        )

    latest_image_date = (
        ee.Date(filtered.sort("system:time_start", False).first().get("system:time_start"))
        .format("YYYY-MM-dd")
        .getInfo()
    )
    earliest_image_date = (
        ee.Date(filtered.sort("system:time_start", True).first().get("system:time_start"))
        .format("YYYY-MM-dd")
        .getInfo()
    )

    return image, {
        "start_date": start_date,
        "end_date": end_date,
        "image_count": count,
        "earliest_image_date": earliest_image_date,
        "latest_image_date": latest_image_date,
    }


def _nearest_s1_image(
    collection: ee.ImageCollection,
    geometry: ee.Geometry,
    target_date: str,
    start: str,
    end_exclusive: str,
) -> tuple[ee.Image, str, int] | None:
    """Return (vh_image, image_date, candidate_count) closest to target_date, or None."""
    filtered = _filtered_collection(collection, start, end_exclusive, geometry)
    count = _collection_size(filtered)
    if count == 0:
        return None

    target_millis = ee.Date(target_date).millis()
    ranked = filtered.map(
        lambda img: img.set(
            "date_diff",
            ee.Number(img.get("system:time_start")).subtract(target_millis).abs(),
        )
    )
    nearest = ee.Image(ranked.sort("date_diff").first()).select("VH").clip(geometry)
    band_names = nearest.bandNames().getInfo() or []
    if not band_names:
        return None
    image_date = (
        ee.Date(nearest.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    )
    return nearest, image_date, count


def _list_nearby_s1_dates(
    collection: ee.ImageCollection,
    geometry: ee.Geometry,
    target_date: str,
    *,
    around_days: int = 365,
    limit: int = 12,
) -> List[str]:
    start = _shift_date(target_date, -around_days)
    end_exclusive = _shift_date(target_date, around_days + 1)
    filtered = (
        _filtered_collection(collection, start, end_exclusive, geometry)
        .sort("system:time_start")
    )
    count = _collection_size(filtered)
    if count == 0:
        return []

    def _feat(img: ee.Image) -> ee.Feature:
        return ee.Feature(
            None,
            {"date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")},
        )

    feats = filtered.limit(limit).map(_feat).getInfo().get("features", [])
    dates = []
    for feat in feats:
        d = (feat.get("properties") or {}).get("date")
        if d:
            dates.append(d)
    return dates


def _mosaic_for_target_date(
    collection: ee.ImageCollection,
    target_date: str,
    geometry: ee.Geometry,
    *,
    label: str,
    window_days: tuple[int, ...] = (0, 3, 7, 12, 20, 45, 60, 90, 120, 180, 365),
) -> tuple[ee.Image, Dict[str, Any]]:
    """
    Find Sentinel-1 VH nearest to target_date.

    Early S1 years (2015–2016) are sparse over some areas — June may have no
    scenes for months. Search expands up to ±1 year when needed.
    """
    for half in window_days:
        if half == 0:
            start = target_date
            end_exclusive = _next_day(target_date)
            window_end_inclusive = target_date
        else:
            start = _shift_date(target_date, -half)
            end_exclusive = _shift_date(target_date, half + 1)
            window_end_inclusive = _shift_date(target_date, half)

        found = _nearest_s1_image(collection, geometry, target_date, start, end_exclusive)
        if found is None:
            continue

        nearest, image_date, count = found
        return nearest, {
            "requested_date": target_date,
            "start_date": target_date,
            "end_date": target_date,
            "image_date": image_date,
            "earliest_image_date": image_date,
            "latest_image_date": image_date,
            "image_count": 1,
            "candidates_in_window": count,
            "search_window_days": half,
            "matched_exact_day": image_date == target_date,
            "search_start": start,
            "search_end": window_end_inclusive,
        }

    nearby = _list_nearby_s1_dates(collection, geometry, target_date)
    hint = (
        f" Nearby available dates (±1 year): {', '.join(nearby)}."
        if nearby
        else " No Sentinel-1 VH scenes found within ±1 year of that date for this KML."
    )
    raise ValueError(
        f"No Sentinel-1 VH images found near {label} date {target_date} "
        f"(searched up to ±{window_days[-1]} days) over your KML area.{hint} "
        "For this region, 2015–mid‑2016 coverage is sparse (often little/no June–Aug). "
        "Prefer dates from mid‑2016 onward, or use a nearby available date from the list."
    )


def _compute_area_ha(mask_img: ee.Image, geometry: ee.Geometry, scale: int = _SCALE) -> float:
    area_img = mask_img.rename("mask").multiply(ee.Image.pixelArea()).rename("area")
    stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e13,
        tileScale=16,
    )
    sum_m2 = ee.Number(
        ee.Algorithms.If(stats.get("area"), stats.get("area"), 0)
    ).getInfo()
    return round(float(sum_m2) / 10000.0, 2)


def _compute_flood_water_areas_ha(
    water_mask: ee.Image,
    flood_mask: ee.Image,
    geometry: ee.Geometry,
    scale: int = _SCALE,
) -> tuple[float, float]:
    """Single Earth Engine round-trip for both water and flood area."""
    stacked = (
        water_mask.rename("water")
        .multiply(ee.Image.pixelArea())
        .addBands(flood_mask.rename("flood").multiply(ee.Image.pixelArea()))
    )
    stats = stacked.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e13,
        tileScale=16,
    )
    info = stats.getInfo() or {}
    water_ha = round(float(info.get("water") or 0) / 10000.0, 2)
    flood_ha = round(float(info.get("flood") or 0) / 10000.0, 2)
    return water_ha, flood_ha


def _batch_plot_stats(
    plots: List[Dict[str, Any]],
    water_mask: ee.Image,
    flood_mask: ee.Image,
) -> List[Dict[str, Any]]:
    """One Earth Engine round-trip for all village/plot areas."""
    features = []
    for plot in plots:
        props: Dict[str, Any] = {"plot_id": plot["plot_id"]}
        village_name = plot.get("village_name")
        if village_name:
            props["village_name"] = village_name
        features.append(ee.Feature(plot["geometry"], props))

    fc = ee.FeatureCollection(features)
    fc = fc.map(
        lambda feat: feat.set(
            "total_area_ha",
            feat.geometry().area().divide(10000),
        )
    )
    stacked = water_mask.rename("water").multiply(ee.Image.pixelArea()).addBands(
        flood_mask.rename("flood").multiply(ee.Image.pixelArea())
    )
    reduced = stacked.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.sum(),
        scale=_SCALE,
        tileScale=16,
    )
    rows = reduced.getInfo().get("features", [])
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        props = row.get("properties") or {}
        plot_id = props.get("plot_id")
        if not plot_id:
            continue
        by_id[plot_id] = props

    results: List[Dict[str, Any]] = []
    for plot in plots:
        props = by_id.get(plot["plot_id"], {})
        results.append(
            {
                "plot_id": plot["plot_id"],
                "village_name": plot.get("village_name"),
                "total_area_ha": round(float(props.get("total_area_ha") or 0), 2),
                "flood_area_ha": round(float(props.get("flood") or 0) / 10000.0, 2),
                "total_water_available_ha": round(float(props.get("water") or 0) / 10000.0, 2),
                "geometry": plot["geojson"],
            }
        )
    return results


def _build_classification(
    before_filtered: ee.Image, after_filtered: ee.Image, geometry: ee.Geometry
) -> tuple[ee.Image, ee.Image, ee.Image]:
    water = before_filtered.lt(_VH_THRESHOLD_DB).And(after_filtered.lt(_VH_THRESHOLD_DB))
    water_mask = water.updateMask(water.eq(1))

    flood = before_filtered.gt(_VH_THRESHOLD_DB).And(after_filtered.lt(_VH_THRESHOLD_DB))
    flood_mask = flood.updateMask(flood.eq(1))

    classification = (
        ee.Image(0)
        .where(water_mask, 1)
        .where(flood_mask, 2)
        .clip(geometry)
    )
    return classification, water_mask, flood_mask


def _tile_url(classification: ee.Image, geometry: ee.Geometry) -> str:
    vis_params = {"min": 0, "max": 2, "palette": ["#ffffff", "#0000ff", "#ff0000"]}
    tile_img = (
        classification.unmask(0)
        .updateMask(ee.Image.constant(1).clip(geometry))
        .reproject(crs="EPSG:4326", scale=12)
    )
    return tile_img.visualize(**vis_params).getMapId()["tile_fetcher"].url_format


def _plot_stats(
    plot_id: str,
    village_name: Optional[str],
    geometry: ee.Geometry,
    geojson: Dict[str, Any],
    water_mask: ee.Image,
    flood_mask: ee.Image,
) -> Dict[str, Any]:
    f_total = _executor.submit(
        lambda: round(geometry.area().divide(10000).getInfo(), 2)
    )
    f_water = _executor.submit(_compute_area_ha, water_mask, geometry)
    f_flood = _executor.submit(_compute_area_ha, flood_mask, geometry)
    return {
        "plot_id": plot_id,
        "village_name": village_name,
        "total_area_ha": f_total.result(),
        "flood_area_ha": f_flood.result(),
        "total_water_available_ha": f_water.result(),
        "geometry": geojson,
    }


def analyze_flood_tile(
    geometry: ee.Geometry,
    pre_date: str,
    post_date: str,
    plots: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run flood/water analysis using single pre and post dates.

    Uses the Sentinel-1 VH scene nearest to each date (exact day first, then
    expands search window). Matches the original GEE before/after approach.
    """
    collection = _s1_collection()

    f_pre = _executor.submit(
        _mosaic_for_target_date,
        collection,
        pre_date,
        geometry,
        label="pre",
    )
    f_post = _executor.submit(
        _mosaic_for_target_date,
        collection,
        post_date,
        geometry,
        label="post",
    )
    before_image, pre_period = f_pre.result()
    after_image, post_period = f_post.result()

    if pre_period.get("image_date") == post_period.get("image_date"):
        raise ValueError(
            f"Pre and post both resolved to the same Sentinel-1 scene "
            f"({pre_period.get('image_date')}). Pick dates farther apart, or choose "
            f"years with denser coverage (this AOI has sparse S1 in 2015–mid‑2016)."
        )

    before_filtered = ee.Image(to_db(refined_lee(to_natural(before_image))))
    after_filtered = ee.Image(to_db(refined_lee(to_natural(after_image))))

    classification, water_mask, flood_mask = _build_classification(
        before_filtered, after_filtered, geometry
    )

    if plots:
        f_stats = _executor.submit(_batch_plot_stats, plots, water_mask, flood_mask)
        f_tile = _executor.submit(_tile_url, classification, geometry)
        village_results = f_stats.result()
        tile_url = f_tile.result()
        total_area_ha = round(sum(v["total_area_ha"] for v in village_results), 2)
        water_area_ha = round(sum(v["total_water_available_ha"] for v in village_results), 2)
        flood_area_ha = round(sum(v["flood_area_ha"] for v in village_results), 2)
    else:
        village_results = []
        f_total = _executor.submit(
            lambda: round(geometry.area().divide(10000).getInfo(), 2)
        )
        f_areas = _executor.submit(_compute_flood_water_areas_ha, water_mask, flood_mask, geometry)
        f_tile = _executor.submit(_tile_url, classification, geometry)
        total_area_ha = f_total.result()
        water_area_ha, flood_area_ha = f_areas.result()
        tile_url = f_tile.result()

    return {
        "total_area_ha": total_area_ha,
        "flood_area_ha": flood_area_ha,
        "total_water_available_ha": water_area_ha,
        "water_area_ha": water_area_ha,
        "plot_count": len(village_results) if plots else 1,
        "villages": village_results,
        "tile_url": tile_url,
        "pre_dates": pre_period,
        "post_dates": post_period,
        "legend": {
            "0": {"color": "#ffffff", "label": "Other"},
            "1": {"color": "#0000ff", "label": "Water body"},
            "2": {"color": "#ff0000", "label": "Flood inundation"},
        },
    }
