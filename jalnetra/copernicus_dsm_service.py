"""
jalnetra.copernicus_dsm_service — Copernicus GLO-30 DSM clipped to uploaded KML.

Same pipeline shape as fabdem_service (KML → bbox → download → merge → clip),
but sources the Copernicus DEM Digital Surface Model (buildings + vegetation)
as public Cloud-Optimized GeoTIFFs from AWS Open Data — no Earth Engine.
"""
from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import rasterio
import rasterio.merge
import requests
from rasterio.mask import mask
from requests import Session

logger = logging.getLogger(__name__)

# GLO-30 Public COGs (DSM). Resolution token "10" = 10 arc-seconds ≈ 30 m.
DSM_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
USER_AGENT = (
    "JalNetra-DSM/1.0 (+https://github.com/planeteyeai/JalNetra-Software; "
    "mailto:support@planeteye.ai)"
)
REQUEST_TIMEOUT = (30, 600)  # connect, read
MAX_RETRIES = 3


def _session() -> Session:
    s = Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        }
    )
    return s


def _hemisphere_lat(lat: int) -> str:
    return f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"


def _hemisphere_lon(lon: int) -> str:
    return f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"


def _tile_stem(lat: int, lon: int) -> str:
    """Copernicus GLO-30 1° tile directory / file stem."""
    return (
        f"Copernicus_DSM_COG_10_{_hemisphere_lat(lat)}_00_"
        f"{_hemisphere_lon(lon)}_00_DEM"
    )


def _tile_url(lat: int, lon: int) -> str:
    stem = _tile_stem(lat, lon)
    return f"{DSM_BASE_URL}/{stem}/{stem}.tif"


def _tiles_for_bounds(
    bounds: Tuple[float, float, float, float],
) -> List[Tuple[int, int]]:
    """1° integer (lat, lon) cells that intersect the WGS84 bbox."""
    west, south, east, north = bounds
    if east < west or north < south:
        raise ValueError("Invalid KML bounding box (east < west or north < south).")

    lat0 = int(math.floor(south))
    lat1 = int(math.floor(north - 1e-12))  # exclude north edge if exactly on integer
    lon0 = int(math.floor(west))
    lon1 = int(math.floor(east - 1e-12))
    if north == south:
        lat1 = lat0
    if east == west:
        lon1 = lon0

    tiles: List[Tuple[int, int]] = []
    for lat in range(lat0, lat1 + 1):
        for lon in range(lon0, lon1 + 1):
            if lat < -90 or lat > 89 or lon < -180 or lon > 179:
                continue
            tiles.append((lat, lon))
    if not tiles:
        # Degenerate / tiny AOI still maps to one cell
        tiles.append((int(math.floor(south)), int(math.floor(west))))
    return tiles


def _purge_invalid_cache(cache_dir: Path) -> List[str]:
    removed: List[str] = []
    if not cache_dir.exists():
        return removed
    for path in cache_dir.glob("*.tif"):
        try:
            with open(path, "rb") as fh:
                magic = fh.read(2)
            if magic not in (b"II", b"MM") or path.stat().st_size < 1024:
                path.unlink(missing_ok=True)
                removed.append(path.name)
        except OSError:
            continue
    return removed


def _download_tile(
    session: Session, url: str, destination_path: Path
) -> None:
    last_exc: Optional[BaseException] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                raise FileNotFoundError(
                    f"DSM tile not published (404): {url}. "
                    "Ocean cells or withheld GLO-30 Public countries return 404."
                )
            response.raise_for_status()
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = destination_path.with_suffix(destination_path.suffix + ".part")
            with open(tmp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        fh.write(chunk)
            with open(tmp_path, "rb") as fh:
                magic = fh.read(2)
            if magic not in (b"II", b"MM"):
                tmp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Downloaded file is not a TIFF (magic={magic!r}): {url}"
                )
            tmp_path.replace(destination_path)
            return
        except FileNotFoundError:
            raise
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "DSM tile download attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
    assert last_exc is not None
    raise last_exc


def _merge_rasters(
    output_path: Path, tiles: List[Path], bounds: Tuple[float, float, float, float]
) -> None:
    rasters = [rasterio.open(tile) for tile in tiles]
    try:
        merged_raster, merged_transform = rasterio.merge.merge(rasters, bounds=bounds)
        source_crs = rasters[0].crs
        if source_crs is None:
            raise ValueError("No CRS present in DSM tile metadata.")
        metadata = {
            "driver": "GTiff",
            "count": merged_raster.shape[0],
            "height": merged_raster.shape[1],
            "width": merged_raster.shape[2],
            "dtype": merged_raster.dtype,
            "crs": source_crs,
            "transform": merged_transform,
        }
        with rasterio.open(output_path, mode="w", **metadata) as dest:
            dest.write(merged_raster)
    finally:
        for raster in rasters:
            raster.close()


def _download_dsm_for_bounds(
    bounds: Tuple[float, float, float, float],
    output_path: Path,
    cache_dir: Path,
) -> Dict[str, Any]:
    """Download intersecting GLO-30 DSM COG tiles and merge to bbox."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    purged = _purge_invalid_cache(cache_dir)
    if purged:
        logger.warning("Purged invalid DSM cache tile(s): %s", ", ".join(purged))

    tile_cells = _tiles_for_bounds(bounds)
    missing_urls: List[str] = []
    tile_paths: List[Path] = []

    with _session() as session:
        for lat, lon in tile_cells:
            stem = _tile_stem(lat, lon)
            dest = cache_dir / f"{stem}.tif"
            url = _tile_url(lat, lon)

            if dest.exists() and dest.stat().st_size > 1024:
                with open(dest, "rb") as fh:
                    if fh.read(2) in (b"II", b"MM"):
                        tile_paths.append(dest)
                        continue
                dest.unlink(missing_ok=True)

            try:
                _download_tile(session, url, dest)
                tile_paths.append(dest)
            except FileNotFoundError:
                missing_urls.append(url)
                logger.info("Skipping unavailable DSM tile: %s", url)

    if not tile_paths:
        raise RuntimeError(
            "No Copernicus GLO-30 DSM tiles available for the KML bounding box. "
            "Ocean cells or withheld public tiles return 404. "
            + (
                f"Tried: {', '.join(missing_urls)}"
                if missing_urls
                else f"Cells: {tile_cells}"
            )
        )

    _merge_rasters(output_path, tile_paths, bounds)
    return {
        "tile_count": len(tile_paths),
        "tile_cells": [{"lat": lat, "lon": lon} for lat, lon in tile_cells],
        "tiles_downloaded": [p.name for p in tile_paths],
        "tiles_missing": missing_urls,
        "cache_purged": purged,
    }


def download_copernicus_dsm_from_kml(kml_bytes: bytes) -> Dict[str, Any]:
    """
    Download Copernicus GLO-30 DSM for the KML bbox and clip to the KML boundary.

    Returns metadata + clipped GeoTIFF bytes (same response shape as FABDEM DTM).
    """
    if not kml_bytes:
        raise ValueError("KML file is empty.")

    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "DSM_Cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dsm_") as tmp:
        tmp_dir = Path(tmp)
        kml_path = tmp_dir / "upload.kml"
        output_dir = tmp_dir / "DSM_Output"
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_file = output_dir / "DSM_Temporary.tif"
        output_file = output_dir / "Copernicus_DSM_KML_Clipped.tif"

        kml_path.write_bytes(kml_bytes)

        # ---- read KML ----
        try:
            aoi = gpd.read_file(kml_path, driver="KML")
        except Exception:
            try:
                aoi = gpd.read_file(kml_path)
            except Exception as error2:
                raise ValueError(f"Failed to read KML: {error2}") from error2

        if aoi.empty:
            raise ValueError("KML contains no geometry.")

        aoi = aoi[aoi.geometry.notnull()]
        if aoi.empty:
            raise ValueError("No valid geometry found in KML.")

        # ---- CRS → WGS84 ----
        if aoi.crs is None:
            aoi = aoi.set_crs("EPSG:4326")
        else:
            aoi = aoi.to_crs("EPSG:4326")

        aoi_geometry = aoi.geometry.union_all()
        west, south, east, north = aoi_geometry.bounds
        bounds = (float(west), float(south), float(east), float(north))

        try:
            dl_meta = _download_dsm_for_bounds(bounds, temp_file, cache_dir)
        except Exception as error:
            raise RuntimeError(f"Copernicus DSM download failed: {error}") from error

        if not temp_file.exists():
            raise RuntimeError("DSM download produced no temporary TIFF.")

        try:
            with rasterio.open(temp_file) as src:
                geometry = [aoi_geometry.__geo_interface__]
                clipped, clipped_transform = mask(src, geometry, crop=True)
                output_meta = src.meta.copy()
                output_meta.update(
                    {
                        "driver": "GTiff",
                        "height": clipped.shape[1],
                        "width": clipped.shape[2],
                        "transform": clipped_transform,
                    }
                )
                with rasterio.open(output_file, "w", **output_meta) as dst:
                    dst.write(clipped)
        except Exception as error:
            raise RuntimeError(f"DSM clipping failed: {error}") from error

        tif_bytes = output_file.read_bytes()

        return {
            "bounds": {
                "west": float(west),
                "south": float(south),
                "east": float(east),
                "north": float(north),
            },
            "geometry_type": str(aoi_geometry.geom_type),
            "crs": "EPSG:4326",
            "width": int(clipped.shape[2]),
            "height": int(clipped.shape[1]),
            "tif_filename": "Copernicus_DSM_KML_Clipped.tif",
            "tif_bytes": tif_bytes,
            "download": dl_meta,
            "notes": {
                "source": (
                    "Copernicus DEM GLO-30 Public (Digital Surface Model — "
                    "includes buildings, infrastructure, and vegetation)"
                ),
                "clip": "Clipped to exact uploaded KML geometry (not only bbox).",
                "method": (
                    "Direct HTTPS download of Cloud-Optimized GeoTIFF tiles "
                    "from AWS Open Data (no Earth Engine)."
                ),
                "attribution": (
                    "Copernicus DEM © ESA / Airbus — "
                    "https://registry.opendata.aws/copernicus-dem/"
                ),
            },
        }
