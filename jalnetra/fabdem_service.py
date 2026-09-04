"""
jalnetra.fabdem_service — FABDEM DTM download clipped to uploaded KML.

Port of the FABDEM KML DTM downloader script:
  1) read KML → WGS84
  2) download FABDEM for KML bounding box
  3) clip raster to exact KML geometry
  4) return clipped GeoTIFF bytes
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import fabdem
import geopandas as gpd
from rasterio.mask import mask
import rasterio


def download_fabdem_dtm_from_kml(kml_bytes: bytes) -> Dict[str, Any]:
    """
    Download FABDEM for the KML bbox and clip to the KML boundary.

    Returns metadata + clipped GeoTIFF bytes (logic matches the standalone script).
    """
    if not kml_bytes:
        raise ValueError("KML file is empty.")

    # Persistent tile cache (same idea as FABDEM_Cache in the script)
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "FABDEM_Cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fabdem_") as tmp:
        tmp_dir = Path(tmp)
        kml_path = tmp_dir / "upload.kml"
        output_dir = tmp_dir / "FABDEM_Output"
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_file = output_dir / "FABDEM_Temporary.tif"
        output_file = output_dir / "FABDEM_DTM_KML_Clipped.tif"

        kml_path.write_bytes(kml_bytes)

        # ---- read KML ----
        try:
            aoi = gpd.read_file(kml_path, driver="KML")
        except Exception as error:
            # Fallback without explicit driver (pyogrio / fiona auto)
            try:
                aoi = gpd.read_file(kml_path)
            except Exception as error2:
                raise ValueError(f"Failed to read KML: {error2}") from error2

        if aoi.empty:
            raise ValueError("KML contains no geometry.")

        aoi = aoi[aoi.geometry.notnull()]
        if aoi.empty:
            raise ValueError("No valid geometry found in KML.")

        # ---- CRS → WGS84 (FABDEM bounds must be lon/lat) ----
        if aoi.crs is None:
            aoi = aoi.set_crs("EPSG:4326")
        else:
            aoi = aoi.to_crs("EPSG:4326")

        # ---- merge geometries + bbox ----
        aoi_geometry = aoi.geometry.union_all()
        west, south, east, north = aoi_geometry.bounds

        # ---- FABDEM download ----
        try:
            fabdem.download(
                (west, south, east, north),
                output_path=str(temp_file),
                cache=cache_dir,
                show_progress=False,
            )
        except Exception as error:
            raise RuntimeError(f"FABDEM download failed: {error}") from error

        if not temp_file.exists():
            raise RuntimeError("FABDEM download produced no temporary TIFF.")

        # ---- clip to exact KML boundary ----
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
            raise RuntimeError(f"FABDEM clipping failed: {error}") from error

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
            "tif_filename": "FABDEM_DTM_KML_Clipped.tif",
            "tif_bytes": tif_bytes,
            "notes": {
                "source": "FABDEM (Forest And Buildings removed Copernicus DEM)",
                "clip": "Clipped to exact uploaded KML geometry (not only bbox).",
            },
        }
