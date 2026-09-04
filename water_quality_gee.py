#!/usr/bin/env python3
"""
water_quality_gee.py — Water quality KML / GeoTIFF export via Google Earth Engine.

Python port of the JalNetra GEE JavaScript workflow (NDWI, NDCI, TSS/turbidity,
WST, classifications with colour palettes).

Setup (one time):
    pip install earthengine-api requests
    earthengine authenticate

Usage:
    python water_quality_gee.py --project YOUR_GCP_PROJECT
    python water_quality_gee.py --mode june --format kml
    python water_quality_gee.py --mode latest --format both
    python water_quality_gee.py --drive   # export GeoTIFFs to Google Drive
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from xml.dom import minidom

try:
    import ee
except ImportError:
    sys.exit("Missing dependency: pip install earthengine-api")

# ── Palettes (match GEE JavaScript Section 9) ────────────────────────────────
NDWI_PALETTE = ["#8D6E63", "#0D47A1"]
TSS_PALETTE = ["#2196F3", "#FFC107", "#F44336"]
NDCI_PALETTE = ["#C8E6C9", "#1B5E20"]
WST_PALETTE = ["#1565C0", "#64B5F6", "#FFD54F", "#E53935", "#8E0000"]

# Continuous-index palettes (Section 6)
NDWI_CONT_PALETTE = ["saddlebrown", "white", "dodgerblue"]
NDCI_CONT_PALETTE = ["blue", "white", "darkgreen"]
TSS_CONT_PALETTE = ["#0000ff", "#00ffff", "#ffff00", "#ff0000"]
WST_CONT_PALETTE = ["#0000FF", "#00BFFF", "#00FFFF", "#FFFF00", "#FF8C00", "#FF0000"]

DEFAULT_KML = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "dde04ea1f6dd4c38adaca705bc2460a4.kml",
)
FALLBACK_KML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dashboard",
    "mula_mutha",
    "river_patch.kml",
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gee_exports")

MAX_CLOUD_S2 = 20
SCALE = 30


# ── KML → ee.Geometry ────────────────────────────────────────────────────────

def parse_kml_polygon(kml_path: str) -> tuple[ee.Geometry, list[list[float]]]:
    """Extract polygon ring from KML → (ee.Geometry, [[lon, lat], ...])."""
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coords_el = root.find(".//kml:coordinates", ns)
    if coords_el is None or not coords_el.text:
        coords_el = root.find(".//{http://www.opengis.net/kml/2.2}coordinates")
    if coords_el is None or not coords_el.text:
        raise ValueError(f"No polygon coordinates found in {kml_path}")

    ring = []
    for token in coords_el.text.strip().split():
        parts = token.split(",")
        lon, lat = float(parts[0]), float(parts[1])
        ring.append([lon, lat])

    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ee.Geometry.Polygon([ring]), ring


def aoi_latlon_box(aoi: ee.Geometry) -> dict[str, float]:
    """Bounding box for KML GroundOverlay LatLonBox."""
    coords = aoi.bounds().getInfo()["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return {
        "north": max(lats),
        "south": min(lats),
        "east": max(lons),
        "west": min(lons),
    }


# ── Earth Engine init ────────────────────────────────────────────────────────

def init_ee(project: str | None = None) -> None:
    """Initialize Earth Engine (project required since ee-api v1.5+)."""
    project = project or os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:
        msg = str(exc)
        if "no project found" in msg.lower() or "not authenticated" in msg.lower():
            print("[auth] Earth Engine needs a GCP project.")
            print("       1) earthengine authenticate")
            print("       2) python water_quality_gee.py --project YOUR_GCP_PROJECT")
            print("          or set EE_PROJECT=YOUR_GCP_PROJECT")
        raise SystemExit(msg) from exc


# ── Date selection ───────────────────────────────────────────────────────────

def june_first_week_dates(year: int = 2026) -> tuple[str, str, str, str]:
    """Sentinel-2 and Landsat 9 windows for June 1st week."""
    return (
        f"{year}-06-01",
        f"{year}-06-08",
        f"{year}-06-01",
        f"{year}-06-30",
    )


def latest_dates(aoi: ee.Geometry, max_cloud: int = MAX_CLOUD_S2) -> tuple[str, str, str, str]:
    """Pick the most recent clear Sentinel-2 scene and a matching Landsat window."""
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .sort("system:time_start", False)
    )
    latest_s2 = ee.Image(s2_col.first())
    s2_ms = ee.Number(latest_s2.get("system:time_start"))
    s2_date = ee.Date(s2_ms)
    s2_start = s2_date.advance(-2, "day").format("YYYY-MM-dd")
    s2_end = s2_date.advance(3, "day").format("YYYY-MM-dd")

    l9_col = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(aoi)
        .sort("system:time_start", False)
    )
    latest_l9 = ee.Image(l9_col.first())
    l9_ms = ee.Number(latest_l9.get("system:time_start"))
    l9_date = ee.Date(l9_ms)
    l9_start = l9_date.advance(-15, "day").format("YYYY-MM-dd")
    l9_end = l9_date.advance(16, "day").format("YYYY-MM-dd")

    return (
        s2_start.getInfo(),
        s2_end.getInfo(),
        l9_start.getInfo(),
        l9_end.getInfo(),
    )


def count_images(aoi: ee.Geometry, start: str, end: str) -> tuple[int, int]:
    s2_n = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_S2))
        .size()
        .getInfo()
    )
    l9_n = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(start, end)
        .size()
        .getInfo()
    )
    return s2_n, l9_n


# ── Pre-processing (Section 1–4) ─────────────────────────────────────────────

def mask_and_scale_s2(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit).eq(0))
    )
    return (
        image.updateMask(mask)
        .divide(10000)
        .copyProperties(image, ["system:time_start"])
    )


def mask_l9_clouds(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    mask = (
        qa.bitwiseAnd(cloud_bit)
        .eq(0)
        .And(qa.bitwiseAnd(shadow_bit).eq(0))
    )
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])


def build_indices(
    aoi: ee.Geometry,
    s2_start: str,
    s2_end: str,
    l9_start: str,
    l9_end: str,
) -> dict:
    """Compute indices, water mask, classifications, and RGB visualisations."""
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(s2_start, s2_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_S2))
        .map(mask_and_scale_s2)
    )
    s2 = s2_col.median()

    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndci = s2.normalizedDifference(["B5", "B4"]).rename("NDCI")
    tss = s2.select("B4").divide(s2.select("B8")).rename("TSS")

    dw_col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(aoi)
        .filterDate(s2_start, s2_end)
    )
    dw_water = dw_col.select("water").median().rename("DW_Water_Probability")
    water_mask = (
        dw_water.gte(0.50)
        .Or(ndwi.gt(0.05))
        .selfMask()
        .rename("Water_Mask")
    )

    l9_col = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(l9_start, l9_end)
        .map(mask_l9_clouds)
    )
    l9 = l9_col.median()
    wst = (
        l9.select("ST_B10")
        .multiply(0.00341802)
        .add(149)
        .subtract(273.15)
        .rename("WST")
    )
    wst_aligned = wst.resample("bilinear").reproject(crs="EPSG:4326", scale=SCALE)
    wst_water = wst_aligned.updateMask(water_mask).rename("WST_Water")

    # ── Classifications (Section 9) ──────────────────────────────────────────
    ndwi_class = (
        ee.Image(1)
        .where(ndwi.gte(0), 2)
        .rename("NDWI_Class")
        .updateMask(ndwi.mask())
    )
    tss_class = (
        ee.Image(1)
        .where(tss.gt(0.62).And(tss.lte(1.17)), 2)
        .where(tss.gt(1.17), 3)
        .rename("Turbidity_Class")
        .updateMask(tss.mask())
    )
    ndci_class = (
        ee.Image(1)
        .where(ndci.gte(0), 2)
        .rename("NDCI_Class")
        .updateMask(ndci.mask())
    )
    wst_class = (
        ee.Image(1)
        .where(wst_water.gte(27).And(wst_water.lt(30)), 2)
        .where(wst_water.gte(30).And(wst_water.lt(33)), 3)
        .where(wst_water.gte(33).And(wst_water.lt(36)), 4)
        .where(wst_water.gte(36), 5)
        .rename("WST_Class")
        .updateMask(wst_water.mask())
    )

    clipped = lambda img: img.clip(aoi)  # noqa: E731

    # Colourful RGB layers (what you see on the GEE map)
    rgb = {
        "NDWI_class_rgb": clipped(ndwi_class).visualize(min=1, max=2, palette=NDWI_PALETTE),
        "Turbidity_class_rgb": clipped(tss_class).visualize(min=1, max=3, palette=TSS_PALETTE),
        "NDCI_class_rgb": clipped(ndci_class).visualize(min=1, max=2, palette=NDCI_PALETTE),
        "WST_class_rgb": clipped(wst_class).visualize(min=1, max=5, palette=WST_PALETTE),
        "NDWI_continuous_rgb": clipped(ndwi).visualize(min=-1, max=1, palette=NDWI_CONT_PALETTE),
        "NDCI_continuous_rgb": clipped(ndci).visualize(min=-1, max=1, palette=NDCI_CONT_PALETTE),
        "Turbidity_continuous_rgb": clipped(tss).visualize(min=0, max=5, palette=TSS_CONT_PALETTE),
        "WST_continuous_rgb": clipped(wst_water).visualize(min=15, max=40, palette=WST_CONT_PALETTE),
    }

    # Raw float stack for analysis
    float_stack = (
        clipped(ndwi)
        .addBands(clipped(ndci))
        .addBands(clipped(tss))
        .addBands(clipped(wst_aligned))
        .addBands(clipped(wst_water))
        .toFloat()
    )

    byte_stack = (
        clipped(ndwi_class)
        .addBands(clipped(tss_class))
        .addBands(clipped(ndci_class))
        .addBands(clipped(wst_class))
        .toByte()
    )

    return {
        "s2_count": s2_col.size(),
        "l9_count": l9_col.size(),
        "rgb": rgb,
        "float_stack": float_stack,
        "byte_stack": byte_stack,
    }


# ── Export ───────────────────────────────────────────────────────────────────

def export_local(image: ee.Image, path: str, aoi: ee.Geometry, scale: int = SCALE) -> None:
    """Download a GeoTIFF via the Earth Engine REST API (small/medium AOIs)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = image.getDownloadURL(
        {
            "scale": scale,
            "crs": "EPSG:4326",
            "region": aoi,
            "format": "GEO_TIFF",
        }
    )
    print(f"  downloading -> {path}")
    urllib.request.urlretrieve(url, path)


def export_drive(
    image: ee.Image,
    description: str,
    prefix: str,
    aoi: ee.Geometry,
    folder: str = "GEE_Exports",
    scale: int = SCALE,
) -> ee.batch.Task:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=prefix,
        scale=scale,
        region=aoi,
        crs="EPSG:4326",
        maxPixels=1e13,
    )
    task.start()
    return task


def wait_for_tasks(tasks: list[ee.batch.Task], poll_sec: int = 30) -> None:
    pending = list(tasks)
    while pending:
        still = []
        for t in pending:
            status = t.status()
            state = status.get("state")
            desc = status.get("description", "?")
            if state in ("READY", "RUNNING"):
                still.append(t)
                print(f"  [{state}] {desc}")
            elif state == "COMPLETED":
                print(f"  [DONE]  {desc}")
            else:
                print(f"  [FAIL]  {desc}: {status.get('error_message', state)}")
        pending = still
        if pending:
            time.sleep(poll_sec)


def export_png(image: ee.Image, path: str, aoi: ee.Geometry, dimensions: int = 1024) -> None:
    """Download PNG thumbnail for KML GroundOverlay."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = image.getThumbURL({"dimensions": dimensions, "region": aoi, "format": "png"})
    print(f"  downloading -> {path}")
    urllib.request.urlretrieve(url, path)


def _kml_el(parent: ET.Element, tag: str, text: str | None = None) -> ET.Element:
    """Create a KML namespaced child element."""
    el = ET.SubElement(parent, f"{{{KML_NS}}}{tag}")
    if text is not None:
        el.text = text
    return el


KML_NS = "http://www.opengis.net/kml/2.2"

# Layer metadata for KML folders (classification layers shown by default)
KML_LAYERS = [
    ("NDWI_class_rgb", "NDWI — Water / Land", True),
    ("Turbidity_class_rgb", "Turbidity — Low / Med / High", True),
    ("NDCI_class_rgb", "NDCI — Chlorophyll", False),
    ("WST_class_rgb", "WST — Water Temperature", False),
    ("NDWI_continuous_rgb", "NDWI (continuous)", False),
    ("NDCI_continuous_rgb", "NDCI (continuous)", False),
    ("Turbidity_continuous_rgb", "Turbidity (continuous)", False),
    ("WST_continuous_rgb", "WST (continuous)", False),
]


def write_water_quality_kml(
    out_path: str,
    png_dir: str,
    box: dict[str, float],
    ring: list[list[float]],
    tag: str,
    s2_start: str,
    s2_end: str,
    l9_start: str,
    l9_end: str,
) -> None:
    """Build a multi-layer KML with colourful GroundOverlays for Google Earth."""
    root = ET.Element(f"{{{KML_NS}}}kml")
    doc = _kml_el(root, "Document")

    _kml_el(doc, "name", f"JalNetra Water Quality — {tag}")
    _kml_el(
        doc,
        "description",
        (
            f"Sentinel-2: {s2_start} to {s2_end}\n"
            f"Landsat 9: {l9_start} to {l9_end}\n\n"
            "Toggle folders to switch layers.\n\n"
            "Legend:\n"
            "NDWI — brown=land, blue=water\n"
            "Turbidity — blue=low, yellow=medium, red=high\n"
            "NDCI — light green=low chlorophyll, dark green=high\n"
            "WST — blue=cool … red=hot (water pixels only)"
        ),
    )

    # AOI boundary outline
    pm = _kml_el(doc, "Placemark")
    _kml_el(pm, "name", "AOI boundary")
    _kml_el(pm, "visibility", "1")
    poly = _kml_el(pm, "Polygon")
    outer = _kml_el(poly, "outerBoundaryIs")
    lr = _kml_el(outer, "LinearRing")
    coord_str = " ".join(f"{lon},{lat},0" for lon, lat in ring)
    _kml_el(lr, "coordinates", coord_str)
    line = _kml_el(pm, "Style")
    ls = _kml_el(line, "LineStyle")
    _kml_el(ls, "color", "ff2dc0fb")
    _kml_el(ls, "width", "3")
    ps = _kml_el(line, "PolyStyle")
    _kml_el(ps, "color", "00000000")

    for key, label, visible in KML_LAYERS:
        png_name = f"{key}.png"
        png_path = os.path.join(png_dir, png_name)
        if not os.path.isfile(png_path):
            continue

        folder = _kml_el(doc, "Folder")
        _kml_el(folder, "name", label)
        _kml_el(folder, "visibility", "1" if visible else "0")

        overlay = _kml_el(folder, "GroundOverlay")
        _kml_el(overlay, "name", label)
        _kml_el(overlay, "visibility", "1" if visible else "0")
        icon = _kml_el(overlay, "Icon")
        _kml_el(icon, "href", png_name)
        llb = _kml_el(overlay, "LatLonBox")
        _kml_el(llb, "north", f"{box['north']:.8f}")
        _kml_el(llb, "south", f"{box['south']:.8f}")
        _kml_el(llb, "east", f"{box['east']:.8f}")
        _kml_el(llb, "west", f"{box['west']:.8f}")

    xml_bytes = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(pretty)
    print(f"  wrote KML -> {out_path}")


def write_legend(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Classification Legend\n")
        f.write("=" * 40 + "\n\n")
        f.write("NDWI Class\n")
        f.write("  1 = Land / Non-water (< 0)     #8D6E63\n")
        f.write("  2 = Water (>= 0)               #0D47A1\n\n")
        f.write("Turbidity (TSS) Class\n")
        f.write("  1 = Low (<= 0.62)              #2196F3\n")
        f.write("  2 = Medium (0.62–1.17)         #FFC107\n")
        f.write("  3 = High (> 1.17)              #F44336\n\n")
        f.write("NDCI (Chlorophyll) Class\n")
        f.write("  1 = Low (< 0)                  #C8E6C9\n")
        f.write("  2 = High (>= 0)                #1B5E20\n\n")
        f.write("WST (Temperature) Class — water pixels only\n")
        f.write("  1 = Very Low (< 27°C)          #1565C0\n")
        f.write("  2 = Low (27–30°C)              #64B5F6\n")
        f.write("  3 = Moderate (30–33°C)         #FFD54F\n")
        f.write("  4 = High (33–36°C)             #E53935\n")
        f.write("  5 = Very High (>= 36°C)        #8E0000\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def resolve_kml(path: str | None) -> str:
    if path and os.path.isfile(path):
        return path
    for candidate in (DEFAULT_KML, FALLBACK_KML):
        if os.path.isfile(candidate):
            return candidate
    sys.exit(
        "KML not found. Pass --kml path/to/file.kml "
        f"(tried {DEFAULT_KML} and {FALLBACK_KML})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export colourful water-quality KML / GeoTIFF from GEE using a KML AOI."
    )
    parser.add_argument("--kml", help="Path to input KML polygon (default: Downloads copy)")
    parser.add_argument(
        "--mode",
        choices=["june", "latest", "auto"],
        default="auto",
        help="Date window: June 1st week 2026, latest imagery, or auto (try June then latest)",
    )
    parser.add_argument(
        "--format",
        choices=["kml", "geotiff", "both"],
        default="kml",
        help="Output format (default: kml — colourful layers for Google Earth)",
    )
    parser.add_argument(
        "--drive",
        action="store_true",
        help="Export GeoTIFFs to Google Drive instead of downloading locally",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help=f"Local output folder (default: {OUTPUT_DIR})",
    )
    parser.add_argument("--year", type=int, default=2026, help="Year for --mode june")
    parser.add_argument(
        "--project",
        help="Google Cloud project ID for Earth Engine (or set EE_PROJECT env var)",
    )
    args = parser.parse_args()

    kml_path = resolve_kml(args.kml)
    init_ee(project=args.project)

    print("=" * 64)
    print("JalNetra — Water Quality GEE Export")
    print("=" * 64)
    print(f"[kml]   {kml_path}")

    aoi, ring = parse_kml_polygon(kml_path)
    box = aoi_latlon_box(aoi)

    if args.mode == "june":
        s2_start, s2_end, l9_start, l9_end = june_first_week_dates(args.year)
    elif args.mode == "latest":
        s2_start, s2_end, l9_start, l9_end = latest_dates(aoi)
    else:
        s2_start, s2_end, l9_start, l9_end = june_first_week_dates(args.year)
        s2_n, l9_n = count_images(aoi, s2_start, s2_end)
        if s2_n == 0:
            print(f"[dates] No S2 images in June week — switching to latest")
            s2_start, s2_end, l9_start, l9_end = latest_dates(aoi)
        else:
            print(f"[dates] Using June 1st week ({s2_n} S2, {l9_n} L9 in range)")

    tag = f"S2_{s2_start}_{s2_end}_L9_{l9_start}_{l9_end}"
    print(f"[dates] Sentinel-2 : {s2_start} -> {s2_end}")
    print(f"[dates] Landsat 9  : {l9_start} -> {l9_end}")

    result = build_indices(aoi, s2_start, s2_end, l9_start, l9_end)
    print(f"[data]  S2 images  : {result['s2_count'].getInfo()}")
    print(f"[data]  L9 images  : {result['l9_count'].getInfo()}")

    if args.drive:
        print("[export] Starting Google Drive tasks …")
        tasks = []
        for name, img in result["rgb"].items():
            tasks.append(
                export_drive(
                    img,
                    description=f"{name}_{tag}",
                    prefix=f"{name}_{tag}",
                    aoi=aoi,
                )
            )
        tasks.append(
            export_drive(
                result["float_stack"],
                description=f"WaterQuality_Indices_{tag}",
                prefix=f"WaterQuality_Indices_{tag}",
                aoi=aoi,
            )
        )
        tasks.append(
            export_drive(
                result["byte_stack"],
                description=f"WaterQuality_Classes_{tag}",
                prefix=f"WaterQuality_Classes_{tag}",
                aoi=aoi,
            )
        )
        wait_for_tasks(tasks)
        print("[export] Drive exports finished — check GEE_Exports folder in Drive")
        return

    out_dir = os.path.join(args.output, tag)
    os.makedirs(out_dir, exist_ok=True)

    want_kml = args.format in ("kml", "both")
    want_tif = args.format in ("geotiff", "both")

    if want_kml:
        print(f"[export] KML + PNG overlays -> {out_dir}")
        for name, img in result["rgb"].items():
            export_png(img, os.path.join(out_dir, f"{name}.png"), aoi)
        kml_path_out = os.path.join(out_dir, "water_quality.kml")
        write_water_quality_kml(
            kml_path_out,
            out_dir,
            box,
            ring,
            tag,
            s2_start,
            s2_end,
            l9_start,
            l9_end,
        )

    if want_tif:
        print(f"[export] GeoTIFFs -> {out_dir}")
        for name, img in result["rgb"].items():
            export_local(img, os.path.join(out_dir, f"{name}.tif"), aoi)
        export_local(
            result["float_stack"],
            os.path.join(out_dir, "WaterQuality_Indices.tif"),
            aoi,
        )
        export_local(
            result["byte_stack"],
            os.path.join(out_dir, "WaterQuality_Classes.tif"),
            aoi,
        )

    write_legend(os.path.join(out_dir, "legend.txt"))

    print(f"[done]  Output folder: {out_dir}")
    if want_kml:
        print(f"  Open in Google Earth: {os.path.join(out_dir, 'water_quality.kml')}")
        print("  Layers: NDWI, Turbidity, NDCI, WST (toggle folders on/off)")
    if want_tif:
        print("  GeoTIFF layers:")
        for name in result["rgb"]:
            print(f"    • {name}.tif")


if __name__ == "__main__":
    main()
