"""Fetch aligned before/after Sentinel-2 crops for an AOI from Earth Search (STAC).

Flow: search STAC for each date window -> measure cloud cover *over the AOI* using the
Scene Classification Layer (SCL) -> pick the clearest scene per window -> read only the AOI
window from the cloud-optimized GeoTIFFs -> write both dates onto one shared pixel grid.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from impactlens.data.preprocess import FLOOD_BAND_NAMES, dn_to_reflectance

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
RESOLUTION = 10.0  # metres; the native resolution of the RGB+NIR bands

# SCL classes: 3 = cloud shadow, 8/9 = medium/high probability cloud, 10 = thin cirrus.
SCL_CLOUD_CLASSES = (3, 8, 9, 10)
SCL_NODATA = 0
MAX_NODATA_FRACTION = 0.05  # scene must cover at least 95% of the AOI
MAX_CANDIDATES = 12  # how many scenes per window we measure (cheapest tile-cloud first)

# Read only what we need from remote COGs.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


# --------------------------------------------------------------------------- AOI and grid


def load_aoi(path: str | Path) -> BaseGeometry:
    """Load an AOI (lon/lat, EPSG:4326) from a GeoJSON Feature, FeatureCollection or geometry."""
    gj = json.loads(Path(path).read_text())
    kind = gj.get("type")
    if kind == "FeatureCollection":
        return unary_union([shape(f["geometry"]) for f in gj["features"]])
    if kind == "Feature":
        return shape(gj["geometry"])
    return shape(gj)


@dataclass(frozen=True)
class Grid:
    """A pixel grid: CRS + affine transform + size. Both dates are read onto the same Grid."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        t = self.transform
        return (t.c, t.f + t.e * self.height, t.c + t.a * self.width, t.f)


def make_grid(aoi: BaseGeometry, crs: CRS, resolution: float = RESOLUTION) -> Grid:
    """Smallest grid in ``crs`` that covers the AOI, with edges snapped to the resolution."""
    west, south, east, north = transform_bounds("EPSG:4326", crs, *aoi.bounds, densify_pts=21)
    left = math.floor(west / resolution) * resolution
    top = math.ceil(north / resolution) * resolution
    width = math.ceil((east - left) / resolution)
    height = math.ceil((top - south) / resolution)
    return Grid(crs, from_origin(left, top, resolution, resolution), width, height)


# ----------------------------------------------------------------------- reading rasters


def read_band(href: str, grid: Grid, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Read one band onto ``grid``. Only the blocks that overlap the grid are downloaded."""
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        with WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=resampling,
            src_nodata=0,
            nodata=0,
        ) as vrt:
            return vrt.read(1)


def cloud_stats(scl: np.ndarray) -> tuple[float, float]:
    """Return (cloud fraction among valid pixels, nodata fraction) from an SCL array."""
    nodata = float(np.mean(scl == SCL_NODATA))
    valid = scl != SCL_NODATA
    if not valid.any():
        return 1.0, 1.0
    cloudy = np.isin(scl, SCL_CLOUD_CLASSES) & valid
    return float(cloudy.sum() / valid.sum()), nodata


def _href(item, key: str) -> str:
    href = item.assets[key].href
    # Some catalogs return s3:// URLs for the public Sentinel COG bucket; use plain HTTPS.
    if href.startswith("s3://sentinel-cogs/"):
        return href.replace(
            "s3://sentinel-cogs/", "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
        )
    return href


def item_crs(item) -> CRS:
    """The scene's native CRS (UTM zone), from STAC metadata when available."""
    props = item.properties
    if props.get("proj:epsg"):
        return CRS.from_epsg(int(props["proj:epsg"]))
    if props.get("proj:code"):
        return CRS.from_string(props["proj:code"])
    with rasterio.Env(**GDAL_ENV), rasterio.open(_href(item, "scl")) as src:
        return src.crs


def reflectance_params(item, key: str, dn: np.ndarray) -> tuple[float, float, str]:
    """Return (scale, offset, source) for converting one band's DN to reflectance.

    Sentinel-2 processing baseline 04.00+ stores DN with a +1000 shift (offset -0.1 in STAC),
    but Earth Search may already have removed it. Applying it twice clips most pixels to zero
    (we hit this on Valencia: 92% zeros). Decision order:

    1. ``earthsearch:boa_offset_applied`` in the item properties (authoritative).
    2. ``s2:processing_baseline``: >= 04.00 means the shift is in the data, older means no shift.
    3. Fallback: look at the pixels. Real reflectance can't be negative, so non-zero DNs well
       below 1000 mean the shift is already gone.

    ``source`` says which rule decided, so it can be logged.
    """
    bands = item.assets[key].extra_fields.get("raster:bands") or [{}]
    scale = float(bands[0].get("scale", 1e-4))
    shift = float(bands[0].get("offset", -0.1))  # the offset to apply if the shift is present
    props = item.properties

    applied = props.get("earthsearch:boa_offset_applied")
    if applied is not None:
        # "applied" = Earth Search already removed the shift, so we must not subtract again.
        return (
            scale,
            (0.0 if applied else shift),
            f"metadata (earthsearch:boa_offset_applied={applied})",
        )

    baseline = props.get("s2:processing_baseline")
    try:
        needs_shift = float(baseline) >= 4.0
    except (TypeError, ValueError):
        needs_shift = None
    if needs_shift is not None:
        return (
            scale,
            (shift if needs_shift else 0.0),
            f"metadata (s2:processing_baseline={baseline})",
        )

    valid = dn[dn > 0]
    shifted_in_data = not (valid.size and np.percentile(valid, 1) < 0.9 * (-shift / scale))
    return scale, (shift if shifted_in_data else 0.0), "fallback (pixel values)"


# --------------------------------------------------------------------- scene selection


@dataclass
class Candidate:
    item: object = field(repr=False)
    item_id: str
    datetime: str
    tile_cloud: float | None  # whole-tile cloud % from STAC metadata
    aoi_cloud: float  # cloud % over the AOI, measured from SCL
    aoi_nodata: float  # % of the AOI with no data in this scene

    @property
    def usable(self) -> bool:
        return self.aoi_nodata <= MAX_NODATA_FRACTION * 100


class NoSceneError(RuntimeError):
    """No scene met the cloud limit. Carries the best candidates per window for display."""

    def __init__(self, max_cloud: float, failures: dict[str, list[Candidate]]):
        self.max_cloud = max_cloud
        self.failures = failures
        super().__init__(f"No scene with <= {max_cloud:g}% cloud over the AOI in: {list(failures)}")


def search_items(aoi: BaseGeometry, window: str, stac_url: str = EARTH_SEARCH_URL) -> list:
    """All sentinel-2-l2a scenes intersecting the AOI in a 'start/end' date window."""
    from pystac_client import Client

    search = Client.open(stac_url).search(
        collections=[COLLECTION], intersects=mapping(aoi), datetime=window, max_items=200
    )
    return list(search.items())


def score_candidate(item, aoi: BaseGeometry) -> Candidate:
    """Measure cloud and nodata over the AOI by reading only the SCL window."""
    grid = make_grid(aoi, item_crs(item))
    scl = read_band(_href(item, "scl"), grid, Resampling.nearest)
    cloud, nodata = cloud_stats(scl)
    return Candidate(
        item=item,
        item_id=item.id,
        datetime=item.datetime.isoformat() if item.datetime else "",
        tile_cloud=item.properties.get("eo:cloud_cover"),
        aoi_cloud=round(cloud * 100, 2),
        aoi_nodata=round(nodata * 100, 2),
    )


def select_scene(
    aoi: BaseGeometry, window: str, max_cloud: float, stac_url: str = EARTH_SEARCH_URL
) -> tuple[Candidate | None, list[Candidate]]:
    """Return (best passing scene or None, all measured candidates sorted clearest first)."""
    items = search_items(aoi, window, stac_url)
    items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    scored = [score_candidate(i, aoi) for i in items[:MAX_CANDIDATES]]
    scored.sort(key=lambda c: (not c.usable, c.aoi_cloud))
    passing = [c for c in scored if c.usable and c.aoi_cloud <= max_cloud]
    return (passing[0] if passing else None), scored


def format_candidates(cands: list[Candidate]) -> str:
    if not cands:
        return "    (no scenes found in this window)"
    lines = [f"    {'scene':<42} {'date':<20} {'tile%':>6} {'AOI cloud%':>11} {'AOI nodata%':>12}"]
    for c in cands:
        tile = "-" if c.tile_cloud is None else f"{c.tile_cloud:.1f}"
        lines.append(
            f"    {c.item_id:<42} {c.datetime[:19]:<20} {tile:>6} {c.aoi_cloud:>11.1f} "
            f"{c.aoi_nodata:>12.1f}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------------------ output


def write_scene(item, grid: Grid, path: Path) -> dict:
    """Read RGB+NIR+SWIR for ``item`` onto ``grid``; save a 6-band float32 reflectance GeoTIFF.

    Band order is ``FLOOD_BAND_NAMES``: blue, green, red, nir, swir16, swir22. The change model
    uses the first four; the flood model uses all six. The 20 m SWIR bands are resampled
    (bilinear) onto the 10 m grid when read.

    Returns which reflectance offset was used and why (see ``reflectance_params``).
    """
    bands = []
    decisions = {}
    for key in FLOOD_BAND_NAMES:
        dn = read_band(_href(item, key), grid, Resampling.bilinear)
        scale, offset, source = reflectance_params(item, key, dn)
        decisions[key] = {"scale": scale, "offset": offset, "source": source}
        bands.append(dn_to_reflectance(dn, scale, offset))
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": len(bands),
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
        "transform": grid.transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        for i, (arr, name) in enumerate(zip(bands, FLOOD_BAND_NAMES, strict=True), start=1):
            dst.write(arr, i)
            dst.set_band_description(i, name)
        dst.update_tags(
            scene_id=item.id, datetime=item.datetime.isoformat() if item.datetime else ""
        )
    return decisions


def fetch_pair(
    aoi_path: str | Path,
    before: str,
    after: str,
    out_dir: str | Path,
    max_cloud: float = 20.0,
    stac_url: str = EARTH_SEARCH_URL,
    log=print,
) -> dict:
    """Pick the clearest scene in each window and save aligned before.tif / after.tif.

    Raises NoSceneError (with the best candidates) if either window has no scene that is
    <= ``max_cloud`` percent cloudy over the AOI.
    """
    aoi = load_aoi(aoi_path)
    out_dir = Path(out_dir)

    chosen: dict[str, Candidate] = {}
    failures: dict[str, list[Candidate]] = {}
    for label, window in (("before", before), ("after", after)):
        log(f"[{label}] searching {window} ...")
        best, cands = select_scene(aoi, window, max_cloud, stac_url)
        log(format_candidates(cands))
        if best is None:
            failures[f"{label} ({window})"] = cands
        else:
            chosen[label] = best
            log(f"[{label}] picked {best.item_id} ({best.aoi_cloud:.1f}% cloud over AOI)")
    if failures:
        raise NoSceneError(max_cloud, failures)

    # Shared grid in the "before" scene's UTM zone; the "after" scene is warped onto it.
    grid = make_grid(aoi, item_crs(chosen["before"].item))
    offsets = {}
    for label in ("before", "after"):
        log(f"[{label}] reading AOI window ({grid.width}x{grid.height} px) ...")
        offsets[label] = write_scene(chosen[label].item, grid, out_dir / f"{label}.tif")
        first = offsets[label]["blue"]
        log(f"[{label}] reflectance offset {first['offset']:+g} via {first['source']}")

    summary = {
        "aoi": str(aoi_path),
        "grid": {
            "crs": grid.crs.to_string(),
            "bounds": grid.bounds,
            "width": grid.width,
            "height": grid.height,
            "resolution_m": RESOLUTION,
        },
        "max_cloud_pct": max_cloud,
        **{
            label: {
                "scene_id": c.item_id,
                "datetime": c.datetime,
                "tile_cloud_pct": c.tile_cloud,
                "aoi_cloud_pct": c.aoi_cloud,
                "window": window,
                "reflectance_offset": offsets[label],
            }
            for (label, c), window in zip(chosen.items(), (before, after), strict=True)
        },
    }
    (out_dir / "scenes.json").write_text(json.dumps(summary, indent=2))
    log(f"Saved {out_dir}/before.tif, after.tif, scenes.json")
    return summary
