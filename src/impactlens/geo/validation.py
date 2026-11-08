"""External validation: our footprint layers vs the Copernicus EMS EMSR773 flood delineations.

Copernicus EMS Rapid Mapping mapped the Valencia floods (activation EMSR773) from satellite
images taken between Oct 30 and Nov 8, 2024. Our after scene is Nov 10. We compare, pixel by
pixel on our 10 m grid:

    ours                                     theirs (EMS "observed event" polygons)
    standing water (flooded layer)      vs   flooded area on each date (Oct 31 ... Nov 8)
    footprint = standing water          vs   maximum flood extent (union of all their dates)
                + vegetation loss (NDVI drop outside rice paddies)
                (+ OSCD change on residential / industrial land, as a variant)

Both sides are only compared inside one common domain: land (OSM coastline) inside our AOI,
inside the EMS area of interest and image footprint, valid pixels, and not water on both of
our dates (the Albufera lake, canals: "permanent water" is not a flood for either side).

Metrics (all in hectares on that domain):
    precision = share of our pixels inside their polygons    (1 - over-call rate)
    recall    = share of their pixels we also flag            (1 - miss rate)
    IoU       = intersection / union
    lift      = precision / (share of the domain they flag): how many times more likely our
                pixel is to be in their footprint than a random land pixel. 1 = no skill.

Source: Copernicus Emergency Management Service (© 2024 European Union), [EMSR773].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FLOOD_KINDS = ("Flooded area", "Flood trace")  # EMS `notation` values in observedEventA
AOI01_PRODUCTS = {  # product folder -> (date of the image(s) EMS analysed, sensor)
    "PRODUCT": ("2024-10-31", "Sentinel-2 / Landsat-8 / Sentinel-1"),
    "MONIT01": ("2024-11-03", "ICEYE (SAR)"),
    "MONIT02": ("2024-11-05", "Sentinel-2"),
    "MONIT03": ("2024-11-06", "Sentinel-1 (SAR)"),
    "MONIT04": ("2024-11-08", "COSMO-SkyMed (SAR)"),
}


# ----------------------------------------------------------------- pure, tested pieces


def overlap(pred: np.ndarray, ref: np.ndarray, domain: np.ndarray, px_ha: float) -> dict:
    """Agreement of two boolean masks inside ``domain`` (see the module docstring)."""
    p, r = pred & domain, ref & domain
    both, either = (p & r).sum(), (p | r).sum()
    n_domain, n_p, n_r = domain.sum(), p.sum(), r.sum()
    precision = both / n_p if n_p else None
    ref_share = n_r / n_domain if n_domain else None
    return {
        "ours_ha": round(float(n_p * px_ha), 1),
        "ems_ha": round(float(n_r * px_ha), 1),
        "both_ha": round(float(both * px_ha), 1),
        "iou": round(float(both / either), 3) if either else None,
        "precision": round(float(precision), 3) if precision is not None else None,
        "recall": round(float(both / n_r), 3) if n_r else None,
        "lift": round(float(precision / ref_share), 1) if precision and ref_share else None,
    }


def ems_geometries(geojson: dict, kinds: tuple[str, ...] = FLOOD_KINDS) -> list[dict]:
    """Polygons of an EMS observedEventA / maximumFloodExtentA file, filtered by `notation`.

    Maximum-flood-extent files use notation "Maximum flood extent"; pass kinds=() for all."""
    from shapely.geometry import mapping, shape
    from shapely.validation import make_valid

    out = []
    for f in geojson["features"]:
        if kinds and f["properties"].get("notation") not in kinds:
            continue
        out.append(mapping(make_valid(shape(f["geometry"]))))
    return out


def rasterize_lonlat(geoms: list[dict], transform, hw: tuple[int, int], crs: str) -> np.ndarray:
    """Burn lon/lat polygons into a boolean mask on our raster grid (pixel centres inside)."""
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom

    if not geoms:
        return np.zeros(hw, bool)
    shapes = ((transform_geom("EPSG:4326", crs, g), 1) for g in geoms)
    return rasterize(shapes, out_shape=hw, transform=transform, dtype="uint8") > 0


def spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation of two equally long lists (None for fewer than 3 pairs)."""
    if len(a) < 3:
        return None
    from scipy.stats import spearmanr

    return round(float(spearmanr(a, b).statistic), 2)


# ----------------------------------------------------------------- loading


@dataclass
class Grid:
    transform: object
    hw: tuple[int, int]
    crs: str
    px_ha: float

    def burn(self, geoms: list[dict]) -> np.ndarray:
        return rasterize_lonlat(geoms, self.transform, self.hw, self.crs)


def _read(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1)


def _ems_file(ems_dir: Path, aoi: str, product: str, layer: str) -> dict:
    name = f"EMSR773_{aoi}_DEL_{product}_{layer}_v1.json"
    return json.loads((ems_dir / name).read_text())


def _ems_domain(ems_dir: Path, aoi: str, product: str, grid: Grid) -> np.ndarray:
    """Where EMS actually looked: their area of interest AND their image footprint(s)."""
    area = grid.burn(ems_geometries(_ems_file(ems_dir, aoi, product, "areaOfInterestA"), ()))
    seen = grid.burn(ems_geometries(_ems_file(ems_dir, aoi, product, "imageFootprintA"), ()))
    return area & seen


# ----------------------------------------------------------------- the comparison


def compare_emsr773(scene_dir: Path, aoi_path: Path, osm_dir: Path, ems_dir: Path,
                    log=print) -> tuple[dict, dict]:  # fmt: skip
    """Run every comparison. Returns (results JSON, masks for the map)."""
    import rasterio
    from shapely.geometry import box, mapping

    from impactlens.data.stac import load_aoi
    from impactlens.geo import osm
    from impactlens.geo.layers import _masks

    layers = scene_dir / "layers"
    with rasterio.open(layers / "flooded.tif") as src:
        t = src.transform
        grid = Grid(t, (src.height, src.width), src.crs.to_string(), abs(t.a * t.e) / 10_000)

    # --- our side
    flooded = _read(layers / "flooded.tif") > 0
    ndvi_dropped = _read(layers / "ndvi_dropped.tif") > 0
    changed = _read(layers / "changed.tif") > 0
    permanent = _read(layers / "permanent_water.tif") > 0
    valid = _read(layers / "valid.tif") > 0

    aoi = load_aoi(aoi_path)
    west, south, east, north = aoi.bounds
    bbox = (south, west, north, east)
    cache = lambda what: osm_dir / f"{aoi_path.stem}_{what}.json"  # noqa: E731
    landuse = osm.landuse_features(osm.fetch_overpass(osm.overpass_query(*bbox), cache("landuse")))
    coast = osm.fetch_overpass(osm.coastline_query(*bbox), cache("coastline"))
    land = osm.land_polygon(coast, box(west, south, east, north))
    towns = osm.municipalities(
        osm.fetch_overpass(osm.municipality_query(*bbox), cache("municipalities"))
    )
    aoi_utm_mask, on_land = _masks(
        _to(grid, mapping(aoi.segmentize(0.001))), _to(grid, mapping(land)), grid.hw, t
    )
    by_class = lambda *c: grid.burn([f["geometry"] for f in landuse if f["lu_class"] in c])  # noqa: E731
    paddy = by_class("rice_paddy")
    urban = by_class("residential", "industrial") & ~by_class("port")

    standing = flooded
    veg_loss = ndvi_dropped & ~paddy
    footprint = standing | veg_loss
    ours = {
        "standing water": standing,
        "vegetation loss (NDVI drop outside paddies)": veg_loss,
        "footprint = standing water + vegetation loss": footprint,
        "footprint + OSCD change on urban land": footprint | (changed & urban),
    }
    base = valid & on_land & ~permanent  # our side of the common domain
    log(f"domain before EMS clipping: {base.sum() * grid.px_ha:,.0f} ha of land")

    # --- their side, AOI01 (whole of our AOI, five dates)
    results = {"source": "Copernicus Emergency Management Service (© 2024 European Union), "
               "[EMSR773] Valencia province (AOI01) and Horta Sud (AOI03): Delineation Maps",
               "our_after_scene": "2024-11-10", "by_date": [], "max_extent": {},
               "horta_sud_vhr": {}, "by_town": []}  # fmt: skip
    for product, (date, sensor) in AOI01_PRODUCTS.items():
        domain = base & _ems_domain(ems_dir, "AOI01", product, grid)
        ref = grid.burn(ems_geometries(_ems_file(ems_dir, "AOI01", product, "observedEventA")))
        row = {"product": f"AOI01 {product}", "ems_image_date": date, "sensor": sensor,
               "days_before_ours": (np.datetime64("2024-11-10") - np.datetime64(date)).item().days,
               "domain_ha": round(float(domain.sum() * grid.px_ha)),
               "standing_water": overlap(standing, ref, domain, grid.px_ha),
               "standing_water_outside_paddies":
                   overlap(standing, ref, domain & ~paddy, grid.px_ha),
               "footprint": overlap(footprint, ref, domain, grid.px_ha)}  # fmt: skip
        results["by_date"].append(row)
        log(f"EMS {date}: {row['standing_water']}")

    # Maximum flood extent = union of every EMS date up to Nov 8 (from the last monitoring).
    domain = base & _ems_domain(ems_dir, "AOI01", "MONIT04", grid)
    max_ext = grid.burn(ems_geometries(
        _ems_file(ems_dir, "AOI01", "MONIT04", "maximumFloodExtentA"), ()))  # fmt: skip
    results["max_extent"] = {
        "domain_ha": round(float(domain.sum() * grid.px_ha)),
        "all_land": {k: overlap(m, max_ext, domain, grid.px_ha) for k, m in ours.items()},
        "outside_paddies": {
            k: overlap(m, max_ext, domain & ~paddy, grid.px_ha) for k, m in ours.items()
        },  # fmt: skip
        "inside_paddies": {
            k: overlap(m, max_ext, domain & paddy, grid.px_ha) for k, m in ours.items()
        },  # fmt: skip
    }

    # --- their side, AOI03 Horta Sud: very high resolution optical (GeoEye-1 / WorldView-2,
    # 0.5 m) on Oct 31, flooded area AND flood traces (mud). Covers Paiporta and its neighbours.
    vhr_domain = base & _ems_domain(ems_dir, "AOI03", "PRODUCT", grid)
    vhr_file = _ems_file(ems_dir, "AOI03", "PRODUCT", "observedEventA")
    vhr = {k: grid.burn(ems_geometries(vhr_file, (k,))) for k in FLOOD_KINDS}
    vhr_all = vhr["Flooded area"] | vhr["Flood trace"]
    results["horta_sud_vhr"] = {
        "ems_image_date": "2024-10-31",
        "sensor": "GeoEye-1 / WorldView-2 (optical, 0.5 m)",
        "domain_ha": round(float(vhr_domain.sum() * grid.px_ha)),
        "flooded_area_or_trace": {
            k: overlap(m, vhr_all, vhr_domain, grid.px_ha) for k, m in ours.items()
        },  # fmt: skip
        "flood_trace_only": {
            k: overlap(m, vhr["Flood trace"], vhr_domain, grid.px_ha) for k, m in ours.items()
        },  # fmt: skip
    }

    # --- per town, outside the rice paddies (inside them both sides see the same paddy water).
    # Two references: the AOI01 maximum extent (whole AOI, 10-20 m sensors) and the AOI03 VHR
    # delineation (Horta Sud only, 0.5 m, sees mud in streets). Our rate is the published one
    # from summary.json (NDVI drop / vegetated land outside paddies), so the numbers match
    # docs/GEO.md.
    summary = json.loads((scene_dir / "geo/summary.json").read_text())
    our_rate = {r["name"]: r.get("ndvi_drop_pct_of_vegetated") for r in summary["by_town"]
                if r.get("vegetated_ha_outside_rice_paddies", 0) >= 50}  # fmt: skip
    vhr_footprint = ours["footprint + OSCD change on urban land"]
    rows = []
    for town in towns:
        t_all = grid.burn([town["geometry"]])
        t_max, t_vhr = t_all & domain & ~paddy, t_all & vhr_domain & ~paddy
        if t_max.sum() * grid.px_ha < 50:  # too little of it in the common domain
            continue
        row = {"town": town["name"], "our_ndvi_drop_rate_pct": our_rate.get(town["name"]),
               "land_ha_outside_paddies": round(float(t_max.sum() * grid.px_ha)),
               "max_extent": overlap(footprint, max_ext, t_max, grid.px_ha)}  # fmt: skip
        row["max_extent_pct_of_land"] = round(
            100 * row["max_extent"]["ems_ha"] / row["land_ha_outside_paddies"], 1
        )
        if t_vhr.sum() * grid.px_ha >= 50:
            row["vhr"] = overlap(vhr_footprint, vhr_all, t_vhr, grid.px_ha)
            row["vhr_land_ha"] = round(float(t_vhr.sum() * grid.px_ha))
            row["vhr_pct_of_land"] = round(100 * row["vhr"]["ems_ha"] / row["vhr_land_ha"], 1)
        rows.append(row)
    rows.sort(key=lambda r: -(r["our_ndvi_drop_rate_pct"] or -1))
    results["by_town"] = rows

    def corr(key):
        pairs = [(r[key], r["our_ndvi_drop_rate_pct"]) for r in rows
                 if key in r and r["our_ndvi_drop_rate_pct"] is not None]  # fmt: skip
        ems_share, rate = zip(*pairs, strict=True) if pairs else ((), ())
        return {"towns": len(pairs), "spearman": spearman(list(ems_share), list(rate))}

    results["town_rank_correlation"] = {
        "our_rate_vs_max_extent_share_outside_paddies": corr("max_extent_pct_of_land"),
        "our_rate_vs_vhr_share_outside_paddies": corr("vhr_pct_of_land"),
        "note": "our rate = NDVI drop / vegetated land outside paddies (towns with >= 50 ha "
                "vegetated); EMS share = % of the town's non-paddy land in the EMS polygons",
    }  # fmt: skip
    masks = {"footprint": footprint, "max_extent": max_ext, "domain": domain,
             "vhr_footprint": vhr_footprint, "vhr": vhr_all, "vhr_domain": vhr_domain,
             "in_aoi": aoi_utm_mask, "on_land": on_land, "towns": towns, "grid": grid}  # fmt: skip
    return results, masks


def _to(grid: Grid, geom: dict) -> dict:
    from rasterio.warp import transform_geom

    return transform_geom("EPSG:4326", grid.crs, geom)


def validation_map(after_tif: Path, masks: dict, results: dict, path: Path) -> Path:
    """Two agreement panels: both (green), only ours = over-call (orange), only EMS = miss (blue).

    Left: the whole AOI vs the AOI01 maximum flood extent. Right: zoom on Horta Sud vs the AOI03
    very-high-resolution delineation (flooded area + flood traces)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.patches import Patch
    from rasterio.warp import transform_geom
    from shapely.geometry import box, shape

    with rasterio.open(after_tif) as src:
        after, b = src.read([1, 2, 3]), src.bounds
    grid = masks["grid"]
    grey = np.clip(after.mean(0) / 0.25, 0, 1) * 0.6 + 0.3
    town_geoms = [(t["name"], shape(transform_geom("EPSG:4326", grid.crs, t["geometry"])))
                  for t in masks["towns"]]  # fmt: skip

    def panel(ax, ours, theirs, dom, title, view):
        ours, theirs = ours & dom, theirs & dom
        img = np.stack([grey] * 3, -1)
        img[masks["in_aoi"] & ~masks["on_land"]] = (0.85, 0.9, 0.97)
        img[~masks["in_aoi"]] = 1.0
        img[ours & theirs] = (0.1, 0.6, 0.25)
        img[ours & ~theirs] = (0.95, 0.5, 0.05)
        img[~ours & theirs] = (0.2, 0.4, 0.9)
        ax.imshow(img, extent=(b.left, b.right, b.bottom, b.top))
        for name, geom in town_geoms:
            for poly in getattr(geom, "geoms", [geom]):
                ax.plot(*poly.exterior.xy, color="k", lw=0.6)
            inside = geom.intersection(view)
            if inside.area > 1.5e6:  # label towns with > 150 ha in view
                p = inside.representative_point()
                ax.text(p.x, p.y, name, fontsize=8, ha="center",
                        bbox={"fc": "white", "alpha": 0.7, "lw": 0, "pad": 1})  # fmt: skip
        x0, y0, x1, y1 = view.bounds
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 9.5))
    m = results["max_extent"]["all_land"]["footprint = standing water + vegetation loss"]
    panel(left, masks["footprint"], masks["max_extent"], masks["domain"],
          "Whole AOI: our footprint (standing water + vegetation loss, Nov 10)\n"
          f"vs EMS maximum flood extent (Oct 30 - Nov 8). IoU {m['iou']:.2f}, "
          f"precision {m['precision']:.2f}, recall {m['recall']:.2f}",
          box(b.left, b.bottom, b.right, b.top))  # fmt: skip
    v = results["horta_sud_vhr"]["flooded_area_or_trace"]["footprint + OSCD change on urban land"]
    rows, cols = np.nonzero(masks["vhr_domain"])
    (vx0, vy0), (vx1, vy1) = (grid.transform * (cols.min(), rows.max()),
                              grid.transform * (cols.max(), rows.min()))  # fmt: skip
    panel(right, masks["vhr_footprint"], masks["vhr"], masks["vhr_domain"],
          "Horta Sud: footprint + OSCD urban change (Nov 10) vs EMS VHR flooded\n"
          f"area + flood traces (Oct 31, 0.5 m). IoU {v['iou']:.2f}, precision "
          f"{v['precision']:.2f}, recall {v['recall']:.2f}",
          box(vx0, vy0, vx1, vy1))  # fmt: skip
    right.legend(handles=[Patch(color=(0.1, 0.6, 0.25), label="both"),
                          Patch(color=(0.95, 0.5, 0.05), label="only ours (over-call)"),
                          Patch(color=(0.2, 0.4, 0.9), label="only EMS (miss)")],
                 loc="lower left", fontsize=9)  # fmt: skip
    fig.text(0.01, 0.01, "Flood delineations: Copernicus Emergency Management Service (© 2024 "
             "European Union), [EMSR773] Valencia province (AOI01), Horta Sud (AOI03).\n"
             "Towns: (c) OpenStreetMap contributors. Imagery: Copernicus Sentinel-2. Compared on "
             "land only, without water present on both of our dates.", fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path
