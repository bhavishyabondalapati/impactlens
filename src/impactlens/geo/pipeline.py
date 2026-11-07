"""End-to-end Geo-AI steps, called by the CLI (``impactlens infer`` / ``impactlens rollup``)."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from impactlens.geo import layers as L
from impactlens.geo.inference import onnx_predictor


def latest_model(pattern: str) -> Path:
    """Newest file matching a glob, e.g. build/models_*/floods_r2/floods.onnx."""
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no model matches {pattern}; pass the path explicitly")
    return Path(hits[-1])


def _tuned(metrics_path: Path, default: float = 0.5) -> float:
    """The val-tuned probability threshold stored by train.py --tune-threshold (else 0.5)."""
    if not metrics_path.exists():
        return default
    return json.loads(metrics_path.read_text()).get("threshold", {}).get("value", default)


def _index_thresholds(metrics_path: Path) -> dict:
    """NDWI / MNDWI: textbook 0 plus the threshold tuned on Sen1Floods11 val."""
    out = {"ndwi": [0.0], "mndwi": [0.0]}
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        for name in out:
            t = m.get(f"baseline_{name}", {}).get("tuned_threshold")
            if t is not None and t != 0:
                out[name].append(float(t))
    return out


def run_infer(scene_dir: Path, floods_model: Path, oscd_model: Path, out_dir: Path,
              quicklook_png: Path | None, overlap: int = 64, log=print) -> dict:  # fmt: skip
    import rasterio

    with rasterio.open(scene_dir / "before.tif") as src:
        before, profile = src.read(), src.profile
    with rasterio.open(scene_dir / "after.tif") as src:
        after = src.read()
    if before.shape[0] != 6:
        raise ValueError("before.tif must have 6 bands; re-run `impactlens fetch`")

    floods_metrics = floods_model.parent / "metrics.json"
    th = L.Thresholds(
        water=_tuned(floods_metrics), change=_tuned(oscd_model.parent / "metrics.json")
    )
    log(f"models: floods={floods_model} (water threshold {th.water}), oscd={oscd_model}")
    layers = L.compute_layers(before, after, onnx_predictor(floods_model),
                              onnx_predictor(oscd_model), th, overlap)  # fmt: skip
    L.write_layers(layers, profile, out_dir)
    log(f"wrote {len(layers)} GeoTIFFs to {out_dir}/")

    valid = layers["valid"]
    px_ha = abs(profile["transform"].a * profile["transform"].e) / 10_000
    stats = {
        "valid_ha": float(valid.sum() * px_ha),
        **{f"{k}_ha": float(layers[k].sum() * px_ha) for k in L.MASKS if k != "valid"},
        "water_before_ha": float(((layers["water_before"] > th.water) & valid).sum() * px_ha),
        "water_after_ha": float(((layers["water_after"] > th.water) & valid).sum() * px_ha),
    }
    agreement = L.index_agreement(before, after, layers, th, _index_thresholds(floods_metrics))
    summary = {
        "scene_dir": str(scene_dir),
        "models": {"floods": str(floods_model), "oscd": str(oscd_model)},
        "thresholds": th,
        "overlap_px": overlap,
        "pixel_area_ha": px_ha,
        "raster_stats": stats,
        "water_index_agreement": agreement,
    }
    L.save_json(summary, out_dir / "infer_summary.json")
    if quicklook_png:
        scenes = json.loads((scene_dir / "scenes.json").read_text())
        title = (f"Valencia / Albufera  before {scenes['before']['datetime'][:10]}  "
                 f"after {scenes['after']['datetime'][:10]}")  # fmt: skip
        L.quicklook(before, after, layers, quicklook_png, title)
        log(f"quick-look: {quicklook_png}")
    return summary


IMPACT_LAYERS = {  # kind -> (mask layer, confidence layer)
    "flooded": ("flooded", "flood_prob"),
    "changed": ("changed", "change_prob"),
    "ndvi_drop": ("ndvi_dropped", "ndvi_drop"),
}


def run_rollup(scene_dir: Path, layers_dir: Path, aoi_path: Path, out_dir: Path,
               osm_dir: Path, map_png: Path | None = None, ndvi_map_png: Path | None = None,
               log=print) -> dict:  # fmt: skip
    import rasterio
    from rasterio.warp import transform_geom
    from shapely.geometry import box, mapping, shape

    from impactlens.data.stac import load_aoi
    from impactlens.geo import osm, postgis
    from impactlens.geo.polygonize import polygonize

    impacts, profile = {}, None
    for kind, (mask_name, conf_name) in IMPACT_LAYERS.items():
        with rasterio.open(layers_dir / f"{mask_name}.tif") as src:
            mask, profile = src.read(1), src.profile
        with rasterio.open(layers_dir / f"{conf_name}.tif") as src:
            conf = np.clip(src.read(1), 0, 1)  # NDVI drop of 0.4 -> "confidence" 0.4
        impacts[kind] = polygonize(mask, conf, profile["transform"])
        log(f"polygonized {kind}: {len(impacts[kind])} polygons")
    epsg = profile["crs"].to_epsg()

    # The analysis area is the AOI polygon itself (not the raster's UTM bounding box, whose
    # corners stick out of the lon/lat rectangle and have no OSM data).
    aoi = load_aoi(aoi_path)
    aoi_utm = transform_geom("EPSG:4326", f"EPSG:{epsg}", mapping(aoi.segmentize(0.001)))
    west, south, east, north = aoi.bounds
    cache = lambda what: osm_dir / f"{aoi_path.stem}_{what}.json"  # noqa: E731
    bbox = (south, west, north, east)
    landuse = osm.landuse_features(
        osm.fetch_overpass(osm.overpass_query(*bbox), cache("landuse"), log=log)
    )
    coast = osm.fetch_overpass(osm.coastline_query(*bbox), cache("coastline"), log=log)
    land = osm.land_polygon(coast, box(west, south, east, north))
    towns = osm.municipalities(
        osm.fetch_overpass(osm.municipality_query(*bbox), cache("municipalities"), log=log)
    )
    log(f"OSM: {len(landuse)} land-use polygons, {len(towns)} municipalities, "
        f"{len(coast.get('elements', []))} coastline ways (cache {osm_dir}/)")  # fmt: skip

    with postgis.connect() as conn:
        postgis.load(conn, shape(aoi_utm).wkt, epsg, landuse, impacts, mapping(land), towns)
        log("loaded into PostGIS; running spatial SQL ...")
        postgis.rollup(conn)
        infer = json.loads((layers_dir / "infer_summary.json").read_text())
        scenes = json.loads((scene_dir / "scenes.json").read_text())
        context = {
            "aoi": str(aoi_path),
            "before_scene": scenes["before"]["scene_id"],
            "after_scene": scenes["after"]["scene_id"],
            "models": infer["models"],
            "thresholds": infer["thresholds"],
            "raster_stats_ha": infer["raster_stats"],
            "water_index_agreement": infer["water_index_agreement"],
        }
        summary = postgis.build_summary(conn, context)
        paddies = postgis.rice_paddy_outlines(conn)
        summary["rice_paddies"].update(paddy_water_share(layers_dir, paddies, infer))
        town_vegetation(layers_dir, postgis.town_shapes(conn), paddies, summary["by_town"])
        summary["impacts"]["vegetation_loss_likely_mud"]["pct_of_vegetated_land_by_town"] = {
            t["name"]: t["ndvi_drop_pct_of_vegetated"] for t in summary["by_town"]
            if t.get("vegetated_ha_outside_rice_paddies", 0) >= 50  # tiny bases are noise
        }  # fmt: skip
        land_utm = postgis.land_geometry(conn, epsg)
        summary["excluded"].update(sea_check(scene_dir, aoi_utm, land_utm))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        keep = "severity <> 'none' OR changed_ha > 0 OR ndvi_drop_ha > 0"
        postgis.export_geojson(conn, out_dir / "impact_plots.geojson", "plot_impact", keep)
        postgis.export_geojson(conn, out_dir / "hotspots.geojson", "hotspot")
        if map_png:
            L.rollup_map(layers_dir, paddies, postgis.hotspot_shapes(conn), summary, map_png,
                         aoi_utm, land_utm)  # fmt: skip
            log(f"rollup map: {map_png}")
        if ndvi_map_png:
            L.ndvi_drop_map(scene_dir / "after.tif", layers_dir, paddies,
                            postgis.town_shapes(conn), aoi_utm, land_utm, summary,
                            ndvi_map_png)  # fmt: skip
            log(f"NDVI-drop map: {ndvi_map_png}")
    log(f"wrote {out_dir}/summary.json, impact_plots.geojson, hotspots.geojson")
    return summary


def town_vegetation(layers_dir: Path, towns: list[dict], paddies: list[dict], rows: list[dict]):
    """Add each town's vegetated land outside paddies, and NDVI drop as a % of it (in place).

    Raw hectares favour big towns; the rate (drop / ground that COULD show a drop) is the fair
    way to compare a hard-hit town with a quiet one.
    """
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(layers_dir / "ndvi_before.tif") as src:
        vegetated, transform = src.read(1) > 0.3, src.transform
    with rasterio.open(layers_dir / "valid.tif") as src:
        vegetated &= src.read(1).astype(bool)
    hw = vegetated.shape
    if paddies:
        vegetated &= rasterize(((g, 1) for g in paddies), out_shape=hw, transform=transform) == 0
    px_ha = abs(transform.a * transform.e) / 10_000
    shapes = {t["name"]: t["geometry"] for t in towns}
    for row in rows:
        if row["name"] not in shapes:
            continue
        inside = rasterize([(shapes[row["name"]], 1)], out_shape=hw, transform=transform) > 0
        veg_ha = float((vegetated & inside).sum() * px_ha)
        row["vegetated_ha_outside_rice_paddies"] = round(veg_ha, 1)
        drop = row["ndvi_drop_ha_outside_rice_paddies"]
        row["ndvi_drop_pct_of_vegetated"] = round(100 * drop / veg_ha, 1) if veg_ha else None


def sea_check(scene_dir: Path, aoi_utm: dict, land_utm: dict) -> dict:
    """Cross-check the OSM sea mask with the before-date MNDWI (an index, no model involved).

    Nearly all of the OSM sea should be MNDWI water before the event; if it is, the coastline
    mask is right and the 'sea flooding' was the model's domain gap.
    """
    import rasterio
    from rasterio.features import rasterize

    from impactlens.geo.indices import mndwi

    with rasterio.open(scene_dir / "before.tif") as src:
        before, transform = src.read(), src.transform
    hw = before.shape[1:]
    land = rasterize([(land_utm, 1)], out_shape=hw, transform=transform) > 0
    in_aoi = rasterize([(aoi_utm, 1)], out_shape=hw, transform=transform) > 0
    index = mndwi(before)
    sea = in_aoi & ~land & (before.sum(0) > 0)  # nodata pixels are 0 in every band
    n = max(int(sea.sum()), 1)
    return {"mndwi_before_water_pct_of_sea": round(100 * float((index[sea] > 0).sum()) / n, 1)}


def paddy_water_share(layers_dir: Path, paddies: list[dict], infer: dict) -> dict:
    """% of rice-paddy area that the model sees as water on each date.

    If many paddies are already water BEFORE the event, that points at deliberate seasonal
    flooding rather than (only) the disaster.
    """
    import rasterio
    from rasterio.features import rasterize

    t = infer["thresholds"]["water"]
    with rasterio.open(layers_dir / "water_before.tif") as src:
        before, transform, shape = src.read(1), src.transform, src.shape
    with rasterio.open(layers_dir / "water_after.tif") as src:
        after = src.read(1)
    if not paddies:
        return {}
    inside = rasterize(((g, 1) for g in paddies), out_shape=shape, transform=transform) > 0
    n = max(int(inside.sum()), 1)
    return {
        "water_before_pct_of_paddy_area": round(100 * float((before[inside] > t).sum()) / n, 1),
        "water_after_pct_of_paddy_area": round(100 * float((after[inside] > t).sum()) / n, 1),
    }
