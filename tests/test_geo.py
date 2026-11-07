"""Geo-AI on tiny synthetic rasters: blending, flood logic, polygonize, OSM parsing, PostGIS."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape

from impactlens.data.preprocess import FLOOD_BAND_CLIPS
from impactlens.geo import indices, layers, osm, postgis
from impactlens.geo.inference import blend_weights, sliding_window
from impactlens.geo.polygonize import polygonize

UTM = "EPSG:32630"
LEFT, TOP = 720_000.0, 4_360_000.0
TRANSFORM = from_origin(LEFT, TOP, 10, 10)


def _pointwise(x):
    """A fake 'model' that only looks at each pixel: lets us check blending is seamless."""
    return 1 / (1 + np.exp(-(x[:, :1] * 4 - 2)))


# ------------------------------------------------------------------------ inference


def test_blend_weights_are_positive_and_peak_in_the_middle():
    w = blend_weights(64)
    assert w.min() > 0 and w[32, 32] == w.max() and w[0, 0] < w[32, 32]


@pytest.mark.parametrize("h,w", [(300, 260), (50, 40), (64, 64)])
def test_sliding_window_matches_the_pointwise_model_everywhere(h, w):
    img = np.random.default_rng(0).random((6, h, w), dtype=np.float32)
    out = sliding_window(_pointwise, [img], tile=64, overlap=16, batch_size=3)
    assert out.shape == (h, w)
    assert np.allclose(out, _pointwise(img[None])[0, 0], atol=1e-5)  # no seams, edges covered


# --------------------------------------------------------------------- flood logic


def _scene():
    """64x64 reflectance pair: a lake (left) on both dates, new water (top right) after."""
    land = np.array([0.05, 0.08, 0.06, 0.35, 0.25, 0.15], np.float32)[:, None, None]
    water = np.array([0.06, 0.08, 0.04, 0.02, 0.01, 0.01], np.float32)[:, None, None]
    before = np.broadcast_to(land, (6, 64, 64)).copy()
    before[:, :, :20] = water
    after = before.copy()
    after[:, :16, 40:] = water  # new water
    return before, after


def _water_model(x):
    """Fake flood model on NORMALIZED input: water where green beats SWIR1 (like MNDWI)."""
    clips = np.array(FLOOD_BAND_CLIPS, np.float32)[None, :, None, None]
    refl = x * clips
    return (refl[:, 1:2] > refl[:, 4:5]).astype(np.float32) * 0.98 + 0.01


def _no_change(a, b):
    return np.zeros((a.shape[0], 1, *a.shape[-2:]), np.float32)


def test_flooded_is_new_water_only_and_the_lake_is_not_flooded():
    before, after = _scene()
    out = layers.compute_layers(before, after, _water_model, _no_change)
    assert out["flooded"][:16, 40:].all()  # the new water
    assert out["flooded"].sum() == 16 * 24
    assert not out["flooded"][:, :20].any() and out["permanent_water"][:, :20].all()  # lake
    assert out["flood_prob"][:16, 40:].min() > 0.9 and out["flood_prob"][:, :20].max() < 0.05
    assert not out["changed"].any()


def test_ndvi_drop_needs_vegetation_before():
    before, after = _scene()
    out = layers.compute_layers(before, after, _water_model, _no_change)
    # vegetated land turned into water -> big NDVI drop; the lake (not vegetated) does not count
    assert out["ndvi_dropped"][:16, 40:].all() and not out["ndvi_dropped"][:, :20].any()


def test_nodata_pixels_are_never_flooded():
    before, after = _scene()
    before[:, 30:, 40:] = 0  # no data on one date
    after[:, 30:, 40:] = after[:, 0:1, 0:1]  # water after
    out = layers.compute_layers(before, after, _water_model, _no_change)
    assert not out["valid"][30:, 40:].any() and not out["flooded"][30:, 40:].any()


def test_written_layers_keep_georeferencing(tmp_path):
    before, after = _scene()
    out = layers.compute_layers(before, after, _water_model, _no_change)
    profile = {"driver": "GTiff", "width": 64, "height": 64, "crs": UTM, "transform": TRANSFORM}
    layers.write_layers(out, profile, tmp_path)
    with rasterio.open(tmp_path / "flooded.tif") as src:
        assert src.crs.to_string() == UTM and src.transform == TRANSFORM
        assert src.dtypes[0] == "uint8" and src.read(1).sum() == out["flooded"].sum()


def test_town_ndvi_drop_rate_ignores_paddies_and_bare_ground(tmp_path):
    from impactlens.geo.pipeline import town_vegetation

    # 64x64 px of 10 m. Vegetated (NDVI 0.6) in the west 32 columns only; a paddy covers
    # the top half of that. So the town's vegetated land outside the paddy = 32x32 px = 10.24 ha.
    ndvi = np.zeros((64, 64), np.float32)
    ndvi[:, :32] = 0.6
    profile = {"driver": "GTiff", "width": 64, "height": 64, "crs": UTM, "transform": TRANSFORM,
               "count": 1, "dtype": "float32"}  # fmt: skip
    with rasterio.open(tmp_path / "ndvi_before.tif", "w", **profile) as dst:
        dst.write(ndvi, 1)
    with rasterio.open(tmp_path / "valid.tif", "w", **{**profile, "dtype": "uint8"}) as dst:
        dst.write(np.ones((64, 64), np.uint8), 1)
    town = {"name": "Albal", "geometry": mapping(box(LEFT, TOP - 640, LEFT + 640, TOP))}
    paddy = mapping(box(LEFT, TOP - 320, LEFT + 320, TOP))
    rows = [{"name": "Albal", "ndvi_drop_ha_outside_rice_paddies": 2.56}]
    town_vegetation(tmp_path, [town], [paddy], rows)
    assert rows[0]["vegetated_ha_outside_rice_paddies"] == pytest.approx(10.24, abs=0.05)
    assert rows[0]["ndvi_drop_pct_of_vegetated"] == pytest.approx(25.0, abs=0.2)


def test_index_agreement_is_perfect_for_an_index_like_model():
    before, after = _scene()
    out = layers.compute_layers(before, after, _water_model, _no_change)
    agree = layers.index_agreement(before, after, out, layers.Thresholds(), {"ndwi": [0.0],
                                   "mndwi": [0.0]})  # fmt: skip
    assert agree["after/mndwi>0"]["iou"] == 1.0


def test_indices_handle_nodata():
    refl = np.zeros((6, 2, 2), np.float32)
    assert (indices.ndvi(refl) == 0).all() and (indices.mndwi(refl) == 0).all()


# ---------------------------------------------------------------------- polygonize


def test_polygonize_regions_confidence_and_speckle():
    mask = np.zeros((20, 20), bool)
    mask[2:6, 2:6] = True  # 16 px
    mask[10:13, 10:15] = True  # 15 px
    mask[18, 18] = True  # 1 px speckle -> dropped
    conf = np.where(mask, 0.5, 0.0)
    conf[2:6, 2:6] = 0.9
    feats = polygonize(mask, conf, TRANSFORM, min_pixels=4)
    assert len(feats) == 2
    big = max(feats, key=lambda f: f["pixels"])
    assert big["pixels"] == 16 and big["mean_conf"] == pytest.approx(0.9)
    assert shape(big["geometry"]).area == pytest.approx(16 * 100)  # 10 m pixels
    assert shape(big["geometry"]).bounds[0] == LEFT + 20


# ---------------------------------------------------------------------------- OSM


def test_classify_rice_orchard_and_others():
    assert osm.classify({"landuse": "farmland", "crop": "rice"}) == "rice_paddy"
    assert osm.classify({"landuse": "paddy"}) == "rice_paddy"
    assert osm.classify({"landuse": "orchard", "trees": "orange_trees"}) == "orchard"
    assert osm.classify({"landuse": "farmland"}) == "farmland"
    assert osm.classify({"landuse": "retail"}) == "industrial"
    assert osm.classify({"natural": "water"}) == "water"
    assert osm.classify({"landuse": "grass"}) == "other"


def _way(coords):
    return [{"lon": x, "lat": y} for x, y in coords]


def test_relation_multipolygon_with_a_hole():
    outer = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
    inner = [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]
    rel = {"type": "relation", "id": 1, "tags": {"landuse": "farmland", "crop": "rice"},
           "members": [{"type": "way", "role": "outer", "geometry": _way(outer[:3])},
                       {"type": "way", "role": "outer", "geometry": _way(outer[2:])},
                       {"type": "way", "role": "inner", "geometry": _way(inner)}]}  # fmt: skip
    (feat,) = osm.landuse_features({"elements": [rel]})
    assert feat["lu_class"] == "rice_paddy" and shape(feat["geometry"]).area == pytest.approx(15)


def _coast(*pts):
    return {"elements": [{"type": "way", "id": 7, "tags": {"natural": "coastline"},
                          "geometry": [{"lon": x, "lat": y} for x, y in pts]}]}  # fmt: skip


def test_land_is_left_of_the_coastline():
    bbox = box(0, 0, 1, 1)
    # drawn south -> north at x = 0.6: land on the LEFT = west; the way overshoots the bbox
    land = osm.land_polygon(_coast((0.6, -1), (0.6, 2)), bbox)
    assert land.area == pytest.approx(0.6) and land.centroid.x < 0.5
    # drawn north -> south: the sea is west instead
    land = osm.land_polygon(_coast((0.6, 2), (0.6, -1)), bbox)
    assert land.area == pytest.approx(0.4) and land.centroid.x > 0.6
    # a bent coast still works, and no coastline at all means an inland AOI
    land = osm.land_polygon(_coast((0.5, -1), (0.5, 0.5), (0.8, 0.6), (0.8, 2)), bbox)
    assert land.contains(box(0, 0, 0.45, 1)) and not land.intersects(box(0.85, 0, 1, 1))
    assert osm.land_polygon({"elements": []}, bbox).equals(bbox)


def test_municipalities_keep_name_and_outline():
    rel = {"type": "relation", "id": 3, "tags": {"name": "Catarroja", "admin_level": "8"},
           "members": [{"type": "way", "role": "outer",
                        "geometry": [{"lon": x, "lat": y} for x, y in
                                     [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]}]}  # fmt: skip
    (town,) = osm.municipalities({"elements": [rel]})
    assert town["name"] == "Catarroja" and shape(town["geometry"]).area == pytest.approx(1)


def test_open_way_is_skipped():
    way = {"type": "way", "id": 2, "tags": {"landuse": "farmland"},
           "geometry": _way([(0, 0), (1, 0), (1, 1)])}  # fmt: skip
    assert osm.landuse_features({"elements": [way]}) == []


# --------------------------------------------------------------- PostGIS (optional)


@pytest.fixture
def db():
    """A throwaway database in the local PostGIS container; skipped if it is not running."""
    try:
        admin = postgis.connect("impactlens")
    except Exception:
        pytest.skip("PostGIS not running (docker compose up -d)")
    with admin:
        admin.execute("DROP DATABASE IF EXISTS impactlens_test")
        admin.execute("CREATE DATABASE impactlens_test")
    conn = postgis.connect("impactlens_test")
    yield conn
    conn.close()
    with postgis.connect("impactlens") as admin:
        admin.execute("DROP DATABASE IF EXISTS impactlens_test")


def _lonlat(geom):
    return transform_geom(UTM, "EPSG:4326", mapping(geom))


def test_rollup_hectares_severity_paddies_and_hotspots(db):
    # 1 km x 1 km AOI. Rice paddy = west half (50 ha), orchard = east half (50 ha).
    aoi = box(LEFT, TOP - 1000, LEFT + 1000, TOP)
    paddy = box(LEFT, TOP - 1000, LEFT + 500, TOP)
    orchard = box(LEFT + 500, TOP - 1000, LEFT + 1000, TOP)
    landuse = [
        {"osm_id": "way/1", "landuse": "farmland", "crop": "rice", "name": None,
         "lu_class": "rice_paddy", "geometry": _lonlat(paddy)},
        {"osm_id": "way/2", "landuse": "orchard", "crop": None, "name": None,
         "lu_class": "orchard", "geometry": _lonlat(orchard)},
    ]  # fmt: skip
    # Flooded: 20 ha of the paddy (conf 0.8) as 4 nearby blocks, and 5 ha of orchard (conf 0.6).
    blocks = [box(LEFT + x, TOP - 500, LEFT + x + 100, TOP) for x in (0, 120, 240, 360)]
    impacts = {
        "flooded": [{"geometry": mapping(b), "mean_conf": 0.8, "pixels": 500} for b in blocks]
        + [{"geometry": mapping(box(LEFT + 600, TOP - 500, LEFT + 700, TOP)),
            "mean_conf": 0.6, "pixels": 500}],
        "changed": [], "ndvi_drop": [],
    }  # fmt: skip
    postgis.load(db, aoi.wkt, 32630, landuse, impacts)
    postgis.rollup(db)
    s = postgis.build_summary(db, {})

    assert s["totals"]["aoi_ha"] == pytest.approx(100, abs=0.5)
    by = s["by_landuse"]
    assert by["rice_paddy"]["flooded_ha"] == pytest.approx(20, abs=0.2)
    assert by["orchard"]["flooded_ha"] == pytest.approx(5, abs=0.1)
    assert by["orchard"]["flood_mean_conf"] == pytest.approx(0.6, abs=1e-3)
    rp = s["rice_paddies"]
    assert rp["flag"] == "seasonal_flooding_confound" and rp["share_of_flooded_land_pct"] == 80
    assert s["totals"]["flooded_ha_on_land_excluding_rice_paddies"] == pytest.approx(5, abs=0.1)
    # severity: paddy 20/50 * 0.8 = 0.32 -> high; orchard 5/50 * 0.6 = 0.06 -> low
    sev = {p["osm_id"]: p for p in s["top_plots"]}
    assert sev["way/1"]["severity"] == "high" and sev["way/1"]["seasonal_flooding_flag"] is True
    assert sev["way/2"]["severity"] == "low" and sev["way/2"]["seasonal_flooding_flag"] is False
    # hotspots: the 4 paddy blocks are 20 m apart -> one cluster; the lone orchard block is noise
    assert s["n_hotspots"] == 1 and s["hotspots"][0]["main_lu_class"] == "rice_paddy"
    assert s["hotspots"][0]["flooded_ha"] == pytest.approx(20, abs=0.2)


def test_sea_never_enters_a_total_and_impacts_split_by_town(db):
    # 1 km x 1 km AOI: west 800 m is land (paddy 0-400 m, residential orchard town 400-800 m),
    # east 200 m is sea. One town covers the east half of the land (400-800 m).
    aoi = box(LEFT, TOP - 1000, LEFT + 1000, TOP)
    land = box(LEFT, TOP - 1000, LEFT + 800, TOP)
    paddy, orchard = (
        box(LEFT, TOP - 1000, LEFT + 400, TOP),
        box(LEFT + 400, TOP - 500, LEFT + 800, TOP),
    )
    houses = box(LEFT + 400, TOP - 1000, LEFT + 800, TOP - 500)
    landuse = [
        {"osm_id": "way/1", "landuse": "paddy", "crop": None, "name": None,
         "lu_class": "rice_paddy", "geometry": _lonlat(paddy)},
        {"osm_id": "way/2", "landuse": "orchard", "crop": None, "name": None,
         "lu_class": "orchard", "geometry": _lonlat(orchard)},
        {"osm_id": "way/3", "landuse": "residential", "crop": None, "name": None,
         "lu_class": "residential", "geometry": _lonlat(houses)},
    ]  # fmt: skip
    town = {"osm_id": "relation/5", "name": "Massanassa",
            "geometry": _lonlat(box(LEFT + 400, TOP - 1000, LEFT + 1000, TOP))}  # fmt: skip

    def poly(x0, y0, x1, y1, conf=0.9):
        g = box(LEFT + x0, TOP - y1, LEFT + x1, TOP - y0)
        return {"geometry": mapping(g), "mean_conf": conf, "pixels": 100}

    impacts = {
        # 4 ha flooded in the paddy, 10 ha flooded in the SEA (must not count anywhere)
        "flooded": [poly(0, 0, 200, 200), poly(820, 0, 1000, 555.6)],
        # 2 ha NDVI drop in the orchard, 1 ha in the paddy
        "ndvi_drop": [poly(500, 0, 600, 200), poly(0, 900, 100, 1000)],
        # 3 ha change on residential, 1 ha on the orchard (not urban)
        "changed": [poly(400, 600, 700, 700), poly(700, 0, 800, 100)],
    }
    postgis.load(db, aoi.wkt, 32630, landuse, impacts, _lonlat(land), [town])
    postgis.rollup(db)
    s = postgis.build_summary(db, {})

    assert "sea" not in s["by_landuse"]
    assert s["totals"]["land_ha"] == pytest.approx(80, abs=0.5)
    assert s["totals"]["flooded_ha_on_land"] == pytest.approx(4, abs=0.1)
    assert s["excluded"]["sea_flooded_ha"] == pytest.approx(10, abs=0.2)
    imp = s["impacts"]
    assert imp["standing_water"]["in_rice_paddies_ha"] == pytest.approx(4, abs=0.1)
    assert imp["standing_water"]["outside_rice_paddies_ha"] == pytest.approx(0, abs=0.01)
    veg = imp["vegetation_loss_likely_mud"]
    assert veg["ndvi_drop_ha_outside_rice_paddies"] == pytest.approx(2, abs=0.05)
    assert veg["ndvi_drop_ha_in_rice_paddies_not_counted"] == pytest.approx(1, abs=0.05)
    assert veg["by_town"]["Massanassa"] == pytest.approx(2, abs=0.05)
    assert imp["urban_change"]["changed_ha_on_urban_land"] == pytest.approx(3, abs=0.05)
    (t,) = s["by_town"]
    # the town is 60 ha, but only its 40 ha of land is inside the AOI (20 ha is sea)
    assert t["land_in_aoi_ha"] == pytest.approx(40, abs=0.5)
    assert t["ndvi_drop_by_landuse"] == {"orchard": pytest.approx(2, abs=0.05)}
    assert t["urban_changed_ha"] == pytest.approx(3, abs=0.05)


def test_geojson_export_is_lonlat(db, tmp_path):
    aoi = box(LEFT, TOP - 200, LEFT + 200, TOP)
    land = [{"osm_id": "way/9", "landuse": "orchard", "crop": None, "name": "x",
             "lu_class": "orchard", "geometry": _lonlat(aoi)}]  # fmt: skip
    flood = {"geometry": mapping(box(LEFT, TOP - 100, LEFT + 100, TOP)), "mean_conf": 0.9,
             "pixels": 100}  # fmt: skip
    imp = {"flooded": [flood], "changed": [], "ndvi_drop": []}
    postgis.load(db, aoi.wkt, 32630, land, imp)
    postgis.rollup(db)
    path = postgis.export_geojson(db, tmp_path / "plots.geojson")
    fc = json.loads(path.read_text())
    (feat,) = fc["features"]
    lon, lat = shape(feat["geometry"]).centroid.coords[0]
    assert -4 < lon < 0 and 39 < lat < 40  # Spain, not UTM metres
    assert feat["properties"]["flooded_ha"] == pytest.approx(1, abs=0.05)
    assert "geom" not in feat["properties"]


def test_osm_cache_is_refetched_when_the_query_changes(tmp_path, monkeypatch):
    cache = tmp_path / "aoi_landuse.json"
    cache.write_text(json.dumps({"_query": "old bbox", "elements": [1]}))
    assert osm.fetch_overpass("old bbox", cache) == {"_query": "old bbox", "elements": [1]}

    class Resp:  # a fake Overpass answer
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"elements": [2]}'

    monkeypatch.setattr(osm.urllib.request, "urlopen", lambda *a, **k: Resp())
    data = osm.fetch_overpass("new bbox", cache, log=lambda *a: None)
    assert data["elements"] == [2] and json.loads(cache.read_text())["_query"] == "new bbox"
