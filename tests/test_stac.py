"""STAC fetcher tests. No network: scenes are local GeoTIFFs wrapped in fake STAC items."""

import json
from datetime import UTC, datetime

import numpy as np
import pystac
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import box

from impactlens.cli import build_parser, main
from impactlens.data import stac
from impactlens.data.stac import Candidate, NoSceneError, cloud_stats, load_aoi, make_grid

UTM = CRS.from_epsg(32630)
LEFT, TOP, SIZE = 750_000.0, 4_370_000.0, 200  # 200x200 px @10 m = 2 km square


def _write(path, arr, dtype):
    with rasterio.open(
        path, "w", driver="GTiff", height=SIZE, width=SIZE, count=1, dtype=dtype,
        crs=UTM, transform=from_origin(LEFT, TOP, 10, 10),
    ) as dst:  # fmt: skip
        dst.write(arr.astype(dtype), 1)


def _fake_item(tmp_path, name, day, dn, scl_class, **props):
    """A STAC item whose assets are local files: 4 bands of constant DN plus an SCL layer."""
    item = pystac.Item(
        id=name,
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 10, day, tzinfo=UTC),
        properties={"proj:epsg": 32630, "eo:cloud_cover": 5.0, **props},
    )
    raster = [{"scale": 0.0001, "offset": -0.1}]
    for key in ("blue", "green", "red", "nir", "swir16", "swir22"):
        path = tmp_path / f"{name}_{key}.tif"
        _write(path, np.full((SIZE, SIZE), dn), "uint16")
        item.add_asset(key, pystac.Asset(href=str(path), extra_fields={"raster:bands": raster}))
    scl_path = tmp_path / f"{name}_scl.tif"
    _write(scl_path, np.full((SIZE, SIZE), scl_class), "uint8")
    item.add_asset("scl", pystac.Asset(href=str(scl_path)))
    return item


@pytest.fixture
def aoi_file(tmp_path):
    """An AOI (in lon/lat) well inside the fake 2 km scene."""
    west, south, east, north = transform_bounds(UTM, "EPSG:4326", LEFT + 500, TOP - 1500,
                                                LEFT + 1500, TOP - 500)  # fmt: skip
    path = tmp_path / "aoi.geojson"
    path.write_text(json.dumps(box(west, south, east, north).__geo_interface__))
    return path


def test_load_aoi_accepts_feature_collection_and_bare_geometry(tmp_path):
    poly = box(0, 0, 1, 1)
    bare = tmp_path / "bare.geojson"
    bare.write_text(json.dumps(poly.__geo_interface__))
    fc = tmp_path / "fc.geojson"
    fc.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": poly.__geo_interface__}]}))  # fmt: skip
    assert load_aoi(bare).equals(poly)
    assert load_aoi(fc).equals(poly)


def test_repo_aoi_is_about_15_by_19km():
    # ~15 km wide; ~19 km tall since the north edge moved to 39.44 N to take in Paiporta
    aoi = load_aoi("aoi/valencia_albufera.geojson")
    grid = make_grid(aoi, UTM)
    assert 13_000 < grid.width * 10 < 17_000
    assert 17_000 < grid.height * 10 < 21_000


def test_make_grid_is_snapped_and_covers_aoi(aoi_file):
    aoi = load_aoi(aoi_file)
    grid = make_grid(aoi, UTM)
    assert grid.transform.a == 10 and grid.transform.e == -10
    assert grid.transform.c % 10 == 0 and grid.transform.f % 10 == 0
    w, s, e, n = transform_bounds("EPSG:4326", UTM, *aoi.bounds)
    left, bottom, right, top = grid.bounds
    assert left <= w and bottom <= s and right >= e and top >= n


def test_cloud_stats():
    scl = np.array([[0, 4, 4, 9], [8, 4, 3, 10]])  # 1 nodata, 7 valid, 4 cloudy classes
    cloud, nodata = cloud_stats(scl)
    assert nodata == pytest.approx(1 / 8)
    assert cloud == pytest.approx(4 / 7)
    assert cloud_stats(np.zeros((2, 2), dtype=np.uint8)) == (1.0, 1.0)


def test_read_band_matches_grid_and_values(tmp_path, aoi_file):
    item = _fake_item(tmp_path, "s", 1, dn=2000, scl_class=4)
    grid = make_grid(load_aoi(aoi_file), UTM)
    arr = stac.read_band(item.assets["red"].href, grid, Resampling.nearest)
    assert arr.shape == (grid.height, grid.width)
    assert (arr == 2000).all()


def test_read_band_reprojects_between_utm_zones(tmp_path, aoi_file):
    """A scene in a different CRS is warped onto the target grid (the 'after' scene case)."""
    item = _fake_item(tmp_path, "s", 1, dn=1234, scl_class=4)
    grid31 = make_grid(load_aoi(aoi_file), CRS.from_epsg(32631))
    arr = stac.read_band(item.assets["red"].href, grid31, Resampling.nearest)
    assert arr.shape == (grid31.height, grid31.width)
    assert (arr[arr > 0] == 1234).all() and (arr > 0).mean() > 0.9


def _patch_search(monkeypatch, items):
    monkeypatch.setattr(stac, "search_items", lambda aoi, window, url=None: items[window])


def test_fetch_pair_picks_clearest_and_writes_aligned_tifs(tmp_path, aoi_file, monkeypatch):
    cloudy = _fake_item(tmp_path, "cloudy", 5, dn=3000, scl_class=9)
    clear_a = _fake_item(tmp_path, "clear_before", 10, dn=2000, scl_class=4)
    clear_b = _fake_item(tmp_path, "clear_after", 20, dn=3000, scl_class=4)
    _patch_search(monkeypatch, {"b": [cloudy, clear_a], "a": [cloudy, clear_b]})

    out = tmp_path / "out"
    summary = stac.fetch_pair(aoi_file, "b", "a", out, max_cloud=20, log=lambda *_: None)

    assert summary["before"]["scene_id"] == "clear_before"
    assert summary["after"]["scene_id"] == "clear_after"
    with rasterio.open(out / "before.tif") as b, rasterio.open(out / "after.tif") as a:
        assert b.count == a.count == 6
        assert [b.descriptions[i] for i in range(6)] == [
            "blue",
            "green",
            "red",
            "nir",
            "swir16",
            "swir22",
        ]
        assert b.read(6).mean() == pytest.approx(0.1, abs=1e-5)  # SWIR is read like the others
        assert b.transform == a.transform and b.crs == a.crs and b.shape == a.shape
        # DN 2000 * 1e-4 - 0.1 offset = 0.1 reflectance
        assert b.read(1).mean() == pytest.approx(0.1, abs=1e-5)
        assert a.read(1).mean() == pytest.approx(0.2, abs=1e-5)
    assert json.loads((out / "scenes.json").read_text())["grid"]["crs"] == "EPSG:32630"


def test_fetch_pair_reports_candidates_when_nothing_clear(tmp_path, aoi_file, monkeypatch):
    cloudy = _fake_item(tmp_path, "cloudy", 5, dn=3000, scl_class=9)
    _patch_search(monkeypatch, {"b": [cloudy], "a": [cloudy]})
    with pytest.raises(NoSceneError) as err:
        stac.fetch_pair(aoi_file, "b", "a", tmp_path / "out", max_cloud=20, log=lambda *_: None)
    assert set(err.value.failures) == {"before (b)", "after (a)"}
    cand = err.value.failures["before (b)"][0]
    assert isinstance(cand, Candidate) and cand.aoi_cloud == 100.0
    assert not (tmp_path / "out" / "before.tif").exists()  # nothing written on failure


def test_cli_parser_and_no_scene_exit_code(tmp_path, aoi_file, monkeypatch, capsys):
    args = build_parser().parse_args(
        ["fetch", "--aoi", "a.geojson", "--before", "x/y", "--after", "z/w", "--max-cloud", "10"]
    )
    assert args.max_cloud == 10 and args.command == "fetch"

    cloudy = _fake_item(tmp_path, "cloudy", 5, dn=3000, scl_class=9)
    _patch_search(monkeypatch, {"b": [cloudy], "a": [cloudy]})
    code = main(["fetch", "--aoi", str(aoi_file), "--before", "b", "--after", "a",
                 "--out", str(tmp_path / "o")])  # fmt: skip
    assert code == 2
    assert "Best candidates" in capsys.readouterr().out


def _offset_for(tmp_path, dn, **props):
    """Run reflectance_params on a fake item and return (offset, source)."""
    item = _fake_item(tmp_path, "s", 1, dn=dn, scl_class=4, **props)
    _, offset, source = stac.reflectance_params(item, "blue", np.full((4, 4), dn))
    return offset, source


def test_offset_metadata_flag_wins_over_pixels(tmp_path):
    # Flag says already applied -> no shift, even though DN 2000 "looks" shifted.
    offset, source = _offset_for(tmp_path, 2000, **{"earthsearch:boa_offset_applied": True})
    assert offset == 0.0 and "boa_offset_applied=True" in source
    # Flag says not applied -> subtract 0.1, even though DN 300 would trigger the pixel check.
    offset, source = _offset_for(tmp_path, 300, **{"earthsearch:boa_offset_applied": False})
    assert offset == pytest.approx(-0.1) and "boa_offset_applied=False" in source


def test_offset_falls_back_to_processing_baseline(tmp_path):
    offset, source = _offset_for(tmp_path, 300, **{"s2:processing_baseline": "05.11"})
    assert offset == pytest.approx(-0.1) and "processing_baseline=05.11" in source
    offset, source = _offset_for(tmp_path, 2000, **{"s2:processing_baseline": "03.01"})
    assert offset == 0.0 and "processing_baseline=03.01" in source


def test_offset_pixel_check_is_last_resort(tmp_path):
    offset, source = _offset_for(tmp_path, 300)  # no metadata; DN 300 < 1000 -> already shifted
    assert offset == 0.0 and source.startswith("fallback")
    offset, source = _offset_for(tmp_path, 2000)  # DN well above 1000 -> shift still in data
    assert offset == pytest.approx(-0.1) and source.startswith("fallback")


def test_fetch_pair_logs_and_records_offset_source(tmp_path, aoi_file, monkeypatch):
    item = _fake_item(
        tmp_path, "s", 1, dn=300, scl_class=4, **{"earthsearch:boa_offset_applied": True}
    )
    _patch_search(monkeypatch, {"b": [item], "a": [item]})
    lines = []
    out = tmp_path / "out"
    summary = stac.fetch_pair(aoi_file, "b", "a", out, log=lines.append)
    assert any("boa_offset_applied=True" in line for line in lines)
    assert summary["before"]["reflectance_offset"]["blue"]["offset"] == 0.0
    with rasterio.open(out / "before.tif") as b:
        assert b.read(1).mean() == pytest.approx(0.03, abs=1e-5)  # not clipped to 0
