"""External-validation helpers (geo/validation.py): metrics, EMS GeoJSON parsing, rasterizing.

The full EMSR773 comparison needs the downloaded EMS files and the Valencia layers, so it is a
`needs_data` test; the pieces it is built from are tested here with tiny synthetic inputs.
"""

from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from impactlens.geo import validation as V

REPO = Path(__file__).resolve().parents[1]


def test_overlap_counts_only_inside_the_domain():
    pred = np.zeros((10, 10), bool)
    ref = np.zeros((10, 10), bool)
    pred[:, :6] = True  # 60 px
    ref[:, 4:] = True  # 60 px, 20 px shared with pred
    domain = np.ones((10, 10), bool)
    o = V.overlap(pred, ref, domain, px_ha=0.01)
    assert (o["ours_ha"], o["ems_ha"], o["both_ha"]) == (0.6, 0.6, 0.2)
    assert o["iou"] == pytest.approx(0.2, abs=1e-3)
    assert o["precision"] == pytest.approx(20 / 60, abs=1e-3)  # values are rounded to 3 places
    assert o["recall"] == pytest.approx(20 / 60, abs=1e-3)
    # lift = precision / share of the domain EMS flags = (1/3) / 0.6
    assert o["lift"] == pytest.approx(0.6, abs=0.05)

    domain[:, 8:] = False  # cut away 20 ref-only pixels: recall rises, precision unchanged
    o = V.overlap(pred, ref, domain, px_ha=0.01)
    assert o["ems_ha"] == 0.4 and o["recall"] == pytest.approx(0.5)
    assert o["precision"] == pytest.approx(20 / 60, abs=1e-3)


def test_overlap_handles_empty_sides():
    empty = np.zeros((4, 4), bool)
    full = np.ones((4, 4), bool)
    o = V.overlap(empty, full, full, 1.0)
    assert o["precision"] is None and o["recall"] == 0 and o["lift"] is None
    o = V.overlap(full, empty, full, 1.0)
    assert o["precision"] == 0 and o["recall"] is None and o["iou"] == 0


def _feature(notation, coords):
    return {"type": "Feature", "properties": {"notation": notation},
            "geometry": {"type": "Polygon", "coordinates": [coords]}}  # fmt: skip


def test_ems_geometries_filters_by_notation_and_repairs_polygons():
    square = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    bowtie = [(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]  # self-intersecting: invalid as drawn
    gj = {"features": [_feature("Flooded area", square), _feature("Flood trace", bowtie),
                       _feature("Not analysed", square)]}  # fmt: skip
    from shapely.geometry import shape

    kept = V.ems_geometries(gj)
    assert len(kept) == 2 and all(shape(g).is_valid for g in kept)
    assert len(V.ems_geometries(gj, ("Flood trace",))) == 1
    assert len(V.ems_geometries(gj, ())) == 3  # () = every feature


def test_rasterize_lonlat_burns_on_the_utm_grid():
    # A 100 x 100 px, 10 m grid in UTM 30N near Valencia, and a lon/lat box over its west half.
    from rasterio.warp import transform

    transform_ = from_origin(720_000, 4_360_000, 10, 10)
    (lon0, lon1), (lat0, lat1) = transform("EPSG:32630", "EPSG:4326",
                                           [720_000, 720_500], [4_359_000, 4_360_000])  # fmt: skip
    west_half = {"type": "Polygon", "coordinates": [[(lon0, lat0), (lon1, lat0), (lon1, lat1),
                                                     (lon0, lat1), (lon0, lat0)]]}  # fmt: skip
    mask = V.rasterize_lonlat([west_half], transform_, (100, 100), "EPSG:32630")
    assert mask[:, :45].mean() > 0.95 and mask[:, 55:].sum() == 0
    assert not V.rasterize_lonlat([], transform_, (100, 100), "EPSG:32630").any()


def test_spearman_ranks():
    assert V.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert V.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert V.spearman([1, 2], [1, 2]) is None


@pytest.mark.needs_data
@pytest.mark.skipif(
    not (REPO / "data/raw/emsr773").exists()
    or not (REPO / "data/processed/valencia_albufera/geo/summary.json").exists(),
    reason="needs data/raw/emsr773 (see docs/VALIDATION.md) and the Valencia rollup",
)
def test_emsr773_comparison_on_valencia():
    results, _ = V.compare_emsr773(REPO / "data/processed/valencia_albufera",
                                   REPO / "aoi/valencia_albufera.geojson", REPO / "data/raw/osm",
                                   REPO / "data/raw/emsr773", log=lambda *a: None)  # fmt: skip
    assert len(results["by_date"]) == 5
    vhr = results["town_rank_correlation"]["our_rate_vs_vhr_share_outside_paddies"]
    assert vhr["towns"] >= 10 and vhr["spearman"] > 0.7
    top = max((r for r in results["by_town"] if "vhr" in r), key=lambda r: r["vhr_pct_of_land"])
    assert top["town"] == "Paiporta"
