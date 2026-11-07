"""Step 0-1 of Geo-AI: turn the before/after scene into impact rasters.

Layers (all on the scene's own 10 m UTM grid, so georeferencing is kept):
  water_before, water_after  P(water) from the 6-band flood U-Net
  flood_prob                 P(water after) x P(not water before): confidence of NEW water
  flooded                    water after AND NOT water before. The Albufera lake is water on both
                             dates, so it lands in ``permanent_water``, not in ``flooded``.
  permanent_water            water on BOTH dates: lake, sea, and paddies already flooded on the
                             before date (so "permanent" means "not new", not "always water")
  change_prob, changed       P(change) from the 4-band Siamese U-Net (OSCD, mostly urban change)
  ndvi_drop, ndvi_dropped    NDVI before - after, flagged where the land was vegetated before
                             (NDVI > 0.3) and NDVI fell by more than 0.2: crop stress or damage
  valid                      both dates have data
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from impactlens.data.preprocess import BAND_CLIPS, FLOOD_BAND_CLIPS, normalize
from impactlens.geo.indices import mndwi, ndvi, ndwi
from impactlens.geo.inference import Predictor, sliding_window

MASKS = ("flooded", "permanent_water", "changed", "ndvi_dropped", "valid")


@dataclass
class Thresholds:
    water: float = 0.5  # use the val-tuned value from the floods metrics.json when available
    change: float = 0.5
    ndvi_drop: float = 0.2
    ndvi_vegetated: float = 0.3


def compute_layers(
    before: np.ndarray,
    after: np.ndarray,
    water_model: Predictor,
    change_model: Predictor,
    th: Thresholds | None = None,
    overlap: int = 64,
) -> dict[str, np.ndarray]:
    """before/after: (6, H, W) reflectance (blue, green, red, nir, swir16, swir22)."""
    th = th or Thresholds()
    valid = (before[:4].sum(0) > 0) & (after[:4].sum(0) > 0)
    water_b = sliding_window(water_model, [normalize(before, FLOOD_BAND_CLIPS)], overlap=overlap)
    water_a = sliding_window(water_model, [normalize(after, FLOOD_BAND_CLIPS)], overlap=overlap)
    change = sliding_window(
        change_model,
        [normalize(before[:4], BAND_CLIPS), normalize(after[:4], BAND_CLIPS)],
        overlap=overlap,
    )
    ndvi_b, ndvi_a = ndvi(before), ndvi(after)
    drop = ndvi_b - ndvi_a
    wet_b, wet_a = water_b > th.water, water_a > th.water
    return {
        "water_before": water_b,
        "water_after": water_a,
        "flood_prob": (water_a * (1 - water_b)) * valid,
        "flooded": wet_a & ~wet_b & valid,
        "permanent_water": wet_a & wet_b & valid,
        "change_prob": change * valid,
        "changed": (change > th.change) & valid,
        "ndvi_before": ndvi_b,
        "ndvi_after": ndvi_a,
        "ndvi_drop": drop * valid,
        "ndvi_dropped": (drop > th.ndvi_drop) & (ndvi_b > th.ndvi_vegetated) & valid,
        "valid": valid,
    }


# ------------------------------------------------------------------------- sanity check


def _agreement(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    a, b = a & valid, b & valid
    union = (a | b).sum()
    n = max(int(valid.sum()), 1)
    return {
        "iou": float((a & b).sum() / union) if union else 1.0,
        "model_water_pct": 100 * float(a.sum()) / n,
        "index_water_pct": 100 * float(b.sum()) / n,
        "pixel_agreement_pct": 100 * float(((a == b) & valid).sum()) / n,
    }


def index_agreement(
    before: np.ndarray, after: np.ndarray, layers: dict, th: Thresholds, index_thresholds: dict
) -> dict:
    """How well do the model's water masks agree with NDWI / MNDWI on this scene?

    The model was trained on L1C (top-of-atmosphere) chips but runs on L2A (surface) data here.
    The indices are simple ratios that barely care about that, so a big disagreement would point
    at the domain gap. ``index_thresholds`` = {"ndwi": [0.0, t_val], "mndwi": [0.0, t_val]}.
    """
    valid = layers["valid"]
    out = {}
    for date, refl, prob in (("before", before, "water_before"), ("after", after, "water_after")):
        model = layers[prob] > th.water
        for name, fn in (("ndwi", ndwi), ("mndwi", mndwi)):
            idx = fn(refl)
            for t in index_thresholds[name]:
                out[f"{date}/{name}>{t:g}"] = _agreement(model, idx > t, valid)
    return out


# ------------------------------------------------------------------------------ outputs


def write_layers(layers: dict, profile: dict, out_dir: Path) -> list[Path]:
    """One single-band GeoTIFF per layer, same CRS/transform as the input scene."""
    import rasterio

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, arr in layers.items():
        is_mask = name in MASKS
        prof = {**profile, "count": 1, "dtype": "uint8" if is_mask else "float32", "nodata": None}
        path = out_dir / f"{name}.tif"
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(arr.astype(prof["dtype"]), 1)
            dst.set_band_description(1, name)
        paths.append(path)
    return paths


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: asdict(o)))


def quicklook(before: np.ndarray, after: np.ndarray, layers: dict, path: Path, title: str = ""):
    """2x3 panel PNG for eyeballing the result."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rgb(refl):
        return np.clip(refl[[2, 1, 0]].transpose(1, 2, 0) / 0.25, 0, 1)

    base = np.clip(after[[2, 1, 0]].mean(0) / 0.25, 0, 1) * 0.7  # grey after-image backdrop
    overlay = np.stack([base] * 3, -1)
    overlay[layers["permanent_water"]] = (0.15, 0.35, 0.9)  # blue: water on both dates
    overlay[layers["flooded"]] = (0.95, 0.15, 0.1)  # red: NEW water

    panels = [
        ("Before (RGB)", rgb(before), None),
        ("After (RGB)", rgb(after), None),
        ("Flooded (red) vs permanent water (blue)", overlay, None),
        ("P(water) after", layers["water_after"], "Blues"),
        ("P(change), OSCD model", layers["change_prob"], "magma"),
        ("NDVI drop (before - after)", layers["ndvi_drop"], "RdYlGn_r"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    for ax, (name, img, cmap) in zip(axes.flat, panels, strict=True):
        if cmap == "RdYlGn_r":
            im = ax.imshow(img, cmap=cmap, vmin=-0.5, vmax=0.5)
        elif cmap:
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            im = ax.imshow(img)
        if cmap:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(name, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def rollup_map(layers_dir: Path, paddies: list[dict], hotspots: list[dict], summary: dict,
               path: Path, aoi: dict | None = None, land: dict | None = None) -> Path:  # fmt: skip
    """Flooded pixels with rice-paddy outlines and the hotspot boxes: shows the confound."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from shapely.geometry import shape

    with rasterio.open(layers_dir / "flooded.tif") as src:
        flooded, b, transform = src.read(1).astype(bool), src.bounds, src.transform
    with rasterio.open(layers_dir / "permanent_water.tif") as src:
        perm = src.read(1).astype(bool)
    img = np.ones((*flooded.shape, 3)) * 0.93
    img[perm] = (0.6, 0.72, 0.95)
    img[flooded] = (0.9, 0.15, 0.1)
    if aoi is not None and land is not None:  # grey out the sea and the area outside the AOI
        in_aoi, on_land = _masks(aoi, land, flooded.shape, transform)
        img[in_aoi & ~on_land] = (0.8, 0.86, 0.95)
        img[~in_aoi] = 1.0
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, extent=(b.left, b.right, b.bottom, b.top))
    for g in paddies:
        geom = shape(g)
        for poly in getattr(geom, "geoms", [geom]):
            ax.plot(*poly.exterior.xy, color="#1a7f37", lw=0.8)
    for h in sorted(hotspots, key=lambda h: -h["flooded_ha"])[:5]:
        x0, y0, x1, y1 = shape(h["geometry"]).bounds
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="k", lw=1.2, ls="--"))
        ax.text(x0, y1 + 80, f"hotspot {h['id']}: {h['flooded_ha']:.0f} ha", fontsize=9)
    rp, t = summary["rice_paddies"], summary["totals"]
    ax.set_title(
        f"Flooded (red), permanent water (blue), OSM rice paddies (green outline)\n"
        f"flooded on land {t['flooded_ha_on_land']:.0f} ha, of which in rice paddies "
        f"{rp['flooded_ha']:.0f} ha ({rp['share_of_flooded_land_pct']:.0f}%): seasonal flooding "
        "confound. Sea (pale) is masked out",
        fontsize=11,
    )
    ax.set_xlabel("UTM 30N easting (m)")
    ax.set_ylabel("northing (m)")
    ax.text(0.01, 0.01, "Land use (c) OpenStreetMap contributors", transform=ax.transAxes,
            fontsize=8)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def _masks(aoi: dict, land: dict, hw, transform):
    from rasterio.features import rasterize

    in_aoi = rasterize([(aoi, 1)], out_shape=hw, transform=transform) > 0
    return in_aoi, in_aoi & (rasterize([(land, 1)], out_shape=hw, transform=transform) > 0)


def ndvi_drop_map(after_tif: Path, layers_dir: Path, paddies: list[dict], towns: list[dict],
                  aoi: dict, land: dict, summary: dict, path: Path) -> Path:  # fmt: skip
    """NDVI drop outside paddies (orange) vs inside paddies (purple), with town outlines.

    Background = the after image in grey, sea masked out. Each town is labelled with its NDVI
    drop outside the paddies, the candidate mud / damage footprint.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import shape

    with rasterio.open(after_tif) as src:
        after, b, transform = src.read([1, 2, 3]), src.bounds, src.transform
    with rasterio.open(layers_dir / "ndvi_dropped.tif") as src:
        dropped = src.read(1).astype(bool)
    shape_hw = dropped.shape
    in_aoi, on_land = _masks(aoi, land, shape_hw, transform)
    in_paddy = (rasterize(((g, 1) for g in paddies), out_shape=shape_hw, transform=transform) > 0
                if paddies else np.zeros(shape_hw, bool))  # fmt: skip

    grey = np.clip(after.mean(0) / 0.25, 0, 1) * 0.6 + 0.3
    img = np.stack([grey] * 3, -1)
    img[in_aoi & ~on_land] = (0.85, 0.9, 0.97)  # sea: masked out, never counted
    img[~in_aoi] = 1.0  # outside the AOI
    img[dropped & on_land & ~in_paddy] = (0.93, 0.45, 0.05)  # orange: candidate mud / damage
    img[dropped & on_land & in_paddy] = (0.55, 0.3, 0.75)  # purple: in paddies, not counted

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(img, extent=(b.left, b.right, b.bottom, b.top))
    by_name = {t["name"]: t for t in summary["by_town"]}
    for t in towns:
        if t["name"] not in by_name:
            continue
        geom = shape(t["geometry"])
        for poly in getattr(geom, "geoms", [geom]):
            ax.plot(*poly.exterior.xy, color="k", lw=0.7)
        row = by_name[t["name"]]
        rate = row.get("ndvi_drop_pct_of_vegetated")
        label = f"{t['name']}\n{row['ndvi_drop_ha_outside_rice_paddies']:.0f} ha"
        if rate is not None and row.get("vegetated_ha_outside_rice_paddies", 0) >= 50:
            label += f" ({rate:.0f}%)"  # % of its vegetated land; skipped for tiny bases
        if geom.area > 2e6:  # label towns with more than 200 ha in the AOI
            p = geom.representative_point()
            ax.text(p.x, p.y, label, fontsize=8, ha="center",
                    bbox={"fc": "white", "alpha": 0.7, "lw": 0, "pad": 1})  # fmt: skip
    v = summary["impacts"]["vegetation_loss_likely_mud"]
    ax.set_title(
        "NDVI drop > 0.2 on vegetated land: outside rice paddies (orange), in paddies (purple)\n"
        f"outside paddies {v['ndvi_drop_ha_outside_rice_paddies']:.0f} ha, in paddies "
        f"{v['ndvi_drop_ha_in_rice_paddies_not_counted']:.0f} ha; sea masked (pale blue); "
        "black = municipalities",
        fontsize=11,
    )
    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    ax.set_xlabel("UTM 30N easting (m)")
    ax.set_ylabel("northing (m)")
    ax.text(0.01, 0.01, "Boundaries and land use (c) OpenStreetMap contributors",
            transform=ax.transAxes, fontsize=8)  # fmt: skip
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path
