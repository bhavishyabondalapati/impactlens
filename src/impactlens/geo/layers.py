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
               path: Path) -> Path:  # fmt: skip
    """Flooded pixels with rice-paddy outlines and the hotspot boxes: shows the confound."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from shapely.geometry import shape

    with rasterio.open(layers_dir / "flooded.tif") as src:
        flooded, b = src.read(1).astype(bool), src.bounds
    with rasterio.open(layers_dir / "permanent_water.tif") as src:
        perm = src.read(1).astype(bool)
    img = np.ones((*flooded.shape, 3)) * 0.93
    img[perm] = (0.6, 0.72, 0.95)
    img[flooded] = (0.9, 0.15, 0.1)
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
        "confound",
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
