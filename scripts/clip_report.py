"""How many pixels sit at the reflectance clip ceiling, per band, per dataset?

    uv run python scripts/clip_report.py

Reads RAW reflectance (before any clipping): OSCD (L1C), Sen1Floods11 (L1C) and the Valencia
scenes fetched from STAC (L2A). Percentiles use a 1-in-64 pixel subsample; the "above X"
fractions are exact for OSCD and Valencia and subsampled for Sen1Floods11.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import rasterio

from impactlens.data import oscd
from impactlens.data.preprocess import BAND_CLIPS, BAND_NAMES, dn_to_reflectance
from impactlens.data.sen1floods11 import S2_BAND_INDEXES, read_chip_ids

warnings.filterwarnings("ignore")

OSCD_ROOT = Path("data/raw/oscd")
S1F_ROOT = Path("data/raw/sen1floods11")
VALENCIA = Path("data/processed/valencia_albufera")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.8)


class Acc:
    """Accumulates exceedance counts and a subsample for one dataset, per band."""

    def __init__(self):
        self.total = 0
        self.over = np.zeros((4, len(THRESHOLDS)))
        self.sample = [[] for _ in range(4)]
        self.veg_total = 0
        self.veg_nir_over = np.zeros(len(THRESHOLDS))

    def add(self, refl: np.ndarray) -> None:
        """refl: (4, H, W) reflectance (blue, green, red, nir) before clipping."""
        refl = refl.reshape(4, -1)
        self.total += refl.shape[1]
        for b in range(4):
            for k, thr in enumerate(THRESHOLDS):
                self.over[b, k] += (refl[b] > thr).sum()
            self.sample[b].append(refl[b][::64])
        # "healthy vegetation" proxy: NDVI > 0.6
        red, nir = refl[2], refl[3]
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        veg = ndvi > 0.6
        self.veg_total += veg.sum()
        for k, thr in enumerate(THRESHOLDS):
            self.veg_nir_over[k] += (nir[veg] > thr).sum()

    def print(self, title: str) -> None:
        print(f"\n=== {title}  ({self.total / 1e6:.1f} M pixels) ===")
        head = "  band   " + "".join(f">{t:<6}" for t in THRESHOLDS) + "  p99    p99.9  max"
        print(head + "   (% of pixels above threshold)")
        for b, name in enumerate(BAND_NAMES):
            s = np.concatenate(self.sample[b])
            row = "".join(
                f"{100 * self.over[b, k] / self.total:<7.2f}" for k in range(len(THRESHOLDS))
            )
            p99, p999 = np.percentile(s, 99), np.percentile(s, 99.9)
            print(f"  {name:<6} {row} {p99:.3f}  {p999:.3f}  {s.max():.2f}")
        if self.veg_total:
            row = "".join(
                f"{100 * self.veg_nir_over[k] / self.veg_total:<7.1f}"
                for k in range(len(THRESHOLDS))
            )
            share = 100 * self.veg_total / self.total
            print(f"  NIR of vegetation (NDVI>0.6, {share:.1f}% of pixels): % above -> {row}")
        clipped = []
        for b, name in enumerate(BAND_NAMES):
            k = THRESHOLDS.index(BAND_CLIPS[b])
            clipped.append(f"{name}@{BAND_CLIPS[b]} {100 * self.over[b, k] / self.total:.2f}%")
        print("  currently clipped (per-band ceilings): " + ", ".join(clipped))


def main() -> None:
    acc = Acc()
    for split in ("train", "test"):
        for city in oscd.read_split(OSCD_ROOT, split):
            d = oscd._find_city_dir(OSCD_ROOT, city)
            for which in (1, 2):
                acc.add(oscd._read_bands(oscd._band_folder(d, which)))
    acc.print("OSCD (L1C, all 24 cities, both dates)")

    acc = Acc()
    for split in ("train", "valid", "test", "bolivia"):
        for chip in read_chip_ids(S1F_ROOT, split):
            with rasterio.open(S1F_ROOT / "S2Hand" / f"{chip}_S2Hand.tif") as src:
                acc.add(dn_to_reflectance(src.read(list(S2_BAND_INDEXES))))
    acc.print("Sen1Floods11 (L1C, all 446 chips)")

    acc = Acc()
    for name in ("before", "after"):
        with rasterio.open(VALENCIA / f"{name}.tif") as src:
            acc.add(src.read())
    acc.print("Valencia AOI (L2A, before + after)")


if __name__ == "__main__":
    main()
