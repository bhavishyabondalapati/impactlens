"""Spectral indices on REFLECTANCE arrays (bands first: blue, green, red, nir, swir16, swir22).

All are normalized differences (a - b) / (a + b) in [-1, 1]:
  NDVI  = (nir - red) / (nir + red)        vegetation vigour; a drop = crop stress or damage
  NDWI  = (green - nir) / (green + nir)    open water (McFeeters)
  MNDWI = (green - swir16) / (green + swir16)  open water, better against shadows (Xu)
"""

from __future__ import annotations

import numpy as np

BLUE, GREEN, RED, NIR, SWIR16, SWIR22 = range(6)


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float32), b.astype(np.float32)
    total = a + b
    out = np.zeros_like(total)
    np.divide(a - b, total, out=out, where=total > 1e-6)  # nodata (0, 0) -> 0
    return out


def ndvi(refl: np.ndarray) -> np.ndarray:
    return normalized_difference(refl[NIR], refl[RED])


def ndwi(refl: np.ndarray) -> np.ndarray:
    return normalized_difference(refl[GREEN], refl[NIR])


def mndwi(refl: np.ndarray) -> np.ndarray:
    return normalized_difference(refl[GREEN], refl[SWIR16])
