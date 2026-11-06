"""Raster masks -> polygons with a mean confidence each (connected pixel regions)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

EIGHT = np.ones((3, 3), dtype=int)  # diagonal neighbours count as connected


def polygonize(mask: np.ndarray, confidence: np.ndarray, transform, min_pixels: int = 4):
    """One feature per connected region of ``mask`` with at least ``min_pixels`` pixels.

    Each feature: {"geometry": GeoJSON dict in the raster CRS, "mean_conf", "pixels"}.
    Regions smaller than ``min_pixels`` (4 px = 0.04 ha at 10 m) are speckle and dropped.
    """
    from rasterio.features import shapes

    labels, n = ndimage.label(mask.astype(bool), structure=EIGHT)
    if n == 0:
        return []
    index = np.arange(1, n + 1)
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index)
    means = ndimage.mean(confidence.astype(np.float64), labels, index)
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_pixels
    labels = np.where(keep[labels], labels, 0).astype(np.int32)
    feats = []
    for geom, value in shapes(labels, mask=labels > 0, transform=transform, connectivity=8):
        i = int(value) - 1
        feats.append({"geometry": geom, "mean_conf": float(means[i]), "pixels": int(sizes[i])})
    return feats
