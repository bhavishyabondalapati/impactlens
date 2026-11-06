"""Water-index baselines for flood water: the bar the U-Net must beat.

Water reflects green light and absorbs infrared, so a ratio of green to an infrared band is a
classic hand-made water detector. No training. Two indices:

  NDWI  = (green - NIR)   / (green + NIR)     (McFeeters 1996)
  MNDWI = (green - SWIR1) / (green + SWIR1)   (Xu 2006): SWIR absorbs water almost completely,
                                               so it is usually the stronger of the two.

For each index we report two versions:
  - fixed:  threshold 0 (the textbook value)
  - tuned:  the threshold with the best F1 on the *validation* split (never on test)
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from impactlens.data.preprocess import FLOOD_BAND_CLIPS
from impactlens.models.metrics import ConfusionMeter

THRESHOLDS = [round(-0.3 + 0.05 * i, 2) for i in range(17)]  # -0.30 ... 0.50


def _ratio(img: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """(band a - band b) / (band a + band b) on reflectance. img: (B, 6, H, W) normalized."""
    clips = torch.tensor(FLOOD_BAND_CLIPS, dtype=img.dtype).view(1, -1, 1, 1)
    refl = img * clips  # undo the per-band scaling so the two bands are comparable
    return (refl[:, a : a + 1] - refl[:, b : b + 1]) / (
        refl[:, a : a + 1] + refl[:, b : b + 1] + 1e-6
    )


def ndwi(img: torch.Tensor) -> torch.Tensor:
    """(B, 6, H, W) -> (B, 1, H, W): green (band 1) vs NIR (band 3)."""
    return _ratio(img, 1, 3)


def mndwi(img: torch.Tensor) -> torch.Tensor:
    """(B, 6, H, W) -> (B, 1, H, W): green (band 1) vs SWIR B11 (band 4)."""
    return _ratio(img, 1, 4)


INDEXES = {"ndwi": ndwi, "mndwi": mndwi}


def collect(dataset, index_fn=ndwi, batch_size: int = 32, max_batches: int | None = None):
    """Index, mask and valid for a whole split as three tensors."""
    outs = ([], [], [])
    for i, batch in enumerate(DataLoader(dataset, batch_size=batch_size)):
        if max_batches is not None and i >= max_batches:
            break
        parts = (index_fn(batch["img"]), batch["mask"], batch["valid"])
        for lst, t in zip(outs, parts, strict=True):
            lst.append(t)
    return tuple(torch.cat(o) for o in outs)


def score(data, threshold: float) -> dict[str, float]:
    index, mask, valid = data
    meter = ConfusionMeter()
    meter.update_pred((index > threshold).float(), mask, valid)
    return meter.compute()


def run_baseline(val_ds, other: dict, index_fn=ndwi, max_batches: int | None = None) -> dict:
    """``other`` maps names (e.g. "test", "bolivia") to datasets. Returns a JSON-ready dict."""
    val = collect(val_ds, index_fn, max_batches=max_batches)
    best = max(THRESHOLDS, key=lambda t: score(val, t)["f1"])
    result = {"tuned_threshold": best, "val": {"fixed": score(val, 0.0), "tuned": score(val, best)}}
    for name, ds in other.items():
        data = collect(ds, index_fn, max_batches=max_batches)
        result[name] = {"fixed": score(data, 0.0), "tuned": score(data, best)}
    return result
