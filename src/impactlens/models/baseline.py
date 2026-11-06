"""NDWI threshold baseline for flood water: the bar the U-Net must beat.

NDWI = (green - NIR) / (green + NIR). Water reflects green and absorbs NIR, so NDWI > 0 is a
classic hand-made water detector. No training. We report two versions:
  - fixed:  threshold 0 (the textbook value)
  - tuned:  the threshold with the best F1 on the *validation* split (never on test)
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from impactlens.data.preprocess import BAND_CLIPS
from impactlens.models.metrics import ConfusionMeter

THRESHOLDS = [round(-0.3 + 0.05 * i, 2) for i in range(17)]  # -0.30 ... 0.50


def ndwi(img: torch.Tensor) -> torch.Tensor:
    """img: (B, 4, H, W) normalized [blue, green, red, nir] -> (B, 1, H, W) NDWI in [-1, 1]."""
    clips = torch.tensor(BAND_CLIPS, dtype=img.dtype).view(1, 4, 1, 1)
    refl = img * clips  # undo the per-band scaling so the two bands are comparable
    green, nir = refl[:, 1:2], refl[:, 3:4]
    return (green - nir) / (green + nir + 1e-6)


def collect(dataset, batch_size: int = 32, max_batches: int | None = None):
    """NDWI, mask and valid for a whole split as three tensors."""
    outs = ([], [], [])
    for i, batch in enumerate(DataLoader(dataset, batch_size=batch_size)):
        if max_batches is not None and i >= max_batches:
            break
        for lst, t in zip(outs, (ndwi(batch["img"]), batch["mask"], batch["valid"]), strict=True):
            lst.append(t)
    return tuple(torch.cat(o) for o in outs)


def score(data, threshold: float) -> dict[str, float]:
    index, mask, valid = data
    meter = ConfusionMeter()
    meter.update_pred((index > threshold).float(), mask, valid)
    return meter.compute()


def run_baseline(val_ds, other: dict, max_batches: int | None = None) -> dict:
    """``other`` maps names (e.g. "test", "bolivia") to datasets. Returns a JSON-ready dict."""
    val = collect(val_ds, max_batches=max_batches)
    best = max(THRESHOLDS, key=lambda t: score(val, t)["f1"])
    result = {"tuned_threshold": best, "val": {"fixed": score(val, 0.0), "tuned": score(val, best)}}
    for name, ds in other.items():
        data = collect(ds, max_batches=max_batches)
        result[name] = {"fixed": score(data, 0.0), "tuned": score(data, best)}
    return result
