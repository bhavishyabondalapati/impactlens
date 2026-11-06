"""Shared pixel preprocessing, so OSCD training data and STAC inference data look the same.

Both sources are converted to surface/top-of-atmosphere *reflectance* (0..~1) and then
normalized with the same fixed function (fixed per-band ceilings). We deliberately use a
fixed global scale (not a per-image stretch) so that a real brightness change between the
before/after dates is preserved instead of being "normalized away".
"""

from __future__ import annotations

import numpy as np
import torch

# Order of the 4 channels everywhere in this project: RGB + near-infrared.
BAND_NAMES = ("blue", "green", "red", "nir")
# Sentinel-2 band IDs for the same four channels (used by the OSCD file names).
OSCD_BAND_FILES = ("B02", "B03", "B04", "B08")

# Per-band reflectance ceilings (blue, green, red, nir). Anything brighter is clipped.
# Chosen from the data (scripts/clip_report.py) so that every band has <= ~1% of pixels at the
# ceiling in OSCD, Sen1Floods11 and our Valencia scenes. A single 0.4 ceiling clipped 5.1% of
# Sen1Floods11 NIR (and 6-8% of healthy-vegetation NIR), so a model could not tell healthy
# crops from merely bright ones. NIR gets more headroom because vegetation is NIR-bright.
BAND_CLIPS = (0.5, 0.5, 0.5, 0.6)


def dn_to_reflectance(dn: np.ndarray, scale: float = 1e-4, offset: float = 0.0) -> np.ndarray:
    """Convert Sentinel-2 digital numbers to reflectance. DN 0 means 'no data' -> 0.0."""
    refl = dn.astype(np.float32) * np.float32(scale) + np.float32(offset)
    refl[dn == 0] = 0.0
    return np.clip(refl, 0.0, None)


def normalize(reflectance: np.ndarray) -> np.ndarray:
    """Map reflectance to [0, 1] with fixed per-band ceilings (``BAND_CLIPS``).

    Works on (4, H, W) or batched (N, 4, H, W) arrays: the band axis is always third from last.
    """
    if reflectance.ndim < 3 or reflectance.shape[-3] != len(BAND_CLIPS):
        raise ValueError(f"expected (..., 4, H, W) bands, got shape {reflectance.shape}")
    clips = np.asarray(BAND_CLIPS, dtype=np.float32).reshape(-1, 1, 1)
    return (np.clip(reflectance, 0.0, clips) / clips).astype(np.float32)


def random_flip_rotate(x: torch.Tensor) -> torch.Tensor:
    """Random horizontal/vertical flip and 90-degree rotation of a (C, H, W) tensor.

    Stack every layer that must stay aligned (images, mask, valid-mask) into ONE tensor
    before calling this, so they all get exactly the same transform.
    """
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(-1,))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(-2,))
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(-2, -1))
