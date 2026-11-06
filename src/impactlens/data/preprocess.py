"""Shared pixel preprocessing, so OSCD training data and STAC inference data look the same.

Both sources are converted to surface/top-of-atmosphere *reflectance* (0..~1) and then
normalized with the same fixed function. We deliberately use a fixed global scale (not a
per-image stretch) so that a real brightness change between the before/after dates is
preserved instead of being "normalized away".
"""

from __future__ import annotations

import numpy as np
import torch

# Order of the 4 channels everywhere in this project: RGB + near-infrared.
BAND_NAMES = ("blue", "green", "red", "nir")
# Sentinel-2 band IDs for the same four channels (used by the OSCD file names).
OSCD_BAND_FILES = ("B02", "B03", "B04", "B08")

# Reflectance above this is almost always cloud/bright roof; clip so it doesn't dominate.
REFLECTANCE_CLIP = 0.4


def dn_to_reflectance(dn: np.ndarray, scale: float = 1e-4, offset: float = 0.0) -> np.ndarray:
    """Convert Sentinel-2 digital numbers to reflectance. DN 0 means 'no data' -> 0.0."""
    refl = dn.astype(np.float32) * np.float32(scale) + np.float32(offset)
    refl[dn == 0] = 0.0
    return np.clip(refl, 0.0, None)


def normalize(reflectance: np.ndarray) -> np.ndarray:
    """Map reflectance to [0, 1] with a fixed global scale."""
    return (np.clip(reflectance, 0.0, REFLECTANCE_CLIP) / REFLECTANCE_CLIP).astype(np.float32)


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
