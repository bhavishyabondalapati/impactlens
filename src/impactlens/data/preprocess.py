"""Shared pixel preprocessing, so OSCD training data and STAC inference data look the same.

Both sources are converted to surface/top-of-atmosphere *reflectance* (0..~1) and then
normalized with the same fixed function (fixed per-band ceilings). We deliberately use a
fixed global scale (not a per-image stretch) so that a real brightness change between the
before/after dates is preserved instead of being "normalized away".
"""

from __future__ import annotations

import numpy as np

# Order of the 4 channels everywhere in this project: RGB + near-infrared.
BAND_NAMES = ("blue", "green", "red", "nir")
# The flood model also sees the two shortwave-infrared bands (B11 ~1.6 um, B12 ~2.2 um): water
# absorbs SWIR almost completely, so it separates water from dark shadows and wet soil far
# better than RGB+NIR alone. They are native 20 m bands; Sen1Floods11 chips and our STAC fetch
# both resample them to the 10 m grid. OSCD stays at 4 bands (the first four of these).
FLOOD_BAND_NAMES = (*BAND_NAMES, "swir16", "swir22")
# Sentinel-2 band IDs for the same four channels (used by the OSCD file names).
OSCD_BAND_FILES = ("B02", "B03", "B04", "B08")

# Per-band reflectance ceilings (blue, green, red, nir). Anything brighter is clipped.
# Chosen from the data (scripts/clip_report.py) so that every band has <= ~1% of pixels at the
# ceiling in OSCD, Sen1Floods11 and our Valencia scenes. A single 0.4 ceiling clipped 5.1% of
# Sen1Floods11 NIR (and 6-8% of healthy-vegetation NIR), so a model could not tell healthy
# crops from merely bright ones. NIR gets more headroom because vegetation is NIR-bright.
BAND_CLIPS = (0.5, 0.5, 0.5, 0.6)
# SWIR (measured, see docs/clip_report.txt): B11 clips 0.97% of Sen1Floods11 pixels at 0.5; B12 is
# darker (p99.9 = 0.47, max 0.67), so 0.4 keeps resolution and clips only 0.5%.
SWIR16_CLIP, SWIR22_CLIP = 0.5, 0.4
# Ceilings for the six flood-model bands: the four above plus B11 and B12 (set from
# scripts/clip_report.py, see docs/clip_report.txt).
FLOOD_BAND_CLIPS = (*BAND_CLIPS, SWIR16_CLIP, SWIR22_CLIP)


def dn_to_reflectance(dn: np.ndarray, scale: float = 1e-4, offset: float = 0.0) -> np.ndarray:
    """Convert Sentinel-2 digital numbers to reflectance. DN 0 means 'no data' -> 0.0."""
    refl = dn.astype(np.float32) * np.float32(scale) + np.float32(offset)
    refl[dn == 0] = 0.0
    return np.clip(refl, 0.0, None)


def normalize(reflectance: np.ndarray, band_clips: tuple[float, ...] = BAND_CLIPS) -> np.ndarray:
    """Map reflectance to [0, 1] with fixed per-band ceilings (default ``BAND_CLIPS``).

    Works on (C, H, W) or batched (N, C, H, W) arrays: the band axis is always third from last.
    Pass ``FLOOD_BAND_CLIPS`` for the 6-band flood input.
    """
    if reflectance.ndim < 3 or reflectance.shape[-3] != len(band_clips):
        raise ValueError(
            f"expected (..., {len(band_clips)}, H, W) bands, got shape {reflectance.shape}"
        )
    clips = np.asarray(band_clips, dtype=np.float32).reshape(-1, 1, 1)
    return (np.clip(reflectance, 0.0, clips) / clips).astype(np.float32)


def random_flip_rotate(x: torch.Tensor) -> torch.Tensor:  # noqa: F821
    """Random horizontal/vertical flip and 90-degree rotation of a (C, H, W) tensor.

    Stack every layer that must stay aligned (images, mask, valid-mask) into ONE tensor
    before calling this, so they all get exactly the same transform.
    """
    import torch  # training only; inference (and the AWS Lambdas) run without PyTorch

    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(-1,))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(-2,))
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(-2, -1))


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets along one axis. The last tile is shifted back so the edge is covered."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts
