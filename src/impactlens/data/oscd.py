"""Onera Satellite Change Detection (OSCD) loader.

Expected folder layout (you download and unzip OSCD into data/raw/oscd yourself):

    data/raw/oscd/
        images/
            train.txt                  # comma-separated city names (official train split)
            test.txt                   # comma-separated city names (official test split)
            <city>/imgs_1_rect/B02.tif # "before" bands (B03, B04, B08 too)
            <city>/imgs_2_rect/B02.tif # "after" bands
        train_labels/<city>/cm/cm.png  # change mask, 255 = change
        test_labels/<city>/cm/cm.png

The loader is forgiving: it also accepts imgs_1/imgs_2 (non-rectified), a ``labels/`` folder,
and ``<city>-cm.tif`` label files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from impactlens.data.preprocess import (
    OSCD_BAND_FILES,
    dn_to_reflectance,
    normalize,
    random_flip_rotate,
)

TILE_SIZE = 256


def read_split(root: str | Path, split: str) -> list[str]:
    """Return the official list of city names for 'train' or 'test'."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = Path(root)
    for candidate in (root / "images" / f"{split}.txt", root / f"{split}.txt"):
        if candidate.exists():
            text = candidate.read_text()
            return [c.strip() for c in text.replace("\n", ",").split(",") if c.strip()]
    raise FileNotFoundError(f"Could not find {split}.txt under {root} (looked in images/ and root)")


def _find_city_dir(root: Path, city: str) -> Path:
    for base in (root / "images", root):
        if (base / city).is_dir():
            return base / city
    raise FileNotFoundError(f"No image folder for city {city!r} under {root}")


def _read_bands(folder: Path) -> np.ndarray:
    """Read B02,B03,B04,B08 from a folder into a (4, H, W) reflectance array."""
    arrays = []
    for name in OSCD_BAND_FILES:
        path = folder / f"{name}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Missing band file {path}")
        with rasterio.open(path) as src:
            arrays.append(dn_to_reflectance(src.read(1)))
    # Bands should be identical in size, but guard against off-by-one resampling differences.
    h = min(a.shape[0] for a in arrays)
    w = min(a.shape[1] for a in arrays)
    return np.stack([a[:h, :w] for a in arrays])


def _band_folder(city_dir: Path, which: int) -> Path:
    for name in (f"imgs_{which}_rect", f"imgs_{which}"):
        if (city_dir / name).is_dir():
            return city_dir / name
    raise FileNotFoundError(f"No imgs_{which}_rect or imgs_{which} folder in {city_dir}")


def _read_label(root: Path, city: str, split: str) -> np.ndarray:
    """Read the change mask as uint8 (1 = change, 0 = no change)."""
    for label_dir in (root / f"{split}_labels", root / "labels", root / "images"):
        for path in (
            label_dir / city / "cm" / "cm.png",
            label_dir / city / "cm" / f"{city}-cm.tif",
            label_dir / city / f"{city}-cm.tif",
        ):
            if path.exists():
                with rasterio.open(path) as src:
                    arr = src.read(1)
                if path.suffix == ".tif" and arr.max() <= 2:
                    # OSCD tif labels use 1 = no change, 2 = change.
                    return (arr == 2).astype(np.uint8)
                return (arr > 127).astype(np.uint8) if arr.max() > 2 else (arr > 0).astype(np.uint8)
    raise FileNotFoundError(f"No change mask found for {city!r} in {split} labels under {root}")


@dataclass
class City:
    name: str
    img1: np.ndarray  # (4, H, W) normalized, before
    img2: np.ndarray  # (4, H, W) normalized, after
    mask: np.ndarray  # (H, W) uint8


def load_city(root: str | Path, city: str, split: str) -> City:
    """Load one OSCD city: both dates (RGB+NIR, normalized) and its change mask."""
    root = Path(root)
    city_dir = _find_city_dir(root, city)
    img1 = _read_bands(_band_folder(city_dir, 1))
    img2 = _read_bands(_band_folder(city_dir, 2))
    mask = _read_label(root, city, split)
    # Crop everything to the common size so shapes always agree.
    h = min(img1.shape[1], img2.shape[1], mask.shape[0])
    w = min(img1.shape[2], img2.shape[2], mask.shape[1])
    return City(city, normalize(img1[:, :h, :w]), normalize(img2[:, :h, :w]), mask[:h, :w])


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets along one axis. The last tile is shifted back so the edge is covered."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _pad_to(arr: np.ndarray, size: int) -> np.ndarray:
    """Zero-pad the last two axes up to at least ``size`` (only matters for tiny images)."""
    h, w = arr.shape[-2:]
    if h >= size and w >= size:
        return arr
    pad = [(0, 0)] * (arr.ndim - 2) + [(0, max(0, size - h)), (0, max(0, size - w))]
    return np.pad(arr, pad)


class OSCDDataset(Dataset):
    """256x256 tiles of OSCD city pairs.

    Each item is a dict: ``img1`` and ``img2`` (4, T, T) float32 in [0, 1], ``mask`` (1, T, T)
    float32 in {0, 1}, plus ``city``. With ``augment=True`` random flips and 90-degree
    rotations are applied identically to both images and the mask.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        tile: int = TILE_SIZE,
        stride: int | None = None,
        augment: bool = False,
    ):
        self.tile = tile
        self.stride = stride or tile
        self.augment = augment
        self.cities = [load_city(root, name, split) for name in read_split(root, split)]
        # Pre-compute (city index, row, col) for every tile.
        self.index: list[tuple[int, int, int]] = []
        for i, c in enumerate(self.cities):
            h, w = c.mask.shape
            for r in tile_starts(h, tile, self.stride):
                for col in tile_starts(w, tile, self.stride):
                    self.index.append((i, r, col))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ci, r, c = self.index[idx]
        city = self.cities[ci]
        t = self.tile
        img1 = _pad_to(city.img1[:, r : r + t, c : c + t], t)
        img2 = _pad_to(city.img2[:, r : r + t, c : c + t], t)
        mask = _pad_to(city.mask[r : r + t, c : c + t].astype(np.float32), t)[None]

        # Stack so every transform hits all three identically: (4 + 4 + 1, T, T).
        x = torch.from_numpy(np.concatenate([img1, img2, mask], axis=0))
        if self.augment:
            x = random_flip_rotate(x)
        return {"img1": x[:4], "img2": x[4:8], "mask": x[8:9], "city": city.name}
