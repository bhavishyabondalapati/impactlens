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

The original download folder names also work as-is, no renaming needed:

    data/raw/oscd/Onera Satellite Change Detection dataset - Images/
    data/raw/oscd/Onera Satellite Change Detection dataset - Train Labels/
    data/raw/oscd/Onera Satellite Change Detection dataset - Test Labels/

The loader is forgiving: it also accepts imgs_1/imgs_2 (non-rectified), a ``labels/`` folder,
and ``<city>-cm.tif`` label files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from impactlens.data.preprocess import (
    OSCD_BAND_FILES,
    dn_to_reflectance,
    normalize,
    random_flip_rotate,
)

TILE_SIZE = 256

# 3 of the 14 official training cities are held out for validation / checkpoint selection:
# Europe (pisa), Asia (mumbai), South America (aguasclaras); ~20% of training pixels, mid-range
# change rates (1.6-2.6%). The official TEST cities are never used for model selection, only
# for the final reported number.
VAL_CITIES = ("pisa", "mumbai", "aguasclaras")
SPLITS = ("train", "val", "test", "train_all")


def split_cities(official_train: list[str], official_test: list[str], split: str) -> list[str]:
    """City names for our four splits.

    train      official train minus VAL_CITIES   (what the model learns from)
    val        VAL_CITIES                        (checkpoint selection, early stopping)
    test       official test                     (final evaluation ONLY)
    train_all  full official train               (optional final retrain, no validation)
    """
    if split == "train":
        return [c for c in official_train if c not in VAL_CITIES]
    if split == "val":
        val = [c for c in official_train if c in VAL_CITIES]
        if not val:
            raise ValueError(f"none of the validation cities {VAL_CITIES} are in the train split")
        return val
    if split == "test":
        return list(official_test)
    if split == "train_all":
        return list(official_train)
    raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


def _find_dir(root: Path, names: tuple[str, ...], suffix: str) -> Path | None:
    """First existing folder from ``names``, else any folder whose name ends with ``suffix``."""
    for name in names:
        if (root / name).is_dir():
            return root / name
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if child.is_dir() and child.name.lower().endswith(suffix):
            return child
    return None


def _images_dir(root: Path) -> Path:
    """'images/' or the original '... - Images' folder; falls back to root itself."""
    return _find_dir(root, ("images",), "- images") or root


def _labels_dir(root: Path, split: str) -> Path | None:
    """'<split>_labels/', 'labels/' or the original '... - Train Labels' folder."""
    return _find_dir(root, (f"{split}_labels", "labels"), f"- {split} labels")


def read_split(root: str | Path, split: str) -> list[str]:
    """Return the official list of city names for 'train' or 'test'."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = Path(root)
    for candidate in (_images_dir(root) / f"{split}.txt", root / f"{split}.txt"):
        if candidate.exists():
            text = candidate.read_text()
            return [c.strip() for c in text.replace("\n", ",").split(",") if c.strip()]
    raise FileNotFoundError(f"Could not find {split}.txt under {root} (looked in images/ and root)")


def _find_city_dir(root: Path, city: str) -> Path:
    for base in (_images_dir(root), root):
        if (base / city).is_dir():
            return base / city
    raise FileNotFoundError(f"No image folder for city {city!r} under {root}")


def _read_bands(folder: Path) -> np.ndarray:
    """Read B02,B03,B04,B08 from a folder into a (4, H, W) reflectance array."""
    import rasterio  # imported lazily: the packed-npz path (Kaggle) does not need rasterio

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
    import rasterio

    label_dirs = [d for d in (_labels_dir(root, split), _images_dir(root)) if d is not None]
    for label_dir in label_dirs:
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


def _load_cities(source: str | Path, split: str) -> list[City]:
    """Load the cities of one of our splits from raw OSCD folders or a packed .npz file."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if Path(source).suffix == ".npz":
        from impactlens.data.packed import load_oscd_cities

        return load_oscd_cities(source, split)
    names = split_cities(read_split(source, "train"), read_split(source, "test"), split)
    label_split = "test" if split == "test" else "train"  # where the label files live
    return [load_city(source, name, label_split) for name in names]


class OSCDDataset(Dataset):
    """256x256 tiles of OSCD city pairs.

    ``source`` is either the raw OSCD folder or a packed ``oscd.npz`` (see
    scripts/pack_for_kaggle.py). ``split`` is one of ``train`` / ``val`` / ``test`` /
    ``train_all`` (see ``split_cities``).

    Each item is a dict: ``img1`` and ``img2`` (4, T, T) float32 in [0, 1], ``mask`` (1, T, T)
    float32 in {0, 1}, ``valid`` (1, T, T) float32 (0 on padding), plus ``city``. With
    ``augment=True`` random flips and 90-degree rotations are applied identically to the
    images, mask and valid-mask.

    Two sampling modes:
    - default: fixed tiles on a grid (deterministic; use for val and test).
    - ``crops_per_epoch=N``: every epoch yields N *new* random crops (city picked in proportion
      to its area, position uniform). With only 11 training cities, fixed tiles would let the
      model memorize them; random crops show it different windows each epoch.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        tile: int = TILE_SIZE,
        stride: int | None = None,
        augment: bool = False,
        crops_per_epoch: int | None = None,
    ):
        if crops_per_epoch is not None and split in ("val", "test"):
            raise ValueError("random crops are for training only; val/test use fixed tiles")
        self.tile = tile
        self.stride = stride or tile
        self.augment = augment
        self.crops_per_epoch = crops_per_epoch
        self.cities = _load_cities(root, split)
        # Pre-compute (city index, row, col) for every tile.
        self.index: list[tuple[int, int, int]] = []
        for i, c in enumerate(self.cities):
            h, w = c.mask.shape
            for r in tile_starts(h, tile, self.stride):
                for col in tile_starts(w, tile, self.stride):
                    self.index.append((i, r, col))
        # Random-crop sampling weights: number of possible crop positions per city.
        areas = [max(c.mask.shape[0] - tile + 1, 1) * max(c.mask.shape[1] - tile + 1, 1)
                 for c in self.cities]  # fmt: skip
        self._crop_weights = torch.tensor(areas, dtype=torch.float)

    def __len__(self) -> int:
        return self.crops_per_epoch if self.crops_per_epoch is not None else len(self.index)

    def _random_position(self) -> tuple[int, int, int]:
        ci = int(torch.multinomial(self._crop_weights, 1))
        h, w = self.cities[ci].mask.shape
        r = int(torch.randint(0, max(h - self.tile, 0) + 1, (1,)))
        c = int(torch.randint(0, max(w - self.tile, 0) + 1, (1,)))
        return ci, r, c

    def __getitem__(self, idx: int) -> dict:
        ci, r, c = self._random_position() if self.crops_per_epoch is not None else self.index[idx]
        city = self.cities[ci]
        t = self.tile
        img1 = _pad_to(city.img1[:, r : r + t, c : c + t], t)
        img2 = _pad_to(city.img2[:, r : r + t, c : c + t], t)
        raw_mask = city.mask[r : r + t, c : c + t].astype(np.float32)
        mask = _pad_to(raw_mask, t)[None]
        # 1 on real pixels, 0 on zero-padding (cities smaller than a tile, e.g. norcia is 241 px
        # tall). Metrics and loss must ignore padding or it counts as easy "no change".
        valid = _pad_to(np.ones_like(raw_mask), t)[None]

        # Stack so every transform hits all layers identically: (4 + 4 + 1 + 1, T, T).
        x = torch.from_numpy(np.concatenate([img1, img2, mask, valid], axis=0))
        if self.augment:
            x = random_flip_rotate(x)
        return {
            "img1": x[:4],
            "img2": x[4:8],
            "mask": x[8:9],
            "valid": x[9:10],
            "city": city.name,
        }
