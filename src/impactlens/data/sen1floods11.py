"""Sen1Floods11 loader: hand-labeled Sentinel-2 flood chips with water masks.

Why this exists: OSCD change labels are mostly urban construction, so a model trained only on
OSCD will not learn "farmland turned into water". Sen1Floods11 teaches a plain U-Net what
flood water looks like in the same four bands (RGB + NIR) we use everywhere else.

Expected folder layout (you download it into data/raw/sen1floods11, see the README/summary):

    data/raw/sen1floods11/
        S2Hand/<chip>_S2Hand.tif          # 13-band Sentinel-2 L1C chips, 512x512, int16
        LabelHand/<chip>_LabelHand.tif    # 1 = water, 0 = not water, -1 = no label (cloud etc.)
        splits/flood_train_data.csv       # official splits: train / valid / test / bolivia
        splits/flood_valid_data.csv
        splits/flood_test_data.csv
        splits/flood_bolivia_data.csv

Notes:
- The split CSVs list ``<chip>_S1Hand.tif,<chip>_LabelHand.tif``; we only need the chip id and
  derive the S2 file name from it.
- Chips are Sentinel-2 *L1C* (top-of-atmosphere), while our live data from STAC is L2A. They
  are close for water vs not-water, but it is a known small domain gap.
- Pixels labelled -1 are returned with ``valid = 0`` so the loss can ignore them. Never train
  with -1 treated as "not water".
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from impactlens.data.oscd import tile_starts
from impactlens.data.preprocess import (
    FLOOD_BAND_CLIPS,
    dn_to_reflectance,
    normalize,
    random_flip_rotate,
)

SPLITS = ("train", "valid", "test", "bolivia")
# 1-based positions of B2, B3, B4, B8, B11, B12 (blue, green, red, nir, swir16, swir22) in the
# 13-band chip (order: B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12). Same channel order as the
# STAC fetch; the first four match OSCD. The chips are already resampled to 10 m.
FLOOD_BAND_INDEXES = (2, 3, 4, 8, 12, 13)
CHIP_BANDS = 13
TILE_SIZE = 256


def read_chip_ids(root: str | Path, split: str) -> list[str]:
    """Chip ids (e.g. 'Ghana_103272') of an official split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    root = Path(root)
    for path in (
        root / "splits" / f"flood_{split}_data.csv",
        root / f"flood_{split}_data.csv",
        root / "splits" / "flood_handlabeled" / f"flood_{split}_data.csv",
    ):
        if path.exists():
            ids = []
            for row in csv.reader(path.read_text().splitlines()):
                if not row:
                    continue
                name = row[-1].strip()  # the *_LabelHand.tif column
                ids.append(name.removesuffix("_LabelHand.tif"))
            return ids
    raise FileNotFoundError(f"Could not find flood_{split}_data.csv under {root}/splits")


def _chip_paths(root: Path, chip: str) -> tuple[Path, Path]:
    image = root / "S2Hand" / f"{chip}_S2Hand.tif"
    label = root / "LabelHand" / f"{chip}_LabelHand.tif"
    for path in (image, label):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
    return image, label


def read_chip(root: str | Path, chip: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one whole chip: normalized (6, H, W) float32 image, water uint8, valid uint8."""
    import rasterio  # lazy: the packed-npz path (Kaggle) does not need rasterio

    image_path, label_path = _chip_paths(Path(root), chip)
    with rasterio.open(image_path) as src:
        if src.count != CHIP_BANDS:
            raise ValueError(f"{image_path} has {src.count} bands, expected {CHIP_BANDS}")
        dn = src.read(list(FLOOD_BAND_INDEXES))
    with rasterio.open(label_path) as src:
        label = src.read(1)
    return (
        normalize(dn_to_reflectance(dn), FLOOD_BAND_CLIPS),
        (label == 1).astype(np.uint8),
        (label != -1).astype(np.uint8),
    )


def _pad_to(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    if h >= size and w >= size:
        return arr
    pad = [(0, 0)] * (arr.ndim - 2) + [(0, max(0, size - h)), (0, max(0, size - w))]
    return np.pad(arr, pad)


class Sen1Floods11Dataset(Dataset):
    """Tiles of Sen1Floods11 hand-labeled chips for water segmentation.

    ``root`` is the raw Sen1Floods11 folder (tiles are read lazily from disk) or a packed
    ``sen1floods11.npz`` made by scripts/pack_for_kaggle.py (everything held in memory).

    Each item is a dict: ``img`` (6, T, T) float32 in [0, 1] (B2,B3,B4,B8,B11,B12; ceilings in
    ``FLOOD_BAND_CLIPS``), ``mask`` (1, T, T) float32 with 1 = water, ``valid`` (1, T, T) float32
    with 0 where the label is -1 (ignore), and ``chip``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        tile: int = TILE_SIZE,
        stride: int | None = None,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.tile = tile
        self.stride = stride or tile
        self.augment = augment
        self._packed = None
        sizes: list[tuple[int, int]] = []  # (height, width) of every chip
        if self.root.suffix == ".npz":
            from impactlens.data.packed import load_sen1floods11_arrays

            self._packed = load_sen1floods11_arrays(self.root, split)
            self.chips = [str(c) for c in self._packed["ids"]]
            sizes = [self._packed["mask"].shape[1:]] * len(self.chips)
        else:
            import rasterio

            self.chips = read_chip_ids(self.root, split)
            for chip in self.chips:
                image, _ = _chip_paths(self.root, chip)
                with rasterio.open(image) as src:
                    if src.count != CHIP_BANDS:
                        raise ValueError(f"{image} has {src.count} bands, expected {CHIP_BANDS}")
                    sizes.append((src.height, src.width))
        # (chip index, row, col) for every tile. Only file headers are read here.
        self.index: list[tuple[int, int, int]] = []
        for i, (h, w) in enumerate(sizes):
            for r in tile_starts(h, tile, self.stride):
                for c in tile_starts(w, tile, self.stride):
                    self.index.append((i, r, c))

    def __len__(self) -> int:
        return len(self.index)

    def _read_window(self, ci: int, r: int, c: int):
        """Normalized image (6,h,w) float32, water (h,w) and valid (h,w) for one window."""
        t = self.tile
        if self._packed is not None:
            from impactlens.data.packed import dequantize

            p = self._packed
            img = dequantize(p["img"][ci, :, r : r + t, c : c + t])
            return img, p["mask"][ci, r : r + t, c : c + t], p["valid"][ci, r : r + t, c : c + t]

        import rasterio
        from rasterio.windows import Window

        image_path, label_path = _chip_paths(self.root, self.chips[ci])
        window = Window(c, r, t, t)  # rasterio clips this at the chip edge
        with rasterio.open(image_path) as src:
            dn = src.read(list(FLOOD_BAND_INDEXES), window=window)
        with rasterio.open(label_path) as src:
            label = src.read(1, window=window)
        return normalize(dn_to_reflectance(dn), FLOOD_BAND_CLIPS), label == 1, label != -1

    def __getitem__(self, idx: int) -> dict:
        ci, r, c = self.index[idx]
        t = self.tile
        img, water, valid = self._read_window(ci, r, c)
        img = _pad_to(img, t)
        water = _pad_to(water.astype(np.float32), t)[None]
        valid = _pad_to(valid.astype(np.float32), t)[None]  # padding counts as invalid

        # Stack so every transform hits all three identically: (6 + 1 + 1, T, T).
        x = torch.from_numpy(np.concatenate([img, water, valid], axis=0))
        if self.augment:
            x = random_flip_rotate(x)
        return {"img": x[:6], "mask": x[6:7], "valid": x[7:8], "chip": self.chips[ci]}
