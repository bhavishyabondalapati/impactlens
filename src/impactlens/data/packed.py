"""Compact .npz versions of the training datasets, for uploading to Kaggle.

Why: the raw OSCD + Sen1Floods11 downloads are ~1 GB of 13-band GeoTIFFs and we only use four
bands. Packing keeps just the normalized RGB+NIR bands, masks and valid-masks, stored as uint8
(value = round(normalized * 255), so the round-trip error is at most 0.5/255 = 0.2% of the band's
range) and compressed. Training on Kaggle then needs no rasterio and no raw data.

Files (made by scripts/pack_for_kaggle.py):

    oscd.npz           per city: "<city>__img1", "<city>__img2" (4,H,W) uint8,
                                 "<city>__mask" (H,W) uint8 (1 = change), plus "meta" (JSON text)
                       OSCD has no separate valid-mask: every real pixel is valid (the dataset
                       adds padding, marked invalid, if a city is smaller than a tile).
    sen1floods11.npz   per split: "<split>_img" (N,4,H,W) uint8, "<split>_mask" (N,H,W) uint8
                       (1 = water), "<split>_valid" (N,H,W) uint8 (0 = unlabeled), "<split>_ids",
                       plus "meta"

``meta`` records the clip ceilings used. Loading refuses a file packed with different ceilings
than the current code, so a stale pack can never silently feed the model differently
normalized data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from impactlens.data import oscd
from impactlens.data.preprocess import BAND_CLIPS, BAND_NAMES

FORMAT_VERSION = 1


def quantize(x: np.ndarray) -> np.ndarray:
    """Normalized float [0, 1] -> uint8."""
    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def dequantize(q: np.ndarray) -> np.ndarray:
    """uint8 -> normalized float32 in [0, 1]."""
    return q.astype(np.float32) / 255.0


def _meta(**extra) -> np.ndarray:
    meta = {
        "format_version": FORMAT_VERSION,
        "band_names": list(BAND_NAMES),
        "band_clips": list(BAND_CLIPS),
        **extra,
    }
    return np.array(json.dumps(meta))  # a 0-d unicode array: no pickle needed


def _read_meta(data, path: Path) -> dict:
    meta = json.loads(str(data["meta"]))
    if meta["format_version"] != FORMAT_VERSION:
        raise ValueError(f"{path} has format {meta['format_version']}, expected {FORMAT_VERSION}")
    if tuple(meta["band_clips"]) != tuple(BAND_CLIPS):
        raise ValueError(
            f"{path} was packed with band clips {meta['band_clips']} but the code uses "
            f"{list(BAND_CLIPS)}. Re-run scripts/pack_for_kaggle.py."
        )
    return meta


# ------------------------------------------------------------------------------------ OSCD


def pack_oscd(raw_root: str | Path, out_path: str | Path, log=print) -> Path:
    """Pack every OSCD city (official train + test) into one compressed npz."""
    raw_root, out_path = Path(raw_root), Path(out_path)
    official = {s: oscd.read_split(raw_root, s) for s in ("train", "test")}
    arrays: dict[str, np.ndarray] = {}
    for split, names in official.items():
        for name in names:
            city = oscd.load_city(raw_root, name, split)
            arrays[f"{name}__img1"] = quantize(city.img1)
            arrays[f"{name}__img2"] = quantize(city.img2)
            arrays[f"{name}__mask"] = city.mask.astype(np.uint8)
            log(f"  oscd {split:5} {name:12} {city.mask.shape[0]}x{city.mask.shape[1]}")
    arrays["meta"] = _meta(
        dataset="oscd",
        official_train=official["train"],
        official_test=official["test"],
        val_cities=list(oscd.VAL_CITIES),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path


def load_oscd_cities(path: str | Path, split: str) -> list[oscd.City]:
    """Load the cities of one split (train/val/test/train_all) from a packed oscd.npz."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = _read_meta(data, path)
        names = oscd.split_cities(
            meta["official_train"], meta["official_test"], split, meta["val_cities"]
        )
        return [
            oscd.City(
                name,
                dequantize(data[f"{name}__img1"]),
                dequantize(data[f"{name}__img2"]),
                data[f"{name}__mask"],
            )
            for name in names
        ]


# ---------------------------------------------------------------------------- Sen1Floods11


def pack_sen1floods11(raw_root: str | Path, out_path: str | Path, log=print) -> Path:
    """Pack all four official Sen1Floods11 splits into one compressed npz."""
    from impactlens.data.sen1floods11 import SPLITS, read_chip, read_chip_ids

    raw_root, out_path = Path(raw_root), Path(out_path)
    arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        ids = read_chip_ids(raw_root, split)
        imgs, masks, valids = [], [], []
        for chip in ids:
            img, water, valid = read_chip(raw_root, chip)
            imgs.append(quantize(img))
            masks.append(water)
            valids.append(valid)
        shapes = {m.shape for m in masks}
        if len(shapes) != 1:
            raise ValueError(f"chips in split {split!r} differ in size: {sorted(shapes)}")
        arrays[f"{split}_img"] = np.stack(imgs)
        arrays[f"{split}_mask"] = np.stack(masks)
        arrays[f"{split}_valid"] = np.stack(valids)
        arrays[f"{split}_ids"] = np.array(ids)
        log(f"  sen1floods11 {split:8} {len(ids)} chips")
    arrays["meta"] = _meta(dataset="sen1floods11", splits=list(SPLITS))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path


def load_sen1floods11_arrays(path: str | Path, split: str) -> dict[str, np.ndarray]:
    """Arrays of one split: img (N,4,H,W) uint8, mask (N,H,W), valid (N,H,W), ids (N,)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        _read_meta(data, path)
        if f"{split}_img" not in data.files:
            raise ValueError(f"split {split!r} not in {path} (has: {sorted(data.files)})")
        return {k: data[f"{split}_{k}"] for k in ("img", "mask", "valid", "ids")}
