"""Sen1Floods11 loader tests on a tiny fake dataset (300x300 chips, 13 int16 bands)."""

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from impactlens.data.preprocess import FLOOD_BAND_CLIPS
from impactlens.data.sen1floods11 import Sen1Floods11Dataset, read_chip_ids

SIZE = 300


def _write(path, arr, dtype="int16"):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.atleast_3d(arr) if arr.ndim == 3 else arr[None]
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
        dtype=dtype, crs="EPSG:4326", transform=from_origin(0, 0, 1e-4, 1e-4),
    ) as dst:  # fmt: skip
        dst.write(arr.astype(dtype))


def _make_chip(root, chip):
    # Band k (1-based) holds the constant DN k*100, so we can tell which bands were picked.
    bands = np.stack([np.full((SIZE, SIZE), (k + 1) * 100) for k in range(13)])
    _write(root / "S2Hand" / f"{chip}_S2Hand.tif", bands)
    label = np.zeros((SIZE, SIZE), dtype=np.int16)
    label[:100, :] = 1  # water in the top strip
    label[100:150, :] = -1  # unlabeled strip (cloud)
    _write(root / "LabelHand" / f"{chip}_LabelHand.tif", label)


@pytest.fixture
def root(tmp_path):
    for chip in ("Ghana_1", "Ghana_2", "Bolivia_3"):
        _make_chip(tmp_path, chip)
    splits = tmp_path / "splits"
    splits.mkdir()
    # Real CSVs list S1Hand names in column one.
    (splits / "flood_train_data.csv").write_text(
        "Ghana_1_S1Hand.tif,Ghana_1_LabelHand.tif\nGhana_2_S1Hand.tif,Ghana_2_LabelHand.tif\n"
    )
    (splits / "flood_bolivia_data.csv").write_text("Bolivia_3_S1Hand.tif,Bolivia_3_LabelHand.tif\n")
    return tmp_path


def test_read_chip_ids(root):
    assert read_chip_ids(root, "train") == ["Ghana_1", "Ghana_2"]
    assert read_chip_ids(root, "bolivia") == ["Bolivia_3"]
    with pytest.raises(FileNotFoundError):
        read_chip_ids(root, "test")
    with pytest.raises(ValueError):
        read_chip_ids(root, "val")


def test_shapes_and_tiling(root):
    ds = Sen1Floods11Dataset(root, "train", tile=128)
    assert len(ds) == 2 * 9  # 300 px -> starts 0, 128, 172 -> 3x3 tiles per chip
    item = ds[0]
    assert item["img"].shape == (6, 128, 128)
    assert item["mask"].shape == item["valid"].shape == (1, 128, 128)
    assert item["chip"] == "Ghana_1"


def test_uses_b2_b3_b4_b8_b11_b12_with_shared_normalization(root):
    img = Sen1Floods11Dataset(root, "train", tile=128)[0]["img"]
    # DNs of B2, B3, B4, B8, B11, B12 are 200, 300, 400, 800, 1200, 1300 -> reflectance 0.02 ...
    # 0.13, then divided by each band's clip ceiling.
    expected = np.array([200, 300, 400, 800, 1200, 1300]) * 1e-4 / np.array(FLOOD_BAND_CLIPS)
    assert np.allclose(img.mean(dim=(1, 2)).numpy(), expected, atol=1e-6)


def test_water_and_ignore_labels(root):
    # tile=300 -> one tile covering the whole chip
    item = Sen1Floods11Dataset(root, "train", tile=300)[0]
    mask, valid = item["mask"][0], item["valid"][0]
    assert mask[:100].min() == 1 and mask[100:].max() == 0
    assert valid[100:150].max() == 0  # -1 pixels are ignored, not treated as "not water"
    assert valid[:100].min() == 1 and valid[150:].min() == 1
    assert (mask * (1 - valid)).sum() == 0  # ignored pixels never count as water


def test_padding_is_marked_invalid(root):
    item = Sen1Floods11Dataset(root, "train", tile=320)[0]  # bigger than the 300 px chip
    assert item["img"].shape == (6, 320, 320)
    assert item["valid"][0, 300:, :].max() == 0 and item["valid"][0, :, 300:].max() == 0


def test_augment_keeps_layers_aligned(root):
    plain = Sen1Floods11Dataset(root, "train", tile=300)
    aug = Sen1Floods11Dataset(root, "train", tile=300, augment=True)
    torch.manual_seed(1)
    for _ in range(5):
        a, b = plain[0], aug[0]
        assert a["mask"].sum() == b["mask"].sum() and a["valid"].sum() == b["valid"].sum()
        # Water is the top strip in `a`; wherever it lands in `b`, mask and valid move together.
        assert (b["mask"] * (1 - b["valid"])).sum() == 0
