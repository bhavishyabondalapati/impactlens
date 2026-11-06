"""OSCD loader tests, using a tiny fake OSCD folder written to a temp dir."""

import warnings

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from impactlens.data.oscd import OSCDDataset, load_city, read_split, tile_starts
from impactlens.data.preprocess import OSCD_BAND_FILES

# PNG masks carry no georeferencing; that is expected.
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def _write_tif(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=arr.dtype, crs="EPSG:32630", transform=from_origin(0, 0, 10, 10),
    ) as dst:  # fmt: skip
        dst.write(arr, 1)


def _make_city(root, city, split, h, w, seed):
    rng = np.random.default_rng(seed)
    for which in (1, 2):
        for band in OSCD_BAND_FILES:
            dn = rng.integers(500, 3000, size=(h, w)).astype(np.uint16)
            _write_tif(root / "images" / city / f"imgs_{which}_rect" / f"{band}.tif", dn)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:40, 20:60] = 255  # a changed rectangle
    label = root / f"{split}_labels" / city / "cm" / "cm.png"  # the real OSCD label format
    label.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        label, "w", driver="PNG", height=h, width=w, count=1, dtype="uint8"
    ) as dst:  # fmt: skip
        dst.write(mask, 1)


@pytest.fixture
def oscd_root(tmp_path):
    _make_city(tmp_path, "cityA", "train", 300, 280, seed=1)
    _make_city(tmp_path, "cityB", "test", 260, 260, seed=2)
    (tmp_path / "images" / "train.txt").write_text("cityA\n")
    (tmp_path / "images" / "test.txt").write_text("cityB")
    return tmp_path


def test_read_split(oscd_root):
    assert read_split(oscd_root, "train") == ["cityA"]
    assert read_split(oscd_root, "test") == ["cityB"]
    with pytest.raises(ValueError):
        read_split(oscd_root, "val")


def test_tile_starts_cover_edges():
    assert tile_starts(100, 256, 256) == [0]
    assert tile_starts(300, 256, 256) == [0, 44]  # last tile shifted back to hit the edge
    assert tile_starts(512, 256, 256) == [0, 256]


def test_load_city_shapes_and_range(oscd_root):
    city = load_city(oscd_root, "cityA", "train")
    assert city.img1.shape == (4, 300, 280) == city.img2.shape
    assert city.mask.shape == (300, 280)
    assert 0.0 <= city.img1.min() and city.img1.max() <= 1.0
    assert set(np.unique(city.mask)) <= {0, 1}
    assert city.mask.sum() == 30 * 40


def test_dataset_tiles(oscd_root):
    ds = OSCDDataset(oscd_root, "train")
    assert len(ds) == 4  # 300x280 -> 2 x 2 tiles of 256
    item = ds[0]
    assert item["img1"].shape == (4, 256, 256)
    assert item["img2"].shape == (4, 256, 256)
    assert item["mask"].shape == (1, 256, 256)
    assert item["city"] == "cityA"


def test_small_image_is_padded(oscd_root):
    ds = OSCDDataset(oscd_root, "test", tile=300)  # larger than the 260x260 image
    assert ds[0]["img1"].shape == (4, 300, 300)


def test_augment_keeps_mask_aligned_with_images(oscd_root):
    """Flip/rotate must move image and mask together: change-pixel count is preserved."""
    plain = OSCDDataset(oscd_root, "train", tile=128)
    aug = OSCDDataset(oscd_root, "train", tile=128, augment=True)
    torch.manual_seed(0)
    for i in range(len(plain)):
        a, b = plain[i], aug[i]
        assert a["mask"].sum() == b["mask"].sum()
        # Whatever happened to the mask also happened to the image: sorted values match.
        assert torch.allclose(a["img1"].flatten().sort().values, b["img1"].flatten().sort().values)
