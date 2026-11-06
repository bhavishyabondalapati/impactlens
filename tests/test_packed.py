"""Packed .npz datasets (for Kaggle): round-trip fidelity, splits, safety checks."""

import json
import subprocess
import sys

import numpy as np
import pytest

from impactlens.data import oscd, packed
from impactlens.data.oscd import OSCDDataset
from impactlens.data.sen1floods11 import Sen1Floods11Dataset
from test_oscd import _make_city
from test_sen1floods11 import _make_chip

TOL = 0.5 / 255 + 1e-6  # uint8 rounding error bound


@pytest.fixture
def raw_oscd(tmp_path, monkeypatch):
    monkeypatch.setattr(oscd, "VAL_CITIES", ("cityC",))
    root = tmp_path / "oscd_raw"
    for name, split, h, w, seed in (
        ("cityA", "train", 300, 280, 1),
        ("cityC", "train", 270, 260, 3),
        ("cityD", "test", 260, 260, 4),  # smaller height than a tile: exercises padding
    ):
        _make_city(root, name, split, h, w, seed)
    (root / "images" / "train.txt").write_text("cityA,cityC")
    (root / "images" / "test.txt").write_text("cityD")
    return root


@pytest.fixture
def raw_s1f(tmp_path):
    root = tmp_path / "s1f_raw"
    for chip in ("Ghana_1", "Ghana_2", "Bolivia_3"):
        _make_chip(root, chip)
    splits = root / "splits"
    splits.mkdir()
    for split, chips in (("train", ["Ghana_1", "Ghana_2"]), ("valid", ["Ghana_2"]),
                         ("test", ["Ghana_1"]), ("bolivia", ["Bolivia_3"])):  # fmt: skip
        rows = "".join(f"{c}_S1Hand.tif,{c}_LabelHand.tif\n" for c in chips)
        (splits / f"flood_{split}_data.csv").write_text(rows)
    return root


def test_quantize_roundtrip_error_is_bounded():
    x = np.random.default_rng(0).uniform(0, 1, (4, 50, 50)).astype(np.float32)
    q = packed.quantize(x)
    assert q.dtype == np.uint8
    assert np.abs(packed.dequantize(q) - x).max() <= TOL
    # exact endpoints and clipping
    assert packed.quantize(np.array([-1.0, 0.0, 1.0, 2.0])).tolist() == [0, 0, 255, 255]


def test_oscd_packed_matches_raw_for_every_split(raw_oscd, tmp_path):
    npz = packed.pack_oscd(raw_oscd, tmp_path / "oscd.npz", log=lambda *_: None)
    for split in ("train", "val", "test", "train_all"):
        raw, pk = OSCDDataset(raw_oscd, split), OSCDDataset(npz, split)
        assert [c.name for c in raw.cities] == [c.name for c in pk.cities]
        assert len(raw) == len(pk)
        for i in range(len(raw)):
            a, b = raw[i], pk[i]
            assert (a["mask"] == b["mask"]).all() and (a["valid"] == b["valid"]).all()
            assert (a["img1"] - b["img1"]).abs().max() <= TOL
            assert (a["img2"] - b["img2"]).abs().max() <= TOL


def test_packed_oscd_carries_its_own_val_cities(raw_oscd, tmp_path, monkeypatch):
    npz = packed.pack_oscd(raw_oscd, tmp_path / "oscd.npz", log=lambda *_: None)
    monkeypatch.setattr(oscd, "VAL_CITIES", ("cityA",))  # code changes later; the pack must not
    assert [c.name for c in OSCDDataset(npz, "val").cities] == ["cityC"]
    assert [c.name for c in OSCDDataset(npz, "train").cities] == ["cityA"]


def test_packed_oscd_random_crops_work(raw_oscd, tmp_path):
    npz = packed.pack_oscd(raw_oscd, tmp_path / "oscd.npz", log=lambda *_: None)
    ds = OSCDDataset(npz, "train", crops_per_epoch=3, tile=128)
    assert len(ds) == 3 and ds[0]["img1"].shape == (4, 128, 128)


def test_s1f_packed_matches_raw_for_every_split(raw_s1f, tmp_path):
    npz = packed.pack_sen1floods11(raw_s1f, tmp_path / "s1f.npz", log=lambda *_: None)
    for split in ("train", "valid", "test", "bolivia"):
        raw, pk = (
            Sen1Floods11Dataset(raw_s1f, split, tile=128),
            Sen1Floods11Dataset(npz, split, tile=128),
        )
        assert len(raw) == len(pk) > 0
        for i in range(len(raw)):
            a, b = raw[i], pk[i]
            assert a["chip"] == b["chip"]
            assert (a["mask"] == b["mask"]).all() and (a["valid"] == b["valid"]).all()
            assert (a["img"] - b["img"]).abs().max() <= TOL


def test_s1f_packed_keeps_ignore_pixels_ignored(raw_s1f, tmp_path):
    npz = packed.pack_sen1floods11(raw_s1f, tmp_path / "s1f.npz", log=lambda *_: None)
    item = Sen1Floods11Dataset(npz, "train", tile=300)[0]
    assert item["valid"][0, 100:150].max() == 0 and (item["mask"] * (1 - item["valid"])).sum() == 0
    with pytest.raises(ValueError):
        Sen1Floods11Dataset(npz, "nope")


def test_stale_pack_with_different_clips_is_refused(raw_oscd, tmp_path, monkeypatch):
    npz = packed.pack_oscd(raw_oscd, tmp_path / "oscd.npz", log=lambda *_: None)
    monkeypatch.setattr(packed, "BAND_CLIPS", (0.4, 0.4, 0.4, 0.4))
    with pytest.raises(ValueError, match="Re-run scripts/pack_for_kaggle.py"):
        OSCDDataset(npz, "train")


def test_pack_metadata_is_plain_json_and_files_need_no_pickle(raw_oscd, tmp_path):
    npz = packed.pack_oscd(raw_oscd, tmp_path / "oscd.npz", log=lambda *_: None)
    with np.load(npz, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
    assert meta["band_names"] == ["blue", "green", "red", "nir"] and meta["dataset"] == "oscd"


def test_packed_path_does_not_need_rasterio(raw_oscd, tmp_path):
    """On Kaggle we install without rasterio: loading packed data must never import it."""
    npz = packed.pack_oscd(raw_oscd, tmp_path / "oscd.npz", log=lambda *_: None)
    code = (
        "import sys; sys.modules['rasterio'] = None  # any import of rasterio now fails\n"
        "from impactlens.data.oscd import OSCDDataset\n"
        "from impactlens.data.sen1floods11 import Sen1Floods11Dataset\n"
        f"print(len(OSCDDataset({str(npz)!r}, 'test')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
