"""Pack OSCD + Sen1Floods11 into small .npz files for upload to Kaggle.

    uv run python scripts/pack_for_kaggle.py              # both files
    uv run python scripts/pack_for_kaggle.py --only floods  # just sen1floods11.npz

Output (default build/kaggle/, gitignored):
    oscd.npz, sen1floods11.npz, dataset-metadata.json

After packing, each file is loaded back through the real dataset classes and compared with the
raw-data loaders, so you know the packed copy trains on the same pixels (within uint8 rounding).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from impactlens.data.oscd import OSCDDataset
from impactlens.data.packed import pack_oscd, pack_sen1floods11
from impactlens.data.sen1floods11 import Sen1Floods11Dataset

warnings.filterwarnings("ignore")

DESCRIPTION = (
    "Preprocessed RGB+NIR tiles for the ImpactLens project (github.com/<you>/impactlens). "
    "oscd.npz: Onera Satellite Change Detection (Daudt et al.) change masks. "
    "sen1floods11.npz: Sen1Floods11 hand-labeled flood water masks (Bonafilia et al.). "
    "oscd.npz has 4 bands (blue, green, red, nir); sen1floods11.npz has 6 (adds swir16, swir22). "
    "Bands are normalized to [0,1] with fixed per-band ceilings and stored as uint8. "
    "Derived from the original datasets; see their licenses before redistributing."
)


def _mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def verify(
    oscd_raw: Path, s1f_raw: Path, oscd_npz: Path, s1f_npz: Path, only: str = "both"
) -> None:
    """Compare packed vs raw datasets: same tile counts, same masks, pixels within 1/255."""
    tol = 0.5 / 255 + 1e-6
    for split in ("train", "val", "test") if only in ("both", "oscd") else ():
        raw, packed = OSCDDataset(oscd_raw, split), OSCDDataset(oscd_npz, split)
        assert len(raw) == len(packed), f"oscd {split}: tile count differs"
        for i in range(0, len(raw), max(1, len(raw) // 10)):  # a spread of ~10 tiles
            a, b = raw[i], packed[i]
            assert a["city"] == b["city"] and (a["mask"] == b["mask"]).all()
            err = max((a[k] - b[k]).abs().max().item() for k in ("img1", "img2"))
            assert err <= tol, f"oscd {split} tile {i}: pixel error {err}"
        print(f"  verified oscd {split}: {len(raw)} tiles")
    for split in ("train", "valid", "test", "bolivia") if only in ("both", "floods") else ():
        raw, packed = Sen1Floods11Dataset(s1f_raw, split), Sen1Floods11Dataset(s1f_npz, split)
        assert len(raw) == len(packed), f"sen1floods11 {split}: tile count differs"
        for i in range(0, len(raw), max(1, len(raw) // 10)):
            a, b = raw[i], packed[i]
            assert a["chip"] == b["chip"]
            assert (a["mask"] == b["mask"]).all() and (a["valid"] == b["valid"]).all()
            assert np.abs((a["img"] - b["img"]).numpy()).max() <= tol
        print(f"  verified sen1floods11 {split}: {len(raw)} tiles")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--oscd-root", type=Path, default=Path("data/raw/oscd"))
    p.add_argument("--s1f-root", type=Path, default=Path("data/raw/sen1floods11"))
    p.add_argument("--out", type=Path, default=Path("build/kaggle"))
    p.add_argument("--kaggle-user", default="bhavishya12345678")
    p.add_argument("--slug", default="impactlens-packed", help="Kaggle dataset name")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument(
        "--only", choices=("both", "oscd", "floods"), default="both",
        help="pack just one file (e.g. 'floods' after the SWIR change; oscd.npz is unchanged)",
    )  # fmt: skip
    args = p.parse_args()

    oscd_npz, s1f_npz = args.out / "oscd.npz", args.out / "sen1floods11.npz"
    if args.only in ("both", "oscd"):
        pack_oscd(args.oscd_root, oscd_npz)
    if args.only in ("both", "floods"):
        pack_sen1floods11(args.s1f_root, s1f_npz)
    if not args.no_verify:
        verify(args.oscd_root, args.s1f_root, oscd_npz, s1f_npz, args.only)

    meta = {
        "title": "ImpactLens packed datasets",
        "id": f"{args.kaggle_user}/{args.slug}",
        "licenses": [{"name": "other"}],
        "subtitle": "OSCD (4-band) + Sen1Floods11 (6-band) tiles, packed for training",
        "description": DESCRIPTION,
    }
    (args.out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nWrote to {args.out}/:")
    for f in (oscd_npz, s1f_npz, args.out / "dataset-metadata.json"):
        if not f.exists():
            continue
        print(f"  {f.name:28} {_mb(f):8.1f} MB")
    print(f"  total upload: {sum(_mb(f) for f in args.out.iterdir()):.1f} MB")
    print(f"Kaggle dataset id: {meta['id']}  (input path: /kaggle/input/{args.slug})")


if __name__ == "__main__":
    main()
