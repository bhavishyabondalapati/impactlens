"""Run both dataset loaders on the REAL data and print sanity statistics.

    uv run python scripts/dataset_report.py

Also saves one sample (images + mask) per dataset to docs/images/ for eyeballing.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import rasterio

from impactlens.data.oscd import OSCDDataset
from impactlens.data.preprocess import BAND_NAMES
from impactlens.data.sen1floods11 import SPLITS, Sen1Floods11Dataset

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

OSCD_ROOT = Path("data/raw/oscd")
S1F_ROOT = Path("data/raw/sen1floods11")
OUT = Path("docs/images")


def band_range(mins, maxs, means) -> str:
    return "  ".join(
        f"{n}: [{lo:.3f}, {hi:.3f}] mean {m:.3f}"
        for n, lo, hi, m in zip(BAND_NAMES, mins, maxs, means, strict=True)
    )


def to_rgb(img: np.ndarray, gain: float = 2.0) -> np.ndarray:
    """(4,H,W) in [0,1] -> (3,H,W) uint8 for display (R,G,B order, brightened by ``gain``)."""
    return (np.clip(img[[2, 1, 0]] * gain, 0, 1) * 255).astype(np.uint8)


def save_png(path: Path, panels: list[np.ndarray], scale: int = 2) -> None:
    """Place (3,H,W) uint8 panels side by side with a white gap and save as PNG."""
    gap = np.full((3, panels[0].shape[1], 4), 255, np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.concatenate([row, gap, p], axis=2)
    row = np.kron(row, np.ones((1, scale, scale), np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="PNG", height=row.shape[1], width=row.shape[2], count=3, dtype="uint8"
    ) as dst:
        dst.write(row)
    print(f"  saved {path}")


def mask_panel(mask: np.ndarray, color: tuple[int, int, int], valid=None) -> np.ndarray:
    out = np.zeros((3, *mask.shape), np.uint8)
    if valid is not None:
        out[:] = np.where(valid > 0, 0, 110)[None]  # grey = unlabeled (ignored) pixels
    for c in range(3):
        out[c][mask > 0] = color[c]
    return out


def report_oscd() -> None:
    print("\n=== OSCD (change detection) ===")
    best = (-1.0, None)
    for split in ("train", "val", "test"):
        ds = OSCDDataset(OSCD_ROOT, split)
        mins, maxs, sums = np.full(4, 9.0), np.full(4, -9.0), np.zeros(4)
        changed = total = tiles_with_change = 0
        for i in range(len(ds)):
            t = ds[i]
            both = np.concatenate([t["img1"].numpy(), t["img2"].numpy()], axis=0)  # 8 channels
            for k in range(2):
                x = both[4 * k : 4 * k + 4]
                mins, maxs = np.minimum(mins, x.min((1, 2))), np.maximum(maxs, x.max((1, 2)))
                sums += x.mean((1, 2)) / 2
            m, v = t["mask"].numpy(), t["valid"].numpy()
            changed += (m * v).sum()
            total += v.sum()  # count real pixels only, not zero-padding
            tiles_with_change += int(m.sum() > 0)
            frac = m.mean()
            # prefer a tile that is not a washed-out desert so the eyeball image is readable
            if split == "test" and t["img1"].mean() < 0.4 and frac > best[0]:
                best = (frac, t)
        n = len(ds)
        print(f"\n[{split}] {len(ds.cities)} cities, {n} tiles (256x256, stride 256)")
        print("  cities:", ", ".join(c.name for c in ds.cities))
        print("  normalized range:", band_range(mins, maxs, sums / n))
        print(
            f"  changed pixels: {100 * changed / total:.2f}%   "
            f"tiles with any change: {100 * tiles_with_change / n:.1f}%"
        )
        sizes = sorted({c.mask.shape for c in ds.cities})
        print(f"  city sizes (H,W): min {sizes[0]}, max {sizes[-1]}")
    frac, t = best
    print(f"\n  sample tile: {t['city']} ({100 * frac:.1f}% changed)")
    save_png(
        OUT / "oscd_sample.png",
        [
            to_rgb(t["img1"].numpy(), 1.5),
            to_rgb(t["img2"].numpy(), 1.5),
            mask_panel(t["mask"][0].numpy(), (255, 255, 255)),
        ],
    )


def report_s1f() -> None:
    print("\n=== Sen1Floods11 (flood water) ===")
    best = (-1.0, None)
    for split in SPLITS:
        ds = Sen1Floods11Dataset(S1F_ROOT, split)
        mins, maxs, sums = np.full(4, 9.0), np.full(4, -9.0), np.zeros(4)
        water = valid = total = tiles_with_water = 0
        for i in range(len(ds)):
            t = ds[i]
            x, m, v = t["img"].numpy(), t["mask"].numpy(), t["valid"].numpy()
            mins, maxs, sums = (
                np.minimum(mins, x.min((1, 2))),
                np.maximum(maxs, x.max((1, 2))),
                sums + x.mean((1, 2)),
            )
            water += (m * v).sum()
            valid += v.sum()
            total += v.size
            tiles_with_water += int((m * v).sum() > 0)
            frac = (m * v).sum() / max(v.sum(), 1)
            if split == "train" and v.mean() > 0.95 and 0.2 < frac < 0.7 and frac > best[0]:
                best = (frac, t)
        n = len(ds)
        print(f"\n[{split}] {len(ds.chips)} chips, {n} tiles (256x256, stride 256)")
        print("  normalized range:", band_range(mins, maxs, sums / n))
        print(
            f"  valid pixels: {100 * valid / total:.2f}%   water (of valid): "
            f"{100 * water / max(valid, 1):.2f}%   "
            f"tiles with water: {100 * tiles_with_water / n:.1f}%"
        )
    frac, t = best
    print(f"\n  sample tile: {t['chip']} ({100 * frac:.1f}% water)")
    img = t["img"].numpy()
    false_color = np.stack([img[3], img[2], img[1]])  # NIR,R,G: water shows up dark
    save_png(
        OUT / "sen1floods11_sample.png",
        [to_rgb(img), (np.clip(false_color * 2, 0, 1) * 255).astype(np.uint8),
         mask_panel(t["mask"][0].numpy(), (40, 120, 255), t["valid"][0].numpy())],
    )  # fmt: skip


if __name__ == "__main__":
    report_oscd()
    report_s1f()
