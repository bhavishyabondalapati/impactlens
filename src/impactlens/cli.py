"""Command line entry point: ``impactlens fetch | infer | rollup``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from impactlens.data.stac import EARTH_SEARCH_URL, NoSceneError, fetch_pair, format_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="impactlens")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch", help="Download aligned before/after Sentinel-2 crops for an AOI"
    )
    fetch.add_argument("--aoi", required=True, type=Path, help="AOI GeoJSON (lon/lat)")
    fetch.add_argument("--before", required=True, help="date window, e.g. 2024-10-01/2024-10-28")
    fetch.add_argument("--after", required=True, help="date window, e.g. 2024-10-30/2024-11-20")
    fetch.add_argument("--max-cloud", type=float, default=20.0, help="max %% cloud over the AOI")
    fetch.add_argument("--out", type=Path, default=None, help="default: data/processed/<aoi name>")
    fetch.add_argument("--stac-url", default=EARTH_SEARCH_URL)

    infer = sub.add_parser("infer", help="Run both models over before/after; write GeoTIFF layers")
    infer.add_argument("--scene", type=Path, default=Path("data/processed/valencia_albufera"))
    infer.add_argument("--floods-model", type=Path, default=None,
                       help="default: newest build/models_*/floods_r2/floods.onnx")  # fmt: skip
    infer.add_argument("--oscd-model", type=Path, default=Path("build/models/oscd/oscd.onnx"))
    infer.add_argument("--overlap", type=int, default=64, help="sliding-window overlap (px)")
    infer.add_argument("--quicklook", type=Path, default=None, help="PNG path (optional)")

    rollup = sub.add_parser("rollup", help="Polygonize, OSM land use, PostGIS stats, export")
    rollup.add_argument("--scene", type=Path, default=Path("data/processed/valencia_albufera"))
    rollup.add_argument("--aoi", type=Path, default=Path("aoi/valencia_albufera.geojson"))
    rollup.add_argument("--map", type=Path, default=None, help="rollup map PNG (optional)")
    rollup.add_argument("--ndvi-map", type=Path, default=None,
                        help="NDVI-drop by town map PNG (optional)")  # fmt: skip
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fetch":
        out = args.out or Path("data/processed") / args.aoi.stem
        try:
            fetch_pair(args.aoi, args.before, args.after, out, args.max_cloud, args.stac_url)
        except NoSceneError as err:
            print(f"\nNo scene passed the {err.max_cloud:g}% cloud limit. Best candidates:")
            for window, cands in err.failures.items():
                print(f"\n  {window}")
                print(format_candidates(cands))
            return 2
    elif args.command == "infer":
        from impactlens.geo.pipeline import latest_model, run_infer

        floods = args.floods_model or latest_model("build/models_*/floods_r2/floods.onnx")
        run_infer(args.scene, floods, args.oscd_model, args.scene / "layers", args.quicklook,
                  args.overlap)  # fmt: skip
    elif args.command == "rollup":
        from impactlens.geo.pipeline import run_rollup

        run_rollup(args.scene, args.scene / "layers", args.aoi, args.scene / "geo",
                   Path("data/raw/osm"), args.map, args.ndvi_map)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
