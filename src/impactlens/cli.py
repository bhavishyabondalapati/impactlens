"""Command line entry point: ``impactlens fetch ...``."""

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
