"""Write the Valencia request JSON (AOI + date windows) that starts one AWS run.

Usage: python scripts/make_request.py <out.json> [--event]

Request format (read by src/impactlens/aws/steps.py):
  aoi        GeoJSON (Feature, FeatureCollection or geometry), lon/lat, at most 2,500 km2
  before     date window for the before image, "YYYY-MM-DD/YYYY-MM-DD"
  after      date window for the after image
  max_cloud  optional: max % cloud over the AOI (default 20)
  event      optional: {"name", "event_date"} used in the reports (default: Valencia DANA)
"""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("--event", action="store_true", help="include the event name and date")
args = parser.parse_args()

request = {
    "aoi": json.loads(Path("aoi/valencia_albufera.geojson").read_text()),
    "before": "2024-10-01/2024-10-28",
    "after": "2024-10-30/2024-11-12",
    "max_cloud": 20,
}
if args.event:
    request["event"] = {"name": "DANA (cut-off low) floods, Valencia province, Spain",
                        "event_date": "2024-10-29"}  # fmt: skip
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(request))
print(f"wrote {args.out}")
