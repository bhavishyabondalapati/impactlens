"""OpenStreetMap land-use polygons for the AOI, via the Overpass API.

One query per AOI, cached to data/raw/osm/<aoi>_landuse.json (OSM data is ODbL-licensed:
keep the attribution "(c) OpenStreetMap contributors" wherever results are shown).

Every polygon gets a ``lu_class`` used in the rollups:
  rice_paddy  landuse=paddy, or crop=rice (rice fields are flooded ON PURPOSE in late
              autumn/winter, so "new water" there may be farming, not a disaster: we report
              them separately and flag them, never drop them)
  orchard     landuse=orchard (around Valencia these are mostly citrus)
  farmland    other farmland, meadow, greenhouses, farmyards, vineyards
  residential residential
  industrial  industrial, commercial, retail
  water       natural=water (e.g. the Albufera lake)
  wetland     natural=wetland (marsh, reed beds; paddies tagged wetland=rice -> rice_paddy)
  other       anything else (forest, grass, construction, ...)
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from shapely.geometry import LineString, MultiPolygon, Polygon, mapping
from shapely.ops import linemerge, polygonize, unary_union

ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USER_AGENT = "impactlens/0.1 (open-source research project)"
FARMLAND = {"farmland", "meadow", "greenhouse_horticulture", "farmyard", "vineyard", "allotments"}
INDUSTRIAL = {"industrial", "commercial", "retail"}


def overpass_query(south: float, west: float, north: float, east: float) -> str:
    bbox = f"{south},{west},{north},{east}"
    return (
        "[out:json][timeout:180];("
        f'way["landuse"]({bbox});relation["landuse"]({bbox});'
        f'way["natural"~"^(water|wetland)$"]({bbox});'
        f'relation["natural"~"^(water|wetland)$"]({bbox});'
        ");out geom;"
    )


def fetch_overpass(query: str, cache: Path, retries: int = 5, log=print) -> dict:
    """Run the query (or read the cache). Retries with exponential backoff across mirrors."""
    if cache.exists():
        return json.loads(cache.read_text())
    body = urllib.parse.urlencode({"data": query}).encode()
    wait, last_err = 10, None
    for attempt in range(retries):
        url = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            req = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=240) as resp:
                data = json.loads(resp.read())
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
            return data
        except Exception as err:  # 429/502/504 busy, timeouts, bad JSON
            last_err = err
            log(f"  overpass {url} failed ({err}); retrying in {wait}s")
            time.sleep(wait)
            wait *= 2
    raise RuntimeError(f"Overpass failed after {retries} tries: {last_err}")


def classify(tags: dict) -> str:
    landuse, natural = tags.get("landuse"), tags.get("natural")
    if landuse == "paddy" or "rice" in tags.get("crop", "") or tags.get("wetland") == "rice":
        return "rice_paddy"
    if landuse == "orchard":
        return "orchard"
    if landuse in FARMLAND:
        return "farmland"
    if landuse == "residential":
        return "residential"
    if landuse in INDUSTRIAL:
        return "industrial"
    if natural == "water" or landuse in ("reservoir", "basin"):
        return "water"
    if natural == "wetland":
        return "wetland"
    return "other"


def _ring(coords) -> list[tuple[float, float]]:
    return [(p["lon"], p["lat"]) for p in coords]


def element_geometry(el: dict):
    """Polygon for a closed way, (Multi)Polygon for a multipolygon relation, else None."""
    if el["type"] == "way":
        pts = _ring(el.get("geometry", []))
        if len(pts) >= 4 and pts[0] == pts[-1]:
            poly = Polygon(pts)
            return poly if poly.is_valid else poly.buffer(0)
        return None
    outers, inners = [], []
    for m in el.get("members", []):
        if m.get("type") == "way" and m.get("geometry"):
            (inners if m.get("role") == "inner" else outers).append(
                LineString(_ring(m["geometry"]))
            )
    if not outers:
        return None
    outer = unary_union(list(polygonize(linemerge(outers))))
    if inners:
        outer = outer.difference(unary_union(list(polygonize(linemerge(inners)))))
    if outer.is_empty:
        return None
    return outer if isinstance(outer, Polygon | MultiPolygon) else None


def landuse_features(data: dict) -> list[dict]:
    """GeoJSON-like features (lon/lat) with osm id, tags we keep, and lu_class."""
    feats = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        geom = element_geometry(el)
        if geom is None or geom.is_empty:
            continue
        feats.append({
            "osm_id": f"{el['type']}/{el['id']}",
            "landuse": tags.get("landuse") or tags.get("natural"),
            "crop": tags.get("crop"),
            "name": tags.get("name"),
            "lu_class": classify(tags),
            "geometry": mapping(geom),
        })  # fmt: skip
    return feats
