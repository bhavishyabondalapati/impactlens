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


def coastline_query(south: float, west: float, north: float, east: float) -> str:
    bbox = f"{south},{west},{north},{east}"
    return f'[out:json][timeout:180];way["natural"="coastline"]({bbox});out geom;'


def municipality_query(south: float, west: float, north: float, east: float) -> str:
    """Municipalities (admin_level 8 in Spain) touching the bbox, with their full outline."""
    bbox = f"{south},{west},{north},{east}"
    return (
        '[out:json][timeout:180];relation["boundary"="administrative"]["admin_level"="8"]'
        f"({bbox});out geom;"
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


def municipalities(data: dict) -> list[dict]:
    """Town outlines (lon/lat) from a municipality_query result."""
    towns = []
    for el in data.get("elements", []):
        geom = element_geometry(el)
        if geom is not None and not geom.is_empty:
            name = el.get("tags", {}).get("name", f"relation/{el['id']}")
            towns.append({"osm_id": f"relation/{el['id']}", "name": name,
                          "geometry": mapping(geom)})  # fmt: skip
    return towns


def land_polygon(data: dict, bbox: Polygon):
    """The land part of ``bbox`` (lon/lat), cut out along OSM's natural=coastline ways.

    OSM convention: a coastline way is drawn with the LAND ON ITS LEFT and the sea on its right.
    We cut the bbox into pieces along the coastline, then keep each piece whose coastline edge
    has the piece on its left. No coastline in the bbox = all land (an inland AOI).
    """
    lines = [LineString(_ring(el["geometry"])) for el in data.get("elements", [])
             if el.get("type") == "way" and len(el.get("geometry", [])) >= 2]  # fmt: skip
    if not lines:
        return bbox
    coast = unary_union(lines).intersection(bbox)
    segments = []  # every straight piece of coastline inside the bbox, in drawing order
    for line in getattr(coast, "geoms", [coast]):
        if isinstance(line, LineString):
            segments += list(zip(line.coords[:-1], line.coords[1:], strict=True))
    # unary_union "nodes" the lines (splits them where they cross) so polygonize finds the pieces
    pieces = list(polygonize(unary_union([bbox.boundary, coast])))
    land = [p for p in pieces if _left_of_coast(p, segments)]
    return unary_union(land) if land else Polygon()


def _left_of_coast(piece: Polygon, segments, eps: float = 1e-7) -> bool:
    from shapely.geometry import Point

    edge = piece.boundary
    for (x0, y0), (x1, y1) in segments:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if edge.distance(Point(mx, my)) > eps:
            continue  # this coastline segment is not on the piece's edge
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 or 1.0
        left = Point(mx - (y1 - y0) / length * 1e-6, my + (x1 - x0) / length * 1e-6)
        return piece.contains(left)  # a step to the LEFT of the coast lands inside = land
    return True  # touches no coastline: an inland piece
