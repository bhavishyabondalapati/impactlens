"""Load polygons into PostGIS, run the spatial SQL, export the summary JSON + GeoJSON."""

from __future__ import annotations

import json
import os
from pathlib import Path

SQL_DIR = Path(__file__).parent / "sql"

PADDY_NOTE = (
    "Rice paddies around the Albufera are traditionally flooded on purpose in late autumn and "
    "winter, after the harvest. The Oct 26 -> Nov 10 window overlaps that, so new water in "
    "paddies may be farming practice, not flood damage. Reported separately and flagged; not "
    "dropped and not merged into the farmland totals. OSM only tags some paddies with "
    "crop=rice, so untagged paddies count as 'farmland' (the confound may be larger)."
)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader (KEY=VALUE lines); existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def dsn(dbname: str = "impactlens") -> str:
    load_dotenv()
    if os.environ.get("DATABASE_URL") and dbname == "impactlens":
        return os.environ["DATABASE_URL"]
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD missing: copy .env.example to .env")
    port = os.environ.get("POSTGRES_PORT", "5433")
    return f"host=127.0.0.1 port={port} dbname={dbname} user=impactlens password={password}"


def connect(dbname: str = "impactlens"):
    import psycopg

    return psycopg.connect(dsn(dbname), autocommit=True, connect_timeout=3)


def run_sql_file(conn, name: str) -> None:
    conn.execute((SQL_DIR / name).read_text())


def load(conn, aoi_wkt: str, epsg: int, landuse: list[dict], impacts: dict[str, list[dict]],
         land: dict | None = None, towns: list[dict] = ()):  # fmt: skip
    """aoi_wkt / impact geometries are in the raster CRS (``epsg``); landuse, land and towns are
    lon/lat. ``land`` = land side of the coastline; None means the whole AOI is land."""
    run_sql_file(conn, "schema.sql")
    to_utm = "ST_Transform(ST_SetSRID({g}, {s}), 32630)"
    conn.execute(f"INSERT INTO aoi VALUES ({to_utm.format(g='ST_GeomFromText(%s)', s=epsg)})",
                 (aoi_wkt,))  # fmt: skip
    from_lonlat = ("ST_Multi(ST_CollectionExtract(ST_MakeValid(ST_Transform(ST_SetSRID("
                   "ST_GeomFromGeoJSON(%s), 4326), 32630)), 3))")  # fmt: skip
    if land is None:
        conn.execute("INSERT INTO land SELECT ST_Multi(geom) FROM aoi")
    else:
        conn.execute(f"INSERT INTO land VALUES ({from_lonlat})", (json.dumps(land),))
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO town (osm_id, name, geom) VALUES (%s, %s, {from_lonlat})",
            [(t["osm_id"], t["name"], json.dumps(t["geometry"])) for t in towns],
        )
        cur.executemany(
            "INSERT INTO landuse (osm_id, landuse, crop, name, lu_class, geom) VALUES "
            "(%s, %s, %s, %s, %s, ST_Multi(ST_CollectionExtract(ST_MakeValid("
            "ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 32630)), 3)))",
            [(f["osm_id"], f["landuse"], f["crop"], f["name"], f["lu_class"],
              json.dumps(f["geometry"])) for f in landuse],
        )  # fmt: skip
        geom = to_utm.format(g="ST_GeomFromGeoJSON(%s)", s=epsg)
        cur.executemany(
            "INSERT INTO impact (kind, mean_conf, pixels, geom) VALUES "
            f"(%s, %s, %s, ST_Multi(ST_CollectionExtract(ST_MakeValid({geom}), 3)))",
            [(kind, f["mean_conf"], f["pixels"], json.dumps(f["geometry"]))
             for kind, feats in impacts.items() for f in feats],
        )  # fmt: skip
    conn.execute("DELETE FROM landuse WHERE ST_IsEmpty(geom)")
    conn.execute("DELETE FROM impact WHERE ST_IsEmpty(geom)")


def rollup(conn) -> None:
    run_sql_file(conn, "rollup.sql")


def _rows(conn, sql: str, params=()) -> list[dict]:
    cur = conn.execute(sql, params)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def build_summary(conn, context: dict) -> dict:
    """The metrics summary the report agents (Phase 5) must stay consistent with."""
    cover = {r["lu_class"]: r for r in _rows(conn,
             "SELECT lu_class, ST_Area(geom) / 1e4 AS area_ha FROM landcover")}  # fmt: skip
    by_type = {}
    for lu, c in sorted(cover.items(), key=lambda kv: -kv[1]["area_ha"]):
        row = {"area_ha": _r(c["area_ha"])}
        for kind in ("flooded", "changed", "ndvi_drop"):
            hit = _rows(conn, "SELECT impacted_ha, mean_conf FROM type_impact "
                              "WHERE lu_class = %s AND kind = %s", (lu, kind))  # fmt: skip
            ha = hit[0]["impacted_ha"] if hit else 0.0
            row[f"{kind}_ha"] = _r(ha)
            row[f"{kind}_pct"] = _r(100 * ha / c["area_ha"], 1) if c["area_ha"] else 0.0
            if kind == "flooded":
                row["flood_mean_conf"] = _r(hit[0]["mean_conf"], 3) if hit else None
        by_type[lu] = row

    sea = by_type.pop("sea", None)  # never part of any total; reported under "excluded"
    land = [k for k in by_type if k != "water"]
    flooded_land = sum(by_type[k]["flooded_ha"] for k in land)
    zero = {"area_ha": 0.0, "flooded_ha": 0.0, "flooded_pct": 0.0, "changed_ha": 0.0,
            "ndvi_drop_ha": 0.0}  # fmt: skip
    paddy = by_type.get("rice_paddy", zero)
    n_paddy = _rows(conn, "SELECT COUNT(*) AS n FROM landuse WHERE lu_class = 'rice_paddy'")[0]["n"]

    severity = {r["severity"]: r["n"] for r in _rows(conn,
                "SELECT severity, COUNT(*) AS n FROM plot_impact GROUP BY severity")}  # fmt: skip
    top_plots = _rows(
        conn,
        """
        SELECT osm_id, lu_class, landuse, name, area_ha, flooded_ha, flooded_frac, flood_conf,
               severity_score, severity, lu_class = 'rice_paddy' AS seasonal_flooding_flag
        FROM plot_impact WHERE severity <> 'none'
        ORDER BY flooded_ha DESC LIMIT 15  -- biggest impact first; severity says how bad""",
    )
    hotspots = _rows(
        conn,
        """
        SELECT cid AS hotspot_id, n_polygons, flooded_ha, mean_conf, main_lu_class, rice_paddy_ha,
               ST_X(ST_Transform(ST_Centroid(geom), 4326)) AS lon,
               ST_Y(ST_Transform(ST_Centroid(geom), 4326)) AS lat
        FROM hotspot ORDER BY flooded_ha DESC""",
    )
    for h in hotspots:
        h["rice_paddy_share"] = (
            _r(h["rice_paddy_ha"] / h["flooded_ha"], 3) if h["flooded_ha"] else 0
        )
    num = lambda d: {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
    towns = town_rows(conn)
    ndvi_land = sum(by_type[k]["ndvi_drop_ha"] for k in land)
    urban = {k: by_type.get(k, zero)["changed_ha"] for k in URBAN}

    return {
        **context,
        "crs": "EPSG:32630 (UTM 30N); areas in hectares",
        "definitions": {
            "flooded": "water AFTER and NOT water BEFORE (permanent water such as the Albufera "
            "lake is excluded)",
            "severity": "per OSM plot: flooded area fraction x mean flood confidence; "
            "high >= 0.30, medium >= 0.10, low > 0",
            "hotspot": "ST_ClusterDBSCAN of flooded polygons on land, eps 100 m, minpoints 3",
            "landcover": "OSM polygons overlap, so per-type totals give each piece of ground to "
            "one class by priority: rice_paddy > water > port > residential > industrial "
            "> orchard > farmland > wetland > other; 'unmapped' = land with no OSM land use",
            "land": "the land side of the OSM coastline (natural=coastline); the open sea is "
            "outside it and enters no total. 'on land' also leaves out the water class "
            "(the Albufera lake, canals)",
            "ndvi_drop": "NDVI before > 0.3 (vegetated) and NDVI before - after > 0.2",
            "urban": "OSM residential + industrial/commercial/retail land; the port "
            "(landuse=harbour, industrial=port) is not urban and is reported apart",
        },  # fmt: skip
        "totals": {
            "aoi_ha": _r(sum(c["area_ha"] for c in cover.values())),
            "land_ha": _r(sum(by_type[k]["area_ha"] for k in land)),
            "flooded_ha_on_land": _r(flooded_land),
            "flooded_ha_on_land_excluding_rice_paddies": _r(flooded_land - paddy["flooded_ha"]),
            "flooded_ha_on_water_class": by_type.get("water", {}).get("flooded_ha", 0.0),
            "changed_ha_on_land": _r(sum(by_type[k]["changed_ha"] for k in land)),
            "ndvi_drop_ha_on_land": _r(ndvi_land),
        },
        "impacts": {
            "standing_water": {
                "label": "standing water (paddy-confounded)",
                "flooded_ha_on_land": _r(flooded_land),
                "in_rice_paddies_ha": paddy["flooded_ha"],
                "outside_rice_paddies_ha": _r(flooded_land - paddy["flooded_ha"]),
                "flag": "seasonal_flooding_confound",
                "note": "New water on the after date. Almost all of it is in rice paddies, where "
                "it may be DANA inflow or deliberate seasonal flooding (see rice_paddies). "
                "The after image is 12 days after the DANA, when most floodwater outside the "
                "paddies had drained, so this layer is NOT the event footprint.",
            },
            "vegetation_loss_likely_mud": {
                "label": "vegetation loss / likely mud (outside rice paddies)",
                "ndvi_drop_ha_outside_rice_paddies": _r(ndvi_land - paddy["ndvi_drop_ha"]),
                "ndvi_drop_ha_in_rice_paddies_not_counted": paddy["ndvi_drop_ha"],
                "by_landuse": {
                    k: by_type[k]["ndvi_drop_ha"]
                    for k in land
                    if k != "rice_paddy" and by_type[k]["ndvi_drop_ha"] > 0
                },
                "by_town": {
                    t["name"]: t["ndvi_drop_ha_outside_rice_paddies"]
                    for t in towns
                    if t["ndvi_drop_ha_outside_rice_paddies"] > 0
                },
                "note": "Vegetated ground (NDVI > 0.3 before) that lost > 0.2 NDVI. Consistent "
                "with mud deposits and dead or flattened vegetation, but not proof: harvest, "
                "ploughing and pruning also drop NDVI. Mud on streets and roofs is invisible to "
                "this layer, because it was not vegetated before.",
            },
            "urban_change": {
                "label": "change on urban land only",
                "changed_ha_on_urban_land": _r(sum(urban.values())),
                "by_class": urban,
                "port_excluded": {
                    "changed_ha": by_type.get("port", zero)["changed_ha"],
                    "area_ha": by_type.get("port", zero)["area_ha"],
                    "note": "Port of Valencia (OSM landuse=harbour / industrial=port): ships "
                    "and containers moving between the dates look like change. Not damage, "
                    "not in changed_ha_on_urban_land.",
                },
                "by_town": {
                    t["name"]: t["urban_changed_ha"] for t in towns if t["urban_changed_ha"] > 0
                },
                "note": "OSCD Siamese U-Net change (trained on urban change, test F1 0.487) "
                "restricted to OSM residential and industrial land. Not damage by itself: "
                "construction and mud-covered streets both count as change.",
            },
        },  # fmt: skip
        "excluded": {
            "sea_ha": sea["area_ha"] if sea else 0.0,
            "sea_flooded_ha": sea["flooded_ha"] if sea else 0.0,
            "sea_changed_ha": sea["changed_ha"] if sea else 0.0,
            "note": "Open sea outside the OSM coastline. Not in any total. The flood model "
            "under-calls sea water on the before date (domain gap), so sea pixels can look "
            "'flooded'; this is the diagnostic, not damage.",
        },
        "by_landuse": by_type,
        "by_town": towns,
        "rice_paddies": {
            "flag": "seasonal_flooding_confound",
            "note": PADDY_NOTE,
            "osm_polygons": n_paddy,
            "area_ha": paddy["area_ha"],
            "flooded_ha": paddy["flooded_ha"],
            "flooded_pct": paddy["flooded_pct"],
            "share_of_flooded_land_pct": _r(100 * paddy["flooded_ha"] / flooded_land, 1)
            if flooded_land
            else 0.0,
        },
        "severity_counts": severity,
        "top_plots": [num(p) for p in top_plots],
        "hotspots": [num(h) for h in hotspots[:10]],
        "n_hotspots": len(hotspots),
        "attribution": "Land use (c) OpenStreetMap contributors, ODbL. Imagery: Copernicus "
        "Sentinel-2 via Earth Search.",  # fmt: skip
    }


URBAN = ("residential", "industrial")


def town_rows(conn) -> list[dict]:
    """Per municipality: land inside the AOI, and its impacts split the way the summary is."""
    rows = _rows(conn, """
        SELECT t.osm_id, t.name, ST_Area(t.geom) / 1e4 AS municipality_ha,
               COALESCE((SELECT SUM(ST_Area(c.geom)) FROM town_cover c
                         WHERE c.osm_id = t.osm_id AND c.lu_class <> 'water'), 0) / 1e4
                   AS land_in_aoi_ha
        FROM town t""")  # fmt: skip
    hits = _rows(conn, "SELECT osm_id, lu_class, kind, impacted_ha FROM town_impact")
    out = []
    for t in rows:
        if t["land_in_aoi_ha"] < 1:  # touches the AOI by a sliver only
            continue
        mine = [h for h in hits if h["osm_id"] == t["osm_id"] and h["lu_class"] != "water"]

        def ha(kind, keep=lambda lu: True, mine=mine):
            hit = [h for h in mine if h["kind"] == kind and keep(h["lu_class"])]
            return _r(sum(h["impacted_ha"] for h in hit))

        paddy, not_paddy = (lambda lu: lu == "rice_paddy"), (lambda lu: lu != "rice_paddy")
        ndvi_by_lu = {}
        for h in mine:
            if h["kind"] == "ndvi_drop" and h["lu_class"] != "rice_paddy":
                ndvi_by_lu[h["lu_class"]] = _r(h["impacted_ha"])
        out.append({
            "name": t["name"],
            "osm_id": t["osm_id"],
            "municipality_ha": _r(t["municipality_ha"]),
            "land_in_aoi_ha": _r(t["land_in_aoi_ha"]),
            "pct_of_municipality_in_aoi": _r(100 * t["land_in_aoi_ha"] / t["municipality_ha"], 1),
            "ndvi_drop_ha_outside_rice_paddies": ha("ndvi_drop", not_paddy),
            "ndvi_drop_by_landuse": dict(sorted(ndvi_by_lu.items(), key=lambda kv: -kv[1])),
            "ndvi_drop_ha_in_rice_paddies": ha("ndvi_drop", paddy),
            "flooded_ha_outside_rice_paddies": ha("flooded", not_paddy),
            "flooded_ha_in_rice_paddies": ha("flooded", paddy),
            "urban_changed_ha": ha("changed", lambda lu: lu in URBAN),
        })  # fmt: skip
    return sorted(out, key=lambda t: -t["ndvi_drop_ha_outside_rice_paddies"])


def town_shapes(conn) -> list[dict]:
    """Each town's part of the AOI (land only), as UTM GeoJSON, for the NDVI-drop map."""
    rows = conn.execute(
        "SELECT name, ST_AsGeoJSON(ST_SimplifyPreserveTopology(ST_UnaryUnion(ST_Collect(geom)), "
        "5)) FROM town_cover GROUP BY name"
    ).fetchall()
    return [{"name": r[0], "geometry": json.loads(r[1])} for r in rows if r[1]]


def land_geometry(conn, epsg: int) -> dict:
    """The AOI's land part in the raster CRS (to rasterize a land mask)."""
    row = conn.execute("SELECT ST_AsGeoJSON(ST_Transform(geom, %s)) FROM land_aoi",
                       (epsg,)).fetchone()  # fmt: skip
    return json.loads(row[0])


def export_geojson(conn, path: Path, table: str = "plot_impact", where: str = "TRUE") -> Path:
    """A table as a GeoJSON FeatureCollection in lon/lat (EPSG:4326, the GeoJSON standard)."""
    cols = {"plot_impact": "osm_id, lu_class, landuse, crop, name, area_ha, flooded_ha, "
                           "changed_ha, ndvi_drop_ha, flooded_frac, flood_conf, severity_score, "
                           "severity, lu_class = 'rice_paddy' AS seasonal_flooding_flag",
            "hotspot": "cid AS hotspot_id, n_polygons, flooded_ha, mean_conf, main_lu_class, "
                       "rice_paddy_ha"}[table]  # fmt: skip
    sql = f"""
        SELECT json_build_object('type', 'FeatureCollection', 'features',
               COALESCE(json_agg(json_build_object(
                   'type', 'Feature',
                   'geometry', ST_AsGeoJSON(ST_Transform(geom, 4326), 6)::json,
                   'properties', to_jsonb(t) - 'geom')), '[]'::json))
        FROM (SELECT {cols}, ST_SimplifyPreserveTopology(geom, 2) AS geom
              FROM {table} WHERE {where}) t"""
    data = conn.execute(sql).fetchone()[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def rice_paddy_outlines(conn) -> list[dict]:
    """Rice-paddy land cover as GeoJSON geometries in UTM (for the rollup map)."""
    rows = conn.execute(
        "SELECT ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, 5)) FROM landcover "
        "WHERE lu_class = 'rice_paddy'"
    ).fetchall()
    return [json.loads(r[0]) for r in rows if r[0]]


def hotspot_shapes(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT cid, flooded_ha, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, 5)) FROM hotspot"
    ).fetchall()
    return [{"id": r[0], "flooded_ha": float(r[1]), "geometry": json.loads(r[2])} for r in rows]
