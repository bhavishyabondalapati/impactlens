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


def load(conn, aoi_wkt: str, epsg: int, landuse: list[dict], impacts: dict[str, list[dict]]):
    """aoi_wkt / impact geometries are in the raster CRS (``epsg``); landuse is lon/lat."""
    run_sql_file(conn, "schema.sql")
    to_utm = "ST_Transform(ST_SetSRID({g}, {s}), 32630)"
    conn.execute(f"INSERT INTO aoi VALUES ({to_utm.format(g='ST_GeomFromText(%s)', s=epsg)})",
                 (aoi_wkt,))  # fmt: skip
    with conn.cursor() as cur:
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

    land = [k for k in by_type if k not in ("water", "unmapped")]
    flooded_land = sum(by_type[k]["flooded_ha"] for k in land)
    paddy = by_type.get("rice_paddy", {"area_ha": 0.0, "flooded_ha": 0.0, "flooded_pct": 0.0})
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
            "one class by priority: rice_paddy > water > residential > industrial "
            "> orchard > farmland > wetland > other; 'unmapped' = no OSM land use "
            "(includes the open sea)",
        },  # fmt: skip
        "totals": {
            "aoi_ha": _r(sum(c["area_ha"] for c in cover.values())),
            "flooded_ha_on_land": _r(flooded_land),
            "flooded_ha_on_land_excluding_rice_paddies": _r(flooded_land - paddy["flooded_ha"]),
            "flooded_ha_unmapped_incl_sea": by_type.get("unmapped", {}).get("flooded_ha", 0.0),
            "flooded_ha_on_water_class": by_type.get("water", {}).get("flooded_ha", 0.0),
            "changed_ha_on_land": _r(sum(by_type[k]["changed_ha"] for k in land)),
            "ndvi_drop_ha_on_land": _r(sum(by_type[k]["ndvi_drop_ha"] for k in land)),
        },
        "by_landuse": by_type,
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
