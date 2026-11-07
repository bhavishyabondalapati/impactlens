"""Load polygons into PostGIS, run the spatial SQL, export the summary JSON + GeoJSON."""

from __future__ import annotations

import json
import os
from pathlib import Path

from impactlens.geo.summary import RollupTables, assemble_summary

SQL_DIR = Path(__file__).parent / "sql"


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


def tables(conn) -> RollupTables:
    """Read the tables rollup.sql built into the shared RollupTables (see geo/summary.py)."""
    cover = {r["lu_class"]: float(r["area_ha"]) for r in _rows(conn,
             "SELECT lu_class, ST_Area(geom) / 1e4 AS area_ha FROM landcover")}  # fmt: skip
    type_impact = {
        (r["lu_class"], r["kind"]): (float(r["impacted_ha"]), r["mean_conf"])
        for r in _rows(conn, "SELECT lu_class, kind, impacted_ha, mean_conf FROM type_impact")
    }
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
    towns = _rows(conn, """
        SELECT t.osm_id, t.name, ST_Area(t.geom) / 1e4 AS municipality_ha,
               COALESCE((SELECT SUM(ST_Area(c.geom)) FROM town_cover c
                         WHERE c.osm_id = t.osm_id AND c.lu_class <> 'water'), 0) / 1e4
                   AS land_in_aoi_ha
        FROM town t""")  # fmt: skip
    hits = _rows(conn, "SELECT osm_id, lu_class, kind, impacted_ha FROM town_impact")
    return RollupTables(cover, type_impact, n_paddy, severity, top_plots, hotspots, towns, hits)


def build_summary(conn, context: dict) -> dict:
    return assemble_summary(tables(conn), context)


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


PLOT_KEEP = "severity <> 'none' OR changed_ha > 0 OR ndvi_drop_ha > 0"


class PostgisBackend:
    """The rollup run by PostGIS (local docker). Same methods as rollup_shapely.ShapelyBackend,
    so geo/pipeline.py can use either."""

    name = "postgis"

    def __init__(self, conn=None):
        self.conn = conn or connect()

    def load(self, aoi_wkt, epsg, landuse, impacts, land=None, towns=()):
        load(self.conn, aoi_wkt, epsg, landuse, impacts, land, towns)

    def rollup(self):
        rollup(self.conn)

    def tables(self) -> RollupTables:
        return tables(self.conn)

    def build_summary(self, context: dict) -> dict:
        return build_summary(self.conn, context)

    def rice_paddy_outlines(self):
        return rice_paddy_outlines(self.conn)

    def town_shapes(self):
        return town_shapes(self.conn)

    def land_geometry(self, epsg):
        return land_geometry(self.conn, epsg)

    def hotspot_shapes(self):
        return hotspot_shapes(self.conn)

    def export_geojson(self, path, table="plot_impact", impacted_only=False):
        return export_geojson(self.conn, path, table, PLOT_KEEP if impacted_only else "TRUE")

    def close(self):
        self.conn.close()
