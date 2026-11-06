-- ImpactLens Geo-AI schema. Everything is stored in UTM zone 30N (EPSG:32630), a projected CRS
-- in metres, so ST_Area gives square metres (/10 000 = hectares) without distortion.
CREATE EXTENSION IF NOT EXISTS postgis;
DROP TABLE IF EXISTS impact, landuse, landcover, aoi, landmask CASCADE;

CREATE TABLE aoi (geom geometry(Polygon, 32630));

-- One row per OSM land-use polygon = one "plot".
CREATE TABLE landuse (
    id serial PRIMARY KEY,
    osm_id text, landuse text, crop text, name text,
    lu_class text NOT NULL,
    geom geometry(MultiPolygon, 32630) NOT NULL
);

-- One row per polygonized impact region. kind: flooded | changed | ndvi_drop.
CREATE TABLE impact (
    id serial PRIMARY KEY,
    kind text NOT NULL,
    mean_conf real NOT NULL,
    pixels integer NOT NULL,
    geom geometry(MultiPolygon, 32630) NOT NULL
);
