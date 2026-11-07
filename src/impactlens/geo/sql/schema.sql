-- ImpactLens Geo-AI schema. Everything is stored in UTM zone 30N (EPSG:32630), a projected CRS
-- in metres, so ST_Area gives square metres (/10 000 = hectares) without distortion.
CREATE EXTENSION IF NOT EXISTS postgis;
DROP TABLE IF EXISTS impact, landuse, landcover, aoi, landmask, land, land_aoi, town, town_cover,
    town_impact CASCADE;

CREATE TABLE aoi (geom geometry(Polygon, 32630));

-- Land side of the OSM coastline (one row). Anything in the AOI outside it is open sea.
CREATE TABLE land (geom geometry(MultiPolygon, 32630) NOT NULL);

-- Municipalities (OSM admin_level 8), full outline (may reach outside the AOI).
CREATE TABLE town (
    id serial PRIMARY KEY,
    osm_id text, name text NOT NULL,
    geom geometry(MultiPolygon, 32630) NOT NULL
);

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
