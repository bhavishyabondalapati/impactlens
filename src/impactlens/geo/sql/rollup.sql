-- Spatial rollups. Run after landuse + impact are loaded. All areas in hectares (m2 / 10 000).

CREATE INDEX ON landuse USING gist (geom);
CREATE INDEX ON impact USING gist (geom);
ANALYZE landuse; ANALYZE impact;

-- 0. Land part of the AOI: AOI clipped to the land side of the OSM coastline. Everything else
--    in the AOI is open sea, which becomes its own class and never enters a land total.
DROP TABLE IF EXISTS land_aoi;
CREATE TABLE land_aoi AS
SELECT ST_Multi(ST_CollectionExtract(ST_MakeValid(ST_Intersection(
           (SELECT ST_UnaryUnion(ST_Collect(geom)) FROM land), a.geom)), 3)) AS geom
FROM aoi a;

-- 1. Exclusive land cover: OSM polygons overlap (an orchard inside a big farmland polygon), so
--    for per-TYPE totals each piece of ground goes to exactly one class, by priority.
DROP TABLE IF EXISTS landcover;
CREATE TABLE landcover AS
WITH prio(lu_class, rank) AS (VALUES
    ('rice_paddy', 1), ('water', 2), ('port', 3), ('residential', 4), ('industrial', 5),
    ('orchard', 6), ('farmland', 7), ('wetland', 8), ('other', 9)),
merged AS (
    SELECT p.lu_class, p.rank, ST_UnaryUnion(ST_Collect(ST_Intersection(l.geom, a.geom))) AS geom
    FROM landuse l JOIN prio p USING (lu_class) CROSS JOIN land_aoi a
    WHERE ST_Intersects(l.geom, a.geom)
    GROUP BY p.lu_class, p.rank)
SELECT m.lu_class, m.rank,
       ST_Multi(ST_CollectionExtract(ST_MakeValid(COALESCE(
           ST_Difference(m.geom, (SELECT ST_UnaryUnion(ST_Collect(h.geom)) FROM merged h
                                  WHERE h.rank < m.rank)),
           m.geom)), 3)) AS geom
FROM merged m;
-- "unmapped": land inside the AOI with no OSM land-use polygon (roads, beaches, gaps in OSM)
INSERT INTO landcover
SELECT 'unmapped', 99, ST_Multi(ST_CollectionExtract(ST_MakeValid(COALESCE(
    ST_Difference(a.geom, (SELECT ST_UnaryUnion(ST_Collect(geom)) FROM landcover)), a.geom)), 3))
FROM land_aoi a;
-- "sea": the AOI outside the coastline. Kept as a class only so we can show what we excluded.
INSERT INTO landcover
SELECT 'sea', 100, ST_Multi(ST_CollectionExtract(ST_MakeValid(COALESCE(
    ST_Difference(a.geom, l.geom), a.geom)), 3))
FROM aoi a CROSS JOIN land_aoi l;
DELETE FROM landcover WHERE ST_IsEmpty(geom);
CREATE INDEX ON landcover USING gist (geom);

-- Land = everything except open water and the sea. Hotspots are searched here only.
DROP TABLE IF EXISTS landmask;
CREATE TABLE landmask AS
SELECT ST_UnaryUnion(ST_Collect(geom)) AS geom FROM landcover
WHERE lu_class NOT IN ('water', 'sea');

-- 2. Hectares per land-use TYPE and impact kind (+ area-weighted mean confidence).
DROP TABLE IF EXISTS type_impact;
CREATE TABLE type_impact AS
SELECT c.lu_class, i.kind,
       SUM(ST_Area(ST_Intersection(c.geom, i.geom))) / 1e4 AS impacted_ha,
       SUM(ST_Area(ST_Intersection(c.geom, i.geom)) * i.mean_conf)
         / NULLIF(SUM(ST_Area(ST_Intersection(c.geom, i.geom))), 0) AS mean_conf
FROM landcover c JOIN impact i ON ST_Intersects(c.geom, i.geom)
GROUP BY c.lu_class, i.kind;

-- 3. Per PLOT (each OSM polygon): impacted hectares per kind, flood severity.
--    severity score = (flooded area / plot area) x mean flood confidence
--    level: high >= 0.30, medium >= 0.10, low > 0, none = 0
DROP TABLE IF EXISTS plot_impact;
CREATE TABLE plot_impact AS
WITH clipped AS (
    SELECT l.id, ST_Intersection(l.geom, a.geom) AS geom
    FROM landuse l CROSS JOIN land_aoi a WHERE ST_Intersects(l.geom, a.geom)),
hits AS (
    SELECT c.id, i.kind,
           SUM(ST_Area(ST_Intersection(c.geom, i.geom))) AS m2,
           SUM(ST_Area(ST_Intersection(c.geom, i.geom)) * i.mean_conf) AS m2_conf
    FROM clipped c JOIN impact i ON ST_Intersects(c.geom, i.geom)
    GROUP BY c.id, i.kind)
SELECT l.id, l.osm_id, l.lu_class, l.landuse, l.crop, l.name,
       ST_Area(c.geom) / 1e4 AS area_ha,
       COALESCE(MAX(h.m2) FILTER (WHERE h.kind = 'flooded'), 0) / 1e4 AS flooded_ha,
       COALESCE(MAX(h.m2) FILTER (WHERE h.kind = 'changed'), 0) / 1e4 AS changed_ha,
       COALESCE(MAX(h.m2) FILTER (WHERE h.kind = 'ndvi_drop'), 0) / 1e4 AS ndvi_drop_ha,
       COALESCE(MAX(h.m2_conf / NULLIF(h.m2, 0)) FILTER (WHERE h.kind = 'flooded'), 0)
           AS flood_conf,
       c.geom
FROM landuse l JOIN clipped c USING (id) LEFT JOIN hits h USING (id)
GROUP BY l.id, c.geom;
ALTER TABLE plot_impact ADD COLUMN flooded_frac double precision, ADD COLUMN severity_score
    double precision, ADD COLUMN severity text;
UPDATE plot_impact SET flooded_frac = flooded_ha / NULLIF(area_ha, 0);
UPDATE plot_impact SET severity_score = COALESCE(flooded_frac, 0) * flood_conf;
UPDATE plot_impact SET severity = CASE
    WHEN severity_score >= 0.30 THEN 'high'
    WHEN severity_score >= 0.10 THEN 'medium'
    WHEN severity_score > 0 THEN 'low'
    ELSE 'none' END;

-- 4. Hotspots: DBSCAN over the flooded polygons ON LAND. Polygons closer than 100 m are
--    neighbours; a cluster needs at least 3 of them. Noise (cluster id NULL) is dropped.
DROP TABLE IF EXISTS hotspot;
CREATE TABLE hotspot AS
WITH on_land AS (
    SELECT i.id, i.mean_conf,
           ST_CollectionExtract(ST_Intersection(i.geom, m.geom), 3) AS geom
    FROM impact i CROSS JOIN landmask m
    WHERE i.kind = 'flooded' AND ST_Intersects(i.geom, m.geom)),
clustered AS (
    SELECT *, ST_ClusterDBSCAN(geom, eps := 100, minpoints := 3) OVER () AS cid
    FROM on_land WHERE NOT ST_IsEmpty(geom)),
agg AS (
    SELECT cid, COUNT(*) AS n_polygons, SUM(ST_Area(geom)) / 1e4 AS flooded_ha,
           SUM(ST_Area(geom) * mean_conf) / NULLIF(SUM(ST_Area(geom)), 0) AS mean_conf,
           ST_Multi(ST_UnaryUnion(ST_Collect(geom))) AS geom
    FROM clustered WHERE cid IS NOT NULL GROUP BY cid)
SELECT a.*,
       (SELECT c.lu_class FROM landcover c WHERE ST_Intersects(c.geom, a.geom)
        ORDER BY ST_Area(ST_Intersection(c.geom, a.geom)) DESC LIMIT 1) AS main_lu_class,
       (SELECT COALESCE(SUM(ST_Area(ST_Intersection(c.geom, a.geom))), 0) / 1e4
        FROM landcover c WHERE c.lu_class = 'rice_paddy' AND ST_Intersects(c.geom, a.geom))
        AS rice_paddy_ha
FROM agg a;

-- 5. Per TOWN (municipality) and land-cover class: impacted hectares per kind.
--    town_cover = each town's land inside the AOI, split by exclusive land-cover class.
DROP TABLE IF EXISTS town_cover;
CREATE TABLE town_cover AS
SELECT t.osm_id, t.name, c.lu_class,
       ST_Multi(ST_CollectionExtract(ST_Intersection(t.geom, c.geom), 3)) AS geom
FROM town t JOIN landcover c ON ST_Intersects(t.geom, c.geom)
WHERE c.lu_class <> 'sea';
DELETE FROM town_cover WHERE ST_IsEmpty(geom);
CREATE INDEX ON town_cover USING gist (geom);

DROP TABLE IF EXISTS town_impact;
CREATE TABLE town_impact AS
SELECT tc.osm_id, tc.name, tc.lu_class, i.kind,
       SUM(ST_Area(ST_Intersection(tc.geom, i.geom))) / 1e4 AS impacted_ha
FROM town_cover tc JOIN impact i ON ST_Intersects(tc.geom, i.geom)
GROUP BY tc.osm_id, tc.name, tc.lu_class, i.kind;
