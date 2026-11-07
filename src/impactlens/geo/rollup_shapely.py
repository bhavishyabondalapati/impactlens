"""The PostGIS rollup (sql/rollup.sql) redone in plain Shapely, for AWS Lambda.

Lambda has no database, and running PostGIS just for one rollup per request would cost money
while idle. Shapely 2 wraps GEOS, the same geometry engine PostGIS uses, so the same steps give
the same hectares:

  0. land_aoi   = AOI on the land side of the coastline
  1. landcover  = exclusive classes by priority (+ 'unmapped' land, + 'sea')
  2. type_impact= impacted hectares per class and impact kind (+ area-weighted confidence)
  3. plots      = per OSM polygon: flooded / changed / NDVI-drop hectares, flood severity
  4. hotspots   = DBSCAN of flooded polygons on land (eps 100 m, minpoints 3)
  5. towns      = each municipality's land in the AOI, and its impacts per class

Speed trick used throughout: big multipolygons (a class can cover thousands of hectares) are
split into their parts and indexed in an STRtree, so each small impact polygon is only
intersected with the few parts it actually touches. Parts of a valid multipolygon do not
overlap, so summing over parts gives the same area as one big intersection.

tests/test_geo.py runs the same synthetic cases through both backends, and
tests/test_aws.py checks this backend against the PostGIS summary.json on the Valencia data.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, mapping, shape

from impactlens.geo.summary import KINDS, RollupTables, assemble_summary

PRIORITY = ("rice_paddy", "water", "port", "residential", "industrial", "orchard", "farmland",
            "wetland", "other")  # fmt: skip
EMPTY = MultiPolygon()


def polys(g) -> MultiPolygon:
    """PostGIS ST_Multi(ST_CollectionExtract(ST_MakeValid(g), 3)): only the polygon parts."""
    if g is None or g.is_empty:
        return EMPTY
    if not g.is_valid:
        g = shapely.make_valid(g)
    parts, stack = [], [g]
    while stack:
        x = stack.pop()
        if isinstance(x, Polygon):
            if not x.is_empty:
                parts.append(x)
        elif x.geom_type in ("MultiPolygon", "GeometryCollection"):
            stack.extend(shapely.get_parts(x))
    out = MultiPolygon(parts)
    return out if out.is_valid else polys(shapely.make_valid(out))


def _union(geoms) -> MultiPolygon:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    return polys(shapely.union_all(geoms)) if geoms else EMPTY


def _overlaps(a_geoms: list, b_geoms: list):
    """Every (i, j) with a_geoms[i] intersecting b_geoms[j], and the area of the overlap (m2).
    a_geoms are split into parts first (see the module docstring)."""
    if not a_geoms or not b_geoms:
        return np.array([], int), np.array([], int), np.array([])
    parts, owner = shapely.get_parts(np.array(a_geoms, dtype=object), return_index=True)
    b = np.array(b_geoms, dtype=object)
    hit_part, hit_b = shapely.STRtree(b).query(parts, predicate="intersects")
    area = shapely.area(shapely.intersection(parts[hit_part], b[hit_b]))
    return owner[hit_part], hit_b, area


def _to_crs(geojsons: list[dict], src: str, dst: str) -> list:
    from rasterio.warp import transform_geom

    if not geojsons:
        return []
    return [shape(g) for g in transform_geom(src, dst, geojsons)]


class ShapelyBackend:
    """Same methods as postgis.PostgisBackend; everything is held in memory."""

    name = "shapely"

    # ------------------------------------------------------------------------ load
    def load(self, aoi_wkt, epsg, landuse, impacts, land=None, towns=()):
        """aoi_wkt and impacts are in the raster CRS ``epsg``; landuse, land, towns lon/lat."""
        self.crs = f"EPSG:{epsg}"
        self.aoi = polys(shapely.from_wkt(aoi_wkt))
        lonlat = lambda gs: _to_crs(gs, "EPSG:4326", self.crs)  # noqa: E731
        self.land = polys(lonlat([land])[0]) if land is not None else self.aoi
        self.landuse = []
        for f, g in zip(landuse, lonlat([f["geometry"] for f in landuse]), strict=True):
            g = polys(g)
            if not g.is_empty:
                self.landuse.append({**{k: f.get(k) for k in ("osm_id", "landuse", "crop", "name",
                                                              "lu_class")}, "geom": g})  # fmt: skip
        self.towns = []
        for t, g in zip(towns, lonlat([t["geometry"] for t in towns]), strict=True):
            self.towns.append({"osm_id": t["osm_id"], "name": t["name"], "geom": polys(g)})
        self.impacts = []  # (kind, mean_conf, geom)
        for kind, feats in impacts.items():
            for f in feats:
                g = polys(shape(f["geometry"]))
                if not g.is_empty:
                    self.impacts.append((kind, float(f["mean_conf"]), g))

    # ---------------------------------------------------------------------- rollup
    def rollup(self):
        self._land_cover()
        self._type_impact()
        self._plots()
        self._hotspots()
        self._towns()

    def _land_cover(self):
        """Steps 0 and 1: land part of the AOI, then one class per piece of ground."""
        self.land_aoi = polys(shapely.intersection(self.land, self.aoi))
        merged = {}
        for lu in PRIORITY:
            geoms = [l["geom"] for l in self.landuse if l["lu_class"] == lu]  # noqa: E741
            geoms = [shapely.intersection(g, self.land_aoi) for g in geoms
                     if shapely.intersects(g, self.land_aoi)]  # fmt: skip
            if geoms:
                merged[lu] = _union([polys(g) for g in geoms])
        self.cover, taken = {}, []
        for lu in PRIORITY:  # each class minus every higher-priority class
            if lu not in merged:
                continue
            higher = _union(taken)
            self.cover[lu] = polys(shapely.difference(merged[lu], higher))
            taken.append(merged[lu])
        self.cover["unmapped"] = polys(shapely.difference(self.land_aoi,
                                                          _union(self.cover.values())))  # fmt: skip
        self.cover["sea"] = polys(shapely.difference(self.aoi, self.land_aoi))
        self.cover = {k: g for k, g in self.cover.items() if not g.is_empty}
        self.landmask = _union(g for k, g in self.cover.items() if k not in ("water", "sea"))

    def _type_impact(self):
        """Step 2: hectares per (land-cover class, impact kind) and mean confidence."""
        classes = list(self.cover)
        geoms = [self.impacts[j][2] for j in range(len(self.impacts))]
        i, j, area = _overlaps([self.cover[c] for c in classes], geoms)
        m2, m2conf = defaultdict(float), defaultdict(float)
        for ci, ij, a in zip(i, j, area, strict=True):
            kind, conf, _ = self.impacts[ij]
            m2[(classes[ci], kind)] += a
            m2conf[(classes[ci], kind)] += a * conf
        self.type_impact = {k: (v / 1e4, m2conf[k] / v if v > 0 else None) for k, v in m2.items()}

    def _plots(self):
        """Step 3: per OSM polygon (clipped to land), impacted hectares and flood severity."""
        self.plots = []
        for l in self.landuse:  # noqa: E741
            if shapely.intersects(l["geom"], self.land_aoi):
                clipped = shapely.intersection(l["geom"], self.land_aoi)
                self.plots.append({**l, "geom": clipped, "area_ha": clipped.area / 1e4})
        i, j, area = _overlaps([p["geom"] for p in self.plots], [g for *_, g in self.impacts])
        m2, m2conf = defaultdict(float), defaultdict(float)
        for pi, ij, a in zip(i, j, area, strict=True):
            kind, conf, _ = self.impacts[ij]
            m2[(pi, kind)] += a
            m2conf[(pi, kind)] += a * conf
        for pi, p in enumerate(self.plots):
            for kind in KINDS:
                p[f"{kind}_ha"] = m2.get((pi, kind), 0.0) / 1e4
            fl = m2.get((pi, "flooded"), 0.0)
            p["flood_conf"] = m2conf[(pi, "flooded")] / fl if fl > 0 else 0.0
            p["flooded_frac"] = p["flooded_ha"] / p["area_ha"] if p["area_ha"] else None
            score = (p["flooded_frac"] or 0.0) * p["flood_conf"]
            p["severity_score"] = score
            p["severity"] = ("high" if score >= 0.30 else "medium" if score >= 0.10
                             else "low" if score > 0 else "none")  # fmt: skip

    def _hotspots(self, eps: float = 100.0, minpoints: int = 3):
        """Step 4: DBSCAN like PostGIS ST_ClusterDBSCAN on the flooded polygons on land.

        A polygon is a CORE if at least ``minpoints`` polygons (itself included) are within
        ``eps`` metres; cores within eps of each other share a cluster; a non-core polygon
        within eps of a core joins that core's cluster (a 'border'); the rest is noise.
        """
        flooded = [(conf, g) for kind, conf, g in self.impacts if kind == "flooded"]
        i, j, _ = _overlaps([g for _, g in flooded], [self.landmask]) if flooded else ([], [], [])
        on_land = []
        for fi in sorted(set(int(x) for x in i)):
            g = polys(shapely.intersection(flooded[fi][1], self.landmask))
            if not g.is_empty:
                on_land.append((flooded[fi][0], g))
        geoms = np.array([g for _, g in on_land], dtype=object)
        n = len(geoms)
        a, b = (shapely.STRtree(geoms).query(geoms, predicate="dwithin", distance=eps)
                if n else (np.array([], int), np.array([], int)))  # fmt: skip
        neighbours = [[] for _ in range(n)]
        for x, y in zip(a, b, strict=True):
            neighbours[x].append(int(y))  # includes itself
        core = [len(nb) >= minpoints for nb in neighbours]
        parent = list(range(n))

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for x in range(n):
            if core[x]:
                for y in neighbours[x]:
                    if core[y]:
                        parent[root(y)] = root(x)
        cluster = {}
        for x in range(n):
            if core[x]:
                cluster[x] = root(x)
            else:
                cores = sorted(y for y in neighbours[x] if core[y])
                if cores:
                    cluster[x] = root(cores[0])
        members = defaultdict(list)
        for x, c in cluster.items():
            members[c].append(x)

        self.hotspot_rows = []
        for cid, (_, xs) in enumerate(sorted(members.items(), key=lambda kv: min(kv[1]))):
            areas = np.array([geoms[x].area for x in xs])
            confs = np.array([on_land[x][0] for x in xs])
            geom = _union(geoms[x] for x in xs)
            overlap = {lu: shapely.intersection(g, geom).area for lu, g in self.cover.items()
                       if shapely.intersects(g, geom)}  # fmt: skip
            self.hotspot_rows.append({
                "hotspot_id": cid,
                "n_polygons": len(xs),
                "flooded_ha": areas.sum() / 1e4,
                "mean_conf": float((areas * confs).sum() / areas.sum()) if areas.sum() else None,
                "main_lu_class": max(overlap, key=overlap.get) if overlap else None,
                "rice_paddy_ha": overlap.get("rice_paddy", 0.0) / 1e4,
                "geom": geom,
            })  # fmt: skip
        self.hotspot_rows.sort(key=lambda h: -h["flooded_ha"])

    def _towns(self):
        """Step 5: each town's land in the AOI per class, and impacted hectares per class."""
        self.town_cover = []  # (town index, class, geom)
        for ti, t in enumerate(self.towns):
            for lu, g in self.cover.items():
                if lu != "sea" and shapely.intersects(t["geom"], g):
                    piece = polys(shapely.intersection(t["geom"], g))
                    if not piece.is_empty:
                        self.town_cover.append((ti, lu, piece))
        i, j, area = _overlaps([g for *_, g in self.town_cover], [g for *_, g in self.impacts])
        m2 = defaultdict(float)
        for ci, ij, a in zip(i, j, area, strict=True):
            ti, lu, _ = self.town_cover[ci]
            m2[(ti, lu, self.impacts[ij][0])] += a
        self.town_hits = [{"osm_id": self.towns[ti]["osm_id"], "lu_class": lu, "kind": kind,
                           "impacted_ha": v / 1e4} for (ti, lu, kind), v in m2.items()]  # fmt: skip

    # ------------------------------------------------------------------- outputs
    def tables(self) -> RollupTables:
        from rasterio.warp import transform

        severity = defaultdict(int)
        for p in self.plots:
            severity[p["severity"]] += 1
        cols = ("osm_id", "lu_class", "landuse", "name", "area_ha", "flooded_ha", "flooded_frac",
                "flood_conf", "severity_score", "severity")  # fmt: skip
        hit = sorted((p for p in self.plots if p["severity"] != "none"),
                     key=lambda p: -p["flooded_ha"])[:15]  # fmt: skip
        top = [{**{c: p[c] for c in cols}, "seasonal_flooding_flag": p["lu_class"] == "rice_paddy"}
               for p in hit]  # fmt: skip
        hotspots = []
        for h in self.hotspot_rows:
            c = h["geom"].centroid
            (lon,), (lat,) = transform(self.crs, "EPSG:4326", [c.x], [c.y])
            hotspots.append({**{k: v for k, v in h.items() if k != "geom"}, "lon": lon,
                             "lat": lat})  # fmt: skip
        towns = []
        for ti, t in enumerate(self.towns):
            land = sum(g.area for i, lu, g in self.town_cover if i == ti and lu != "water")
            towns.append({"osm_id": t["osm_id"], "name": t["name"],
                          "municipality_ha": t["geom"].area / 1e4,
                          "land_in_aoi_ha": land / 1e4})  # fmt: skip
        return RollupTables(
            cover={k: g.area / 1e4 for k, g in self.cover.items()},
            type_impact=self.type_impact,
            n_paddy=sum(l["lu_class"] == "rice_paddy" for l in self.landuse),  # noqa: E741
            severity=dict(severity),
            top_plots=top,
            hotspots=hotspots,
            towns=towns,
            town_hits=self.town_hits,
        )

    def build_summary(self, context: dict) -> dict:
        return assemble_summary(self.tables(), context)

    def rice_paddy_outlines(self) -> list[dict]:
        g = self.cover.get("rice_paddy")
        return [mapping(shapely.simplify(g, 5, preserve_topology=True))] if g else []

    def town_shapes(self) -> list[dict]:
        by_name = defaultdict(list)
        for ti, _, g in self.town_cover:
            by_name[self.towns[ti]["name"]].append(g)
        return [{"name": name, "geometry": mapping(shapely.simplify(_union(gs), 5,
                                                                    preserve_topology=True))}
                for name, gs in by_name.items()]  # fmt: skip

    def land_geometry(self, epsg: int) -> dict:
        if f"EPSG:{epsg}" == self.crs:
            return mapping(self.land_aoi)
        return mapping(_to_crs([mapping(self.land_aoi)], self.crs, f"EPSG:{epsg}")[0])

    def hotspot_shapes(self) -> list[dict]:
        return [{"id": h["hotspot_id"], "flooded_ha": h["flooded_ha"],
                 "geometry": mapping(shapely.simplify(h["geom"], 5, preserve_topology=True))}
                for h in self.hotspot_rows]  # fmt: skip

    def export_geojson(self, path: Path, table="plot_impact", impacted_only=False) -> Path:
        """Plots or hotspots as a GeoJSON FeatureCollection in lon/lat, 6 decimals."""
        if table == "plot_impact":
            cols = ("osm_id", "lu_class", "landuse", "crop", "name", "area_ha", "flooded_ha",
                    "changed_ha", "ndvi_drop_ha", "flooded_frac", "flood_conf", "severity_score",
                    "severity")  # fmt: skip
            rows = [p for p in self.plots if not impacted_only or p["severity"] != "none"
                    or p["changed_ha"] > 0 or p["ndvi_drop_ha"] > 0]  # fmt: skip
            flag = "seasonal_flooding_flag"
            props = [{**{c: p[c] for c in cols}, flag: p["lu_class"] == "rice_paddy"}
                     for p in rows]  # fmt: skip
        else:
            rows = self.hotspot_rows
            keys = ("hotspot_id", "n_polygons", "flooded_ha", "mean_conf", "main_lu_class",
                    "rice_paddy_ha")  # fmt: skip
            props = [{k: h[k] for k in keys} for h in rows]
        simple = [mapping(shapely.simplify(r["geom"], 2, preserve_topology=True)) for r in rows]
        lonlat = _to_crs(simple, self.crs, "EPSG:4326")
        feats = [{"type": "Feature",
                  "geometry": mapping(shapely.set_precision(g, 1e-6)) if not g.is_empty else None,
                  "properties": pr} for g, pr in zip(lonlat, props, strict=True)]  # fmt: skip
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        return path

    def close(self):
        pass
