"""summary.json from the rollup tables, shared by both rollup backends.

The spatial work (land cover, overlaps, hotspots, towns) runs either in PostGIS (local:
``geo/postgis.py`` + ``sql/rollup.sql``) or in plain Shapely (AWS Lambda:
``geo/rollup_shapely.py``).
Both produce the same small ``RollupTables``; this module turns them into summary.json, so the
two backends cannot drift apart in how numbers are added up, rounded or labelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

KINDS = ("flooded", "changed", "ndvi_drop")
URBAN = ("residential", "industrial")

PADDY_NOTE = (
    "Rice paddies around the Albufera are traditionally flooded on purpose in late autumn and "
    "winter, after the harvest. The Oct 26 -> Nov 10 window overlaps that, so new water in "
    "paddies may be farming practice, not flood damage. Reported separately and flagged; not "
    "dropped and not merged into the farmland totals. OSM only tags some paddies with "
    "crop=rice, so untagged paddies count as 'farmland' (the confound may be larger)."
)


@dataclass
class RollupTables:
    """What the spatial step produces (hectares unless noted)."""

    cover: dict[str, float]  # land-cover class -> area_ha (exclusive classes incl. sea)
    type_impact: dict[tuple[str, str], tuple[float, float | None]]  # (class, kind) -> ha, conf
    n_paddy: int  # OSM rice-paddy polygons
    severity: dict[str, int]  # severity level -> number of plots
    top_plots: list[dict]  # 15 plots with severity != none, most flooded first
    hotspots: list[dict]  # all hotspots, most flooded first (lon/lat of the centroid)
    towns: list[dict] = field(default_factory=list)  # osm_id, name, municipality_ha, land_in_aoi_ha
    town_hits: list[dict] = field(default_factory=list)  # osm_id, lu_class, kind, impacted_ha


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def assemble_summary(t: RollupTables, context: dict) -> dict:
    """The metrics summary the report agents (Phase 5) must stay consistent with."""
    by_type = {}
    for lu, area in sorted(t.cover.items(), key=lambda kv: -kv[1]):
        row = {"area_ha": _r(area)}
        for kind in KINDS:
            ha, conf = t.type_impact.get((lu, kind), (0.0, None))
            row[f"{kind}_ha"] = _r(ha)
            row[f"{kind}_pct"] = _r(100 * ha / area, 1) if area else 0.0
            if kind == "flooded":
                row["flood_mean_conf"] = _r(conf, 3) if (lu, kind) in t.type_impact else None
        by_type[lu] = row

    sea = by_type.pop("sea", None)  # never part of any total; reported under "excluded"
    land = [k for k in by_type if k != "water"]
    flooded_land = sum(by_type[k]["flooded_ha"] for k in land)
    zero = {"area_ha": 0.0, "flooded_ha": 0.0, "flooded_pct": 0.0, "changed_ha": 0.0,
            "ndvi_drop_ha": 0.0}  # fmt: skip
    paddy = by_type.get("rice_paddy", zero)

    hotspots = [dict(h) for h in t.hotspots]
    for h in hotspots:
        h["rice_paddy_share"] = (
            _r(h["rice_paddy_ha"] / h["flooded_ha"], 3) if h["flooded_ha"] else 0
        )
    num = lambda d: {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
    towns = town_rows(t.towns, t.town_hits)
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
            "aoi_ha": _r(sum(t.cover.values())),
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
            "osm_polygons": t.n_paddy,
            "area_ha": paddy["area_ha"],
            "flooded_ha": paddy["flooded_ha"],
            "flooded_pct": paddy["flooded_pct"],
            "share_of_flooded_land_pct": _r(100 * paddy["flooded_ha"] / flooded_land, 1)
            if flooded_land
            else 0.0,
        },
        "severity_counts": t.severity,
        "top_plots": [num(p) for p in t.top_plots],
        "hotspots": [num(h) for h in hotspots[:10]],
        "n_hotspots": len(hotspots),
        "attribution": "Land use (c) OpenStreetMap contributors, ODbL. Imagery: Copernicus "
        "Sentinel-2 via Earth Search.",  # fmt: skip
    }


def town_rows(towns: list[dict], hits: list[dict]) -> list[dict]:
    """Per municipality: land inside the AOI, and its impacts split the way the summary is."""
    out = []
    for t in towns:
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
