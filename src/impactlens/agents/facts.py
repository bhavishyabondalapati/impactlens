"""The fact sheet: a compact view of summary.json that the Reporter writes from.

summary.json is large (every plot, every hotspot). The reports only need the headline numbers,
the three impact layers, the per-town table, the caveat inputs and the event dates, so we copy
those VALUES out unchanged (no new numbers are computed here, except days after the event).
The Validator checks every number in a report against this sheet.
"""

from __future__ import annotations

from datetime import date

EVENT = {
    "name": "DANA (cut-off low) floods, Valencia province, Spain",
    "event_date": "2024-10-29",
}

TOWN_KEYS = ("name", "land_in_aoi_ha", "pct_of_municipality_in_aoi",
             "vegetated_ha_outside_rice_paddies", "ndvi_drop_ha_outside_rice_paddies",
             "ndvi_drop_pct_of_vegetated", "flooded_ha_outside_rice_paddies",
             "flooded_ha_in_rice_paddies", "urban_changed_ha")  # fmt: skip


def build_facts(summary: dict, aoi_bounds: list[float] | None = None) -> dict:
    before = summary.get("before_scene", "")
    after = summary.get("after_scene", "")
    before_date, after_date = _scene_date(before), _scene_date(after)
    days_after = None
    if after_date:
        days_after = (date.fromisoformat(after_date) - date.fromisoformat(EVENT["event_date"])).days
    imp = summary["impacts"]
    no_notes = lambda d: {k: v for k, v in d.items() if k not in ("note", "label")}  # noqa: E731
    urban = no_notes(imp["urban_change"])
    urban["port_excluded"] = no_notes(urban["port_excluded"])
    rice = no_notes(summary["rice_paddies"])
    return {
        "event": {
            **EVENT,
            "before_image_date": before_date,
            "after_image_date": after_date,
            "after_image_days_after_event": days_after,
            "before_scene": before,
            "after_scene": after,
        },  # fmt: skip
        "aoi": {
            "bounds_lon_lat": aoi_bounds,
            "area_ha": summary["totals"]["aoi_ha"],
            "land_ha": summary["totals"]["land_ha"],
            "sea_ha_excluded": summary["excluded"]["sea_ha"],
        },  # fmt: skip
        "crs": summary.get("crs"),
        "definitions": summary["definitions"],
        "totals": summary["totals"],
        "standing_water": no_notes(imp["standing_water"]),
        "vegetation_loss_likely_mud": no_notes(imp["vegetation_loss_likely_mud"]),
        "urban_change": urban,
        "rice_paddies": rice,
        "by_town": [
            {k: t.get(k) for k in TOWN_KEYS}
            for t in summary["by_town"]
            if t.get("vegetated_ha_outside_rice_paddies", 0) >= 50
            or t.get("urban_changed_ha", 0) >= 5
        ],  # fmt: skip
        "excluded_sea": no_notes(summary["excluded"]),
        "thresholds": summary.get("thresholds"),
        "n_hotspots": summary.get("n_hotspots"),
        "largest_hotspot": {
            k: summary["hotspots"][0].get(k)
            for k in ("flooded_ha", "main_lu_class", "rice_paddy_share")
        }
        if summary.get("hotspots")
        else None,  # fmt: skip
        "attribution": summary.get("attribution"),
    }


def _scene_date(scene_id: str) -> str | None:
    """S2A_30SYJ_20241026_0_L2A -> 2024-10-26."""
    for part in scene_id.split("_"):
        if len(part) == 8 and part.isdigit():
            return f"{part[:4]}-{part[4:6]}-{part[6:]}"
    return None
