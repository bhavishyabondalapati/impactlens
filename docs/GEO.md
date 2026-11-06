# Geo-AI (Phase 4): from pixels to hectares per plot

```bash
docker compose up -d                                   # local PostGIS (password in .env)
uv run impactlens infer --quicklook docs/images/valencia_quicklook.png
uv run impactlens rollup --map docs/images/valencia_rollup.png
```

Outputs (gitignored, under `data/processed/valencia_albufera/`):
- `layers/*.tif`: one GeoTIFF per layer on the scene's 10 m UTM 30N grid: `water_before`,
  `water_after`, `flood_prob`, `flooded`, `permanent_water`, `change_prob`, `changed`,
  `ndvi_before`, `ndvi_after`, `ndvi_drop`, `ndvi_dropped`, `valid`; plus `infer_summary.json`.
- `geo/summary.json`: the numbers the report agents must stay consistent with (Phase 5).
- `geo/impact_plots.geojson` (OSM plots with hectares + severity) and `geo/hotspots.geojson`.

## How it works

1. **Sliding-window inference** (`geo/inference.py`): 256 px windows, 64 px overlap, Hann-window
   blending (no seams), through the exported ONNX models. Flood U-Net (6 bands) on each date;
   Siamese U-Net (4 bands) on the pair. Water threshold 0.45 = the value tuned on Sen1Floods11 val.
2. **Flooded = water after AND NOT water before.** The Albufera lake is water on both dates, so it
   is `permanent_water`, not flooded. Confidence = P(water after) x P(not water before).
3. **NDVI drop**: NDVI before - after > 0.2 where NDVI before > 0.3 (it was vegetated).
4. **Polygonize** (`geo/polygonize.py`): connected regions (8-neighbour), >= 4 px, with mean
   confidence. **OSM land use** via Overpass (`geo/osm.py`), cached in `data/raw/osm/`.
5. **Spatial SQL** (`geo/sql/rollup.sql`, all in EPSG:32630 metres):
   - exclusive land cover per type (OSM polygons overlap; priority rice_paddy > water >
     residential > industrial > orchard > farmland > wetland > other; rest = `unmapped`, incl. sea)
   - hectares per plot and per type for flooded / changed / NDVI drop
   - severity per plot = flooded fraction x mean confidence: high >= 0.30, medium >= 0.10, low > 0
   - hotspots: `ST_ClusterDBSCAN` (eps 100 m, minpoints 3) over flooded polygons on land

## Valencia results (before 2024-10-26, after 2024-11-10)

| | hectares |
|---|---|
| Flooded on land | 2,048 |
| of which in OSM rice paddies | 2,024 (98.9%) |
| Flooded on land outside rice paddies | 23 |
| Flooded in `unmapped` (mostly open sea, see below) | 736 |
| Changed (OSCD model) on land | 2,024 |
| NDVI drop on land | 633 |

**Known confound: rice paddies.** Albufera paddies are flooded on purpose in late autumn and
winter. 40.8% of paddy area was already water on Oct 26 and 69.8% on Nov 10. So most "new water"
here is probably farming, not the DANA flood (which hit mostly further north and west, on Oct 29).
The paddies are reported separately (`rice_paddies` in the summary, `seasonal_flooding_flag` per
plot), never dropped. The OSCD change model also fires on the paddies (1,941 ha of its change), as
expected for a model trained on urban change: flooding a field is a big visual change.

**Sanity check (model vs water indices on Valencia, IoU of the water masks):**

| date | NDWI > 0 | MNDWI > 0 | MNDWI > 0.15 |
|---|---|---|---|
| before | 0.78 | 0.84 | 0.85 |
| after | 0.90 | 0.95 | 0.95 |

Agreement is high, so the L1C-train / L2A-inference gap does not break the model overall. It does
show up in one place: part of the **open sea on Oct 26** gets P(water) ~0.37 from the model while
MNDWI there is 0.74 (clearly water). That sea strip then looks "flooded" on Nov 10. It falls in
`unmapped` (no OSM land use), is reported apart from the land totals, and is excluded from the
hotspot search, which only runs on land.

![rollup](images/valencia_rollup.png)

Land use (c) OpenStreetMap contributors (ODbL). Imagery: Copernicus Sentinel-2 via Earth Search.
