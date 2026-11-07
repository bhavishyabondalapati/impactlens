# Geo-AI (Phase 4): from pixels to hectares per plot

```bash
docker compose up -d                                   # local PostGIS (password in .env)
uv run impactlens infer --quicklook docs/images/valencia_quicklook.png
uv run impactlens rollup --map docs/images/valencia_rollup.png --ndvi-map docs/images/valencia_ndvi_drop.png
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
   confidence. **OSM** via Overpass (`geo/osm.py`), cached in `data/raw/osm/`: land use,
   municipalities (admin_level 8) and the coastline. **Land mask**: the AOI is cut along the
   OSM `natural=coastline` ways (OSM draws them with land on the left); the sea side enters no
   total. Cross-check: 98.7% of that sea is MNDWI water on the before date.
5. **Spatial SQL** (`geo/sql/rollup.sql`, all in EPSG:32630 metres):
   - exclusive land cover per type (OSM polygons overlap; priority rice_paddy > water >
     residential > industrial > orchard > farmland > wetland > other; rest of the land =
     `unmapped`, mostly huerta fields OSM has not tagged; the sea is its own excluded class)
   - hectares per plot and per type for flooded / changed / NDVI drop
   - severity per plot = flooded fraction x mean confidence: high >= 0.30, medium >= 0.10, low > 0
   - hotspots: `ST_ClusterDBSCAN` (eps 100 m, minpoints 3) over flooded polygons on land
   - per municipality and class: hectares of each impact kind (`town_cover`, `town_impact`)
6. **Impacts split three ways** in `summary.json` -> `impacts`: `standing_water`
   (paddy-confounded), `vegetation_loss_likely_mud` (NDVI drop outside paddies, with a per-town
   rate = drop / vegetated land outside paddies), `urban_change` (OSCD change on residential +
   industrial land only). Sea numbers live under `excluded`, never in a total.

## Valencia results (before 2024-10-26, after 2024-11-10)

AOI: -0.457 to -0.283 E, 39.273 to 39.440 N (the north edge was moved up from 39.407 to take in
Paiporta and the other l'Horta Sud towns; re-fetching picked the same two scenes). 27,819 ha:
20,757 ha land, 2,785 ha inland water (Albufera lake, canals, the Turia channel), 4,277 ha sea.

| impact (on land) | hectares | what it means |
|---|---|---|
| Standing water (new water) | 2,104 | 98% of it (2,059 ha) is in rice paddies: confounded, see below |
| ...outside rice paddies | 45 | |
| Vegetation loss / likely mud (NDVI drop outside paddies) | 643 | candidate damage footprint |
| NDVI drop inside paddies (not counted above) | 542 | harvest / paddy flooding, ambiguous |
| Change on urban land (OSCD model) | 222 | 163 ha industrial, 59 ha residential |
| *Excluded: sea "flooded"* | *688* | *model domain gap on the before date; never in a total* |

**Day 12 is too late for water.** The DANA hit on Oct 29. By Nov 10 the standing water in
l'Horta Sud had drained and left mud, so a water detector correctly finds almost nothing in the
towns (under 1.5 ha of new water outside the paddies in each of Paiporta, Sedaví, Alfafar and
Massanassa). The
remaining new water is in the paddies.

**The damage shows up as vegetation loss along the Barranco del Poyo.** NDVI drop as a share of
each town's vegetated land outside the paddies (towns with >= 50 ha vegetated in the AOI):

| town | vegetated ha (outside paddies) | NDVI drop ha | rate |
|---|---|---|---|
| Paiporta | 167 | 69 | **42%** |
| Sedaví | 55 | 22 | **41%** |
| Alfafar | 157 | 58 | **37%** |
| Massanassa | 122 | 37 | **31%** |
| Beniparrell | 188 | 30 | **16%** |
| Catarroja | 510 | 68 | **13%** |
| Albal | 344 | 36 | **10%** |
| València (southern districts) | 1,722 | 164 | 9.5% |
| Picanya (62% of it in the AOI) | 310 | 29 | 9.3% |
| Alcàsser | 490 | 27 | 5.5% |
| Almussafes | 395 | 12 | 3.0% |
| Silla | 934 | 23 | 2.5% |
| Benifaió | 675 | 17 | 2.5% |
| Sollana | 528 | 12 | 2.2% |
| Torrent (8% of it in the AOI) | 392 | 8 | 2.0% |
| Picassent | 1,177 | 22 | 1.9% |

Benetússer (11 ha vegetated) and Sueca (35 ha in the AOI) are too small to rank.

Paiporta, where the Poyo overflowed, has the highest rate, followed by the towns downstream of it
on the way to the lake (Sedaví, Alfafar, Massanassa) at 31-41%. Similar huerta towns to the south
and west (Silla, Sollana, Benifaió, Almussafes, Picassent) sit at 2-3%, ten to twenty times
lower. On the map the orange is concentrated around Paiporta and in the strip between the towns
and the lake. The OSCD change on urban land follows the same towns: Paiporta 34 ha, Alfafar
33 ha, Catarroja 29 ha, Massanassa 22 ha, against 1-2 ha in Silla, Picassent and Almussafes.

This is consistent with mud deposits and flattened vegetation, but NDVI alone can't prove it:
harvest, ploughing and pruning also drop NDVI, and mud on streets and roofs is invisible here
because that ground was not vegetated before. Most of the drop (422 of 643 ha) is on land with no
OSM land-use tag, so the per-class split is weaker than the per-town one. Urban change is not
damage by itself either: the Port of València (134 ha of industrial land in the AOI) adds about
21 ha of "change" from ships and containers moving.

![NDVI drop by town](images/valencia_ndvi_drop.png)

**AOI coverage.** Paiporta, Sedaví, Alfafar, Massanassa, Benetússer, Catarroja and Albal are now
96-100% inside the AOI. Picanya is 62% inside and Torrent 8%; both reach further west and north.
Aldaia, Alaquàs, Xirivella and the rest of València's city districts are outside.

**Known confound: rice paddies.** Albufera paddies are flooded on purpose in late autumn and
winter. 39.0% of paddy area was already water on Oct 26 (three days BEFORE the Oct 29 DANA) and
68.4% on Nov 10. The DANA floodwater also drained into the Albufera through l'Horta Sud, so the
extra paddy water can be either cause, and two optical dates cannot tell them apart. That is why
the paddies are a flagged confound rather than counted as damage or as farming.
The paddies are reported separately (`rice_paddies` in the summary, `seasonal_flooding_flag` per
plot), never dropped. The OSCD change model also fires on the paddies (2,025 ha of its change), as
expected for a model trained on urban change: flooding a field is a big visual change.

**Sanity check (model vs water indices on Valencia, IoU of the water masks):**

| date | NDWI > 0 | MNDWI > 0 | MNDWI > 0.15 | MNDWI > 0, land only | MNDWI > 0, sea only |
|---|---|---|---|---|---|
| before | 0.79 | 0.84 | 0.85 | 0.86 | 0.82 |
| after | 0.86 | 0.91 | 0.91 | 0.94 | 0.87 |

On land the agreement is high, so the L1C-train / L2A-inference gap does not break the model
there. It shows up at sea: the model under-calls dark open water (part of the **open sea on
Oct 26** gets P(water) ~0.37 while MNDWI is 0.74, clearly water), and the extended AOI adds more
sea and the Port of València. That sea strip then looks "flooded" on Nov 10 (688 ha).
The OSM coastline mask puts it in the `sea` class, so it enters no total and no hotspot; it is
kept under `excluded` in the summary as a diagnostic.

![rollup](images/valencia_rollup.png)

Land use (c) OpenStreetMap contributors (ODbL). Imagery: Copernicus Sentinel-2 via Earth Search.
