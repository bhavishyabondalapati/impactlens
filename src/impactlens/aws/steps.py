"""The pipeline steps as they run on AWS: one Lambda each, chained by Step Functions.

    requests/<run_id>.json -> SQS -> start -> Step Functions:
    Fetch -> Detect -> GeoAnalyze -> Retrieve -> Report -> Validate --valid--> Publish
                                                   ^          |
                                                   +--retry---+ (max 2) --> FlagForReview

Each step is a plain function ``step(store, event, work) -> event``: ``store`` is the bucket
(or a local folder in tests), ``event`` is the small JSON Step Functions passes along
({"run_id": ...} plus a few flags), ``work`` is an empty scratch folder (/tmp on Lambda).
Everything big (images, layers, reports) goes through the store, never through the event:
Step Functions passes at most 256 KB between states.

The science is not duplicated here: these functions call the same code as the local CLI
(stac.fetch_pair, pipeline.run_infer, pipeline.run_rollup with the Shapely backend, and the
agent steps in agents/steps.py). Only the plumbing (download, upload, state in S3) is new.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # also a valid Step Functions execution name
MAX_AOI_KM2 = 2500.0  # 50 x 50 km: bounds Lambda time and Overpass load per request
MODELS = {"floods": ("floods.onnx", "metrics.json"), "oscd": ("oscd.onnx", "metrics.json")}
SCENE_FILES = ("before.tif", "after.tif", "scenes.json")
INDEX_FILES = ("sops.faiss", "chunks.jsonl")
EMBEDDER_FILES = ("minilm.onnx", "tokenizer.json")


def _keys(run_id: str) -> dict:
    if not RUN_ID.match(run_id):
        raise ValueError(f"bad run_id {run_id!r}: use letters, digits, '-' and '_' (max 64)")
    return {
        "request": f"requests/{run_id}.json",
        "layers": f"layers/{run_id}",
        "work": f"reports/{run_id}/work",
        "reports": f"reports/{run_id}",
    }


def read_request(store, run_id: str) -> dict:
    """The request JSON: {"aoi": GeoJSON, "before": "YYYY-MM-DD/YYYY-MM-DD", "after": ...,
    optional "max_cloud" (%, default 20) and "event": {"name", "event_date"}}."""
    req = store.read_json(_keys(run_id)["request"])
    for field in ("aoi", "before", "after"):
        if field not in req:
            raise ValueError(f"request {run_id} has no {field!r}")
    for field in ("before", "after"):
        if not re.match(r"^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}$", req[field]):
            raise ValueError(f"{field} must look like 2024-10-01/2024-10-28, got {req[field]!r}")
    return req


def aoi_key(aoi) -> str:
    """A stable name per AOI (from its bounds), so its OSM answers are cached once per AOI."""
    b = ",".join(f"{v:.4f}" for v in aoi.bounds)
    return "aoi_" + hashlib.sha1(b.encode()).hexdigest()[:10]


def _aoi_km2(aoi) -> float:
    from rasterio.warp import transform_geom
    from shapely.geometry import mapping, shape

    lon = (aoi.bounds[0] + aoi.bounds[2]) / 2
    lat = (aoi.bounds[1] + aoi.bounds[3]) / 2
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return shape(transform_geom("EPSG:4326", f"EPSG:{epsg}", mapping(aoi))).area / 1e6


# ------------------------------------------------------------------------------- steps


def fetch(store, event: dict, work: Path, fetch_pair=None) -> dict:
    """Fetch: the clearest Sentinel-2 scene in each date window (Earth Search STAC)."""
    from impactlens.data.stac import fetch_pair as stac_fetch_pair
    from impactlens.data.stac import load_aoi

    k = _keys(event["run_id"])
    req = read_request(store, event["run_id"])
    aoi_path = work / "aoi.geojson"
    aoi_path.write_text(json.dumps(req["aoi"]))
    km2 = _aoi_km2(load_aoi(aoi_path))
    if km2 > MAX_AOI_KM2:
        raise ValueError(f"AOI is {km2:.0f} km2; the limit is {MAX_AOI_KM2:.0f} km2")
    (fetch_pair or stac_fetch_pair)(aoi_path, req["before"], req["after"], work / "scene",
                                    req.get("max_cloud", 20.0))  # fmt: skip
    for f in SCENE_FILES:
        store.upload(work / "scene" / f, f"{k['layers']}/scene/{f}")
    store.upload(aoi_path, f"{k['layers']}/aoi.geojson")
    return {"run_id": event["run_id"]}


def detect(store, event: dict, work: Path) -> dict:
    """Detect: both ONNX models over the scene (sliding window) -> GeoTIFF layers."""
    from impactlens.geo.pipeline import run_infer

    k = _keys(event["run_id"])
    for f in SCENE_FILES:
        store.download(f"{k['layers']}/scene/{f}", work / "scene" / f)
    for name, files in MODELS.items():  # run_infer reads metrics.json next to the model
        for f in files:
            store.download(f"models/{name}/{f}", work / "models" / name / f)
    out = work / "layers"
    run_infer(work / "scene", work / "models/floods/floods.onnx",
              work / "models/oscd/oscd.onnx", out, None)  # fmt: skip
    for f in sorted(out.iterdir()):
        store.upload(f, f"{k['layers']}/layers/{f.name}")
    return {"run_id": event["run_id"]}


def geo_analyze(store, event: dict, work: Path) -> dict:
    """GeoAnalyze: OSM land use + Shapely rollup -> summary.json, GeoJSON, fact sheet.
    OSM answers are cached in the bucket under osm/, so Overpass is asked once per AOI."""
    from impactlens.agents.facts import build_facts
    from impactlens.data.stac import load_aoi
    from impactlens.geo.pipeline import IMPACT_LAYERS, run_rollup

    k = _keys(event["run_id"])
    req = read_request(store, event["run_id"])
    scene, layers = work / "scene", work / "layers"
    for f in ("before.tif", "scenes.json"):
        store.download(f"{k['layers']}/scene/{f}", scene / f)
    needed = {n for pair in IMPACT_LAYERS.values() for n in pair}
    needed |= {"ndvi_before", "valid", "water_before", "water_after"}
    for name in sorted(needed):
        store.download(f"{k['layers']}/layers/{name}.tif", layers / f"{name}.tif")
    store.download(f"{k['layers']}/layers/infer_summary.json", layers / "infer_summary.json")

    aoi = load_aoi(store.download(f"{k['layers']}/aoi.geojson", work / "aoi.geojson"))
    aoi_path = work / "aoi" / f"{aoi_key(aoi)}.geojson"
    aoi_path.parent.mkdir(parents=True, exist_ok=True)
    aoi_path.write_text((work / "aoi.geojson").read_text())
    osm_dir = work / "osm"
    names = [f"{aoi_path.stem}_{w}.json" for w in ("landuse", "coastline", "municipalities")]
    cached = {n for n in names if store.exists(f"osm/{n}")}
    for n in cached:
        store.download(f"osm/{n}", osm_dir / n)

    out = work / "geo"
    summary = run_rollup(scene, layers, aoi_path, out, osm_dir, backend="shapely")
    for n in names:  # first request for this AOI: keep Overpass's answers for next time
        if n not in cached and (osm_dir / n).exists():
            store.upload(osm_dir / n, f"osm/{n}")
    for f in sorted(out.iterdir()):
        store.upload(f, f"{k['layers']}/geo/{f.name}")
    bounds = [round(b, 3) for b in aoi.bounds]
    store.write_json(f"{k['layers']}/facts.json", build_facts(summary, bounds, req.get("event")))
    return {"run_id": event["run_id"], "osm_cache_hit": len(cached) == len(names)}


def retrieve(store, event: dict, work: Path, embed=None) -> dict:
    """Retrieve: the SOP excerpts S1..Sn from the FAISS index, embedded with ONNX MiniLM."""
    from impactlens.agents.embed_onnx import OnnxEmbedder
    from impactlens.agents.rag import Retriever
    from impactlens.agents.steps import retrieve_chunks

    k = _keys(event["run_id"])
    for f in INDEX_FILES:
        store.download(f"index/{f}", work / "index" / f)
    if embed is None:
        for f in EMBEDDER_FILES:
            store.download(f"index/embedder/{f}", work / "embedder" / f)
        embed = OnnxEmbedder(work / "embedder")
    chunks = retrieve_chunks(Retriever(work / "index", embed=embed))
    store.write_json(f"{k['work']}/chunks.json", chunks)
    state = {"attempts": 0, "reports": {}, "errors": {}, "history": [], "status": "running",
             "log": [f"retriever: {len(chunks)} SOP excerpts"]}  # fmt: skip
    store.write_json(f"{k['work']}/state.json", state)
    store.write_text(f"{k['work']}/tokens.jsonl", "")
    return {"run_id": event["run_id"]}


def _load(store, run_id: str) -> tuple[dict, dict]:
    k = _keys(run_id)
    state = store.read_json(f"{k['work']}/state.json")
    state["facts"] = store.read_json(f"{k['layers']}/facts.json")
    state["chunks"] = store.read_json(f"{k['work']}/chunks.json")
    return state, k


def _save(store, k: dict, state: dict) -> None:
    keep = {key: v for key, v in state.items() if key not in ("facts", "chunks")}
    store.write_json(f"{k['work']}/state.json", keep)


def _tracker(store, k: dict):
    from impactlens.agents.llm import CostTracker

    lines = store.read_text(f"{k['work']}/tokens.jsonl").splitlines()
    calls = [json.loads(line) for line in lines if line.strip()]
    return CostTracker.resume(os.environ.get("MODEL", "claude-haiku-4-5"),
                              float(os.environ.get("BUDGET_USD", "0.10")), calls)  # fmt: skip


def report(store, event: dict, work: Path, make_llm=None) -> dict:
    """Report: Claude writes the reports that are missing or failed validation. The run's
    spend so far is read back from tokens.jsonl, so the $0.10 cap holds across Lambdas."""
    from impactlens.agents.llm import AnthropicLLM
    from impactlens.agents.steps import write_reports

    state, k = _load(store, event["run_id"])
    tracker = _tracker(store, k)
    llm = (make_llm or AnthropicLLM)(tracker)
    update = write_reports(llm, state)
    state["log"] = state.get("log", []) + update.pop("log")
    state.update(update)
    store.write_text(f"{k['work']}/tokens.jsonl",
                     "".join(json.dumps(c) + "\n" for c in tracker.calls))  # fmt: skip
    _save(store, k, state)
    return {"run_id": event["run_id"], "attempt": state["attempts"], "status": state["status"]}


def validate(store, event: dict, work: Path) -> dict:
    """Validate: the deterministic checks. ``valid`` and ``attempt`` drive the Choice state."""
    from impactlens.agents.steps import check_reports

    state, k = _load(store, event["run_id"])
    update = check_reports(state)
    state["log"] = state.get("log", []) + update.pop("log")
    state["history"] = state.get("history", []) + update.pop("history")
    state.update(update)
    _save(store, k, state)
    return {"run_id": event["run_id"], "attempt": state["attempts"],
            "valid": state["status"] == "validated", "status": state["status"]}  # fmt: skip


def finish(store, event: dict, work: Path) -> dict:
    """Publish or FlagForReview: write the reports and run.json to reports/<run_id>/."""
    from impactlens.agents.steps import save_run

    state, k = _load(store, event["run_id"])
    flagged = event.get("outcome") == "flagged"
    if flagged and state["status"] not in ("needs_human_review", "budget_exceeded"):
        state["status"] = "needs_human_review"  # the state machine has the last word
    tracker = _tracker(store, k)
    out = save_run(state, tracker, work / "out")
    for f in sorted(out.iterdir()):
        store.upload(f, f"{k['reports']}/{f.name}")
    return {"run_id": event["run_id"], "status": state["status"],
            "cost_usd": tracker.totals()["cost_usd"]}  # fmt: skip


STEPS = {"fetch": fetch, "detect": detect, "geo_analyze": geo_analyze, "retrieve": retrieve,
         "report": report, "validate": validate, "finish": finish}  # fmt: skip
