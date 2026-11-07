"""The agent graph (LangGraph). Six agents, each a node that reads and updates one shared state:

    Ingest -> Detect -> GeoAnalyst -> Retriever -> Reporter -> Validator --ok--> finish
                                                      ^            |
                                                      +-- errors --+  (max 2 retries)
                                                                   +--still failing--> human review

- Ingest:     check the before/after scene (or fetch it) and read its metadata.
- Detect:     ONNX inference -> GeoTIFF layers (skipped if the layers already exist).
- GeoAnalyst: PostGIS rollup -> summary.json, then the compact fact sheet.
- Retriever:  FAISS search over the SOP corpus; one shared set of excerpts S1..Sn.
- Reporter:   LLM writes three stakeholder reports (only failing ones are rewritten).
- Validator:  deterministic checks (numbers, caveats, citations) -> errors per report.

The first three are tools wrapped as agents (no LLM); only the Reporter calls the LLM.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from impactlens.agents import steps as agent_steps
from impactlens.agents.facts import build_facts
from impactlens.agents.steps import MAX_RETRIES, RETRIEVAL_QUERIES, save_run  # noqa: F401


class State(TypedDict, total=False):
    scene_dir: str
    aoi: str
    out_dir: str
    run_steps: bool  # run Detect/GeoAnalyst even if outputs exist
    scene: dict
    summary: dict
    facts: dict
    chunks: list[dict]
    reports: dict[str, str]
    errors: dict[str, list[str]]  # latest validation result per report
    written: list[str]  # reports (re)written by the latest Reporter attempt
    history: Annotated[list[dict], operator.add]  # every attempt: what was written, its errors
    attempts: int
    status: str  # running | validated | needs_human_review | budget_exceeded
    log: Annotated[list[str], operator.add]


def build_graph(llm, retriever=None, steps=None):
    """``steps`` lets tests replace Ingest/Detect/GeoAnalyst with fakes (no rasters, no DB)."""
    steps = steps or {}

    def ingest(state: State) -> dict:
        if "ingest" in steps:
            return steps["ingest"](state)
        scene = Path(state["scene_dir"])
        needed = ("before.tif", "after.tif", "scenes.json")
        missing = [f for f in needed if not (scene / f).exists()]
        if missing:
            raise FileNotFoundError(f"{scene} lacks {missing}: run `impactlens fetch` first")
        meta = json.loads((scene / "scenes.json").read_text())
        return {"scene": meta, "log": [f"ingest: {meta['before']['scene_id']} -> "
                                       f"{meta['after']['scene_id']}"]}  # fmt: skip

    def detect(state: State) -> dict:
        if "detect" in steps:
            return steps["detect"](state)
        from impactlens.geo.pipeline import latest_model, run_infer

        scene = Path(state["scene_dir"])
        layers = scene / "layers"
        if (layers / "infer_summary.json").exists() and not state.get("run_steps"):
            return {"log": ["detect: reusing existing layers"]}
        run_infer(scene, latest_model("build/models_*/floods_r2/floods.onnx"),
                  Path("build/models/oscd/oscd.onnx"), layers, None)  # fmt: skip
        return {"log": ["detect: ran ONNX inference"]}

    def geo_analyst(state: State) -> dict:
        if "geo_analyst" in steps:
            return steps["geo_analyst"](state)
        from impactlens.data.stac import load_aoi
        from impactlens.geo.pipeline import run_rollup

        scene = Path(state["scene_dir"])
        path = scene / "geo" / "summary.json"
        if not path.exists() or state.get("run_steps"):
            run_rollup(scene, scene / "layers", Path(state["aoi"]), scene / "geo",
                       Path("data/raw/osm"))  # fmt: skip
        summary = json.loads(path.read_text())
        bounds = [round(b, 3) for b in load_aoi(state["aoi"]).bounds]
        return {"summary": summary, "facts": build_facts(summary, bounds),
                "log": ["geo_analyst: summary.json -> fact sheet"]}  # fmt: skip

    def retrieve(state: State) -> dict:
        chunks = agent_steps.retrieve_chunks(retriever)
        return {"chunks": chunks, "log": [f"retriever: {len(chunks)} SOP excerpts"]}

    def write(state: State) -> dict:
        return agent_steps.write_reports(llm, state)

    def check(state: State) -> dict:
        return agent_steps.check_reports(state)

    def route_after_reporter(state: State) -> str:
        return "stop" if state["status"] == "budget_exceeded" else "validate"

    def route_after_validator(state: State) -> str:
        if state["status"] in ("validated", "needs_human_review"):
            return "done"
        return "retry"

    g = StateGraph(State)
    for name, fn in [
        ("ingest", ingest),
        ("detect", detect),
        ("geo_analyst", geo_analyst),
        ("retriever", retrieve),
        ("reporter", write),
        ("validator", check),
    ]:
        g.add_node(name, fn)  # fmt: skip
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "detect")
    g.add_edge("detect", "geo_analyst")
    g.add_edge("geo_analyst", "retriever")
    g.add_edge("retriever", "reporter")
    g.add_conditional_edges("reporter", route_after_reporter,
                            {"validate": "validator", "stop": END})  # fmt: skip
    g.add_conditional_edges("validator", route_after_validator,
                            {"retry": "reporter", "done": END})  # fmt: skip
    return g.compile()
