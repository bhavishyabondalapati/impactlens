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
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from impactlens.agents import reporter
from impactlens.agents.facts import build_facts
from impactlens.agents.llm import BudgetExceeded
from impactlens.agents.validator import validate

MAX_RETRIES = 2
RETRIEVAL_QUERIES = (
    "mapping flood water extent from Sentinel-2 with a water index such as NDWI and a threshold",
    "limitations of optical flood mapping: clouds, urban areas, shadows, timing of the image",
    "damage assessment methods from satellite imagery and how damage grades are assigned",
    "quality control and accuracy of rapid mapping products",
    "estimating agricultural damage and loss for crops after a disaster",
    "post-disaster needs assessment: baseline, damage, losses and recovery needs",
)


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
        seen, chunks = set(), []
        for q in RETRIEVAL_QUERIES:
            for _score, c in retriever.search(q, k=2):
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    chunks.append({"chunk_id": c.chunk_id, "source": c.source,
                                   "title": c.title, "page": c.page, "text": c.text})  # fmt: skip
        return {"chunks": chunks, "log": [f"retriever: {len(chunks)} SOP excerpts"]}

    def write(state: State) -> dict:
        reports = dict(state.get("reports") or {})
        errors = state.get("errors") or {}
        shared = reporter.shared_context(state["facts"], state["chunks"])
        todo = [a for a in reporter.AUDIENCES if a not in reports or errors.get(a)]
        attempt = state.get("attempts", 0) + 1
        try:
            for a in todo:
                task = reporter.task_prompt(a, errors.get(a), reports.get(a))
                reports[a] = llm.complete(reporter.system_prompt(), shared, task,
                                          step=f"reporter:{a}:attempt{attempt}")  # fmt: skip
        except BudgetExceeded as err:
            return {"reports": reports, "attempts": attempt, "status": "budget_exceeded",
                    "log": [f"reporter: STOPPED, {err}"]}  # fmt: skip
        return {"reports": reports, "attempts": attempt, "status": "running", "written": todo,
                "log": [f"reporter: attempt {attempt}, wrote {todo}"]}  # fmt: skip

    def check(state: State) -> dict:
        n = len(state["chunks"])
        errors = {a: validate(r, state["facts"], n) for a, r in state["reports"].items()}
        bad = {a: len(e) for a, e in errors.items() if e}
        status = "validated" if not bad else state.get("status", "running")
        if bad and state["attempts"] > MAX_RETRIES:
            status = "needs_human_review"
        line = f"validator: attempt {state['attempts']}, errors {bad or 'none'}"
        written = state.get("written", [])
        record = {"attempt": state["attempts"], "written": written,
                  "errors": {a: errors[a] for a in written}}  # fmt: skip
        return {"errors": errors, "status": status, "history": [record], "log": [line]}

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


def save_run(state: dict, tracker, out_dir: Path) -> Path:
    """Reports as Markdown + run.json (status, validation errors, tokens, cost, SOP ids)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for audience, text in (state.get("reports") or {}).items():
        flag = ""
        if state.get("status") != "validated" and (state.get("errors") or {}).get(audience):
            flag = "> **FLAGGED FOR HUMAN REVIEW**: failed automatic validation; see run.json.\n\n"
        (out_dir / f"report_{audience}.md").write_text(flag + text)
    history = state.get("history") or []
    per_report = {a: sum(a in h["written"] for h in history) for a in state.get("reports") or {}}
    run = {
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": state.get("status"),
        "attempts": state.get("attempts"),
        "attempts_per_report": per_report,  # how many drafts each report needed
        "validation_errors": state.get("errors"),  # final result
        "validation_history": history,  # errors from every attempt, not just the last
        "report_status": {a: "failed" if (state.get("errors") or {}).get(a) else "passed"
                          for a in state.get("reports") or {}},  # fmt: skip
        "usage": tracker.totals(),
        "sop_excerpts": [{"id": f"S{i}", "chunk_id": c["chunk_id"], "title": c["title"],
                          "page": c["page"]} for i, c in enumerate(state.get("chunks", []), 1)],
        "log": state.get("log"),
    }  # fmt: skip
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    return out_dir
