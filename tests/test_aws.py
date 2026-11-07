"""Phase 6 tests: the AWS pipeline without AWS.

- The real state machine JSON (infra/statemachine/...) is executed by a small local
  interpreter (aws/local.py) against the real step functions, with a folder instead of S3 and
  a fake LLM: happy path, retry loop, human-review flag, cost cap, and failure -> DLQ.
- The starter Lambda (SQS -> Step Functions) with a fake Step Functions client.
- Cloud replacements match the local originals: the Shapely rollup vs the PostGIS summary.json,
  the ONNX MiniLM embedder vs sentence-transformers (both skipped without the local data).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from impactlens.agents import rag
from impactlens.agents.llm import FakeLLM
from impactlens.aws import handlers, local, steps
from impactlens.aws.store import LocalStore
from test_agents import FACTS, GOOD, fake_embed

REPO = Path(__file__).resolve().parents[1]
VALENCIA = REPO / "data/processed/valencia_albufera"
REQUEST = {"aoi": {"type": "Polygon", "coordinates": [[[-0.4, 39.3], [-0.3, 39.3], [-0.3, 39.4],
                                                        [-0.4, 39.4], [-0.4, 39.3]]]},
           "before": "2024-10-01/2024-10-28", "after": "2024-10-30/2024-11-12"}  # fmt: skip


# ----------------------------------------------------------------- the state machine


@pytest.fixture
def store(tmp_path):
    """A bucket (folder) with a request, a tiny 2-excerpt SOP index and no models: Fetch,
    Detect and GeoAnalyze are replaced by fakes that write the fact sheet directly."""
    s = LocalStore(tmp_path / "bucket")
    s.write_json("requests/run1.json", REQUEST)
    chunks = [
        rag.Chunk("a:web:0", "a", "A", None, "flood water mapping with a water index"),
        rag.Chunk("b:p3:0", "b", "B", 3, "limits of optical images: clouds and timing"),
    ]
    index = rag.build_index(chunks, fake_embed, tmp_path / "index")
    for f in steps.INDEX_FILES:
        s.upload(index / f, f"index/{f}")
    return s


def fake_upstream(store, event, work):
    store.write_json(f"layers/{event['run_id']}/facts.json", FACTS)
    return {"run_id": event["run_id"]}


def run_pipeline(store, tmp_path, respond, budget="0.10", monkeypatch=None, **overrides):
    if monkeypatch:
        monkeypatch.setenv("BUDGET_USD", budget)
        monkeypatch.setenv("MODEL", "fake")
    llm_prompts = []

    def report(store, event, work):
        def make(tracker):
            llm = FakeLLM(tracker, respond=respond)
            llm_prompts.append(llm.prompts)
            return llm

        return steps.report(store, event, work, make_llm=make)

    fakes = {"fetch": fake_upstream, "detect": lambda s, e, w: {"run_id": e["run_id"]},
             "geo_analyze": lambda s, e, w: {"run_id": e["run_id"]},
             "retrieve": lambda s, e, w: steps.retrieve(s, e, w, embed=fake_embed),
             "report": report, **overrides}  # fmt: skip
    invoke = local.local_invoke(store, tmp_path / "work", fakes)
    out, trace, dlq = local.run(local.render(), invoke, {"run_id": "run1"})
    return out, trace, dlq, [p for ps in llm_prompts for p in ps]


def test_happy_path_publishes_after_one_retry(store, tmp_path, monkeypatch):
    def respond(system, shared, task, step):
        if "agriculture" in step and "attempt1" in step:
            return GOOD.replace("2,059 ha", "2,350 ha")  # one wrong number, first draft only
        return GOOD

    out, trace, dlq, prompts = run_pipeline(store, tmp_path, respond, monkeypatch=monkeypatch)
    assert trace[:4] == ["Fetch", "Detect", "GeoAnalyze", "Retrieve"]
    assert trace.count("Report") == 2 and trace[-1] == "Publish" and dlq == []
    assert out["status"] == "validated"
    assert len(prompts) == 4  # 3 first drafts + only the failing report rewritten
    run = store.read_json("reports/run1/run.json")
    assert run["attempts_per_report"] == {"emergency": 1, "agriculture": 2, "municipal": 1}
    assert "2,350" in run["validation_history"][0]["errors"]["agriculture"][0]
    assert store.exists("reports/run1/report_emergency.md")


def test_flagged_for_review_after_two_retries(store, tmp_path, monkeypatch):
    bad = GOOD.replace("2,059 ha", "2,350 ha")
    out, trace, dlq, prompts = run_pipeline(store, tmp_path, lambda *a: bad,
                                            monkeypatch=monkeypatch)  # fmt: skip
    assert trace.count("Report") == 3  # first draft + 2 retries, then a person decides
    assert trace[-1] == "FlagForReview" and out["status"] == "needs_human_review"
    text = store.read_text("reports/run1/report_municipal.md")
    assert text.startswith("> **FLAGGED FOR HUMAN REVIEW**")


def test_cost_cap_holds_across_lambdas(store, tmp_path, monkeypatch):
    # each fake call costs $0.007; with a $0.02 cap the worst-case check stops before call 2.
    # The spend is re-read from tokens.jsonl by each Report Lambda, so the cap is per RUN.
    out, trace, _, prompts = run_pipeline(store, tmp_path, lambda *a: GOOD, budget="0.02",
                                          monkeypatch=monkeypatch)  # fmt: skip
    assert "Validate" not in trace and trace[-1] == "FlagForReview"
    assert out["status"] == "budget_exceeded" and out["cost_usd"] <= 0.02
    assert len(prompts) == 1


def test_a_failing_step_sends_the_run_to_the_dlq(store, tmp_path, monkeypatch):
    def broken_detect(store, event, work):
        raise RuntimeError("onnx model missing")

    out, trace, dlq, _ = run_pipeline(store, tmp_path, lambda *a: GOOD, monkeypatch=monkeypatch,
                                      detect=broken_detect)  # fmt: skip
    assert trace[-2:] == ["SendToDLQ", "Failed"]
    assert dlq == [{"run_id": "run1", "error": {"Error": "RuntimeError",
                                                "Cause": "onnx model missing"}}]  # fmt: skip


def test_template_has_every_step_and_the_retry_limit():
    d = local.render(max_retries=2)
    assert {"Fetch", "Detect", "GeoAnalyze", "Retrieve", "Report", "Validate", "Publish",
            "FlagForReview", "SendToDLQ"} <= set(d["States"])  # fmt: skip
    rule = d["States"]["ReportsValid"]["Choices"][1]
    assert rule == {"Variable": "$.attempt", "NumericGreaterThan": 2, "Next": "FlagForReview"}


# --------------------------------------------------------------- requests and starter


def test_request_and_run_id_are_checked(tmp_path):
    s = LocalStore(tmp_path)
    s.write_json("requests/ok.json", {**REQUEST, "before": "last week"})
    with pytest.raises(ValueError, match="before must look like"):
        steps.read_request(s, "ok")
    with pytest.raises(ValueError, match="bad run_id"):
        steps.read_request(s, "../secrets")


def test_aoi_size_limit(tmp_path):
    s = LocalStore(tmp_path)
    ring = [[-2, 38], [0, 38], [0, 40], [-2, 40], [-2, 38]]  # about 170 x 220 km
    huge = {**REQUEST, "aoi": {"type": "Polygon", "coordinates": [ring]}}
    s.write_json("requests/big.json", huge)
    with pytest.raises(ValueError, match="limit is 2500 km2"):
        steps.fetch(s, {"run_id": "big"}, tmp_path, fetch_pair=lambda *a: None)


def test_starter_starts_one_run_per_request_file(monkeypatch):
    monkeypatch.setenv("STATE_MACHINE_ARN", "arn:sm")

    class FakeSfn:
        def __init__(self):
            self.calls = []

        def start_execution(self, **kw):
            self.calls.append(kw)

    def sqs(*keys):
        recs = [{"s3": {"object": {"key": k}}} for k in keys]
        return {"Records": [{"body": json.dumps({"Records": recs})}]}

    sfn = FakeSfn()
    event = sqs("requests/valencia-2026%2D11%2D06.json", "requests/bad%20name.json",
                "layers/x.json")  # fmt: skip
    event["Records"].append({"body": json.dumps({"Event": "s3:TestEvent"})})
    assert handlers.start(event, None, sfn=sfn) == {"started": ["valencia-2026-11-06"]}
    (call,) = sfn.calls
    assert call == {"stateMachineArn": "arn:sm", "name": "valencia-2026-11-06",
                    "input": json.dumps({"run_id": "valencia-2026-11-06"})}  # fmt: skip


# ------------------------------------------- cloud replacements match the originals


needs_valencia = pytest.mark.skipif(
    not (VALENCIA / "geo/summary.json").exists() or not (REPO / "data/raw/osm").exists(),
    reason="needs the local Valencia data (impactlens fetch / infer / rollup)",
)


def _leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _leaves(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, x


@needs_valencia
def test_shapely_rollup_matches_postgis_on_valencia(tmp_path):
    """The Lambda rollup (Shapely, no database) against summary.json from PostGIS: every number
    within rounding (0.25 ha or 0.2%, whichever is larger). Hotspot ids are arbitrary labels."""
    from impactlens.geo.pipeline import run_rollup

    postgis = json.loads((VALENCIA / "geo/summary.json").read_text())
    shapely_ = run_rollup(VALENCIA, VALENCIA / "layers", REPO / "aoi/valencia_albufera.geojson",
                          tmp_path, REPO / "data/raw/osm", log=lambda *a: None,
                          backend="shapely")  # fmt: skip
    a, b = dict(_leaves(json.loads(json.dumps(shapely_)))), dict(_leaves(postgis))
    assert set(a) == set(b)
    compared = 0
    for key, want in b.items():
        got = a[key]
        if key.endswith("hotspot_id"):
            continue
        if isinstance(want, float | int) and not isinstance(want, bool):
            assert got == pytest.approx(want, abs=max(0.25, 0.002 * abs(want))), key
            compared += 1
        elif key not in (".aoi",):
            assert got == want, key
    assert compared > 500


@needs_valencia
def test_geo_step_uses_the_osm_cache_in_the_bucket(tmp_path, monkeypatch):
    """GeoAnalyze with the OSM answers already in osm/: no Overpass request at all."""
    import urllib.request

    from impactlens.data.stac import load_aoi

    def no_network(*a, **k):
        raise AssertionError("Overpass was called although the AOI is cached")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    s = LocalStore(tmp_path / "bucket")
    aoi_path = REPO / "aoi/valencia_albufera.geojson"
    s.write_json("requests/v.json", {**REQUEST, "aoi": json.loads(aoi_path.read_text())})
    key = steps.aoi_key(load_aoi(aoi_path))
    for what in ("landuse", "coastline", "municipalities"):  # the laptop's cache, renamed
        s.upload(REPO / f"data/raw/osm/valencia_albufera_{what}.json", f"osm/{key}_{what}.json")
    s.upload(aoi_path, "layers/v/aoi.geojson")
    for f in ("before.tif", "scenes.json"):
        s.upload(VALENCIA / f, f"layers/v/scene/{f}")
    for f in (VALENCIA / "layers").iterdir():
        s.upload(f, f"layers/v/layers/{f.name}")
    out = steps.geo_analyze(s, {"run_id": "v"}, tmp_path / "work")
    assert out == {"run_id": "v", "osm_cache_hit": True}
    facts = s.read_json("layers/v/facts.json")
    assert facts["totals"]["flooded_ha_on_land"] == pytest.approx(2104.0, abs=0.5)


@pytest.mark.skipif(not (REPO / "build/embedder/minilm.onnx").exists(),
                    reason="run `impactlens export-embedder` first")  # fmt: skip
def test_onnx_embedder_matches_sentence_transformers():
    from impactlens.agents.embed_onnx import OnnxEmbedder
    from impactlens.agents.steps import RETRIEVAL_QUERIES

    texts = [*RETRIEVAL_QUERIES, "short", "word " * 400]  # 400 words: truncated at 256 tokens
    onnx = OnnxEmbedder(REPO / "build/embedder")(texts)
    torch_ = rag.Embedder()(texts)
    assert onnx.shape == (len(texts), 384)
    assert np.min((onnx * torch_).sum(1)) > 0.9999  # cosine similarity of unit vectors
