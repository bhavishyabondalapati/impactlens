"""Phase 5 tests: Validator, cost cap, the LangGraph loop, RAG. A fake LLM, no API calls."""

import json
import re

import numpy as np
import pytest

from impactlens.agents import rag, validator
from impactlens.agents.facts import build_facts
from impactlens.agents.graph import build_graph, save_run
from impactlens.agents.llm import BudgetExceeded, CostTracker, FakeLLM, Usage
from impactlens.agents.sources import SOURCES, sources_markdown

# A tiny summary.json with the same shape as the real one (made-up but consistent numbers).
SUMMARY = {
    "before_scene": "S2A_30SYJ_20241026_0_L2A",
    "after_scene": "S2B_30SYJ_20241110_0_L2A",
    "crs": "EPSG:32630 (UTM 30N); areas in hectares",
    "definitions": {"ndvi_drop": "NDVI before > 0.3 (vegetated) and NDVI before - after > 0.2"},
    "totals": {"aoi_ha": 27818.56, "land_ha": 20757.44, "flooded_ha_on_land": 2104.0,
               "changed_ha_on_land": 2793.8, "ndvi_drop_ha_on_land": 1185.41},
    "impacts": {
        "standing_water": {"label": "x", "note": "x", "flooded_ha_on_land": 2104.0,
                           "in_rice_paddies_ha": 2058.76, "outside_rice_paddies_ha": 45.24},
        "vegetation_loss_likely_mud": {"note": "x", "ndvi_drop_ha_outside_rice_paddies": 643.36},
        "urban_change": {"note": "x", "changed_ha_on_urban_land": 198.41,
                         "port_excluded": {"changed_ha": 23.7, "area_ha": 247.67, "note": "x"}},
    },
    "rice_paddies": {"note": "x", "flooded_ha": 2058.76, "share_of_flooded_land_pct": 97.8,
                     "water_before_pct_of_paddy_area": 39.0, "water_after_pct_of_paddy_area": 68.4},
    "by_town": [{"name": "Paiporta", "land_in_aoi_ha": 382.0, "pct_of_municipality_in_aoi": 96.6,
                 "vegetated_ha_outside_rice_paddies": 166.8,
                 "ndvi_drop_ha_outside_rice_paddies": 69.4, "ndvi_drop_pct_of_vegetated": 41.6,
                 "urban_changed_ha": 33.77}],
    "excluded": {"sea_ha": 4276.62, "sea_flooded_ha": 688.03, "note": "x"},
    "hotspots": [{"flooded_ha": 1979.87, "main_lu_class": "rice_paddy", "rice_paddy_share": 0.989}],
    "n_hotspots": 21,
}  # fmt: skip
FACTS = build_facts(SUMMARY, [-0.457, 39.273, -0.283, 39.44])

GOOD = """# Valencia DANA impact: emergency brief

The after image was taken 12 days after the 29 October 2024 event. Paiporta lost vegetation on
41.6% of its vegetated land (69 ha), and its urban land shows 34 ha of change.

## Findings
- New standing water on land: about 2,100 ha, of which 2,059 ha is in rice paddies.
- Rice paddies are ambiguous: the water may be seasonal, deliberate paddy flooding rather than
  flood damage, and two optical images cannot tell them apart.
- NDVI drop (643 ha outside paddies) is a proxy for mud and damage, not proof.
- The AOI does not cover the whole affected area; towns outside it are not assessed.
- The Port of Valencia (24 ha of change) is excluded from the urban-change total of 198 ha.

## Method
Water was mapped from Sentinel-2 with a water index and a threshold, as in the recommended
practice [S1]. Optical mapping is limited by clouds and by the image date [S2].
"""


def test_good_report_passes():
    assert validator.validate(GOOD, FACTS, n_chunks=2) == []


def test_wrong_number_is_caught():
    bad = GOOD.replace("2,059 ha", "2,350 ha")
    errors = validator.validate(bad, FACTS, n_chunks=2)
    assert len(errors) == 1 and "2,350" in errors[0]


def test_invented_outside_fact_is_caught():
    bad = GOOD.replace("Paiporta lost", "At least 230 people died. Paiporta lost")
    assert any("'230'" in e for e in validator.validate(bad, FACTS, 2))


@pytest.mark.parametrize("cid, drop", [
    ("paddy_ambiguity", "- Rice paddies are ambiguous: the water may be seasonal, deliberate "
                        "paddy flooding rather than\n  flood damage, and two optical images "
                        "cannot tell them apart.\n"),
    ("ndvi_proxy", "- NDVI drop (643 ha outside paddies) is a proxy for mud and damage, "
                   "not proof.\n"),
    ("aoi_coverage", "- The AOI does not cover the whole affected area; towns outside it are not "
                     "assessed.\n"),
    ("port_excluded", "- The Port of Valencia (24 ha of change) is excluded from the "
                      "urban-change total of 198 ha.\n"),
])  # fmt: skip
def test_each_missing_caveat_is_caught(cid, drop):
    assert drop in GOOD
    errors = validator.validate(GOOD.replace(drop, ""), FACTS, 2)
    assert [e for e in errors if cid in e], errors


def test_wrapped_bullet_lines_belong_to_their_bullet():
    text = "- Rice paddy water may be seasonal\n  flooding, so it is ambiguous.\n- Next item.\n"
    assert validator.units(text) == [
        "- Rice paddy water may be seasonal flooding, so it is ambiguous.",
        "- Next item.",
    ]
    assert validator.check_caveats(text, ["paddy_ambiguity"]) == []


def test_caveat_words_must_be_in_the_same_paragraph():
    # "paddy", "seasonal" and "ambiguous" spread over different bullets do not count
    text = "- Rice paddy water.\n- Seasonal patterns.\n- Some ambiguous results.\n"
    assert validator.check_caveats(text, ["paddy_ambiguity"])


def test_rounding_tolerance():
    ok = validator.check_numbers("about 2,100 ha and 41.6% and 42%", FACTS)
    assert ok == []
    assert validator.check_numbers("about 2,000 ha", FACTS)  # 5% off 2,104: not rounding
    assert validator.check_numbers("41.9%", FACTS)  # not 41.6 to one decimal


def test_method_section_needs_valid_citations():
    no_cite = GOOD.replace(" [S1]", "")
    assert any("without an SOP citation" in e for e in validator.validate(no_cite, FACTS, 2))
    ghost = GOOD.replace("[S2]", "[S9]")
    assert any("[S9] does not exist" in e for e in validator.validate(ghost, FACTS, 2))
    no_method = GOOD.replace("## Method", "## How")
    assert any("Method" in e for e in validator.validate(no_method, FACTS, 2))


def test_facts_dates_and_days_after_event():
    ev = FACTS["event"]
    assert ev["before_image_date"] == "2024-10-26" and ev["after_image_date"] == "2024-11-10"
    assert ev["after_image_days_after_event"] == 12
    assert "note" not in FACTS["urban_change"]["port_excluded"]


# ------------------------------------------------------------------------- cost tracking


def test_cost_math_and_hard_cap(tmp_path):
    t = CostTracker("claude-haiku-4-5", budget_usd=0.10, log_path=tmp_path / "tokens.jsonl")
    # 1,000 input ($1/M) + 1,000 output ($5/M) + 10,000 cache reads ($0.10/M) = $0.007
    assert t.charge("x", Usage(1000, 1000, 0, 10_000)) == pytest.approx(0.007)
    assert json.loads((tmp_path / "tokens.jsonl").read_text())["run_total_usd"] == 0.007
    t.spent_usd = 0.095
    with pytest.raises(BudgetExceeded):
        t.check(prompt_chars=30_000, max_tokens=2500)  # worst case ~$0.025 -> over $0.10


# ---------------------------------------------------------------------------- the graph


class FakeRetriever:
    def search(self, query, k=2):
        chunk = rag.Chunk(f"fake:web:{abs(hash(query)) % 7}", "unspider_steps", "Fake SOP",
                          None, f"guidance about {query}")  # fmt: skip
        return [(1.0, chunk)]


FAKE_STEPS = {
    "ingest": lambda s: {"log": ["ingest: fake"]},
    "detect": lambda s: {"log": ["detect: fake"]},
    "geo_analyst": lambda s: {"summary": SUMMARY, "facts": FACTS, "log": ["geo: fake"]},
}


def run_graph(respond, budget=0.10, usage=None):
    tracker = CostTracker("fake", budget)
    llm = FakeLLM(tracker, respond=respond, usage=usage)
    state = build_graph(llm, FakeRetriever(), FAKE_STEPS).invoke({"log": []})
    return state, llm, tracker


def test_graph_sends_specific_errors_back_and_accepts_the_fix():
    def respond(system, shared, task, step):
        if "attempt1" in step:  # first drafts: one wrong number
            return GOOD.replace("2,059 ha", "2,350 ha")
        assert "2,350" in task and "does not match" in task  # the specific error came back
        return GOOD

    state, llm, _ = run_graph(respond)
    assert state["status"] == "validated" and state["attempts"] == 2
    assert len(llm.prompts) == 6  # 3 reports x 2 attempts
    assert "FACT SHEET" in llm.prompts[0]["shared"] and "[S1]" in llm.prompts[0]["shared"]


def test_graph_flags_for_human_review_after_two_retries(tmp_path):
    missing = GOOD.replace("- The AOI does not cover the whole affected area; towns outside it "
                           "are not assessed.\n", "")  # fmt: skip
    state, llm, tracker = run_graph(lambda *a: missing)
    assert state["status"] == "needs_human_review"
    assert state["attempts"] == 3 and len(llm.prompts) == 9  # 1 try + 2 retries, 3 reports
    assert all("aoi_coverage" in " ".join(e) for e in state["errors"].values())
    out = save_run(state, tracker, tmp_path)
    assert (out / "report_municipal.md").read_text().startswith("> **FLAGGED FOR HUMAN REVIEW**")
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "needs_human_review" and run["usage"]["calls"] == 9


def test_graph_stops_before_exceeding_the_budget():
    # each fake call costs 3000 x $1/M + 800 x $5/M = $0.007; a $0.02 cap allows 2 calls
    state, llm, tracker = run_graph(lambda *a: GOOD, budget=0.02)
    assert state["status"] == "budget_exceeded"
    assert len(llm.prompts) == 1 and tracker.spent_usd <= 0.02  # worst-case check stops call 2


# ------------------------------------------------------------------------------------ RAG


def fake_embed(texts):
    """Bag-of-words hashed to 64 dims, L2-normalised: enough to test the index plumbing."""
    out = np.zeros((len(texts), 64), np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, sum(map(ord, w)) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_chunks_overlap_and_index_finds_the_right_chunk(tmp_path):
    words = [f"w{i}" for i in range(500)]
    pieces = rag.chunk_words(" ".join(words), size=220, overlap=40)
    assert len(pieces) == 3 and pieces[1].split()[0] == "w180"  # 220 - 40 = 180 step
    chunks = [
        rag.Chunk("a:web:0", "a", "A", None, "rice paddy seasonal flooding farming"),
        rag.Chunk("b:p3:0", "b", "B", 3, "building damage grading earthquake"),
    ]
    rag.build_index(chunks, fake_embed, tmp_path)
    r = rag.Retriever(tmp_path, embed=fake_embed)
    (score, best), *_ = r.search("seasonal rice paddy", k=2)
    assert best.chunk_id == "a:web:0" and best.page is None


def test_faiss_matches_brute_force_next_to_torch(tmp_path):
    import torch

    x = torch.randn(1500, 1500)
    float((x.exp() + 1).sum())  # make torch start its own OpenMP runtime first
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(300, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunks = [rag.Chunk(f"c:web:{i}", "c", "C", None, f"t{i}") for i in range(300)]
    rag.build_index(chunks, lambda texts: vecs[: len(texts)], tmp_path)
    query = vecs[[7]] + 0.05 * rng.normal(size=(1, 384)).astype(np.float32)
    query /= np.linalg.norm(query)
    r = rag.Retriever(tmp_path, embed=lambda texts: query)
    got = [c.chunk_id for _s, c in r.search("anything", k=5)]
    want = [f"c:web:{i}" for i in np.argsort(-(vecs @ query[0]))[:5]]
    assert got == want and got[0] == "c:web:7"


def test_html_text_skips_scripts_and_menus(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<nav>Menu</nav><script>var x=1</script><p>Flood mapping with NDWI.</p>")
    assert rag.extract_pages(page) == [(None, "Flood mapping with NDWI.")]


def test_every_source_has_a_url_and_licence_note():
    md = sources_markdown()
    for s in SOURCES:
        assert s.url.startswith("https://") and s.licence and s.sid in md
