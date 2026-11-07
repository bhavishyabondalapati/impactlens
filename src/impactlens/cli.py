"""Command line entry point: ``impactlens fetch | infer | rollup``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from impactlens.data.stac import EARTH_SEARCH_URL, NoSceneError, fetch_pair, format_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="impactlens")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch", help="Download aligned before/after Sentinel-2 crops for an AOI"
    )
    fetch.add_argument("--aoi", required=True, type=Path, help="AOI GeoJSON (lon/lat)")
    fetch.add_argument("--before", required=True, help="date window, e.g. 2024-10-01/2024-10-28")
    fetch.add_argument("--after", required=True, help="date window, e.g. 2024-10-30/2024-11-20")
    fetch.add_argument("--max-cloud", type=float, default=20.0, help="max %% cloud over the AOI")
    fetch.add_argument("--out", type=Path, default=None, help="default: data/processed/<aoi name>")
    fetch.add_argument("--stac-url", default=EARTH_SEARCH_URL)

    infer = sub.add_parser("infer", help="Run both models over before/after; write GeoTIFF layers")
    infer.add_argument("--scene", type=Path, default=Path("data/processed/valencia_albufera"))
    infer.add_argument("--floods-model", type=Path, default=None,
                       help="default: newest build/models_*/floods_r2/floods.onnx")  # fmt: skip
    infer.add_argument("--oscd-model", type=Path, default=Path("build/models/oscd/oscd.onnx"))
    infer.add_argument("--overlap", type=int, default=64, help="sliding-window overlap (px)")
    infer.add_argument("--quicklook", type=Path, default=None, help="PNG path (optional)")

    rollup = sub.add_parser("rollup", help="Polygonize, OSM land use, PostGIS stats, export")
    rollup.add_argument("--scene", type=Path, default=Path("data/processed/valencia_albufera"))
    rollup.add_argument("--aoi", type=Path, default=Path("aoi/valencia_albufera.geojson"))
    rollup.add_argument("--map", type=Path, default=None, help="rollup map PNG (optional)")
    rollup.add_argument("--ndvi-map", type=Path, default=None,
                        help="NDVI-drop by town map PNG (optional)")  # fmt: skip
    rollup.add_argument("--backend", choices=("postgis", "shapely"), default="postgis",
                        help="postgis (docker) or shapely (no database, as in AWS)")  # fmt: skip
    rollup.add_argument("--out", type=Path, default=None, help="default: <scene>/geo")

    sub.add_parser("sops", help="Download the SOP corpus, chunk, embed, build the FAISS index")
    sub.add_parser("export-embedder", help="MiniLM -> ONNX (build/embedder/) for AWS Lambda")

    report = sub.add_parser("report", help="Run the 6-agent graph and write 3 validated reports")
    report.add_argument("--scene", type=Path, default=Path("data/processed/valencia_albufera"))
    report.add_argument("--aoi", type=Path, default=Path("aoi/valencia_albufera.geojson"))
    report.add_argument("--model", default="claude-haiku-4-5")
    report.add_argument("--budget", type=float, default=0.10, help="hard cost cap per run, USD")
    report.add_argument("--rerun-steps", action="store_true",
                        help="re-run inference and rollup even if their outputs exist")  # fmt: skip
    local = sub.add_parser("aws-local", help="Run the AWS state machine locally against a folder")
    local.add_argument("--store", type=Path, required=True, help="folder that stands in for S3")
    local.add_argument("--request", type=Path, required=True, help="request JSON (AOI + dates)")
    local.add_argument("--run-id", default="local-run")
    local.add_argument("--seed", action="store_true", help="copy models + index into the store")
    local.add_argument("--reuse-scene", type=Path, default=None,
                       help="skip Fetch: use this already-fetched scene folder")  # fmt: skip
    local.add_argument("--fake-llm", action="store_true", help="no Anthropic calls (dry run)")
    local.add_argument("--template", type=Path, default=None, help="state machine template")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fetch":
        out = args.out or Path("data/processed") / args.aoi.stem
        try:
            fetch_pair(args.aoi, args.before, args.after, out, args.max_cloud, args.stac_url)
        except NoSceneError as err:
            print(f"\nNo scene passed the {err.max_cloud:g}% cloud limit. Best candidates:")
            for window, cands in err.failures.items():
                print(f"\n  {window}")
                print(format_candidates(cands))
            return 2
    elif args.command == "infer":
        from impactlens.geo.pipeline import latest_model, run_infer

        floods = args.floods_model or latest_model("build/models_*/floods_r2/floods.onnx")
        run_infer(args.scene, floods, args.oscd_model, args.scene / "layers", args.quicklook,
                  args.overlap)  # fmt: skip
    elif args.command == "rollup":
        from impactlens.geo.pipeline import run_rollup

        run_rollup(args.scene, args.scene / "layers", args.aoi, args.out or args.scene / "geo",
                   Path("data/raw/osm"), args.map, args.ndvi_map,
                   backend=args.backend)  # fmt: skip
    elif args.command == "sops":
        from impactlens.agents import rag

        rag.download()
        chunks = rag.build_chunks()
        out = rag.build_index(chunks, rag.Embedder())
        print(f"{len(chunks)} chunks indexed in {out}/ (sources: {rag.SOP_DIR}/sources.md)")
    elif args.command == "export-embedder":
        from impactlens.agents.embed_onnx import export_minilm

        print(f"wrote {export_minilm()}/minilm.onnx and tokenizer.json")
    elif args.command == "report":
        return run_report(args)
    elif args.command == "aws-local":
        return run_aws_local(args)
    return 0


def run_aws_local(args) -> int:
    import shutil
    import time

    from impactlens.aws import local
    from impactlens.aws.store import LocalStore

    store = LocalStore(args.store)
    if args.seed:
        local.seed(store)
    store.upload(args.request, f"requests/{args.run_id}.json")
    overrides = {}
    if args.reuse_scene:
        overrides["fetch"] = local.copy_scene(args.reuse_scene)
    if args.fake_llm:
        from impactlens.agents.llm import FakeLLM
        from impactlens.aws import steps

        def report(store, event, work):
            fake = lambda tracker: FakeLLM(tracker, respond=lambda *a: "# Fake report\n")  # noqa: E731
            return steps.report(store, event, work, make_llm=fake)

        overrides["report"] = report
    work = args.store / "_work" / args.run_id
    shutil.rmtree(work, ignore_errors=True)
    timings, invoke = {}, local.local_invoke(store, work, overrides)

    def timed(step, payload):
        t0 = time.time()
        out = invoke(step, payload)
        timings[step] = timings.get(step, 0) + round(time.time() - t0, 1)
        print(f"{step}: {timings[step]} s  -> {out}")
        return out

    definition = local.render(args.template or local.TEMPLATE)
    out, trace, dlq = local.run(definition, timed, {"run_id": args.run_id})
    print(f"\nstates: {' -> '.join(trace)}\noutput: {out}\nDLQ: {dlq or 'empty'}")
    return 0 if trace[-1] in ("Publish", "FlagForReview") else 3


def run_report(args) -> int:
    import time

    from impactlens.agents.graph import build_graph, save_run
    from impactlens.agents.llm import AnthropicLLM, CostTracker
    from impactlens.agents.rag import Retriever

    out = Path("build/reports") / time.strftime("%Y%m%d-%H%M%S")
    tracker = CostTracker(args.model, args.budget, log_path=out / "tokens.jsonl")
    graph = build_graph(AnthropicLLM(tracker), Retriever())
    state = graph.invoke({"scene_dir": str(args.scene), "aoi": str(args.aoi),
                          "run_steps": args.rerun_steps, "log": []})  # fmt: skip
    save_run(state, tracker, out)
    for line in state["log"]:
        print(line)
    t = tracker.totals()
    print(
        f"\nstatus: {state['status']}  |  {t['calls']} LLM calls, {t['input_tokens']} in / "
        f"{t['output_tokens']} out / {t['cache_read_input_tokens']} cache-read tokens, "
        f"${t['cost_usd']:.4f} of ${t['budget_usd']:.2f} cap  |  {out}/"
    )
    return 0 if state["status"] == "validated" else 3


if __name__ == "__main__":
    sys.exit(main())
