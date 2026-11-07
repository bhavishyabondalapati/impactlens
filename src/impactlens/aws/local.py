"""Run the real state machine definition locally, without AWS.

The Step Functions definition (infra/statemachine/pipeline.asl.json.tftpl) decides the order
of steps, the Report <-> Validate loop and where failures go. To test THAT file, not a copy of
its logic, this module fills in the template and walks it with a tiny interpreter that knows
the few state types we use (Task, Choice, Fail). Each Lambda Task calls the matching function
in aws/steps.py with a LocalStore (a folder) instead of S3.

Used by tests/test_aws.py and by `impactlens aws-local`, which runs the whole cloud pipeline
on the laptop (or inside the Lambda container image) against a folder.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[3] / "infra/statemachine/pipeline.asl.json.tftpl"
SEED = {  # store key -> local file: what `make assets` uploads to the bucket
    "models/floods/floods.onnx": "build/models_*/floods_r2/floods.onnx",
    "models/floods/metrics.json": "build/models_*/floods_r2/metrics.json",
    "models/oscd/oscd.onnx": "build/models/oscd/oscd.onnx",
    "models/oscd/metrics.json": "build/models/oscd/metrics.json",
    "index/sops.faiss": "data/processed/rag/sops.faiss",
    "index/chunks.jsonl": "data/processed/rag/chunks.jsonl",
    "index/embedder/minilm.onnx": "build/embedder/minilm.onnx",
    "index/embedder/tokenizer.json": "build/embedder/tokenizer.json",
}
STEP_NAMES = ("fetch", "detect", "geo_analyze", "retrieve", "report", "validate", "finish")


def render(template: Path = TEMPLATE, max_retries: int = 2) -> dict:
    """Fill the Terraform template with fake ARNs ("local:<step>"), like templatefile() does."""
    values = {f"{s}_arn": f"local:{s}" for s in STEP_NAMES}
    values |= {"dlq_url": "local:dlq", "max_retries": str(max_retries)}
    text = re.sub(r"\$\{(\w+)\}", lambda m: values[m.group(1)], template.read_text())
    return json.loads(text)


def _path(data: dict, path: str):
    """The tiny subset of JSONPath we use: "$" and "$.key"."""
    if path == "$":
        return data
    return data[path.removeprefix("$.")]


def _params(params: dict, data: dict):
    out = {}
    for key, value in params.items():
        if key.endswith(".$"):
            out[key[:-2]] = _path(data, value)
        elif isinstance(value, dict):
            out[key] = _params(value, data)
        else:
            out[key] = value
    return out


def _choice(rule: dict, data: dict) -> bool:
    value = _path(data, rule["Variable"])
    for op, test in rule.items():
        if op == "StringEquals":
            return value == test
        if op == "BooleanEquals":
            return value is test
        if op == "NumericGreaterThan":
            return value > test
        if op == "NumericGreaterThanEquals":
            return value >= test
    raise ValueError(f"unsupported Choice rule {rule}")


def run(definition: dict, invoke: Callable[[str, dict], dict], data: dict,
        max_steps: int = 100) -> tuple[dict, list[str], list[dict]]:  # fmt: skip
    """Execute ``definition``. ``invoke(step_name, payload)`` runs one Lambda.

    Returns (final output, the names of the states visited, messages sent to the DLQ).
    """
    name, trace, dlq = definition["StartAt"], [], []
    for _ in range(max_steps):
        state = definition["States"][name]
        trace.append(name)
        kind = state["Type"]
        if kind == "Fail":
            return data, trace, dlq
        if kind == "Choice":
            name = next((c["Next"] for c in state["Choices"] if _choice(c, data)),
                        state["Default"])  # fmt: skip
            continue
        if kind != "Task":
            raise ValueError(f"unsupported state type {kind}")
        params = _params(state["Parameters"], data)
        try:
            if state["Resource"].endswith(":sqs:sendMessage"):
                dlq.append(params["MessageBody"])
                result = {}
            else:
                step = params["FunctionName"].removeprefix("local:")
                result = {"Payload": invoke(step, params["Payload"])}
        except Exception as err:  # noqa: BLE001  Catch: States.ALL
            catch = next(c for c in state.get("Catch", []) if "States.ALL" in c["ErrorEquals"])
            error = {"Error": type(err).__name__, "Cause": str(err)}
            data = {**data, catch["ResultPath"].removeprefix("$."): error}
            name = catch["Next"]
            continue
        data = _path(result, state["OutputPath"]) if "OutputPath" in state else data
        if state.get("End"):
            return data, trace, dlq
        name = state["Next"]
    raise RuntimeError(f"no end after {max_steps} states: {trace[-10:]}")


def local_invoke(store, work_root: Path, overrides: dict | None = None):
    """``invoke`` for run(): each step gets a fresh work folder, like a Lambda's /tmp."""
    from impactlens.aws import steps

    overrides = overrides or {}
    counter = {"n": 0}

    def invoke(step: str, payload: dict) -> dict:
        counter["n"] += 1
        work = Path(work_root) / f"{counter['n']:02d}_{step}"
        work.mkdir(parents=True, exist_ok=True)
        fn = overrides.get(step, steps.STEPS[step])
        return json.loads(json.dumps(fn(store, payload, work)))  # JSON round trip, like AWS

    return invoke


def seed(store, root: Path = Path(".")) -> None:
    """Copy the models, FAISS index and ONNX embedder into a store (newest floods model)."""
    import glob

    for key, pattern in SEED.items():
        hits = sorted(glob.glob(str(root / pattern)))
        if not hits:
            raise FileNotFoundError(f"{pattern} not found (needed for {key})")
        store.upload(hits[-1], key)


def copy_scene(scene_dir: Path):
    """A Fetch replacement that reuses an already-fetched scene (offline runs and tests)."""
    from impactlens.aws.steps import SCENE_FILES, _keys

    def fetch(store, event, work):
        k = _keys(event["run_id"])
        req = store.read_json(k["request"])
        for f in SCENE_FILES:
            store.upload(Path(scene_dir) / f, f"{k['layers']}/scene/{f}")
        (work / "aoi.geojson").write_text(json.dumps(req["aoi"]))
        store.upload(work / "aoi.geojson", f"{k['layers']}/aoi.geojson")
        return {"run_id": event["run_id"]}

    return fetch
