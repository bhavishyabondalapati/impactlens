"""Kaggle automation: the kernel stays private, and the results table reads metrics.json."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_log import log_text  # noqa: E402
from metrics_summary import summarize  # noqa: E402


def test_kernel_metadata_is_private_gpu_and_uses_our_dataset():
    meta = json.loads((ROOT / "notebooks" / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["code_file"] == "train_kaggle.ipynb"
    assert (ROOT / "notebooks" / meta["code_file"]).exists()
    assert meta["dataset_sources"] == ["bhavishya12345678/impactlens-packed"]


def test_run_script_asks_before_push_upload_and_gpu():
    script = (ROOT / "scripts" / "kaggle_run.sh").read_text()
    confirms = [line for line in script.splitlines() if line.strip().startswith("confirm ")]
    for step in ("git push", "upload a new version", "start a GPU run"):
        assert any(step in line for line in confirms), step
    assert '"is_private": true' in script  # refuses to run if the kernel became public


def _m(f1):
    return {"iou": f1 / 2, "f1": f1, "precision": f1, "recall": f1}


def test_summary_table_has_tuned_threshold_and_both_baselines():
    base = {
        "tuned_threshold": -0.05,
        **{s: {"fixed": _m(0.8), "tuned": _m(0.87)} for s in ("val", "test", "bolivia")},
    }
    m = {"task": "floods", "round": 2, "best_epoch": 3, "history": [{}] * 5,
         "val": _m(0.9), "test": _m(0.88), "bolivia": _m(0.85),
         "threshold": {"value": 0.4, "tuned_on": "valid", "valid": _m(0.91), "test": _m(0.89),
                       "bolivia": _m(0.86)},
         "baseline_ndwi": base, "baseline_mndwi": base,
         "onnx": {"ok": True, "max_abs_logit_diff": 1e-7}, "device": "cuda", "amp": True,
         "round_note": "Round 2: ..."}  # fmt: skip
    text = summarize(m)
    assert "| U-Net, threshold 0.40 (tuned on val) | 0.910 | 0.890 | 0.860 |" in text
    assert "NDWI, threshold 0 |" in text and "MNDWI, threshold -0.05 (tuned on val)" in text
    assert "round 2" in text and "Note: Round 2" in text


def test_log_text_joins_kaggle_json_log():
    raw = json.dumps([{"stream_name": "stdout", "data": "a\n"}, {"data": "Error: b\n"}])
    assert log_text(raw) == "a\nError: b\n"
    assert log_text("plain") == "plain"
