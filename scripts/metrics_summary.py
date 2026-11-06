"""Print the metrics of a downloaded Kaggle run as a markdown table.

    uv run python scripts/metrics_summary.py build/models_<stamp>

Finds every metrics.json below the folder. For each: F1 (and IoU) on val/test/bolivia at the 0.5
threshold, at the val-tuned threshold, and for the NDWI / MNDWI baselines (threshold 0 and
tuned on val).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _f1(d: dict | None) -> str:
    return "-" if d is None else f"{d['f1']:.3f}"


def summarize(m: dict, name: str = "") -> str:
    val_key = "valid" if "bolivia" in m else "val"  # threshold section uses the split name
    splits = [s for s in ("val", "test", "bolivia") if s in m]
    th = m.get("threshold")
    lines = [f"### {name or m['task']} (round {m.get('round', 1)}, best epoch "
             f"{m['best_epoch']} of {len(m['history'])})", ""]  # fmt: skip
    head = ["model", *splits]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    net = "Siamese U-Net" if m["task"] == "oscd" else "U-Net"
    lines.append(f"| {net}, threshold 0.5 | " + " | ".join(_f1(m[s]) for s in splits) + " |")
    if th:
        row = [_f1(th.get(val_key if s == "val" else s)) for s in splits]
        lines.append(
            f"| {net}, threshold {th['value']:.2f} (tuned on val) | " + " | ".join(row) + " |"
        )
    for idx in ("ndwi", "mndwi"):
        b = m.get(f"baseline_{idx}")
        if not b:
            continue
        lines.append(f"| {idx.upper()}, threshold 0 | "
                     + " | ".join(_f1(b[s]["fixed"]) for s in splits) + " |")  # fmt: skip
        lines.append(f"| {idx.upper()}, threshold {b['tuned_threshold']:.2f} (tuned on val) | "
                     + " | ".join(_f1(b[s]["tuned"]) for s in splits) + " |")  # fmt: skip
    lines.append("")
    lines.append(f"ONNX check: {'OK' if m['onnx']['ok'] else 'MISMATCH'} "
                 f"(max logit diff {m['onnx']['max_abs_logit_diff']:.1e}); device {m['device']}, "
                 f"amp {m['amp']}")  # fmt: skip
    if m.get("round_note"):
        lines.append(f"Note: {m['round_note']}")
    return "\n".join(lines)


def main(folder: str) -> None:
    files = sorted(Path(folder).rglob("metrics.json"))
    if not files:
        raise SystemExit(f"no metrics.json under {folder}")
    for f in files:
        print(summarize(json.loads(f.read_text()), f.parent.name) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
