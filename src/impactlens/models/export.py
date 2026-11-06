"""Export a trained model to ONNX and check it gives the same answers as PyTorch.

ONNX is a portable file format, so the model can run later (e.g. in AWS Lambda) without
PyTorch. Batch size, height and width are dynamic (height and width must be divisible by 16).
The dummy input used for tracing is 256x256, the training tile size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

TILE = 256
N_BANDS = {"oscd": 4, "floods": 6}  # input channels per image
TOLERANCE = 1e-3  # max allowed |PyTorch logit - ONNX logit|


def input_names(task: str) -> list[str]:
    return ["img1", "img2"] if task == "oscd" else ["img"]


def export_onnx(model: torch.nn.Module, task: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = input_names(task)
    model = model.cpu().eval()
    dummy = tuple(torch.rand(1, N_BANDS[task], TILE, TILE) for _ in names)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=names,
        output_names=["logits"],
        dynamic_axes={n: {0: "batch", 2: "height", 3: "width"} for n in [*names, "logits"]},
        opset_version=17,
        dynamo=False,
    )
    return path


def verify_onnx(model: torch.nn.Module, task: str, path: str | Path, inputs: list[torch.Tensor]):
    """Run the same inputs through PyTorch and ONNX Runtime and compare the logits."""
    import onnxruntime as ort

    model = model.cpu().eval()
    with torch.no_grad():
        expected = model(*inputs).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {n: x.numpy() for n, x in zip(input_names(task), inputs, strict=True)}
    got = session.run(["logits"], feed)[0]
    diff = float(np.abs(expected - got).max())
    agree = float(((expected > 0) == (got > 0)).mean())
    return {
        "path": str(path),
        "batch_checked": int(inputs[0].shape[0]),
        "max_abs_logit_diff": diff,
        "mask_agreement": agree,
        "tolerance": TOLERANCE,
        "ok": bool(diff <= TOLERANCE),
    }
