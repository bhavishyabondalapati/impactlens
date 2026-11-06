"""Sliding-window inference over a whole scene, with overlap and blending.

The models were trained on 256x256 tiles, but a scene is ~1500x1500 px. We slide a 256 window
with overlap and average the overlapping predictions with a weight that is high in the middle
of each window and low at its edges (a CNN is least reliable at tile borders, where it sees
little context). Without blending you get visible seams on a grid.

The models run through ONNX Runtime (the same .onnx files exported after training), so no
PyTorch is needed at inference time.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from impactlens.data.oscd import tile_starts

Predictor = Callable[..., np.ndarray]  # (B,C,T,T) arrays -> (B,1,T,T) probabilities


def blend_weights(tile: int, floor: float = 0.05) -> np.ndarray:
    """(tile, tile) weights: a 2-D Hann window, never below ``floor`` so edges still count."""
    w = np.hanning(tile + 2)[1:-1]  # strictly positive
    return np.maximum(np.outer(w, w), floor).astype(np.float32)


def sliding_window(
    predict: Predictor,
    images: list[np.ndarray],
    tile: int = 256,
    overlap: int = 64,
    batch_size: int = 8,
) -> np.ndarray:
    """Run ``predict`` on overlapping tiles of the (C, H, W) ``images`` and blend the results.

    ``images`` is a list because the change model takes two inputs (before, after); all must
    share H and W. Images smaller than a tile are zero-padded. Returns (H, W) float32.
    """
    h, w = images[0].shape[-2:]
    ph, pw = max(h, tile), max(w, tile)
    padded = [np.pad(im, ((0, 0), (0, ph - h), (0, pw - w))) for im in images]
    stride = tile - overlap
    windows = [(r, c) for r in tile_starts(ph, tile, stride) for c in tile_starts(pw, tile, stride)]
    weight = blend_weights(tile)
    acc = np.zeros((ph, pw), np.float32)
    norm = np.zeros((ph, pw), np.float32)
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        batches = [np.stack([im[:, r : r + tile, c : c + tile] for r, c in chunk]) for im in padded]
        probs = predict(*batches)[:, 0]
        for (r, c), p in zip(chunk, probs, strict=True):
            acc[r : r + tile, c : c + tile] += p * weight
            norm[r : r + tile, c : c + tile] += weight
    return (acc / norm)[:h, :w]


def onnx_predictor(path: str | Path) -> Predictor:
    """Wrap an exported .onnx model: inputs in, sigmoid probabilities out."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]

    def predict(*arrays: np.ndarray) -> np.ndarray:
        feed = {n: a.astype(np.float32) for n, a in zip(names, arrays, strict=True)}
        logits = session.run(None, feed)[0]
        return 1.0 / (1.0 + np.exp(-logits))

    return predict
