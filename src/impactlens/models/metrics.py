"""Pixel metrics for binary masks, counted over valid pixels only.

We add up true/false positives over the whole dataset and compute the ratios once at the end
(not an average of per-tile scores, which would let near-empty tiles distort the result).
"""

from __future__ import annotations

import torch


class ConfusionMeter:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0

    def update_pred(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        """pred, target, valid: same shape; pred and target are 0/1, valid is 1 on real pixels."""
        p, t, v = pred > 0.5, target > 0.5, valid > 0.5
        self.tp += int((p & t & v).sum())
        self.fp += int((p & ~t & v).sum())
        self.fn += int((~p & t & v).sum())
        self.tn += int((~p & ~t & v).sum())

    def update(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        """logit > 0 is the same as probability > 0.5."""
        self.update_pred((logits > 0).float(), target, valid)

    def compute(self) -> dict[str, float]:
        tp, fp, fn = self.tp, self.fp, self.fn
        div = lambda a, b: a / b if b else 0.0  # noqa: E731
        return {
            "iou": div(tp, tp + fp + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn),
            "precision": div(tp, tp + fp),
            "recall": div(tp, tp + fn),
        }
