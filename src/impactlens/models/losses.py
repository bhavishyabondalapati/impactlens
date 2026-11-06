"""Loss: binary cross-entropy + soft Dice, both ignoring pixels where valid == 0.

Both tasks are imbalanced (change ~2-5% of pixels, water ~10%). BCE alone can score well by
predicting "nothing"; Dice measures overlap with the rare class directly, so we add it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1.0
) -> torch.Tensor:
    """logits, target, valid: (B, 1, H, W). target in {0, 1}; valid is 1 for real labels."""
    logits = logits.float()
    n_valid = valid.sum().clamp(min=1.0)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum()
    bce = bce / n_valid
    prob = torch.sigmoid(logits) * valid  # zero out ignored pixels on both sides
    tgt = target * valid
    dice = 1 - (2 * (prob * tgt).sum() + eps) / (prob.sum() + tgt.sum() + eps)
    return bce + dice
