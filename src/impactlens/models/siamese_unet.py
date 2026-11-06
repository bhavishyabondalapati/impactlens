"""FC-Siam-diff (Daudt et al. 2018): Siamese U-Net for change detection on OSCD.

"Siamese" = the SAME encoder (same weights) processes the before image and the after image.
"diff" = at every skip connection the decoder receives |features_before - features_after|,
so it only sees *what changed* between the two dates, not the raw scenes.

Inputs: img1 and img2, each (B, 4, H, W). Output: (B, 1, H, W) logits (positive = change).
As in the paper, the decoder starts from the deepest features of the first image.
"""

from __future__ import annotations

import torch
from torch import nn

from impactlens.models.blocks import Encoder, Up, check_size


class SiameseUNetDiff(nn.Module):
    def __init__(self, in_ch: int = 4, base: int = 16):
        super().__init__()
        b = base
        self.encoder = Encoder(in_ch, (b, 2 * b, 4 * b, 8 * b))  # shared by both dates
        self.up4 = Up(8 * b, 8 * b, 4 * b)
        self.up3 = Up(4 * b, 4 * b, 2 * b)
        self.up2 = Up(2 * b, 2 * b, b)
        self.up1 = Up(b, b, b)
        self.head = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        check_size(img1)
        skips1, pooled1 = self.encoder(img1)
        skips2, _ = self.encoder(img2)
        d1, d2, d3, d4 = (torch.abs(a - b) for a, b in zip(skips1, skips2, strict=True))
        y = self.up4(pooled1, d4)
        y = self.up3(y, d3)
        y = self.up2(y, d2)
        y = self.up1(y, d1)
        return self.head(y)
