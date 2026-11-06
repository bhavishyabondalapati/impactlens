"""Plain U-Net (Ronneberger et al. 2015) for flood-water segmentation on Sen1Floods11.

One 4-band image in, one logit per pixel out (positive = water). Apply a sigmoid to get a
water probability, which doubles as the per-pixel confidence.
"""

from __future__ import annotations

import torch
from torch import nn

from impactlens.models.blocks import Encoder, Up, check_size, conv_block


class UNet(nn.Module):
    def __init__(self, in_ch: int = 4, base: int = 32):
        super().__init__()
        b = base
        self.encoder = Encoder(in_ch, (b, 2 * b, 4 * b, 8 * b))
        self.bottleneck = conv_block(8 * b, 16 * b)
        self.up4 = Up(16 * b, 8 * b, 8 * b)
        self.up3 = Up(8 * b, 4 * b, 4 * b)
        self.up2 = Up(4 * b, 2 * b, 2 * b)
        self.up1 = Up(2 * b, b, b)
        self.head = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_size(x)
        (s1, s2, s3, s4), pooled = self.encoder(x)
        y = self.bottleneck(pooled)
        y = self.up4(y, s4)
        y = self.up3(y, s3)
        y = self.up2(y, s2)
        y = self.up1(y, s1)
        return self.head(y)
