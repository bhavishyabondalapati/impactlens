"""Building blocks shared by both networks."""

from __future__ import annotations

import torch
from torch import nn


def conv_block(cin: int, cout: int) -> nn.Sequential:
    """Two (3x3 conv -> batch-norm -> ReLU) layers: the basic unit of every U-Net."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Four conv blocks with 2x2 max-pooling between them.

    Returns the four feature maps (full, 1/2, 1/4, 1/8 resolution) used as skip connections,
    and the last one pooled once more (1/16 resolution).
    """

    def __init__(self, in_ch: int, widths: tuple[int, int, int, int]):
        super().__init__()
        chans = (in_ch, *widths)
        self.blocks = nn.ModuleList(conv_block(chans[i], chans[i + 1]) for i in range(4))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            skips.append(x)
            if i < 3:
                x = self.pool(x)
        return skips, self.pool(x)


class Up(nn.Module):
    """Decoder step: double the resolution, concatenate the skip, run a conv block."""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cskip, kernel_size=2, stride=2)
        self.conv = conv_block(2 * cskip, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


def check_size(x: torch.Tensor) -> None:
    """Four 2x poolings need height and width divisible by 16 (256 tiles are fine)."""
    if torch.jit.is_tracing():  # the ONNX export uses a fixed, valid size
        return
    if x.shape[-1] % 16 or x.shape[-2] % 16:
        raise ValueError(f"height and width must be divisible by 16, got {tuple(x.shape[-2:])}")
