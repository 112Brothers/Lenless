"""
Denoising Residual U-Net (DRUNet) architecture.

From: "Plug-and-play image restoration with deep denoiser prior"
      Zhang et al., IEEE TPAMI 2021.

Used as pre- and post-processors in the modular Le-ADMM pipeline
(Bezzam et al., arXiv 2502.01102, Appendix B).

Architecture:
  - 4 scales with strided-conv (2×2) downscaling and transposed-conv (2×2) upscaling
  - Residual blocks: two 3×3 convs + skip connection, NO batch norm, ReLU
  - Identity skip connection between StridedConv and TransposedConv at each scale
  - Sandwiched by Conv layers (no activation function)
  - No sigmoid at output (caller decides normalization)

Channel configs (from paper):
  - ~8.2M params: (32, 64, 128, 256)
  - ~4.1M params: (32, 64, 116, 128)
  - ~2.0M params: (16, 32, 64, 128)
"""

import torch
from torch import nn


class ResBlock(nn.Module):
    """
    Residual block: Conv-ReLU-Conv + skip connection.
    No batch normalization (as in DRUNet paper).
    """

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class DRUNet(nn.Module):
    """
    Denoising Residual U-Net (DRUNet).

    Architecture (4 scales):
      Input → Conv (no act) →
        [StridedConv → ResBlocks] × 4 (encoder) →
        ResBlocks (bottleneck) →
        [TransposedConv → ResBlocks + skip] × 4 (decoder) →
      Conv (no act) → Output

    The identity skip connections are between the StridedConv output
    and the corresponding TransposedConv output (added before ResBlocks).
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        channels=(32, 64, 128, 256),
        n_res_blocks=4,
    ):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            channels: tuple of 4 channel sizes for each scale
            n_res_blocks: number of residual blocks per scale
        """
        super().__init__()
        assert len(channels) == 4, "DRUNet requires exactly 4 scale channel sizes"
        c1, c2, c3, c4 = channels

        # Input conv (no activation)
        self.head = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=True)

        # Encoder: strided conv (2×2) + residual blocks
        self.down1 = nn.Conv2d(c1, c2, kernel_size=2, stride=2, bias=True)
        self.enc1 = nn.Sequential(*[ResBlock(c2) for _ in range(n_res_blocks)])

        self.down2 = nn.Conv2d(c2, c3, kernel_size=2, stride=2, bias=True)
        self.enc2 = nn.Sequential(*[ResBlock(c3) for _ in range(n_res_blocks)])

        self.down3 = nn.Conv2d(c3, c4, kernel_size=2, stride=2, bias=True)
        self.enc3 = nn.Sequential(*[ResBlock(c4) for _ in range(n_res_blocks)])

        # Bottleneck residual blocks (at deepest scale)
        self.bottleneck = nn.Sequential(*[ResBlock(c4) for _ in range(n_res_blocks)])

        # Decoder: transposed conv (2×2) + residual blocks
        # Skip connections add encoder features before residual blocks
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2, bias=True)
        self.dec3 = nn.Sequential(*[ResBlock(c3) for _ in range(n_res_blocks)])

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=True)
        self.dec2 = nn.Sequential(*[ResBlock(c2) for _ in range(n_res_blocks)])

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=True)
        self.dec1 = nn.Sequential(*[ResBlock(c1) for _ in range(n_res_blocks)])

        # Output conv (no activation)
        self.tail = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # Head
        x0 = self.head(x)           # (B, c1, H, W)

        # Encoder
        x1 = self.down1(x0)         # (B, c2, H/2, W/2)
        x1 = self.enc1(x1)

        x2 = self.down2(x1)         # (B, c3, H/4, W/4)
        x2 = self.enc2(x2)

        x3 = self.down3(x2)         # (B, c4, H/8, W/8)
        x3 = self.enc3(x3)

        # Bottleneck
        x4 = self.bottleneck(x3)    # (B, c4, H/8, W/8)

        # Decoder with identity skip connections (add, not concat)
        d3 = self.up3(x4)           # (B, c3, H/4, W/4)
        # Handle size mismatch from odd spatial dims
        if d3.shape != x2.shape:
            d3 = nn.functional.interpolate(d3, size=x2.shape[2:], mode='nearest')
        d3 = self.dec3(d3 + x2)     # skip: add encoder feature

        d2 = self.up2(d3)           # (B, c2, H/2, W/2)
        if d2.shape != x1.shape:
            d2 = nn.functional.interpolate(d2, size=x1.shape[2:], mode='nearest')
        d2 = self.dec2(d2 + x1)     # skip: add encoder feature

        d1 = self.up1(d2)           # (B, c1, H, W)
        if d1.shape != x0.shape:
            d1 = nn.functional.interpolate(d1, size=x0.shape[2:], mode='nearest')
        d1 = self.dec1(d1 + x0)     # skip: add encoder feature

        # Tail (no activation)
        out = self.tail(d1)         # (B, out_channels, H, W)
        return out

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
