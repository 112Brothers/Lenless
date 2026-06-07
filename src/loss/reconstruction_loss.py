"""
Reconstruction loss: MSE + LPIPS.
"""

import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    """
    Combined MSE + LPIPS loss for image reconstruction.
    """

    def __init__(self, lpips_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lpips_weight = lpips_weight

        # Initialize LPIPS (VGG variant)
        import lpips
        self.lpips_fn = lpips.LPIPS(net='vgg')
        # Freeze LPIPS weights
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def forward(self, reconstruction, lensed, **batch):
        """
        Args:
            reconstruction: (B, C, H, W) predicted image in [0, 1]
            lensed: (B, C, H, W) ground truth image in [0, 1]
        Returns:
            dict with "loss", "mse_loss", "lpips_loss"
        """
        mse_val = self.mse(reconstruction, lensed)

        # LPIPS expects [-1, 1] range
        lpips_val = self.lpips_fn(
            reconstruction * 2 - 1, lensed * 2 - 1
        ).mean()

        loss = mse_val + self.lpips_weight * lpips_val

        return {
            "loss": loss,
            "mse_loss": mse_val,
            "lpips_loss": lpips_val,
        }
