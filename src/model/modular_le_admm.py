"""
Modular Le-ADMM with optional U-Net pre/post-processors.
Wraps Le-ADMM with neural network processors.
"""

import torch
from torch import nn

from src.model.le_admm import LeADMM
from src.model.unet import UNet


class ModularLeADMM(nn.Module):
    """
    Modular Le-ADMM with optional U-Net pre/post-processors.

    Architecture:
        y → [Pre-processor] → y' → [Le-ADMM-5] → x̂ → [Post-processor] → x_final
    """

    def __init__(
        self,
        n_iter=5,
        init_mu=1e-4,
        init_tau=2e-4,
        use_pre=True,
        use_post=True,
        unet_base_channels=24,
    ):
        super().__init__()
        self.use_pre = use_pre
        self.use_post = use_post

        # Pre-processor (no sigmoid, can output negative values)
        self.pre_processor = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=unet_base_channels,
            use_sigmoid=False,
        ) if use_pre else nn.Identity()

        # Le-ADMM core
        self.le_admm = LeADMM(n_iter=n_iter, init_mu=init_mu, init_tau=init_tau)

        # Post-processor (with sigmoid to clamp to [0,1])
        self.post_processor = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=unet_base_channels,
            use_sigmoid=True,
        ) if use_post else nn.Identity()

    def forward(self, lensless, mask, **batch):
        """
        Args:
            lensless: (B, C, H, W) lensless measurement
            mask: (B, H_m, W_m) PSF/mask
        Returns:
            dict with "reconstruction": (B, C, H, W)
        """
        # Pre-process the measurement
        y = self.pre_processor(lensless)

        # Run Le-ADMM
        admm_out = self.le_admm(y, mask)["reconstruction"]

        # Post-process the reconstruction
        reconstruction = self.post_processor(admm_out)

        return {"reconstruction": reconstruction}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
