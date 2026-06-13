"""
Modular Le-ADMM with optional DRUNet pre/post-processors.

Architecture (from Bezzam et al., arXiv 2502.01102):
    y → [Pre-processor (DRUNet)] → y' → [Le-ADMM-5] → x̂ → [Post-processor (DRUNet)] → x_final

Pre-processor: residual DRUNet — y' = lensless + pre(lensless)
  At init (tail=0), y' = lensless (identity). Learns to refine the measurement.

Post-processor: residual DRUNet — x_final = clamp(admm_out + post(admm_out), 0, 1)
  At init (tail=0), x_final = admm_out (identity). Learns to refine the reconstruction.
  Using clamp instead of sigmoid avoids the 'gray image' local minimum where
  sigmoid(0) = 0.5 everywhere at initialization.

For ~8M total parameters (matching paper's Pre4+LeADMM5+Post4):
    channels=(32, 64, 116, 128), n_res_blocks=3 → ~3.92M per processor
"""

import torch
from torch import nn

from src.model.drunet import DRUNet
from src.model.le_admm import LeADMM


class ModularLeADMM(nn.Module):
    """
    Modular Le-ADMM with optional DRUNet pre/post-processors.

    Architecture:
        y → [Pre-processor] → y' → [Le-ADMM-5] → x̂ → [Post-processor] → x_final

    Both processors use residual connections and zero-initialized tails so the
    model starts as identity (passes measurement/reconstruction unchanged).
    """

    def __init__(
        self,
        n_iter=5,
        init_mu=1e-4,
        init_tau=2e-4,
        use_pre=True,
        use_post=True,
        drunet_channels=(32, 64, 116, 128),
        drunet_n_res_blocks=3,
    ):
        """
        Args:
            n_iter: number of Le-ADMM unrolled iterations
            init_mu: initial value for mu hyperparameter
            init_tau: initial value for tau hyperparameter
            use_pre: whether to use a pre-processor
            use_post: whether to use a post-processor
            drunet_channels: tuple of 4 channel sizes for DRUNet scales
            drunet_n_res_blocks: number of residual blocks per scale in DRUNet
        """
        super().__init__()
        self.use_pre = use_pre
        self.use_post = use_post

        # Pre-processor: DRUNet with zero-initialized tail for residual learning.
        # At init: pre(lensless) ≈ 0, so y' = lensless + 0 = lensless (identity).
        if use_pre:
            self.pre_processor = DRUNet(
                in_channels=3,
                out_channels=3,
                channels=drunet_channels,
                n_res_blocks=drunet_n_res_blocks,
            )
            # Zero-init tail so pre-processor starts as identity (residual learning)
            nn.init.zeros_(self.pre_processor.tail.weight)
            nn.init.zeros_(self.pre_processor.tail.bias)
        else:
            self.pre_processor = None

        # Le-ADMM core
        self.le_admm = LeADMM(n_iter=n_iter, init_mu=init_mu, init_tau=init_tau)

        # Post-processor: DRUNet with zero-initialized tail for residual learning.
        # At init: post(admm_out) ≈ 0, so x_final = clamp(admm_out + 0, 0, 1) = admm_out.
        if use_post:
            self.post_processor = DRUNet(
                in_channels=3,
                out_channels=3,
                channels=drunet_channels,
                n_res_blocks=drunet_n_res_blocks,
            )
            # Zero-init tail so post-processor starts as identity (residual learning)
            nn.init.zeros_(self.post_processor.tail.weight)
            nn.init.zeros_(self.post_processor.tail.bias)
        else:
            self.post_processor = None

    def forward(self, lensless, mask, **batch):
        """
        Args:
            lensless: (B, C, H, W) lensless measurement
            mask: (B, H_m, W_m) PSF/mask
        Returns:
            dict with "reconstruction": (B, C, H, W)
        """
        # Pre-process: residual refinement of the measurement.
        # y' = lensless + pre(lensless). At init, pre ≈ 0 so y' = lensless.
        if self.use_pre:
            y = lensless + self.pre_processor(lensless)
        else:
            y = lensless

        # Run Le-ADMM on the (possibly refined) measurement
        admm_out = self.le_admm(y, mask)["reconstruction"]

        # Post-process: residual refinement of the reconstruction.
        # x_final = clamp(admm_out + post(admm_out), 0, 1).
        # At init, post ≈ 0 so x_final = admm_out (already in [0,1]).
        # clamp avoids the 'gray image' local minimum of sigmoid(0)=0.5.
        if self.use_post:
            reconstruction = torch.clamp(admm_out + self.post_processor(admm_out), 0.0, 1.0)
        else:
            reconstruction = admm_out

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
