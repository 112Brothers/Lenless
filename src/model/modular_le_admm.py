"""
Modular Le-ADMM with optional DRUNet pre/post-processors.

Architecture (from Bezzam et al., arXiv 2502.01102):
    y → [Pre-processor (DRUNet)] → y' → [Le-ADMM-5] → x̂ → [Post-processor (DRUNet)] → x_final

Pre-processor: DRUNet with no output activation (can output any range)
Post-processor: DRUNet with sigmoid to clamp output to [0, 1]

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

        # Pre-processor: DRUNet, no output activation (can output any range)
        if use_pre:
            self.pre_processor = DRUNet(
                in_channels=3,
                out_channels=3,
                channels=drunet_channels,
                n_res_blocks=drunet_n_res_blocks,
            )
        else:
            self.pre_processor = nn.Identity()

        # Le-ADMM core
        self.le_admm = LeADMM(n_iter=n_iter, init_mu=init_mu, init_tau=init_tau)

        # Post-processor: DRUNet with sigmoid to clamp output to [0, 1]
        if use_post:
            self.post_processor = DRUNet(
                in_channels=3,
                out_channels=3,
                channels=drunet_channels,
                n_res_blocks=drunet_n_res_blocks,
            )
        else:
            self.post_processor = nn.Identity()

        self.use_sigmoid_post = use_post  # apply sigmoid only when post-processor is DRUNet

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

        # Apply sigmoid to clamp to [0, 1] when post-processor is used
        if self.use_sigmoid_post:
            reconstruction = torch.sigmoid(reconstruction)

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
