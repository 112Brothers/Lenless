"""
LPIPS metric (VGG variant).
"""

import torch
from src.metrics.base_metric import BaseMetric


class LPIPSMetric(BaseMetric):
    """LPIPS metric for image reconstruction."""

    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import lpips

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = lpips.LPIPS(net='vgg').to(device)
        self.metric.eval()

    def __call__(self, reconstruction, lensed, **kwargs):
        """
        Args:
            reconstruction: (B, C, H, W) predicted image in [0, 1]
            lensed: (B, C, H, W) ground truth image in [0, 1]
        Returns:
            LPIPS value (float)
        """
        # LPIPS expects [-1, 1] range
        return self.metric(reconstruction * 2 - 1, lensed * 2 - 1).mean().item()
