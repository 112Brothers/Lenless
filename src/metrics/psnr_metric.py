"""
PSNR metric using torchmetrics.
"""

import torch
from src.metrics.base_metric import BaseMetric


class PSNRMetric(BaseMetric):
    """PSNR metric for image reconstruction."""

    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from torchmetrics.image import PeakSignalNoiseRatio

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    def __call__(self, reconstruction, lensed, **kwargs):
        """
        Args:
            reconstruction: (B, C, H, W) predicted image in [0, 1]
            lensed: (B, C, H, W) ground truth image in [0, 1]
        Returns:
            PSNR value (float)
        """
        return self.metric(reconstruction, lensed).item()
