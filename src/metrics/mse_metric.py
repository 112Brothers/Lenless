"""
MSE metric using torchmetrics.
"""

import torch
from src.metrics.base_metric import BaseMetric


class MSEMetric(BaseMetric):
    """MSE metric for image reconstruction."""

    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from torchmetrics import MeanSquaredError

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = MeanSquaredError().to(device)

    def __call__(self, reconstruction, lensed, **kwargs):
        """
        Args:
            reconstruction: (B, C, H, W) predicted image in [0, 1]
            lensed: (B, C, H, W) ground truth image in [0, 1]
        Returns:
            MSE value (float)
        """
        return self.metric(reconstruction, lensed).item()
