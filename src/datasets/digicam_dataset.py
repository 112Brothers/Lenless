"""
Dataset wrapper for DigiCam-Mirflickr-MultiMask-10K from HuggingFace.
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class DigiCamDataset(Dataset):
    """
    Dataset wrapper for DigiCam-Mirflickr-MultiMask-10K from HuggingFace.

    Each sample contains:
    - lensless: (3, H, W) float32 tensor in [0, 1] — lensless measurement
    - lensed: (3, H, W) float32 tensor in [0, 1] — ground truth image
    - mask: (H, W) float32 tensor — PSF/mask (normalized to sum=1)
    """

    def __init__(self, split="train", limit=None, instance_transforms=None):
        """
        Args:
            split: "train" or "test"
            limit: if not None, limit dataset size
            instance_transforms: dict of transforms per tensor name
        """
        from datasets import load_dataset

        self.hf_dataset = load_dataset(
            "bezzam/DigiCam-Mirflickr-MultiMask-10K",
            split=split
        )
        if limit is not None:
            self.hf_dataset = self.hf_dataset.select(range(min(limit, len(self.hf_dataset))))

        self.instance_transforms = instance_transforms

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]

        # Extract and convert to tensors
        # The HF dataset stores images as PIL Images and masks as numpy arrays
        # Check the actual format and adapt accordingly
        lensless = self._to_tensor(sample["lensless"])  # (3, H, W)
        lensed = self._to_tensor(sample["lensed"])      # (3, H, W)
        
        # Load mask if available, otherwise create a dummy uniform mask
        if "mask" in sample:
            mask = self._load_mask(sample["mask"])      # (H, W)
        else:
            # Create a uniform mask (all ones, normalized)
            H, W = lensless.shape[1:]
            mask = torch.ones(H, W, dtype=torch.float32) / (H * W)

        instance_data = {
            "lensless": lensless,
            "lensed": lensed,
            "mask": mask,
        }

        # Apply instance transforms
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name in instance_data:
                    instance_data[transform_name] = self.instance_transforms[
                        transform_name
                    ](instance_data[transform_name])

        return instance_data

    def _to_tensor(self, img):
        """Convert PIL Image or numpy array to (C, H, W) float32 tensor in [0, 1]."""
        if isinstance(img, Image.Image):
            img = np.array(img)
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
            if img.ndim == 3:
                img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
            else:
                img = torch.from_numpy(img)
        return img

    def _load_mask(self, mask_data):
        """Load and normalize PSF mask."""
        if isinstance(mask_data, np.ndarray):
            mask = torch.from_numpy(mask_data.astype(np.float32))
        elif isinstance(mask_data, torch.Tensor):
            mask = mask_data.float()
        else:
            # May be stored differently in HF dataset — adapt as needed
            mask = torch.tensor(np.array(mask_data), dtype=torch.float32)

        # Normalize PSF so it sums to 1
        mask = mask / mask.sum()
        return mask
