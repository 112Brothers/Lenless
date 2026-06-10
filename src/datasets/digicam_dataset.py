"""
Dataset wrapper for DigiCam-Mirflickr-MultiMask-10K from HuggingFace.

The dataset has:
- lensless: PIL Image (380x507 RGB)
- lensed: PIL Image (variable size RGB)
- mask_label: int (index of PSF mask, 0-based)

PSF masks are stored separately as masks/mask_{i}.npy files on HuggingFace.
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
    - mask: (H_m, W_m) float32 tensor — PSF/mask (normalized to sum=1)
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

        # Load all PSF masks from HuggingFace (cached locally)
        self.masks = self._load_all_masks()

    def _load_all_masks(self):
        """Load all PSF masks from HuggingFace dataset files."""
        from huggingface_hub import hf_hub_download

        masks = {}
        # Find unique mask labels in dataset
        unique_labels = set()
        for i in range(len(self.hf_dataset)):
            unique_labels.add(self.hf_dataset[i]["mask_label"])

        logger.info(f"Loading {len(unique_labels)} PSF masks from HuggingFace...")
        for label in sorted(unique_labels):
            try:
                path = hf_hub_download(
                    repo_id="bezzam/DigiCam-Mirflickr-MultiMask-10K",
                    filename=f"masks/mask_{label}.npy",
                    repo_type="dataset",
                )
                mask_np = np.load(path).astype(np.float32)
                # Normalize PSF so it sums to 1
                mask_np = mask_np / mask_np.sum()
                masks[label] = torch.from_numpy(mask_np)
            except Exception as e:
                logger.warning(f"Could not load mask_{label}.npy: {e}")
                masks[label] = None

        return masks

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]

        lensless = self._to_tensor(sample["lensless"])  # (3, H, W)
        lensed = self._to_tensor(sample["lensed"])      # (3, H, W)

        # Load PSF mask by mask_label index
        mask_label = sample["mask_label"]
        mask = self.masks.get(mask_label)
        if mask is None:
            # Fallback: uniform mask
            H, W = lensless.shape[1:]
            mask = torch.ones(H, W, dtype=torch.float32) / (H * W)
            logger.warning(f"Using uniform mask for sample {idx} (mask_label={mask_label})")

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
