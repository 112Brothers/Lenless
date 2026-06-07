"""
Dataset for custom directory structure for inference.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CustomDirDataset(Dataset):
    """
    Dataset for custom directory structure:

    data_dir/
    ├── lensless/   *.png
    ├── masks/      *.npy
    └── lensed/     *.png  (optional, ground truth)
    """

    def __init__(self, data_dir, instance_transforms=None):
        self.data_dir = Path(data_dir)
        self.instance_transforms = instance_transforms

        # Scan lensless directory for image IDs
        lensless_dir = self.data_dir / "lensless"
        assert lensless_dir.exists(), f"lensless directory not found: {lensless_dir}"

        self.image_ids = sorted([
            p.stem for p in lensless_dir.glob("*.png")
        ])

        self.has_lensed = (self.data_dir / "lensed").exists()

        logger.info(
            f"CustomDirDataset: {len(self.image_ids)} samples, "
            f"has_lensed={self.has_lensed}"
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load lensless measurement
        lensless_path = self.data_dir / "lensless" / f"{image_id}.png"
        lensless = self._load_image(lensless_path)

        # Load mask/PSF
        mask_path = self.data_dir / "masks" / f"{image_id}.npy"
        mask = self._load_mask(mask_path)

        instance_data = {
            "lensless": lensless,
            "mask": mask,
            "image_id": image_id,
        }

        # Load ground truth if available
        if self.has_lensed:
            lensed_path = self.data_dir / "lensed" / f"{image_id}.png"
            if lensed_path.exists():
                instance_data["lensed"] = self._load_image(lensed_path)

        # Apply instance transforms
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name in instance_data:
                    instance_data[transform_name] = self.instance_transforms[
                        transform_name
                    ](instance_data[transform_name])

        return instance_data

    def _load_image(self, path):
        img = np.array(Image.open(path)).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)

    def _load_mask(self, path):
        mask = np.load(path).astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask / mask.sum()
        return mask
