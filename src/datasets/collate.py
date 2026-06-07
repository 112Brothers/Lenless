"""
Collate function for lensless datasets.
"""

import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate lensless dataset items into a batch.

    Handles optional fields (lensed may not exist, image_id is a string).
    """
    result_batch = {}

    # Always present
    result_batch["lensless"] = torch.stack([item["lensless"] for item in dataset_items])
    result_batch["mask"] = torch.stack([item["mask"] for item in dataset_items])

    # Optional: ground truth
    if "lensed" in dataset_items[0]:
        result_batch["lensed"] = torch.stack([item["lensed"] for item in dataset_items])

    # Optional: image IDs (for inference saving)
    if "image_id" in dataset_items[0]:
        result_batch["image_id"] = [item["image_id"] for item in dataset_items]

    return result_batch
