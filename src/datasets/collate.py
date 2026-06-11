"""
Collate function for lensless datasets.
"""

import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate lensless dataset items into a batch.

    Handles optional fields (lensed may not exist, image_id is a string).

    Target size is determined by the lensless image (the model input).
    The lensed (ground truth) image is resized to match lensless so that
    loss and metrics are computed at the same resolution.

    Pipeline: lensless (fixed 380×507) → pad to FFT size → ADMM → crop back
    → compare with lensed resized to lensless size.
    """
    result_batch = {}

    # Maximum spatial size to prevent OOM on GPU.
    # DigiCam lensless is fixed at 380×507; cap at 270px to fit T4 VRAM.
    MAX_SIZE = 270

    # Target size is always driven by lensless (the model input), not lensed.
    # This is correct because the forward model operates on lensless resolution.
    target_h, target_w = dataset_items[0]["lensless"].shape[1:]

    # Cap to MAX_SIZE while preserving aspect ratio
    if target_h > MAX_SIZE or target_w > MAX_SIZE:
        scale = MAX_SIZE / max(target_h, target_w)
        target_h = int(target_h * scale)
        target_w = int(target_w * scale)

    # Resize all lensless images to target size
    lensless_list = []
    for item in dataset_items:
        img = item["lensless"]
        if img.shape[1:] != (target_h, target_w):
            img = F.interpolate(
                img.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
            ).squeeze(0)
        lensless_list.append(img)
    result_batch["lensless"] = torch.stack(lensless_list)

    # Resize all masks to target size
    mask_list = []
    for item in dataset_items:
        mask = item["mask"]
        if mask.shape != (target_h, target_w):
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
        mask_list.append(mask)
    result_batch["mask"] = torch.stack(mask_list)

    # Optional: ground truth
    if "lensed" in dataset_items[0]:
        lensed_list = []
        for item in dataset_items:
            img = item["lensed"]
            if img.shape[1:] != (target_h, target_w):
                img = F.interpolate(
                    img.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
                ).squeeze(0)
            lensed_list.append(img)
        result_batch["lensed"] = torch.stack(lensed_list)

    # Optional: image IDs (for inference saving)
    if "image_id" in dataset_items[0]:
        result_batch["image_id"] = [item["image_id"] for item in dataset_items]

    return result_batch
