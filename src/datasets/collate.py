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

    PSF mask is resized proportionally to the image downscaling factor,
    NOT to the same size as the image. The PSF is a small pattern (54×26)
    that represents the mask; it must stay small relative to the image for
    the convolution forward model to be physically correct.
    """
    result_batch = {}

    # Maximum spatial size to prevent OOM on GPU.
    # DigiCam lensless is fixed at 380×507; cap at 270px to fit T4 VRAM.
    MAX_SIZE = 270

    # Original lensless size (before any capping)
    orig_h, orig_w = dataset_items[0]["lensless"].shape[1:]

    # Target size is always driven by lensless (the model input), not lensed.
    # This is correct because the forward model operates on lensless resolution.
    target_h, target_w = orig_h, orig_w

    # Cap to MAX_SIZE while preserving aspect ratio
    if target_h > MAX_SIZE or target_w > MAX_SIZE:
        scale = MAX_SIZE / max(target_h, target_w)
        target_h = int(target_h * scale)
        target_w = int(target_w * scale)
    else:
        scale = 1.0

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

    # Resize PSF mask proportionally to the image downscaling factor.
    # The PSF is a small pattern (e.g. 54×26) that must stay small relative
    # to the image. We scale it by the same factor as the image downscaling.
    # After resize, renormalize so PSF still sums to 1.
    mask_list = []
    for item in dataset_items:
        mask = item["mask"]
        mask_h, mask_w = mask.shape
        if scale != 1.0:
            new_mask_h = max(1, int(round(mask_h * scale)))
            new_mask_w = max(1, int(round(mask_w * scale)))
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(new_mask_h, new_mask_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        # Renormalize so PSF still sums to 1 after resize
        mask_sum = mask.sum().clamp(min=1e-8)
        mask = mask / mask_sum
        mask_list.append(mask)

    # Masks may now have different sizes across items in the batch if orig sizes differ.
    # Stack requires same size — pad to the largest mask in the batch.
    max_mh = max(m.shape[0] for m in mask_list)
    max_mw = max(m.shape[1] for m in mask_list)
    padded_masks = []
    for mask in mask_list:
        mh, mw = mask.shape
        if mh != max_mh or mw != max_mw:
            pad_bottom = max_mh - mh
            pad_right = max_mw - mw
            mask = F.pad(mask, (0, pad_right, 0, pad_bottom), value=0.0)
            # Renormalize after padding
            mask_sum = mask.sum().clamp(min=1e-8)
            mask = mask / mask_sum
        padded_masks.append(mask)
    result_batch["mask"] = torch.stack(padded_masks)

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
