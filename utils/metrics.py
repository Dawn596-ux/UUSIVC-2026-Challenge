# utils/metrics.py

import numpy as np
import torch
from PIL import Image


SEG_DSC_WEIGHT = 0.7
SEG_NSD_WEIGHT = 0.3


@torch.no_grad()
def compute_binary_dice_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    - image:
        logits [B, 2, H, W]
        target [B, H, W]
    - video:
        logits [B, T, 2, H, W]
        target [B, T, H, W]

    The organizers keep this baseline note in English for public release.
    """
    if logits.ndim == 5:
        b, t, c, h, w = logits.shape
        logits = logits.reshape(b * t, c, h, w)
        target = target.reshape(b * t, h, w)

    pred = torch.argmax(logits, dim=1).float()   # [N,H,W]
    target = target.float()

    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denom + eps)
    return float(dice.mean().item())

def compute_binary_dice_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = (np.asarray(pred) > 0).astype(np.float32).reshape(-1)
    target = (np.asarray(target) > 0).astype(np.float32).reshape(-1)
    intersection = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    return float((2.0 * intersection + eps) / (denom + eps))


def restore_ceus_prediction_to_original(pred_mask: np.ndarray, restore_meta: dict) -> np.ndarray:
    shape = tuple(int(x) for x in restore_meta["shape"])
    top, bottom, left, right = [int(x) for x in restore_meta["crop_box"]]
    pred_u8 = (np.asarray(pred_mask) > 0).astype(np.uint8) * 255
    crop_h = max(bottom - top, 1)
    crop_w = max(right - left, 1)
    resized = Image.fromarray(pred_u8).resize((crop_w, crop_h), Image.Resampling.NEAREST)
    full = np.zeros(shape, dtype=np.uint8)
    full[top:bottom, left:right] = (np.asarray(resized) > 0).astype(np.uint8) * 255
    return full


@torch.no_grad()
def compute_ceus_official_dice_from_logits(logits: torch.Tensor, raw_batch: dict) -> float:
    official_gt_masks = raw_batch.get("official_gt_mask")
    restore_metas = raw_batch.get("ceus_restore_meta")
    if official_gt_masks is None or restore_metas is None:
        return compute_binary_dice_from_logits(logits, raw_batch["label_seg"])

    if logits.ndim == 5:
        frame_idx = int(logits.shape[1] // 2)
        pred = torch.argmax(logits[:, frame_idx], dim=1).detach().cpu().numpy()
    elif logits.ndim == 4:
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported CEUS logits shape for official dice: {tuple(logits.shape)}")

    scores = []
    for idx, pred_mask in enumerate(pred):
        full_pred = restore_ceus_prediction_to_original(pred_mask, restore_metas[idx])
        scores.append(compute_binary_dice_np(full_pred, official_gt_masks[idx]))
    return float(np.mean(scores)) if scores else 0.0

def _inner_boundary(mask: np.ndarray) -> np.ndarray:
    mask = (np.asarray(mask) > 0)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & (~eroded)


def _dilate_cross(mask: np.ndarray) -> np.ndarray:
    mask = (np.asarray(mask) > 0)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def compute_nsd_np(pred: np.ndarray, target: np.ndarray, tolerance: int = 1) -> float:
    if tolerance != 1:
        raise ValueError("This lightweight NSD implementation expects tolerance=1.")
    target = (np.asarray(target) > 0).astype(np.uint8)
    pred = (np.asarray(pred) > 0).astype(np.uint8)
    if target.sum() == 0 and pred.sum() == 0:
        return 1.0
    if target.sum() == 0 or pred.sum() == 0:
        return 0.0
    boundary_true = _inner_boundary(target)
    boundary_pred = _inner_boundary(pred)
    if boundary_true.sum() == 0 and boundary_pred.sum() == 0:
        return 1.0
    if boundary_true.sum() == 0 or boundary_pred.sum() == 0:
        return 0.0
    true_match = (boundary_true & _dilate_cross(boundary_pred)).sum()
    pred_match = (boundary_pred & _dilate_cross(boundary_true)).sum()
    return float((true_match + pred_match) / (boundary_true.sum() + boundary_pred.sum()))


def compute_binary_seg_score_np(pred: np.ndarray, target: np.ndarray) -> dict:
    dsc = compute_binary_dice_np(pred, target, eps=0.0)
    nsd = compute_nsd_np(pred, target, tolerance=1)
    return {"dsc": dsc, "nsd": nsd, "score": SEG_DSC_WEIGHT * dsc + SEG_NSD_WEIGHT * nsd}


@torch.no_grad()
def compute_binary_seg_score_from_logits(logits: torch.Tensor, target: torch.Tensor) -> dict:
    if logits.ndim == 5:
        b, t, c, h, w = logits.shape
        logits = logits.reshape(b * t, c, h, w)
        target = target.reshape(b * t, h, w)
    pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    scores = [compute_binary_seg_score_np(p, y) for p, y in zip(pred, target_np)]
    if not scores:
        return {"dsc": 0.0, "nsd": 0.0, "score": 0.0}
    return {key: float(np.mean([item[key] for item in scores])) for key in ("dsc", "nsd", "score")}


@torch.no_grad()
def compute_ceus_official_score_from_logits(logits: torch.Tensor, raw_batch: dict) -> dict:
    official_gt_masks = raw_batch.get("official_gt_mask")
    restore_metas = raw_batch.get("ceus_restore_meta")
    if official_gt_masks is None or restore_metas is None:
        return compute_binary_seg_score_from_logits(logits, raw_batch["label_seg"])

    if logits.ndim == 5:
        frame_idx = int(logits.shape[1] // 2)
        pred = torch.argmax(logits[:, frame_idx], dim=1).detach().cpu().numpy()
    elif logits.ndim == 4:
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported CEUS logits shape for official score: {tuple(logits.shape)}")

    scores = []
    for idx, pred_mask in enumerate(pred):
        full_pred = restore_ceus_prediction_to_original(pred_mask, restore_metas[idx])
        scores.append(compute_binary_seg_score_np(full_pred, official_gt_masks[idx]))
    if not scores:
        return {"dsc": 0.0, "nsd": 0.0, "score": 0.0}
    return {key: float(np.mean([item[key] for item in scores])) for key in ("dsc", "nsd", "score")}

