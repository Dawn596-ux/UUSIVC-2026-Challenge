# models/losses/multi_task_loss.py

from typing import Dict, Any
import torch
import torch.nn as nn

from models.task_defs import (
    LABEL_SEG,
    TASK_TYPE,
    IMAGE_SEG,
    VIDEO_SEG,
)
from .dice import DiceLoss
from .boundary_loss import BoundaryLoss


class SegmentationLoss(nn.Module):
    """
    The organizers keep this baseline note in English for public release.
        total = ce_weight * CE + dice_weight * Dice

    The organizers keep this baseline note in English for public release.
    - image_seg: logits [B, K, H, W], target [B, H, W]
    - video_seg: logits [B, T, K, H, W], target [B, T, H, W]

    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    """

    def __init__(
        self,
        ce_weight: float = 0.4,
        dice_weight: float = 0.6,
        boundary_weight: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight  # α，可被 trainer 按 epoch 递增
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()

    def _flatten_video_if_needed(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        image:
            logits [B, K, H, W]
            target [B, H, W]

        video:
            logits [B, T, K, H, W]
            target [B, T, H, W]
            The organizers keep this baseline note in English for public release.
            logits [B*T, K, H, W]
            target [B*T, H, W]
        """
        if logits.ndim == 4 and target.ndim == 3:
            return logits, target

        if logits.ndim == 5 and target.ndim == 4:
            b, t, k, h, w = logits.shape
            logits = logits.reshape(b * t, k, h, w)
            target = target.reshape(b * t, h, w)
            return logits, target

        raise ValueError(
            f"Unsupported seg logits/target shapes: logits={logits.shape}, target={target.shape}"
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits, target = self._flatten_video_if_needed(logits, target)

        loss_ce = self.ce_loss(logits, target.long())
        loss_dice = self.dice_loss(logits, target, softmax=True)
        loss_total = self.ce_weight * loss_ce + self.dice_weight * loss_dice

        result = {
            "loss_total": loss_total,
            "loss_ce": loss_ce,
            "loss_dice": loss_dice,
        }
        if self.boundary_weight > 0:
            loss_boundary = self.boundary_loss(logits, target)
            result["loss_boundary"] = loss_boundary
            result["loss_total"] = loss_total + self.boundary_weight * loss_boundary
        return result


class Stage1SegLoss(nn.Module):
    """
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
        batch, outputs
    The organizers keep this baseline note in English for public release.
        dict(loss_total=..., loss_ce=..., loss_dice=...)
    """

    def __init__(
        self,
        ce_weight: float = 0.4,
        dice_weight: float = 0.6,
        boundary_weight: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.seg_loss = SegmentationLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            boundary_weight=boundary_weight,
            ignore_index=ignore_index,
        )

    @staticmethod
    def _parse_task_type(batch: Dict[str, Any]) -> str:
        task = batch[TASK_TYPE]
        if isinstance(task, str):
            return task
        if isinstance(task, (list, tuple)):
            if len(task) == 0:
                raise ValueError("Empty task_type list.")
            task0 = task[0]
            if not all(t == task0 for t in task):
                raise ValueError(f"Mixed task types in one batch are not supported: {task}")
            return task0
        raise TypeError(f"Unsupported task_type type: {type(task)}")

    def forward(
        self,
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        task_type = self._parse_task_type(batch)

        if task_type not in {IMAGE_SEG, VIDEO_SEG}:
            raise ValueError(f"Stage1SegLoss only supports seg tasks, got task_type={task_type}")

        if "seg_logits" not in outputs or outputs["seg_logits"] is None:
            raise ValueError("outputs must contain seg_logits for segmentation tasks.")

        logits = outputs["seg_logits"]
        target = batch[LABEL_SEG]

        if target is None:
            raise ValueError("batch[LABEL_SEG] is None for segmentation task.")

        loss_dict = self.seg_loss(logits, target)
        return loss_dict