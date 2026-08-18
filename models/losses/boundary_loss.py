"""Boundary Loss (Kervadec et al., "Boundary loss for highly unbalanced segmentation").

L_B = mean(s_theta * phi_G)，其中 s_theta 是前景 softmax 概率，phi_G 是 GT 的
符号距离图（前景内部为负、外部为正、边界为 0）。对边界距离线性可微，与
Dice/CE 互补，用于提升边界精度（NSD）。
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt


def _compute_sdf(mask: torch.Tensor) -> torch.Tensor:
    """从二值 mask 计算符号距离函数（SDF），前景内部为负、外部为正、边界为 0。

    mask: [B, H, W]（0/1）。返回同 shape、同 device 的 float SDF。
    """
    mask_np = (mask > 0).detach().cpu().numpy().astype(bool)
    b = mask_np.shape[0]
    sdf = np.zeros_like(mask_np, dtype=np.float32)
    for i in range(b):
        pos = mask_np[i]
        if pos.any():
            sdf[i] = (distance_transform_edt(~pos) - distance_transform_edt(pos)).astype(np.float32)
        else:
            sdf[i] = 1e10
    return torch.from_numpy(sdf).to(mask.device)


class BoundaryLoss(nn.Module):
    """Boundary Loss：GT 符号距离图加权前景 softmax 概率。

    早期网络 softmax 趋于饱和时梯度会消失，故标准做法是与区域损失复合，且
    权重 alpha 从很小值（如 0.01）按 epoch 递增。权重由调用方（loss 组合）控制。
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, 2, H, W]；target: [B, H, W] 0/1
        probs = torch.softmax(logits, dim=1)[:, 1]  # 前景概率
        sdf = _compute_sdf(target)
        return (probs * sdf).mean()
