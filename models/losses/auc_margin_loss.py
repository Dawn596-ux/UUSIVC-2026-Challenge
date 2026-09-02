"""Pairwise AUC-Margin Ranking Loss（二分类排序正则）.

L_rank = mean_{(i,j): y_i=1, y_j=0} softplus(m - (s_i - s_j))，s = logit[:,1] - logit[:,0]。

s 是 log-odds 差，与 softmax 正类概率单调等价；该损失直接优化样本对的正负序
（即 AUC 的 pairwise 语义），CE 保留不动作为 Acc 锚。softplus 平滑无死区，
成对项量级 O(1) 有界，无需 ramp；margin 在 logit 空间（无界，不像 prob 空间
两端饱和后梯度消失）。权重由调用方（trainer）控制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AUCMarginLoss(nn.Module):
    """Pairwise AUC-margin ranking loss on binary log-odds difference.

    空侧（批内全正或全负）返回 0 损失张量（保梯度图合法），绝不 raise——
    debug 模式关平衡采样器后单类批真实存在（勿重蹈 balanced sampler 单类崩溃）。
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = float(margin)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, 2]；target: [B] 0/1（显式 ==1/==0 掩码，天然剔除任何非 0/1 值）
        scores = logits[:, 1] - logits[:, 0]  # [B] log-odds 差
        pos_mask = target == 1
        neg_mask = target == 0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return logits.new_zeros(())
        diff = self.margin - (scores[pos_mask][:, None] - scores[neg_mask][None, :])
        return F.softplus(diff).mean()
