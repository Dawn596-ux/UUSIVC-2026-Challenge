"""MemoryVideoSegHead — lightweight cross-frame temporal video segmentation head.

Consumes the full ``feat_dict`` (the slot contract of
``TemporalRouter.forward_video_seg_memory``) and produces per-frame seg logits
``[B, T, num_classes, H', W']``.

v1 design (deliberately minimal, <1K extra parameters): a residual *depthwise
temporal-only* 3D convolution mixes ``seg_feat`` across the frame axis (no
spatial mixing), then the existing per-frame ``SegHead2D`` decodes each frame.
This isolates the effect of cross-frame interaction itself; attention-style
aggregation (cf. ``ClassAwareNeighborAttention``) is a possible v2 and is
intentionally not enabled here.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from models.heads.seg_head import SegHead2D


class MemoryVideoSegHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        upsample_scale: int = 1,
        temporal_kernel: int = 3,
    ):
        super().__init__()
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError(f"temporal_kernel must be a positive odd number, got {temporal_kernel}")
        pad = temporal_kernel // 2
        self.temporal_mix = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(temporal_kernel, 1, 1),
                padding=(pad, 0, 0),
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm3d(in_channels),
            nn.GELU(),
        )
        self.frame_head = SegHead2D(
            in_channels=in_channels,
            mid_channels=mid_channels,
            num_classes=num_classes,
            dropout=dropout,
            upsample_scale=upsample_scale,
        )

    def forward(self, feat_dict: Dict[str, Any]) -> torch.Tensor:
        seg_feat = feat_dict["seg_feat"]  # [B, T, C, H, W]
        if seg_feat.ndim != 5:
            raise ValueError(f"MemoryVideoSegHead expects 5D seg_feat [B,T,C,H,W], got {tuple(seg_feat.shape)}")

        b, t, c, h, w = seg_feat.shape
        x = seg_feat.permute(0, 2, 1, 3, 4)      # [B, C, T, H, W]
        x = x + self.temporal_mix(x)              # residual temporal mixing
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        logits = self.frame_head(x)               # [B*T, K, H', W']
        return logits.reshape(b, t, *logits.shape[1:])
