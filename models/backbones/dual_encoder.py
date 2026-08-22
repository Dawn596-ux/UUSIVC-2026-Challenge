from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class DualEncoder(nn.Module):
    """Parallel Swin + DINOv2 encoder with feature-level fusion.

    Runs both encoders and fuses their global tokens and four hierarchical skip
    features via concat -> Linear / 1x1 conv, preserving the SwinUNetBackbone
    contract (``stage_channels`` + ``forward_image``/``forward_video``). Downstream
    lateral/smooth/classification branches and heads are untouched.
    """

    def __init__(self, swin: nn.Module, dino: nn.Module):
        super().__init__()
        self.swin = swin
        self.dino = dino
        self.stage_channels = list(self.swin.stage_channels)

        out_dim = self.stage_channels[-1]  # 768
        self.out_fuse = nn.Linear(out_dim + out_dim, out_dim)
        self.skip_fuses = nn.ModuleList(
            [nn.Conv2d(ch + ch, ch, kernel_size=1, bias=False) for ch in self.stage_channels]
        )

    def _fuse_image_skips(self, swin_skips, dino_skips):
        return [
            self.skip_fuses[i](torch.cat([s, d], dim=1))
            for i, (s, d) in enumerate(zip(swin_skips, dino_skips))
        ]

    def forward_image(
        self,
        x: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        swin_out, swin_skips = self.swin.forward_image(x, prompts=prompts)
        dino_out, dino_skips = self.dino.forward_image(x, prompts=prompts)
        fused_out = self.out_fuse(torch.cat([swin_out, dino_out], dim=-1))
        return fused_out, self._fuse_image_skips(swin_skips, dino_skips)

    def forward_video(
        self,
        x: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        swin_out, swin_skips, meta = self.swin.forward_video(x, prompts=prompts)
        dino_out, dino_skips, _ = self.dino.forward_video(x, prompts=prompts)

        b, t = swin_out.shape[:2]
        fused_out = self.out_fuse(torch.cat([swin_out, dino_out], dim=-1))  # [B, T, 768]

        fused_skips = []
        for i, (s, d) in enumerate(zip(swin_skips, dino_skips)):
            # s, d: [B, T, C_i, H_i, W_i] -> flatten time into batch for the 2D conv
            b_t = s.shape[0] * s.shape[1]
            s_flat = s.reshape(b_t, s.shape[2], s.shape[3], s.shape[4])
            d_flat = d.reshape(b_t, d.shape[2], d.shape[3], d.shape[4])
            fused = self.skip_fuses[i](torch.cat([s_flat, d_flat], dim=1))
            fused_skips.append(fused.reshape(b, t, *fused.shape[1:]))

        return fused_out, fused_skips, meta
