from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2Encoder(nn.Module):
    """DINOv2 (timm) encoder wrapped to match SwinTinyEncoder's 4-stage contract.

    DINOv2 (patch14) outputs a single-resolution grid of patch tokens (16x16 at
    img_size 224). To plug into the existing SwinUNetBackbone (which expects four
    hierarchical skip features at 56/28/14/7 with channels 96/192/384/768), we
    reshape the token grid to [B, dim, 16, 16], interpolate it to each of the four
    resolutions, and map channels with 1x1 convs. The global encoder_out is the
    mean-pooled patch token passed through a Linear(dim -> 768).

    NOTE: the exact shape of timm ``forward_features`` (whether a prefix cls/register
    token is prepended) is resolved defensively here via ``[:, -num_patches:]``, but
    it should still be asserted during the spike step on the target GPU.
    """

    # Swin-Tiny stage channels (kept identical so downstream laterals work unchanged)
    _TARGET_CHANNELS = [96, 192, 384, 768]
    _PATCH_SIZE = 14

    def __init__(
        self,
        variant: str = "vit_small_patch14_dinov2.lvd142m",
        img_size: int = 224,
        in_channels: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"DinoV2Encoder supports in_channels=3, got {in_channels}")

        try:
            import timm
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "timm is required for DINOv2. Install it with `pip install timm`."
            ) from exc

        self.variant = variant
        self.img_size = img_size
        self.num_patches = (img_size // self._PATCH_SIZE) ** 2  # 256 at 224

        # num_classes=0 -> no classifier head; forward_features returns patch tokens
        self.backbone = timm.create_model(variant, pretrained=pretrained, num_classes=0)
        self.dim = self.backbone.embed_dim

        self.stage_channels = list(self._TARGET_CHANNELS)
        self.adaptors = nn.ModuleList(
            [nn.Conv2d(self.dim, ch, kernel_size=1, bias=False) for ch in self.stage_channels]
        )
        self.out_proj = nn.Linear(self.dim, self._TARGET_CHANNELS[-1])

    def _forward_single_image(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        if x.ndim != 4:
            raise ValueError(f"DinoV2Encoder expects [B,C,H,W], got {tuple(x.shape)}")

        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        # timm forward_features returns [B, num_prefix_tokens + num_patches, dim];
        # prefix tokens (cls/register) come first, so keep only the trailing patches.
        tokens = self.backbone.forward_features(x)
        patches = tokens[:, -self.num_patches:]  # [B, 256, dim]
        grid = int(self.num_patches ** 0.5)
        grid_feat = patches.transpose(1, 2).reshape(-1, self.dim, grid, grid)  # [B, dim, 16, 16]

        encoder_out = self.out_proj(patches.mean(dim=1))  # [B, 768]

        skips = []
        for i, ch in enumerate(self.stage_channels):
            h = w = self.img_size // (4 * (2 ** i))  # 56, 28, 14, 7
            up = F.interpolate(grid_feat, size=(h, w), mode="bilinear", align_corners=False)
            skips.append(self.adaptors[i](up))

        return encoder_out, skips, {
            "image_size": self.img_size,
            "stage_channels": self.stage_channels,
        }

    def forward_image(
        self,
        x: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        encoder_out, skips, _ = self._forward_single_image(x)
        return encoder_out, skips

    def forward_video(
        self,
        x: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        if x.ndim != 5:
            raise ValueError(f"DinoV2Encoder expects [B,T,C,H,W], got {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)
        encoder_out, skips, meta = self._forward_single_image(x_flat)
        encoder_out = encoder_out.view(b, t, -1)
        video_skips = [feat.view(b, t, *feat.shape[1:]) for feat in skips]
        meta.update({"batch_size": b, "num_frames": t, "height": h, "width": w})
        return encoder_out, video_skips, meta
