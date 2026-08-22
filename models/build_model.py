import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from models.backbones.dinov2_encoder import DinoV2Encoder
from models.backbones.dual_encoder import DualEncoder
from models.backbones.swin_tiny_encoder import SwinTinyEncoder
from models.backbones.swin_unet_backbone import SwinUNetBackbone
from models.framework.image_model import ImageTaskModel
from models.framework.unified_model import UnifiedModel
from models.framework.video_model import VideoTaskModel
from models.heads.classification_head import ClsHead
from models.heads.seg_head import SegHead2D, SegHeadVideoSimple
from models.heads.temporal_cls_head import TemporalClsHead
from models.task_defs import (
    BUS_IMAGE,
    BUS_VIDEO,
    CEUS_VIDEO,
    IMAGE,
    IMAGE_CLS,
    IMAGE_SEG,
    MODALITY,
    TASK_TYPE,
    VIDEO_CLS,
    VIDEO_SEG,
)
from models.video.temporal_router import TemporalRouter
from utils.checkpoint import load_swin_branch_checkpoint


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict[str, Any]) -> UnifiedModel:
    model_cfg = cfg["model"]
    backbone_cfg = model_cfg.get("backbone", {})

    img_size = backbone_cfg.get("img_size", cfg.get("debug", {}).get("image_size", 224))

    if backbone_cfg.get("name", "swin_tiny") == "dual_dinov2":
        swin_encoder = SwinTinyEncoder(
            img_size=img_size,
            in_channels=model_cfg["in_channels"],
            pretrained=backbone_cfg.get("pretrained", False),
            pretrained_checkpoint=backbone_cfg.get("pretrained_checkpoint", None),
        )
        dino_encoder = DinoV2Encoder(
            variant=backbone_cfg.get("dinov2_variant", "vit_small_patch14_dinov2.lvd142m"),
            img_size=img_size,
            in_channels=model_cfg["in_channels"],
            pretrained=backbone_cfg.get("dinov2_pretrained", True),
        )
        encoder = DualEncoder(swin=swin_encoder, dino=dino_encoder)
        if backbone_cfg.get("swin_branch_checkpoint"):
            load_swin_branch_checkpoint(encoder, backbone_cfg["swin_branch_checkpoint"])
    else:
        encoder = SwinTinyEncoder(
            img_size=img_size,
            in_channels=model_cfg["in_channels"],
            pretrained=backbone_cfg.get("pretrained", False),
            pretrained_checkpoint=backbone_cfg.get("pretrained_checkpoint", None),
        )

    backbone = SwinUNetBackbone(
        encoder=encoder,
        seg_feature_channels=model_cfg["seg_feature_channels"],
        cls_feature_channels=model_cfg["cls_feature_channels"],
        in_channels=model_cfg["in_channels"],
        use_prompt=model_cfg.get("use_prompt", False),
        img_size=backbone_cfg.get("img_size", cfg.get("debug", {}).get("image_size", 224)),
        pretrained=backbone_cfg.get("pretrained", False),
    )

    image_model = ImageTaskModel(
        backbone=backbone,
        seg_head=SegHead2D(
            in_channels=model_cfg["seg_feature_channels"],
            mid_channels=model_cfg.get("seg_head_mid_channels", 128),
            num_classes=model_cfg["num_seg_classes"],
            dropout=model_cfg.get("seg_dropout", 0.1),
            upsample_scale=model_cfg.get("seg_upsample_scale", 1),
        ),
        cls_head=ClsHead(
            in_channels=model_cfg["cls_feature_channels"],
            hidden_channels=model_cfg.get("cls_hidden_channels", 0),
            num_classes=model_cfg["num_cls_classes"],
            dropout=model_cfg.get("cls_dropout", 0.1),
            mode=model_cfg.get("cls_head_mode", "mlp"),
            num_layers=model_cfg.get("cls_mlp_layers", 2),
        ),
        use_prompt=model_cfg.get("use_prompt", False),
    )

    temporal_router = TemporalRouter(
        simple_video_seg_head=SegHeadVideoSimple(
            in_channels=model_cfg["seg_feature_channels"],
            mid_channels=model_cfg.get("seg_head_mid_channels", 128),
            num_classes=model_cfg["num_seg_classes"],
            dropout=model_cfg.get("seg_dropout", 0.1),
            upsample_scale=model_cfg.get("seg_upsample_scale", 1),
        ),
        memory_video_seg_head=None,
        video_cls_head=TemporalClsHead(
            in_channels=model_cfg["cls_feature_channels"],
            num_classes=model_cfg["num_cls_classes"],
            mode=model_cfg.get("temporal_cls_mode", "mean"),
            dropout=model_cfg.get("cls_dropout", 0.1),
            hidden_channels=model_cfg.get("cls_hidden_channels", 0),
            kernel_size=model_cfg.get("neighbor_kernel_size", 3),
            transformer_layers=model_cfg.get("temporal_transformer_layers", 1),
            transformer_heads=model_cfg.get("temporal_transformer_heads", 4),
            mlp_layers=model_cfg.get("cls_mlp_layers", 2),
        ),
        use_memory_seg=model_cfg.get("use_memory_seg", False),
        seg_mode=model_cfg.get("video_seg_mode", "simple"),
        cls_mode=model_cfg.get("video_cls_mode", "mean"),
    )

    video_model = VideoTaskModel(
        backbone=backbone,
        temporal_router=temporal_router,
        use_prompt=model_cfg.get("use_prompt", False),
    )

    return UnifiedModel(image_model=image_model, video_model=video_model)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def build_dummy_batches(cfg: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    model_cfg = cfg["model"]
    debug_cfg = cfg["debug"]

    batch_size = debug_cfg.get("batch_size", 2)
    num_frames = debug_cfg.get("num_frames", 8)
    image_size = debug_cfg.get("image_size", 224)
    in_channels = model_cfg["in_channels"]

    image_tensor = torch.randn(batch_size, in_channels, image_size, image_size)
    video_tensor = torch.randn(batch_size, num_frames, in_channels, image_size, image_size)

    return {
        IMAGE_SEG: {
            IMAGE: image_tensor.clone(),
            TASK_TYPE: IMAGE_SEG,
            MODALITY: BUS_IMAGE,
        },
        IMAGE_CLS: {
            IMAGE: image_tensor.clone(),
            TASK_TYPE: IMAGE_CLS,
            MODALITY: BUS_IMAGE,
        },
        VIDEO_SEG: {
            IMAGE: video_tensor.clone(),
            TASK_TYPE: VIDEO_SEG,
            MODALITY: BUS_VIDEO,
        },
        VIDEO_CLS: {
            IMAGE: video_tensor.clone(),
            TASK_TYPE: VIDEO_CLS,
            MODALITY: CEUS_VIDEO,
        },
    }


def run_model_self_check(model: UnifiedModel, cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    params = count_parameters(model)
    print(f"Device: {device}")
    print(f"Total params: {params['total']:,}")
    print(f"Trainable params: {params['trainable']:,}")

    batch_size = cfg["debug"].get("batch_size", 2)
    num_frames = cfg["debug"].get("num_frames", 8)
    image_size = cfg["debug"].get("image_size", 224)
    num_seg_classes = cfg["model"]["num_seg_classes"]
    num_cls_classes = cfg["model"]["num_cls_classes"]

    model.eval()
    with torch.no_grad():
        for task_name, batch in build_dummy_batches(cfg).items():
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            outputs = model(batch)
            seg_logits = outputs["seg_logits"]
            cls_logits = outputs["cls_logits"]

            if task_name == IMAGE_SEG:
                assert tuple(seg_logits.shape) == (batch_size, num_seg_classes, image_size, image_size)
            elif task_name == IMAGE_CLS:
                assert tuple(cls_logits.shape) == (batch_size, num_cls_classes)
            elif task_name == VIDEO_SEG:
                assert tuple(seg_logits.shape) == (
                    batch_size,
                    num_frames,
                    num_seg_classes,
                    image_size,
                    image_size,
                )
            elif task_name == VIDEO_CLS:
                assert tuple(cls_logits.shape) == (batch_size, num_cls_classes)

            print(
                f"[OK] {task_name}: "
                f"seg={None if seg_logits is None else tuple(seg_logits.shape)}, "
                f"cls={None if cls_logits is None else tuple(cls_logits.shape)}"
            )


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().parents[1] / "configs" / "stage2_cls.yaml"
    parser = argparse.ArgumentParser(description="Build baseline model and run self-check")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to model config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    model = build_model(cfg)
    run_model_self_check(model, cfg)


if __name__ == "__main__":
    main()
