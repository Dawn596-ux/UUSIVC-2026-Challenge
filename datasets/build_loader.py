# datasets/build_loader.py

from typing import Dict, Any, Tuple, Optional

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.uusivc2026_paths import expand_fixed_uusivc_data_cfg
from datasets.transforms import BasicImageTransform, BasicVideoTransform, build_augmentation

from datasets import (
    BUS_VIDEO,
    CEUS_VIDEO,
    USImageClsDataset,
    USImageSegDataset,
    USVideoClsDataset,
    USVideoSegDataset,
    collate_image_task_batch,
    collate_video_task_batch,
)


def build_balanced_sampler_from_cls_dataset(dataset) -> WeightedRandomSampler:
    """
    The organizers keep this baseline note in English for public release.
    - USImageClsDataset
    - USVideoClsDataset

    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
        image_path label
    The organizers keep this baseline note in English for public release.
        frames_dir label
    """
    labels = []
    for sample in dataset.samples:
        if isinstance(sample, (tuple, list)):
            label = int(sample[-1])
        else:
            parts = str(sample).split()
            label = int(parts[-1])
        labels.append(label)

    labels = np.array(labels, dtype=np.int64)
    class_count = np.bincount(labels)

    if len(class_count) < 2:
        raise ValueError(
            f"Classification dataset must contain at least 2 classes, got class_count={class_count}"
        )

    class_weight = 1.0 / np.maximum(class_count, 1)
    sample_weights = class_weight[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def build_image_seg_loader(
    list_file: str,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    transform=None,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> Tuple[USImageSegDataset, DataLoader]:
    dataset = USImageSegDataset(
        list_file=list_file,
        dataset_name=dataset_name,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_image_task_batch,
    )
    return dataset, loader


def build_image_cls_loader(
    list_file: str,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    transform=None,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    use_balanced_sampler: bool = False,
) -> Tuple[USImageClsDataset, DataLoader]:
    dataset = USImageClsDataset(
        list_file=list_file,
        dataset_name=dataset_name,
        transform=transform,
    )

    sampler = None
    if use_balanced_sampler:
        sampler = build_balanced_sampler_from_cls_dataset(dataset)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_image_task_batch,
    )
    return dataset, loader


def build_video_seg_loader(
    list_file: str,
    dataset_name: str,
    modality: str,
    batch_size: int,
    num_workers: int,
    num_frames: int = 8,
    transform=None,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    ceus_dualview_fusion: bool = True,
    ceus_fusion_mode: str = "baseline_difference",
    ceus_midline_offset: int = 0,
    ceus_crop_top_ratio: float = 0.10,
    ceus_crop_bottom_ratio: float = 0.04,
    ceus_crop_side_ratio: float = 0.04,
    ceus_crop_center_ratio: float = 0.02,
    ceus_gt_side_mode: str = "auto",
    ceus_baseline_frames: int = 5,
) -> Tuple[USVideoSegDataset, DataLoader]:
    dataset = USVideoSegDataset(
        list_file=list_file,
        dataset_name=dataset_name,
        modality=modality,
        num_frames=num_frames,
        transform=transform,
        ceus_dualview_fusion=ceus_dualview_fusion,
        ceus_fusion_mode=ceus_fusion_mode,
        ceus_midline_offset=ceus_midline_offset,
        ceus_crop_top_ratio=ceus_crop_top_ratio,
        ceus_crop_bottom_ratio=ceus_crop_bottom_ratio,
        ceus_crop_side_ratio=ceus_crop_side_ratio,
        ceus_crop_center_ratio=ceus_crop_center_ratio,
        ceus_gt_side_mode=ceus_gt_side_mode,
        ceus_baseline_frames=ceus_baseline_frames,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_video_task_batch,
    )
    return dataset, loader


def build_video_cls_loader(
    list_file: str,
    dataset_name: str,
    modality: str,
    batch_size: int,
    num_workers: int,
    num_frames: int = 8,
    transform=None,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    use_balanced_sampler: bool = False,
) -> Tuple[USVideoClsDataset, DataLoader]:
    dataset = USVideoClsDataset(
        list_file=list_file,
        dataset_name=dataset_name,
        modality=modality,
        num_frames=num_frames,
        transform=transform,
    )

    sampler = None
    if use_balanced_sampler:
        sampler = build_balanced_sampler_from_cls_dataset(dataset)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_video_task_batch,
    )
    return dataset, loader


def build_stage1_seg_loaders(cfg: Dict[str, Any]) -> Dict[str, Dict[str, DataLoader]]:
    cfg = expand_fixed_uusivc_data_cfg(cfg)
    """
    Stage 1: segmentation only

    The organizers keep this baseline note in English for public release.
    {
        "num_workers": 4,
        "num_frames": 8,
        "batch_size_image_seg": 8,
        "batch_size_video_seg": 2,

        "image_seg_train": {"dataset_name": "...", "list_file": "..."},
        "image_seg_val": {"dataset_name": "...", "list_file": "..."},

        "cardiac_video_seg_train": {"dataset_name": "...", "list_file": "..."},
        "cardiac_video_seg_val": {"dataset_name": "...", "list_file": "..."},

        "ceus_video_seg_train": {"dataset_name": "...", "list_file": "..."},
        "ceus_video_seg_val": {"dataset_name": "...", "list_file": "..."},
    }
    """
    loaders = {"train": {}, "val": {}}

    num_workers = cfg.get("num_workers", 4)
    num_frames = cfg.get("num_frames", 8)
    batch_size_image_seg = cfg.get("batch_size_image_seg", 8)
    batch_size_video_seg = cfg.get("batch_size_video_seg", 2)

    image_seg_train_tf = cfg.get("image_seg_train_transform", None)
    image_seg_val_tf = cfg.get("image_seg_val_transform", None)
    video_seg_train_tf = cfg.get("video_seg_train_transform", None)
    video_seg_val_tf = cfg.get("video_seg_val_transform", None)

    # 数据增强：仅训练集增强，验证/推理不增强。
    # 视频任务（cardiac/ceus）帧一致性增强暂缓，避免帧间抖动，先只对 image_seg 增强。
    if cfg.get("augmentation", False):
        aug = build_augmentation(cfg.get("aug_extra", "none"))
        image_seg_train_tf = BasicImageTransform((224, 224), binary_mask=False, augmentation=aug)
    ceus_dualview_fusion = cfg.get("ceus_dualview_fusion", True)
    ceus_fusion_mode = cfg.get("ceus_fusion_mode", "baseline_difference")
    ceus_midline_offset = cfg.get("ceus_midline_offset", 0)
    ceus_crop_top_ratio = cfg.get("ceus_crop_top_ratio", 0.10)
    ceus_crop_bottom_ratio = cfg.get("ceus_crop_bottom_ratio", 0.04)
    ceus_crop_side_ratio = cfg.get("ceus_crop_side_ratio", 0.04)
    ceus_crop_center_ratio = cfg.get("ceus_crop_center_ratio", 0.02)
    ceus_gt_side_mode = cfg.get("ceus_gt_side_mode", "auto")
    ceus_baseline_frames = cfg.get("ceus_baseline_frames", 5)

    # image segmentation
    if "image_seg_train" in cfg:
        _, loaders["train"]["image_seg"] = build_image_seg_loader(
            list_file=cfg["image_seg_train"]["list_file"],
            dataset_name=cfg["image_seg_train"]["dataset_name"],
            batch_size=batch_size_image_seg,
            num_workers=num_workers,
            transform=image_seg_train_tf,
            shuffle=True,
            drop_last=True,
        )

    if "image_seg_val" in cfg:
        _, loaders["val"]["image_seg"] = build_image_seg_loader(
            list_file=cfg["image_seg_val"]["list_file"],
            dataset_name=cfg["image_seg_val"]["dataset_name"],
            batch_size=batch_size_image_seg,
            num_workers=num_workers,
            transform=image_seg_val_tf,
            shuffle=False,
            drop_last=False,
        )

    # cardiac video segmentation
    if "cardiac_video_seg_train" in cfg:
        _, loaders["train"]["cardiac_video_seg"] = build_video_seg_loader(
            list_file=cfg["cardiac_video_seg_train"]["list_file"],
            dataset_name=cfg["cardiac_video_seg_train"]["dataset_name"],
            modality=BUS_VIDEO,
            batch_size=batch_size_video_seg,
            num_workers=num_workers,
            num_frames=num_frames,
            transform=video_seg_train_tf,
            shuffle=True,
            drop_last=True,
        )

    if "cardiac_video_seg_val" in cfg:
        _, loaders["val"]["cardiac_video_seg"] = build_video_seg_loader(
            list_file=cfg["cardiac_video_seg_val"]["list_file"],
            dataset_name=cfg["cardiac_video_seg_val"]["dataset_name"],
            modality=BUS_VIDEO,
            batch_size=batch_size_video_seg,
            num_workers=num_workers,
            num_frames=num_frames,
            transform=video_seg_val_tf,
            shuffle=False,
            drop_last=False,
        )

    # ceus video segmentation
    if "ceus_video_seg_train" in cfg:
        _, loaders["train"]["video_seg_ceus"] = build_video_seg_loader(
            list_file=cfg["ceus_video_seg_train"]["list_file"],
            dataset_name=cfg["ceus_video_seg_train"]["dataset_name"],
            modality=CEUS_VIDEO,
            batch_size=batch_size_video_seg,
            num_workers=num_workers,
            num_frames=num_frames,
            transform=video_seg_train_tf,
            shuffle=True,
            drop_last=True,
            ceus_dualview_fusion=ceus_dualview_fusion,
            ceus_fusion_mode=ceus_fusion_mode,
            ceus_midline_offset=ceus_midline_offset,
            ceus_crop_top_ratio=ceus_crop_top_ratio,
            ceus_crop_bottom_ratio=ceus_crop_bottom_ratio,
            ceus_crop_side_ratio=ceus_crop_side_ratio,
            ceus_crop_center_ratio=ceus_crop_center_ratio,
            ceus_gt_side_mode=ceus_gt_side_mode,
            ceus_baseline_frames=ceus_baseline_frames,
        )

    if "ceus_video_seg_val" in cfg:
        _, loaders["val"]["video_seg_ceus"] = build_video_seg_loader(
            list_file=cfg["ceus_video_seg_val"]["list_file"],
            dataset_name=cfg["ceus_video_seg_val"]["dataset_name"],
            modality=CEUS_VIDEO,
            batch_size=batch_size_video_seg,
            num_workers=num_workers,
            num_frames=num_frames,
            transform=video_seg_val_tf,
            shuffle=False,
            drop_last=False,
            ceus_dualview_fusion=ceus_dualview_fusion,
            ceus_fusion_mode=ceus_fusion_mode,
            ceus_midline_offset=ceus_midline_offset,
            ceus_crop_top_ratio=ceus_crop_top_ratio,
            ceus_crop_bottom_ratio=ceus_crop_bottom_ratio,
            ceus_crop_side_ratio=ceus_crop_side_ratio,
            ceus_crop_center_ratio=ceus_crop_center_ratio,
            ceus_gt_side_mode=ceus_gt_side_mode,
            ceus_baseline_frames=ceus_baseline_frames,
        )

    return loaders


def build_stage2_cls_loaders(cfg: Dict[str, Any]) -> Dict[str, Dict[str, DataLoader]]:
    cfg = expand_fixed_uusivc_data_cfg(cfg)
    """
    Stage 2: classification only

    The organizers keep this baseline note in English for public release.
    {
        "num_workers": 4,
        "num_frames": 8,
        "batch_size_image_cls": 16,
        "batch_size_video_cls": 4,
        "use_balanced_sampler_image_cls": True,
        "use_balanced_sampler_video_cls": True,

        "image_cls_train": {"dataset_name": "...", "list_file": "..."},
        "image_cls_val": {"dataset_name": "...", "list_file": "..."},

        "standard_video_cls_train": {"dataset_name": "...", "list_file": "..."},
        "standard_video_cls_val": {"dataset_name": "...", "list_file": "..."},

        "ceus_video_cls_train": {"dataset_name": "...", "list_file": "..."},
        "ceus_video_cls_val": {"dataset_name": "...", "list_file": "..."},
    }
    """
    loaders = {"train": {}, "val": {}}

    num_workers = cfg.get("num_workers", 4)
    num_frames = cfg.get("num_frames", 8)
    batch_size_image_cls = cfg.get("batch_size_image_cls", 16)
    batch_size_video_cls = cfg.get("batch_size_video_cls", 4)

    use_balanced_sampler_image_cls = cfg.get("use_balanced_sampler_image_cls", False)
    use_balanced_sampler_video_cls = cfg.get("use_balanced_sampler_video_cls", False)

    image_cls_train_tf = cfg.get("image_cls_train_transform", None)
    image_cls_val_tf = cfg.get("image_cls_val_transform", None)
    video_cls_train_tf = cfg.get("video_cls_train_transform", None)
    video_cls_val_tf = cfg.get("video_cls_val_transform", None)

    # image classification
    if "image_cls_train" in cfg:
        _, loaders["train"]["image_cls"] = build_image_cls_loader(
            list_file=cfg["image_cls_train"]["list_file"],
            dataset_name=cfg["image_cls_train"]["dataset_name"],
            batch_size=batch_size_image_cls,
            num_workers=num_workers,
            transform=image_cls_train_tf,
            shuffle=not use_balanced_sampler_image_cls,
            drop_last=True,
            use_balanced_sampler=use_balanced_sampler_image_cls,
        )

    if "image_cls_val" in cfg:
        _, loaders["val"]["image_cls"] = build_image_cls_loader(
            list_file=cfg["image_cls_val"]["list_file"],
            dataset_name=cfg["image_cls_val"]["dataset_name"],
            batch_size=batch_size_image_cls,
            num_workers=num_workers,
            transform=image_cls_val_tf,
            shuffle=False,
            drop_last=False,
            use_balanced_sampler=False,
        )

    # standard video classification
    if "standard_video_cls_train" in cfg:
        _, loaders["train"]["standard_video_cls"] = build_video_cls_loader(
            list_file=cfg["standard_video_cls_train"]["list_file"],
            dataset_name=cfg["standard_video_cls_train"]["dataset_name"],
            modality=BUS_VIDEO,
            batch_size=batch_size_video_cls,
            num_workers=num_workers,
            num_frames=num_frames,
            transform=video_cls_train_tf,
            shuffle=not use_balanced_sampler_video_cls,
            drop_last=True,
            use_balanced_sampler=use_balanced_sampler_video_cls,
        )

    if "standard_video_cls_val" in cfg:
        _, loaders["val"]["standard_video_cls"] = build_video_cls_loader(
            list_file=cfg["standard_video_cls_val"]["list_file"],
            dataset_name=cfg["standard_video_cls_val"]["dataset_name"],
            modality=BUS_VIDEO,
            batch_size=batch_size_video_cls,
            num_workers=num_workers,
            num_frames=num_frames,
            transform=video_cls_val_tf,
            shuffle=False,
            drop_last=False,
            use_balanced_sampler=False,
        )

    # ceus video classification
    if "ceus_video_cls_train" in cfg:
        _, loaders["train"]["video_cls_ceus"] = build_video_cls_loader(
            list_file=cfg["ceus_video_cls_train"]["list_file"],
            dataset_name=cfg["ceus_video_cls_train"]["dataset_name"],
            modality=CEUS_VIDEO,
            batch_size=batch_size_video_cls,
            num_workers=num_workers,
            num_frames=num_frames,
            transform=video_cls_train_tf,
            shuffle=not use_balanced_sampler_video_cls,
            drop_last=True,
            use_balanced_sampler=use_balanced_sampler_video_cls,
        )

    if "ceus_video_cls_val" in cfg:
        _, loaders["val"]["video_cls_ceus"] = build_video_cls_loader(
            list_file=cfg["ceus_video_cls_val"]["list_file"],
            dataset_name=cfg["ceus_video_cls_val"]["dataset_name"],
            modality=CEUS_VIDEO,
            batch_size=batch_size_video_cls,
            num_workers=num_workers,
            num_frames=num_frames,
            transform=video_cls_val_tf,
            shuffle=False,
            drop_last=False,
            use_balanced_sampler=False,
        )

    return loaders


def build_joint_train_loaders(cfg: Dict[str, Any]) -> Dict[str, Dict[str, DataLoader]]:
    """
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    """
    seg_loaders = build_stage1_seg_loaders(cfg)
    cls_loaders = build_stage2_cls_loaders(cfg)

    loaders = {"train": {}, "val": {}}
    loaders["train"].update(seg_loaders["train"])
    loaders["train"].update(cls_loaders["train"])
    loaders["val"].update(seg_loaders["val"])
    loaders["val"].update(cls_loaders["val"])
    return loaders
