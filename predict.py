"""UUSIVC2026 model submission entry.

This script loads one full-model checkpoint, runs inference on the fixed
challenge split, and writes the official submission directory and zip file.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from datasets.task_defs import (
    BUS_IMAGE,
    BUS_VIDEO,
    CASE_ID,
    CEUS_VIDEO,
    DATASET_NAME,
    IMAGE,
    IMAGE_CLS,
    IMAGE_SEG,
    IS_VIDEO,
    LABEL_CLS,
    LABEL_SEG,
    MODALITY,
    NUM_FRAMES,
    PATIENT_ID,
    TASK_TYPE,
    VIDEO_CLS,
    VIDEO_SEG,
    IGNORE_CLS_LABEL,
    IGNORE_SEG_LABEL,
)
from datasets.transforms import BasicImageTransform, BasicVideoTransform
from datasets.us_video_cls_dataset import USVideoClsDataset
from datasets.us_video_seg_dataset import USVideoSegDataset
from datasets.uusivc2026_paths import (
    TASK_DATASET_NAME,
    normalize_rel,
    phase_data_root,
    phase_json_path,
    resolve_data_root,
)
from models.build_model import build_model
from utils.checkpoint import resolve_checkpoint_reference
from utils.metrics import restore_ceus_prediction_to_original


SEG_TASKS = {"image_seg", "ceus_seg", "video_seg"}
CLS_TASKS = {"image_cls", "ceus_cls"}


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_submission_checkpoint(cfg: Dict[str, Any], explicit_path: Optional[str], which: str) -> str:
    if explicit_path:
        return resolve_checkpoint_reference(explicit_path, prefix=None)

    prefix = "best_stage2_cls"
    latest_name = "latest_stage2_cls.pth"
    save_dir = Path(cfg["trainer"].get("save_dir", "outputs/stage2_cls"))
    filename = f"{prefix}.pth" if which == "best" else latest_name
    path = save_dir / filename
    try:
        if not path.exists() and which == "best":
            return resolve_checkpoint_reference(save_dir, prefix=prefix)
        if path.exists():
            return str(path.resolve())
    except FileNotFoundError:
        pass

    fallback_root = Path("outputs")
    if fallback_root.exists():
        patterns = [f"{prefix}_rank*.pth"] if which == "best" else [latest_name]
        candidates = []
        for pattern in patterns:
            candidates.extend(fallback_root.rglob(pattern))
        if candidates:
            candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
            return str(candidates[0].resolve())

    raise FileNotFoundError("Checkpoint not found. Pass --checkpoint explicitly.")


def load_model(cfg: Dict[str, Any], checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def output_rel(entry: Dict[str, Any]) -> Path:
    if entry.get("target_path_relative"):
        return Path(normalize_rel(entry["target_path_relative"]))
    target_name = entry.get("target_name") or entry.get("mask_name") or entry.get("annotation_name")
    if target_name:
        folder = "masks" if entry["task"] == "image_seg" else "annotations"
        return Path(entry["task"]) / entry["dataset_name"] / folder / target_name
    rel = normalize_rel(entry["input_path_relative"])
    rel = rel.replace("/imgs/", "/masks/").replace("/videos/", "/annotations/")
    rel = rel.replace("seg_img_", "seg_mask_").replace("seg_video_", "seg_annotation_")
    if entry["task"] in {"ceus_seg", "video_seg"}:
        rel = os.path.splitext(rel)[0] + ".npz"
    return Path(rel)


def image_hw(entry: Dict[str, Any], phase_root: Path) -> Tuple[int, int]:
    dims = entry.get("input_dimensions") or entry.get("img_dimensions")
    if isinstance(dims, Sequence) and len(dims) >= 2:
        return int(dims[1]), int(dims[0])
    with Image.open(phase_root / normalize_rel(entry["input_path_relative"])) as img:
        w, h = img.size
    return h, w


def video_hw(entry: Dict[str, Any], phase_root: Path) -> Tuple[int, int]:
    shape = entry.get("video_shape") or entry.get("input_dimensions")
    if isinstance(shape, Sequence) and len(shape) >= 3:
        if len(shape) >= 4 and int(shape[0]) in {1, 3}:
            return int(shape[-2]), int(shape[-1])
        return int(shape[-3]), int(shape[-2])
    arr = np.load(phase_root / normalize_rel(entry["input_path_relative"]), mmap_mode="r")
    try:
        if arr.ndim >= 4 and arr.shape[0] in {1, 3}:
            return int(arr.shape[-2]), int(arr.shape[-1])
        return int(arr.shape[-3]), int(arr.shape[-2])
    finally:
        del arr


def class_count(entry: Dict[str, Any]) -> int:
    config = entry.get("class_config") or {}
    max_id = 1
    if config:
        max_id = max(max_id, max(int(k) for k in config.keys()))
    return max_id + 1


def classification_key(entry: Dict[str, Any]) -> str:
    sample_id = entry.get("sample_id")
    if not sample_id:
        raise KeyError("Classification entries must contain sample_id for the official submission format.")
    return str(sample_id)


def resize_binary(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    resized = Image.fromarray(mask).resize((w, h), Image.Resampling.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8) * 255


def _flip_batch_image(batch: Dict[str, Any]) -> Dict[str, Any]:
    """返回水平翻转 image 的 batch 副本（用于 TTA）。

    超声图像左右对称，水平翻转（W 维，dim=-1）是安全的几何不变性。
    """
    flipped = dict(batch)
    flipped[IMAGE] = torch.flip(batch[IMAGE], dims=[-1])
    return flipped


def _seg_tta_prob(model: torch.nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    """单模型 image_seg 概率：softmax(x) + flip(softmax(flip(x)), W)（与现役 TTA 表达式逐字符一致）。"""
    logits = model(batch)["seg_logits"]
    logits_flip = model(_flip_batch_image(batch))["seg_logits"]
    return torch.softmax(logits[0], dim=0) + torch.softmax(logits_flip[0], dim=0).flip(-1)


def _cls_member_prob(model: torch.nn.Module, batch: Dict[str, Any], n: int) -> np.ndarray:
    """单成员 cls 概率：先截断到 n 类再各自归一（与现役逐模型行为一致）。"""
    probs = torch.softmax(model(batch)["cls_logits"], dim=1)[0].detach().cpu().numpy()
    probs = probs[:n]
    return probs / max(float(probs.sum()), 1e-12)


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def make_seg_batch(image_tensor: torch.Tensor, task: str, modality: str, dataset_name: str, case_id: str, device: torch.device) -> Dict[str, Any]:
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
        is_video = False
        task_type = IMAGE_SEG
        num_frames = 1
    else:
        image_tensor = image_tensor.unsqueeze(0)
        is_video = True
        task_type = VIDEO_SEG
        num_frames = image_tensor.shape[1]
    return {
        IMAGE: image_tensor.to(device),
        LABEL_CLS: torch.tensor([IGNORE_CLS_LABEL], dtype=torch.long, device=device),
        LABEL_SEG: None,
        TASK_TYPE: [task_type],
        MODALITY: [modality],
        DATASET_NAME: [dataset_name],
        IS_VIDEO: torch.tensor([is_video], dtype=torch.bool, device=device),
        NUM_FRAMES: torch.tensor([num_frames], dtype=torch.long, device=device),
        PATIENT_ID: [None],
        CASE_ID: [case_id],
    }


def make_cls_batch(image_tensor: torch.Tensor, task: str, modality: str, dataset_name: str, case_id: str, device: torch.device) -> Dict[str, Any]:
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
        is_video = False
        task_type = IMAGE_CLS
        num_frames = 1
    else:
        image_tensor = image_tensor.unsqueeze(0)
        is_video = True
        task_type = VIDEO_CLS
        num_frames = image_tensor.shape[1]
    return {
        IMAGE: image_tensor.to(device),
        LABEL_CLS: torch.tensor([0], dtype=torch.long, device=device),
        LABEL_SEG: IGNORE_SEG_LABEL,
        TASK_TYPE: [task_type],
        MODALITY: [modality],
        DATASET_NAME: [dataset_name],
        IS_VIDEO: torch.tensor([is_video], dtype=torch.bool, device=device),
        NUM_FRAMES: torch.tensor([num_frames], dtype=torch.long, device=device),
        PATIENT_ID: [None],
        CASE_ID: [case_id],
    }


def build_ceus_processor(data_cfg: Dict[str, Any]) -> USVideoSegDataset:
    processor = USVideoSegDataset.__new__(USVideoSegDataset)
    processor.modality = CEUS_VIDEO
    processor.num_frames = int(data_cfg.get("num_frames", 10))
    processor.ceus_dualview_fusion = bool(data_cfg.get("ceus_dualview_fusion", True))
    processor.ceus_fusion_mode = data_cfg.get("ceus_fusion_mode", "baseline_difference")
    processor.ceus_midline_offset = int(data_cfg.get("ceus_midline_offset", 0))
    processor.ceus_crop_top_ratio = float(data_cfg.get("ceus_crop_top_ratio", 0.10))
    processor.ceus_crop_bottom_ratio = float(data_cfg.get("ceus_crop_bottom_ratio", 0.04))
    processor.ceus_crop_side_ratio = float(data_cfg.get("ceus_crop_side_ratio", 0.04))
    processor.ceus_crop_center_ratio = float(data_cfg.get("ceus_crop_center_ratio", 0.02))
    processor.ceus_gt_side_mode = "auto"
    processor.ceus_baseline_frames = int(data_cfg.get("ceus_baseline_frames", 5))
    processor.transform = BasicVideoTransform((224, 224), binary_mask=True)
    return processor


def load_video_npy(path: Path) -> np.ndarray:
    return USVideoClsDataset._to_thwc(np.load(path, allow_pickle=True))


def sampled_frames(video: np.ndarray, num_frames: int) -> List[np.ndarray]:
    if video.shape[0] <= 0:
        raise ValueError("Video contains no frames.")
    if video.shape[0] >= num_frames:
        indices = np.linspace(0, video.shape[0] - 1, num_frames).astype(int).tolist()
    else:
        indices = list(range(video.shape[0]))
        while len(indices) < num_frames:
            indices.append(indices[-1])
    return [video[i] for i in indices]


@torch.no_grad()
def predict_image_seg(model: torch.nn.Module, entry: Dict[str, Any], phase_root: Path, out_dir: Path, device: torch.device, extra_models: Optional[List[Tuple[torch.nn.Module, float, bool]]] = None) -> None:
    path = phase_root / normalize_rel(entry["input_path_relative"])
    image = read_image(path)
    tensor, _ = BasicImageTransform((224, 224), binary_mask=False)(image, None)
    batch = make_seg_batch(tensor, entry["task"], BUS_IMAGE, TASK_DATASET_NAME["image_seg"], path.stem, device)
    if extra_models:
        # 概率级集成：P = [1.0·TTA(主模型) + Σ w_i·TTA_i] / (1 + Σ w_i)；argmax 在平均之后、resize 之前，严禁平均 mask。
        denom = 1.0 + float(sum(weight for _, weight, _ in extra_models))
        prob = _seg_tta_prob(model, batch)
        for extra_model, weight, use_tta in extra_models:
            member = _seg_tta_prob(extra_model, batch) if use_tta else torch.softmax(extra_model(batch)["seg_logits"][0], dim=0)
            prob = prob + member * float(weight)
        pred = torch.argmax(prob / denom, dim=0).detach().cpu().numpy()
    else:
        logits = model(batch)["seg_logits"]
        logits_flip = model(_flip_batch_image(batch))["seg_logits"]
        prob = torch.softmax(logits[0], dim=0) + torch.softmax(logits_flip[0], dim=0).flip(-1)
        pred = torch.argmax(prob, dim=0).detach().cpu().numpy()
    mask = resize_binary(pred, image_hw(entry, phase_root))
    out_path = out_dir / output_rel(entry)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)


@torch.no_grad()
def predict_video_seg(model: torch.nn.Module, entry: Dict[str, Any], phase_root: Path, out_dir: Path, device: torch.device, num_frames: int) -> None:
    path = phase_root / normalize_rel(entry["input_path_relative"])
    video = load_video_npy(path)
    frames = sampled_frames(video, num_frames)
    tensor, _ = BasicVideoTransform((224, 224), binary_mask=True)(frames, None)
    batch = make_seg_batch(tensor, entry["task"], BUS_VIDEO, TASK_DATASET_NAME["video_seg"], path.stem, device)
    logits = model(batch)["seg_logits"]
    pred = torch.argmax(logits[0], dim=1).detach().cpu().numpy()
    frame_indices = [str(x) for x in (entry.get("frame_indices") or range(video.shape[0]))]
    masks = {}
    for key in frame_indices:
        try:
            frame_number = int(key)
            nearest = int(round(frame_number * max(num_frames - 1, 0) / max(video.shape[0] - 1, 1)))
        except ValueError:
            nearest = len(masks) * max(num_frames - 1, 0) // max(len(frame_indices) - 1, 1)
        nearest = max(0, min(pred.shape[0] - 1, nearest))
        masks[key] = resize_binary(pred[nearest], video_hw(entry, phase_root))
    out_path = out_dir / output_rel(entry)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, fnum_mask=masks)


@torch.no_grad()
def predict_ceus_seg(model: torch.nn.Module, entry: Dict[str, Any], phase_root: Path, out_dir: Path, device: torch.device, processor: USVideoSegDataset) -> None:
    path = phase_root / normalize_rel(entry["input_path_relative"])
    video = USVideoSegDataset._load_video_npy(str(path))
    frames = sampled_frames(video, processor.num_frames)
    dummy_masks = [np.zeros(frame.shape[:2], dtype=np.uint8) for frame in frames]
    ceus_side = processor._resolve_ceus_side_from_content(frames)
    side = "left" if ceus_side == "right" else "right"
    fused_frames, _ = processor._fuse_sampled_ceus_video_and_mask(frames, dummy_masks, side)
    tensor, _ = processor.transform(fused_frames, None)
    batch = make_seg_batch(tensor, entry["task"], CEUS_VIDEO, TASK_DATASET_NAME["ceus_seg"], path.stem, device)
    logits = model(batch)["seg_logits"]
    frame_idx = int(logits.shape[1] // 2)
    pred = torch.argmax(logits[0, frame_idx], dim=0).detach().cpu().numpy()
    meta = processor._build_ceus_restore_meta(frames[0].shape, side)
    mask = restore_ceus_prediction_to_original(pred, meta)
    out_path = out_dir / output_rel(entry)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, mask=mask)


@torch.no_grad()
def predict_image_cls(model: torch.nn.Module, entry: Dict[str, Any], phase_root: Path, device: torch.device, extra_models: Optional[List[Tuple[torch.nn.Module, float, bool]]] = None) -> Dict[str, Any]:
    path = phase_root / normalize_rel(entry["input_path_relative"])
    image = read_image(path)
    tensor, _ = BasicImageTransform((224, 224), binary_mask=True)(image, None)
    batch = make_cls_batch(tensor, entry["task"], BUS_IMAGE, TASK_DATASET_NAME["image_cls"], path.stem, device)
    n = class_count(entry)
    if extra_models:
        # 概率级集成：p = [1.0·renorm(softmax_主[:n]) + Σ w_i·renorm(softmax_i[:n])] / (1 + Σ w_i)；截断在平均之前逐成员做。
        probs = _cls_member_prob(model, batch, n)
        denom = 1.0
        for extra_model, weight, _use_tta in extra_models:
            probs = probs + _cls_member_prob(extra_model, batch, n) * float(weight)
            denom += float(weight)
        probs = probs / denom
    else:
        probs = torch.softmax(model(batch)["cls_logits"], dim=1)[0].detach().cpu().numpy()
        probs = probs[:n]
        probs = probs / max(float(probs.sum()), 1e-12)
    return {"prediction": int(np.argmax(probs)), "probability": [float(x) for x in probs]}


@torch.no_grad()
def predict_ceus_cls(model: torch.nn.Module, entry: Dict[str, Any], phase_root: Path, device: torch.device, num_frames: int, extra_models: Optional[List[Tuple[torch.nn.Module, float, bool]]] = None) -> Dict[str, Any]:
    path = phase_root / normalize_rel(entry["input_path_relative"])
    video = load_video_npy(path)
    frames = sampled_frames(video, num_frames)
    tensor, _ = BasicVideoTransform((224, 224), binary_mask=True)(frames, None)
    batch = make_cls_batch(tensor, entry["task"], CEUS_VIDEO, TASK_DATASET_NAME["ceus_cls"], path.stem, device)
    n = class_count(entry)
    if extra_models:
        # 概率级集成：与 predict_image_cls 同式；截断在平均之前逐成员做。
        probs = _cls_member_prob(model, batch, n)
        denom = 1.0
        for extra_model, weight, _use_tta in extra_models:
            probs = probs + _cls_member_prob(extra_model, batch, n) * float(weight)
            denom += float(weight)
        probs = probs / denom
    else:
        probs = torch.softmax(model(batch)["cls_logits"], dim=1)[0].detach().cpu().numpy()
        probs = probs[:n]
        probs = probs / max(float(probs.sum()), 1e-12)
    return {"prediction": int(np.argmax(probs)), "probability": [float(x) for x in probs]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a model-based UUSIVC2026 submission.")
    parser.add_argument("--competition-root", type=str, default=None, help="Optional competition root. If --data-root is omitted, the resolver searches nearby data roots.")
    parser.add_argument("--data-root", type=str, default=None, help="UUSIVC2026 public data root containing TRAIN/ and VAL/ packages.")
    parser.add_argument("--phase", choices=["val", "test"], default="val", help="Submission phase to generate. Public releases provide val; test is for platform/internal use.")
    parser.add_argument("--config", type=str, default="configs/stage2_cls.yaml", help="Model config path. Defaults to the final Stage-2 config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Full-model checkpoint path or directory. If omitted, the best Stage-2 checkpoint is used.")
    parser.add_argument("--secondary-checkpoint", type=str, default=None, help="Optional second full-model checkpoint used for video_seg and image_cls (mixed-checkpoint repair).")
    parser.add_argument("--secondary-config", type=str, default=None, help="Model config for the secondary checkpoint. Defaults to --config. Required when the secondary checkpoint has a different graph (e.g. MemoryVideoSegHead).")
    parser.add_argument("--secondary-tasks", type=str, default="video_seg,image_cls", help="Comma-separated task list served by the secondary checkpoint in mixed mode.")
    parser.add_argument("--ensemble-tasks", type=str, default="", help="Comma-separated tasks to probability-ensemble with the other checkpoint (e.g. image_seg or image_cls,ceus_cls). Empty disables ensembling.")
    parser.add_argument("--ensemble-weight", type=float, default=0.5, help="Weight of the extra (other) checkpoint in the probability average. The serving checkpoint has fixed weight 1.0.")
    parser.add_argument("--ensemble-secondary-tta", action="store_true", help="Also apply flip-TTA to the extra checkpoint for image_seg ensembles (doubles that member's forwards).")
    parser.add_argument("--ensemble-third-checkpoint", type=str, default=None, help="Optional third ensemble member (no flip-TTA), loaded with the primary config. Explicit path only.")
    parser.add_argument("--ensemble-third-weight", type=float, default=0.5, help="Weight of the third ensemble member when --ensemble-third-checkpoint is set.")
    parser.add_argument("--ensemble-third-tasks", type=str, default="", help="Comma-separated tasks the third member applies to. Empty = all --ensemble-tasks tasks (third member is only useful for a subset, e.g. image_seg but not ceus_cls).")
    parser.add_argument("--which", choices=["best", "latest"], default="best", help="Checkpoint choice when --checkpoint is not provided.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Inference device.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output submission directory.")
    parser.add_argument("--zip-path", type=str, default=None, help="Optional upload-ready zip path. The zip contains only the official submission files.")
    parser.add_argument("--no-zip", action="store_true", help="Write the submission directory only.")
    return parser.parse_args()


def write_submission_zip(submission_dir: Path, zip_path: Path) -> None:
    allowed_roots = {"classification.json", "image_seg", "ceus_seg", "video_seg"}
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(submission_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(submission_dir).as_posix()
            top = rel.split("/", 1)[0]
            if top in allowed_roots:
                zf.write(path, rel)


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root or args.competition_root)
    phase_root = phase_data_root(data_root, args.phase)
    entries = load_json(phase_json_path(data_root, args.phase))
    out_dir = Path(args.output_dir or f"outputs/submission_{args.phase}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    cfg = load_yaml(args.config)
    cfg.setdefault("data", {})["data_root"] = str(data_root)
    checkpoint_path = resolve_submission_checkpoint(cfg, args.checkpoint, args.which)

    print(f"[Info] Device: {device}")
    print(f"[Info] Data root: {data_root}")
    print(f"[Info] Phase: {args.phase}")
    print(f"[Info] Config: {args.config}")
    print(f"[Info] Checkpoint: {checkpoint_path}")

    model = load_model(cfg, checkpoint_path, device)
    secondary_model = None
    secondary_tasks: set = set()
    if args.secondary_checkpoint:
        secondary_path = resolve_checkpoint_reference(args.secondary_checkpoint, prefix=None)
        secondary_cfg = load_yaml(args.secondary_config) if args.secondary_config else cfg
        secondary_model = load_model(secondary_cfg, secondary_path, device)
        secondary_tasks = {t.strip() for t in args.secondary_tasks.split(",") if t.strip()}
        print(f"[Info] Secondary Checkpoint: {secondary_path}")
        print(f"[Info] Secondary Config: {args.secondary_config or args.config}")
        print(f"[Info] Secondary Tasks: {sorted(secondary_tasks)}")

    # 概率级集成（flag 门控，默认关闭）：v1 仅 image_seg/image_cls/ceus_cls；extra 永远是"另一个" checkpoint
    ensemble_tasks = {t.strip() for t in args.ensemble_tasks.split(",") if t.strip()}
    if ensemble_tasks:
        bad = sorted(ensemble_tasks & {"video_seg", "ceus_seg"})
        if bad:
            raise SystemExit(f"[Error] --ensemble-tasks v1 scope excludes: {bad}")
        if secondary_model is None:
            raise SystemExit("[Error] --ensemble-tasks requires --secondary-checkpoint: every ensemble task's extra member is the other checkpoint.")
        print(f"[Info] Ensemble Tasks: {sorted(ensemble_tasks)} (weight={args.ensemble_weight}, secondary_tta={bool(args.ensemble_secondary_tta)})")
    ensemble_third_model = None
    ensemble_third_tasks: set = set()
    if args.ensemble_third_checkpoint:
        third_path = resolve_checkpoint_reference(args.ensemble_third_checkpoint, prefix=None)
        ensemble_third_model = load_model(cfg, third_path, device)
        ensemble_third_tasks = {t.strip() for t in args.ensemble_third_tasks.split(",") if t.strip()} or set(ensemble_tasks)
        print(f"[Info] Ensemble Third Checkpoint: {third_path}")
        print(f"[Info] Ensemble Third Tasks: {sorted(ensemble_third_tasks)}")

    ceus_processor = build_ceus_processor(cfg.get("data", {}))
    num_frames = int(cfg.get("data", {}).get("num_frames", 10))

    # 混合 checkpoint：secondary_tasks 内的任务用 secondary（默认 video_seg/image_cls，旧 encoder 未漂移），其余用主 checkpoint
    def _model_for(task: str):
        if secondary_model is not None and task in secondary_tasks:
            return secondary_model
        return model

    # 概率级集成路由：serving 模型仍由 _model_for 决定；extra 是"另一个" checkpoint（权重 w，serving 固定 1.0）
    def _ensemble_extras_for(task: str):
        if task not in ensemble_tasks:
            return None
        extras: List[Tuple[torch.nn.Module, float, bool]] = []
        if task in secondary_tasks:
            extras.append((model, float(args.ensemble_weight), False))
        else:
            extras.append((secondary_model, float(args.ensemble_weight), bool(args.ensemble_secondary_tta)))
        if ensemble_third_model is not None and task in ensemble_third_tasks:
            extras.append((ensemble_third_model, float(args.ensemble_third_weight), False))
        return extras

    classification: Dict[str, Dict[str, Any]] = {}
    counts = {"classification": 0, "segmentation": 0}
    for entry in tqdm(entries, desc="Predict"):
        task = entry.get("task")
        if task == "image_seg":
            predict_image_seg(_model_for(task), entry, phase_root, out_dir, device, extra_models=_ensemble_extras_for(task))
            counts["segmentation"] += 1
        elif task == "video_seg":
            predict_video_seg(_model_for(task), entry, phase_root, out_dir, device, num_frames)
            counts["segmentation"] += 1
        elif task == "ceus_seg":
            predict_ceus_seg(_model_for(task), entry, phase_root, out_dir, device, ceus_processor)
            counts["segmentation"] += 1
        elif task == "image_cls":
            classification[classification_key(entry)] = predict_image_cls(_model_for(task), entry, phase_root, device, extra_models=_ensemble_extras_for(task))
            counts["classification"] += 1
        elif task == "ceus_cls":
            classification[classification_key(entry)] = predict_ceus_cls(_model_for(task), entry, phase_root, device, num_frames, extra_models=_ensemble_extras_for(task))
            counts["classification"] += 1

    write_json(out_dir / "classification.json", classification)
    zip_path = None
    if not args.no_zip:
        zip_path = Path(args.zip_path).resolve() if args.zip_path else out_dir.with_suffix(".zip")
        write_submission_zip(out_dir, zip_path)

    summary = {
        "data_root": str(data_root),
        "phase": args.phase,
        "output_dir": str(out_dir),
        "zip_path": str(zip_path) if zip_path else None,
        "config": args.config,
        "checkpoint": checkpoint_path,
        "secondary_config": (args.secondary_config or args.config) if secondary_model else None,
        "secondary_tasks": sorted(secondary_tasks) if secondary_model else None,
        "ensemble": {
            "tasks": sorted(ensemble_tasks),
            "weight": args.ensemble_weight,
            "secondary_tta": bool(args.ensemble_secondary_tta),
            "third_checkpoint": args.ensemble_third_checkpoint,
            "third_weight": args.ensemble_third_weight if ensemble_third_model is not None else None,
            "third_tasks": sorted(ensemble_third_tasks) if ensemble_third_model is not None else None,
        } if ensemble_tasks else None,
        "classification_samples": counts["classification"],
        "classification_keys": len(classification),
        "segmentation_files": counts["segmentation"],
        "format": "competition submission",
    }
    write_json(out_dir / "submission_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

