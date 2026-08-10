"""Fixed UUSIVC2026 challenge data protocol.

The public release is distributed as split packages under one root:
TRAIN contains labeled training data, VAL contains participant metadata without
labels or masks, and TEST is reserved for the challenge platform/internal use.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PUBLIC_DIR = "Challenge_Data_Public"
PRIVATE_DIR = "Challenge_Data_Private_v2_fully_anonymized"
FINGERPRINT_DIR = "dataset_json_fingerprints_v4"

PHASE_JSON = {
    "train": "private_train_ground_truth.json",
    "val": "private_val_for_participants.json",
    "test": "private_test_for_participants.json",
    "public": "public_all_ground_truth.json",
}

PACKAGE_DIR = {
    "train": "TRAIN",
    "val": "VAL",
    "test": "TEST",
    "public": "TRAIN",
}

PHASE_SPLIT_DIR = {
    "train": "Train",
    "val": "Val",
    "test": "Test",
}

TASK_TO_CFG_KEY = {
    "image_cls": "image_cls",
    "ceus_cls": "ceus_video_cls",
    "image_seg": "image_seg",
    "ceus_seg": "ceus_video_seg",
    "video_seg": "cardiac_video_seg",
}

TASK_DATASET_NAME = {
    "image_cls": "UUSIVC2026_IMAGE_CLS",
    "ceus_cls": "UUSIVC2026_CEUS_CLS",
    "image_seg": "UUSIVC2026_IMAGE_SEG",
    "ceus_seg": "UUSIVC2026_CEUS_SEG",
    "video_seg": "UUSIVC2026_VIDEO_SEG",
}


def normalize_rel(path: str) -> str:
    return str(path).replace("\\", "/").lstrip("/")


def _is_split_package_root(path: Path) -> bool:
    return (path / "TRAIN" / FINGERPRINT_DIR).is_dir() and (path / "VAL" / FINGERPRINT_DIR).is_dir()


def _is_legacy_data_root(path: Path) -> bool:
    return (path / FINGERPRINT_DIR).is_dir()


def default_data_root() -> Path:
    env_root = os.environ.get("UUSIVC2026_DATA_ROOT")
    if env_root:
        return resolve_data_root(env_root)
    raise FileNotFoundError(
        "Could not find the UUSIVC2026 data root. Set UUSIVC2026_DATA_ROOT "
        "or pass --data-root / data.data_root."
    )


def resolve_data_root(root_like: str | None = None) -> Path:
    """Resolve a public split-package root or the older single data root."""
    if root_like is None or str(root_like).strip() == "":
        return default_data_root()
    root = Path(root_like).expanduser().resolve()
    candidates = [root, root / "data", root.parent, root.parent / "data"]
    for parent in root.parents:
        candidates.extend([parent, parent / "data"])
    for candidate in candidates:
        if _is_split_package_root(candidate):
            return candidate
    for candidate in candidates:
        if _is_legacy_data_root(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find a UUSIVC2026 data root. Expected TRAIN/VAL packages "
        f"or {FINGERPRINT_DIR}. Got: {root_like}"
    )


def phase_package_root(data_root: Path, phase: str) -> Path:
    if _is_split_package_root(data_root):
        package = data_root / PACKAGE_DIR[phase]
        if not package.exists():
            raise FileNotFoundError(f"Missing UUSIVC2026 {PACKAGE_DIR[phase]} package: {package}")
        return package
    return data_root


def phase_data_root(data_root: Path, phase: str) -> Path:
    package_root = phase_package_root(data_root, phase)
    if phase == "public":
        return package_root / PUBLIC_DIR
    return package_root / PRIVATE_DIR / PHASE_SPLIT_DIR[phase]


def phase_json_path(data_root: Path, phase: str) -> Path:
    return phase_package_root(data_root, phase) / FINGERPRINT_DIR / PHASE_JSON[phase]


def load_phase_entries(data_root: Path, phase: str) -> List[Dict[str, Any]]:
    path = phase_json_path(data_root, phase)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _manifest_line(entry: Dict[str, Any], root: Path) -> str | None:
    task_family = entry.get("task_family")
    input_rel = entry.get("input_path_relative")
    if not input_rel:
        return None
    input_path = root / normalize_rel(input_rel)

    if task_family == "classification":
        label = entry.get("class_label_index")
        if label is None:
            return None
        return f"{input_path}\t{int(label)}"

    if task_family == "segmentation":
        target_rel = entry.get("target_path_relative")
        if not target_rel:
            return None
        target_path = root / normalize_rel(target_rel)
        return f"{input_path}\t{target_path}"

    return None


def _split_lines(lines: List[str], val_fraction: float, seed: int) -> tuple[List[str], List[str]]:
    if val_fraction <= 0:
        return lines, []
    if len(lines) <= 1:
        return lines, lines[:]
    indices = list(range(len(lines)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(lines) * val_fraction)))
    val_indices = set(indices[:val_count])
    train_lines = [line for idx, line in enumerate(lines) if idx not in val_indices]
    val_lines = [line for idx, line in enumerate(lines) if idx in val_indices]
    if not train_lines:
        train_lines = lines[:]
    return train_lines, val_lines


def _collect_labeled_training_lines(data_root: Path) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for source_phase in ("public", "train"):
        entries = load_phase_entries(data_root, source_phase)
        root = phase_data_root(data_root, source_phase)
        for entry in entries:
            task = entry.get("task")
            if task not in TASK_TO_CFG_KEY:
                continue
            line = _manifest_line(entry, root)
            if line is not None:
                buckets[task].append(line)
    return buckets


def write_fixed_manifests(
    data_root_like: str,
    output_dir: str | Path,
    phases: Iterable[str] = ("train", "val"),
    local_val_fraction: float = 0.1,
    seed: int = 2024,
    max_samples_per_task: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Materialize labeled train/local-val manifests from TRAIN package data."""
    data_root = resolve_data_root(data_root_like)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_buckets = _collect_labeled_training_lines(data_root)
    split_buckets: Dict[str, Dict[str, List[str]]] = {"train": {}, "val": {}}
    for task, lines in labeled_buckets.items():
        train_lines, val_lines = _split_lines(lines, local_val_fraction, seed)
        split_buckets["train"][task] = train_lines
        split_buckets["val"][task] = val_lines

    built: Dict[str, Dict[str, Any]] = {}
    for phase in phases:
        for task, lines in split_buckets.get(phase, {}).items():
            if not lines:
                continue
            if max_samples_per_task is not None and len(lines) > max_samples_per_task:
                lines = lines[:max_samples_per_task]
            cfg_base = TASK_TO_CFG_KEY[task]
            path = output_dir / f"{phase}_{task}.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            built[f"{cfg_base}_{phase}"] = {
                "dataset_name": TASK_DATASET_NAME[task],
                "list_file": str(path),
            }
    return built


def expand_fixed_uusivc_data_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a data cfg populated from labeled TRAIN data and a local holdout."""
    data_root = cfg.get("data_root") or cfg.get("root")
    out_dir = cfg.get("manifest_cache_dir", "./outputs/uusivc2026_fixed/manifests")
    val_fraction = float(cfg.get("local_val_fraction", 0.1))
    seed = int(cfg.get("split_seed", cfg.get("seed", 2024)))
    max_samples = cfg.get("debug_max_samples_per_task", None)
    generated = write_fixed_manifests(
        data_root,
        out_dir,
        phases=("train", "val"),
        local_val_fraction=val_fraction,
        seed=seed,
        max_samples_per_task=max_samples,
    )
    expanded = dict(cfg)
    expanded.update(generated)
    expanded["resolved_data_root"] = str(resolve_data_root(data_root))
    return expanded

