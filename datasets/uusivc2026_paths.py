"""Fixed UUSIVC2026 challenge data protocol.

The public release is distributed as split packages under one root:
TRAIN contains labeled training data, VAL contains participant metadata without
labels or masks, and TEST is reserved for the challenge platform/internal use.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
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


def derive_group_key(entry: Dict[str, Any]) -> str:
    """Return a patient/case group key for a labeled entry.

    The released JSON has no ``patient_id`` field, so patient identity must be
    derived from the input filename. The convention is NOT uniform across the
    dataset — it depends on task (and public vs private source):

    - ``ceus_seg`` (private): ``<patient>_<timestamp>.npy`` -> strip the timestamp.
    - ``video_seg``: private ``X001.npy`` (no suffix) vs public CAMUS
      ``patient0001_2CH.npy`` -> strip the view label when present.
    - ``image_cls`` / ``image_seg``: multi-frame uses ``<label>_<case>_<frame>``
      (a short frame index, 1-2 digits -> strip it); single-image uses
      ``<label>_<case>`` (keep the whole stem). A long zero-padded trailing
      segment (e.g. Prostate ``seg_img_00000``) is a case index, NOT a frame,
      so the whole stem is kept.

    ``data_partition_group`` (``private_train`` vs ``public_all``) is prepended so a
    private and a public sample that happen to share a name are never merged.
    """
    task = entry.get("task") or ""
    input_rel = entry.get("input_path_relative") or ""
    stem = input_rel.split("/")[-1].split(".")[0]
    organ = entry.get("organ") or entry.get("dataset_name") or task
    partition = entry.get("data_partition_group") or ""

    if task in ("ceus_seg", "video_seg"):
        patient = stem.rsplit("_", 1)[0]  # strip trailing timestamp / view label
    elif task in ("image_cls", "image_seg"):
        if stem.count("_") >= 2:
            base, _, suffix = stem.rpartition("_")
            patient = base if suffix.isdigit() and len(suffix) <= 2 else stem
        else:
            patient = stem
    else:  # ceus_cls
        patient = stem

    return f"{partition}/{organ}/{patient}"


def _group_rep_label(group: List[Dict[str, Any]]) -> Optional[int]:
    labels = [e.get("class_label_index") for e in group if e.get("class_label_index") is not None]
    if not labels:
        return None
    return int(max(set(labels), key=labels.count))


def _groups_have_label(groups: Dict[str, List[Dict[str, Any]]]) -> bool:
    return any(_group_rep_label(g) is not None for g in groups.values())


def split_entries(
    entries: List[Dict[str, Any]],
    val_fraction: float,
    seed: int,
    mode: str = "grouped_stratified",
    group_key_fn=derive_group_key,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split one task's entries into (train_entries, val_entries).

    ``mode``:
    - ``random``: sample-level shuffle split (legacy behavior).
    - ``grouped``: whole patient/case groups are assigned to one split only.
    - ``grouped_stratified``: grouped, and classification labels stay balanced
      across splits (groups are bucketed by their representative label first).
    """
    if val_fraction <= 0:
        return list(entries), []
    if len(entries) <= 1:
        return list(entries), list(entries)

    rng = random.Random(seed)

    if mode == "random":
        indices = list(range(len(entries)))
        rng.shuffle(indices)
        val_count = max(1, int(round(len(entries) * val_fraction)))
        val_indices = set(indices[:val_count])
        train = [e for i, e in enumerate(entries) if i not in val_indices]
        val = [e for i, e in enumerate(entries) if i in val_indices]
        if not train:
            train = list(entries)
        return train, val

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        key = str(group_key_fn(e))
        groups.setdefault(key, []).append(e)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    if mode == "grouped_stratified" and _groups_have_label(groups):
        by_label: Dict[int, List[str]] = defaultdict(list)
        for key in group_keys:
            label = _group_rep_label(groups[key])
            if label is not None:
                by_label[label].append(key)
        train_keys: List[str] = []
        val_keys: List[str] = []
        for label in sorted(by_label.keys()):
            keys = by_label[label]
            rng.shuffle(keys)
            val_count = max(1, int(round(len(keys) * val_fraction)))
            if val_count >= len(keys):
                val_count = max(0, len(keys) - 1)
            val_keys.extend(keys[:val_count])
            train_keys.extend(keys[val_count:])
    else:
        val_count = max(1, int(round(len(group_keys) * val_fraction)))
        if val_count >= len(group_keys):
            val_count = max(0, len(group_keys) - 1)
        val_keys = group_keys[:val_count]
        train_keys = group_keys[val_count:]

    train = [e for key in train_keys for e in groups[key]]
    val = [e for key in val_keys for e in groups[key]]
    if not train:
        train = list(entries)
    return train, val


def _collect_labeled_training_entries(data_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Collect full labeled entries (public + train) bucketed by task.

    Each entry keeps all original metadata and gains a ``_root`` field pointing at
    the physical data root of its source phase (public/train resolve to different
    directories), so downstream consumers can resolve absolute input/mask paths.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for source_phase in ("public", "train"):
        entries = load_phase_entries(data_root, source_phase)
        root = phase_data_root(data_root, source_phase)
        for entry in entries:
            task = entry.get("task")
            if task not in TASK_TO_CFG_KEY:
                continue
            if _manifest_line(entry, root) is None:
                continue
            copy = dict(entry)
            copy["_root"] = str(root)
            buckets[task].append(copy)
    return buckets


def write_fixed_manifests(
    data_root_like: str,
    output_dir: str | Path,
    phases: Iterable[str] = ("train", "val"),
    local_val_fraction: float = 0.1,
    seed: int = 2024,
    max_samples_per_task: Optional[int] = None,
    split_mode: str = "grouped_stratified",
) -> Dict[str, Dict[str, Any]]:
    """Materialize labeled train/local-val manifests from TRAIN package data.

    Besides the per-task ``{phase}_{task}.txt`` manifests (unchanged format, kept
    for backward compatibility with train.py/test.py), this also writes:

    - ``val_entries.json``: full metadata for every val entry (used by predict_val.py)
    - ``split_summary.json``: per-task sample/group/label counts for manual inspection
    """
    data_root = resolve_data_root(data_root_like)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_buckets = _collect_labeled_training_entries(data_root)
    split_buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {"train": {}, "val": {}}
    val_entries: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for task, entries in labeled_buckets.items():
        train_entries, val_entries_task = split_entries(
            entries, local_val_fraction, seed, mode=split_mode
        )
        split_buckets["train"][task] = train_entries
        split_buckets["val"][task] = val_entries_task
        val_entries.extend(val_entries_task)
        summary[task] = _summarize_split(train_entries, val_entries_task)

    built: Dict[str, Dict[str, Any]] = {}
    for phase in phases:
        for task, entries in split_buckets.get(phase, {}).items():
            if not entries:
                continue
            lines = []
            for entry in entries:
                line = _manifest_line(entry, Path(entry["_root"]))
                if line is not None:
                    lines.append(line)
            if max_samples_per_task is not None and len(lines) > max_samples_per_task:
                lines = lines[:max_samples_per_task]
            cfg_base = TASK_TO_CFG_KEY[task]
            path = output_dir / f"{phase}_{task}.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            built[f"{cfg_base}_{phase}"] = {
                "dataset_name": TASK_DATASET_NAME[task],
                "list_file": str(path),
            }

    val_path = output_dir / "val_entries.json"
    val_path.write_text(json.dumps(val_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return built


def _summarize_split(
    train_entries: List[Dict[str, Any]], val_entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    def label_counter(entries: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Counter = Counter()
        for e in entries:
            label = e.get("class_label_index")
            if label is not None:
                counts[str(label)] += 1
        return dict(counts)

    def group_count(entries: List[Dict[str, Any]]) -> int:
        return len({derive_group_key(e) for e in entries})

    return {
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "train_groups": group_count(train_entries),
        "val_groups": group_count(val_entries),
        "train_labels": label_counter(train_entries),
        "val_labels": label_counter(val_entries),
    }


def expand_fixed_uusivc_data_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a data cfg populated from labeled TRAIN data and a local holdout."""
    data_root = cfg.get("data_root") or cfg.get("root")
    out_dir = cfg.get("manifest_cache_dir", "./outputs/uusivc2026_fixed/manifests")
    val_fraction = float(cfg.get("local_val_fraction", 0.1))
    seed = int(cfg.get("split_seed", cfg.get("seed", 2024)))
    max_samples = cfg.get("debug_max_samples_per_task", None)
    split_mode = str(cfg.get("split_mode", "grouped_stratified"))
    generated = write_fixed_manifests(
        data_root,
        out_dir,
        phases=("train", "val"),
        local_val_fraction=val_fraction,
        seed=seed,
        max_samples_per_task=max_samples,
        split_mode=split_mode,
    )
    expanded = dict(cfg)
    expanded.update(generated)
    expanded["resolved_data_root"] = str(resolve_data_root(data_root))
    return expanded

