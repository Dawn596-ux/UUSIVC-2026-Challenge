import json
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _checkpoint_payload(model, optimizer=None, epoch=None, metrics=None, scheduler=None) -> Dict[str, Any]:
    save_obj = {
        "model": model.state_dict(),
    }
    if optimizer is not None:
        save_obj["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_obj["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        save_obj["epoch"] = epoch
    if metrics is not None:
        save_obj["metrics"] = metrics
    return save_obj


def save_checkpoint(path, model, optimizer=None, epoch=None, metrics=None, scheduler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_obj = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics=metrics,
        scheduler=scheduler,
    )

    try:
        torch.save(save_obj, str(path))
        return str(path)
    except Exception as exc:
        fallback_path = path.with_name(
            f"{path.stem}_fallback_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"
        )
        try:
            torch.save(save_obj, str(fallback_path))
            warnings.warn(
                f"Could not save checkpoint '{path}' ({exc}). "
                f"Saved fallback checkpoint to '{fallback_path}'.",
                RuntimeWarning,
            )
            return str(fallback_path)
        except Exception:
            raise


def _safe_metric_name(metric_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in metric_name)


def _write_best_manifest(best_dir: Path, prefix: str, monitor_metric: str, records: List[Dict[str, Any]]) -> None:
    manifest_path = best_dir / f"{prefix}_manifest.json"
    payload = {
        "prefix": prefix,
        "monitor_metric": monitor_metric,
        "records": records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_topk_best_checkpoint(
    save_dir,
    prefix: str,
    model,
    optimizer=None,
    scheduler=None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    monitor_metric: str = "mean_dice",
    history: Optional[List[Dict[str, Any]]] = None,
    keep_top_k: int = 3,
) -> Tuple[List[Dict[str, Any]], Optional[str], List[Dict[str, Any]]]:
    if keep_top_k <= 0:
        return history or [], None, []
    if metrics is None or monitor_metric not in metrics:
        raise KeyError(f"Monitor metric '{monitor_metric}' not found in metrics: {metrics}")

    metric_value = float(metrics[monitor_metric])
    best_dir = Path(save_dir) / "best_checkpoints"
    best_dir.mkdir(parents=True, exist_ok=True)

    records = list(history or [])
    current_marker = "__CURRENT_CHECKPOINT__"
    records.append({"epoch": int(epoch or 0), "metric": metric_value, "path": current_marker})
    records.sort(key=lambda item: (float(item["metric"]), int(item["epoch"])), reverse=True)

    kept = records[:keep_top_k]
    removed = records[keep_top_k:]
    if not any(record["path"] == current_marker for record in kept):
        _write_best_manifest(best_dir, prefix, monitor_metric, history or [])
        return history or [], None, removed

    saved_path: Optional[str] = None
    for rank in range(len(kept), 0, -1):
        record = kept[rank - 1]
        target_path = best_dir / f"{prefix}_rank{rank}.pth"
        if record["path"] == current_marker:
            saved_path = save_checkpoint(
                path=target_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
            )
            record["path"] = str(Path(saved_path).resolve())
        else:
            source_path = Path(record["path"])
            if source_path.exists() and source_path.resolve() != target_path.resolve():
                shutil.copyfile(str(source_path), str(target_path))
            record["path"] = str(target_path.resolve())

    _write_best_manifest(best_dir, prefix, monitor_metric, kept)
    return kept, saved_path, removed


def resolve_checkpoint_reference(checkpoint_ref, prefix: Optional[str] = None) -> str:
    ref = Path(checkpoint_ref).expanduser()
    if ref.is_file():
        return str(ref.resolve())

    candidates = []
    if ref.is_dir():
        search_dirs = [ref / "best_checkpoints", ref]
        prefixes = [prefix] if prefix else ["best_stage1_seg", "best_stage2_cls"]
        for search_dir in search_dirs:
            for candidate_prefix in prefixes:
                manifest_path = search_dir / f"{candidate_prefix}_manifest.json"
                if not manifest_path.exists():
                    continue
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                records = manifest.get("records", [])
                if records:
                    best_path = Path(records[0]["path"])
                    if best_path.exists():
                        return str(best_path.resolve())
            if search_dir.exists():
                for candidate_prefix in prefixes:
                    candidates.extend(search_dir.glob(f"{candidate_prefix}_rank*.pth"))

    if not ref.exists() and ref.parent.exists():
        prefixes = [prefix] if prefix else [ref.stem]
        search_dir = ref.parent / "best_checkpoints"
        for candidate_prefix in prefixes:
            manifest_path = search_dir / f"{candidate_prefix}_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                records = manifest.get("records", [])
                if records:
                    best_path = Path(records[0]["path"])
                    if best_path.exists():
                        return str(best_path.resolve())

    candidates = [path for path in candidates if path.exists()]
    if candidates:
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return str(candidates[0].resolve())

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_ref}")


def load_swin_branch_checkpoint(dual_encoder, checkpoint_path, map_location=None):
    """Load the Swin encoder branch from a full Swin-only model checkpoint into a
    DualEncoder, remapping ``image_model.backbone.encoder.backbone.*`` -> ``swin.backbone.*``.

    Used to initialize the (frozen) Swin branch of the DINOv2 fusion model with the
    domain-adapted weights from the 0.6505 Swin baseline, instead of ImageNet-only.
    """
    old_prefix = "image_model.backbone.encoder.backbone."
    new_prefix = "swin.backbone."

    resolved_path = Path(resolve_checkpoint_reference(checkpoint_path)).expanduser().resolve()
    checkpoint = torch.load(str(resolved_path), map_location=map_location)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    remapped = {}
    skipped = 0
    for key, value in state_dict.items():
        if key.startswith(old_prefix):
            remapped[new_prefix + key[len(old_prefix):]] = value
        else:
            skipped += 1

    incompatible = dual_encoder.load_state_dict(remapped, strict=False)
    return {
        "path": str(resolved_path),
        "loaded": len(remapped),
        "skipped": skipped,
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
    }
