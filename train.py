import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from datasets.build_loader import build_stage1_seg_loaders, build_stage2_cls_loaders
from models.build_model import build_model
from trainers.trainer_stage1_seg import Stage1SegTrainer
from trainers.trainer_stage2_cls import Stage2ClsTrainer
from utils.freeze import apply_freeze_config, count_parameters, load_model_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UUSIVC2026 baseline training entry")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to yaml config. Defaults to configs/stage1_seg.yaml or configs/stage2_cls.yaml.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="Training device",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="stage1_seg",
        choices=["stage1_seg", "stage2_cls"],
        help="Training stage",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="UUSIVC2026 public data root containing TRAIN/ and VAL/ packages. If omitted, uses UUSIVC2026_DATA_ROOT.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional model checkpoint used to initialize training. Overrides train.init_checkpoint.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Optional training checkpoint used to resume model, optimizer, epoch, and metrics.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional output directory. Overrides trainer.save_dir.",
    )
    parser.add_argument(
        "--keep-best",
        type=int,
        default=None,
        help="Number of top local-validation checkpoints to keep. Overrides trainer.keep_best.",
    )
    parser.add_argument(
        "--local-val-fraction",
        type=float,
        default=None,
        help="Fraction of labeled TRAIN data held out for local validation. Overrides data.local_val_fraction.",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Use all labeled TRAIN data for training and skip per-epoch validation/best-checkpoint selection.",
    )
    parser.add_argument(
        "--debug",
        nargs="+",
        default=None,
        choices=["fast", "full", "deep"],
        help="Enable debug mode. Levels: fast (run N steps and stop), "
        "full (limited samples + few epochs, full pipeline), "
        "deep (full data strategy + gradient/activation/performance diagnostics). "
        "Combine with a space, e.g. --debug deep fast.",
    )
    parser.add_argument(
        "--debug-steps",
        type=int,
        default=10,
        help="Max training steps for --debug fast (default: 10).",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=100,
        help="Max samples per task for --debug full/deep (default: 100).",
    )
    parser.add_argument(
        "--debug-epochs",
        type=int,
        default=3,
        help="Max epochs for --debug full/deep (default: 3).",
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    optim_cfg = cfg["optimizer"]
    name = optim_cfg.get("name", "adam").lower()
    lr = optim_cfg["lr"]
    weight_decay = optim_cfg.get("weight_decay", 0.0)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters remain after applying freeze config.")

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=optim_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def _compute_schedule_steps(
        train_loaders: Dict[str, Any],
        task_names: List[str],
        task_sampling_ratio: Dict[str, int],
) -> int:
    active_tasks = [name for name in task_names if name in train_loaders]
    if not active_tasks:
        raise ValueError("No active training loaders found for scheduler construction.")

    ratios = {name: max(int(task_sampling_ratio.get(name, 1)), 1) for name in active_tasks}
    loader_lengths = {name: len(train_loaders[name]) for name in active_tasks}
    scale = max(loader_lengths[name] / ratios[name] for name in active_tasks)
    return int(sum(max(1, int(np.ceil(scale * ratios[name]))) for name in active_tasks))


def build_scheduler(
        optimizer: torch.optim.Optimizer,
        cfg: Dict[str, Any],
        stage: str,
        train_loaders: Dict[str, Any],
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    scheduler_cfg = cfg.get("scheduler", {})
    name = str(scheduler_cfg.get("name", "none")).lower()
    if name in {"", "none", "null"}:
        return None

    if stage == "stage1_seg":
        steps_per_epoch = _compute_schedule_steps(
            train_loaders=train_loaders,
            task_names=["image_seg", "cardiac_video_seg", "video_seg_ceus"],
            task_sampling_ratio=cfg.get("sampling", {}).get("stage1_seg_ratio", {}),
        )
    elif stage == "stage2_cls":
        steps_per_epoch = _compute_schedule_steps(
            train_loaders=train_loaders,
            task_names=["image_cls", "standard_video_cls", "video_cls_ceus"],
            task_sampling_ratio=cfg.get("sampling", {}).get("stage2_cls_ratio", {}),
        )
    else:
        raise ValueError(f"Unsupported stage for scheduler: {stage}")

    total_steps = max(int(cfg["train"]["max_epochs"]) * steps_per_epoch, 1)

    if name == "cosine":
        eta_min = float(scheduler_cfg.get("eta_min", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=eta_min,
        )

    if name == "multistep":
        milestones = scheduler_cfg.get("milestones", [])
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        if not milestones:
            return None
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(m) * steps_per_epoch for m in milestones],
            gamma=gamma,
        )

    raise ValueError(f"Unsupported scheduler: {name}")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_debug_config(
        cfg: Dict[str, Any],
        args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Apply debug-mode overrides to cfg and return a debug_cfg dict for the Trainer.

    Returns None when debug is not active (normal training).
    """
    if not args.debug:
        return None

    level = set(args.debug)

    diagnostics = None
    if "deep" in level:
        diagnostics = {
            "grad_monitor": True,
            "act_monitor": True,
            "perf_monitor": True,
            "log_interval": 1,
        }

    debug_cfg: Dict[str, Any] = {
        "enabled": True,
        "levels": sorted(level),
        "max_steps": args.debug_steps if "fast" in level else None,
        "max_epochs": args.debug_epochs if "full" in level or "deep" in level else None,
        "max_samples_per_task": args.debug_samples if "full" in level or "deep" in level else None,
        "diagnostics": diagnostics,
        "keep_best": 1,
    }

    # --- override cfg values (same pattern as --full-train) ---
    if debug_cfg["max_epochs"] is not None:
        cfg.setdefault("train", {})["max_epochs"] = debug_cfg["max_epochs"]

    if "full" in level or "deep" in level:
        cfg.setdefault("trainer", {})["keep_best"] = 1
        cfg.setdefault("trainer", {})["log_interval"] = 1
        # signal the manifest builder to truncate per-task samples
        cfg.setdefault("data", {})["debug_max_samples_per_task"] = debug_cfg["max_samples_per_task"]
        # disable balanced samplers — truncated samples may all be one class
        cfg.setdefault("data", {})["use_balanced_sampler_image_cls"] = False
        cfg.setdefault("data", {})["use_balanced_sampler_video_cls"] = False

    return debug_cfg


def main() -> None:

    args = parse_args()
    stage = args.stage or "stage1_seg"
    if args.config is None:
        config_name = "stage1_seg.yaml" if stage == "stage1_seg" else "stage2_cls.yaml"
        args.config = str(Path(__file__).resolve().parent / "configs" / config_name)
    cfg = load_yaml(args.config)
    stage = args.stage or cfg.get("run", {}).get("stage", "stage1_seg")
    if args.data_root:
        cfg.setdefault("data", {})["data_root"] = args.data_root
    if args.save_dir:
        cfg.setdefault("trainer", {})["save_dir"] = args.save_dir
    if args.keep_best is not None:
        cfg.setdefault("trainer", {})["keep_best"] = args.keep_best
    if args.local_val_fraction is not None:
        cfg.setdefault("data", {})["local_val_fraction"] = args.local_val_fraction
    if args.full_train:
        cfg.setdefault("data", {})["local_val_fraction"] = 0.0
        cfg.setdefault("train", {})["require_validation"] = False
        cfg.setdefault("trainer", {})["keep_best"] = 0

    debug_cfg = apply_debug_config(cfg, args)

    save_dir = cfg["trainer"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    set_seed(
        seed=cfg["train"]["seed"],
        deterministic=cfg["train"].get("deterministic", True),
    )

    device = resolve_device(args.device)
    print(f"[Info] Using device: {device}")
    print(f"[Info] Loading config: {args.config}")
    print(f"[Info] Running stage: {stage}")
    if debug_cfg:
        print(f"[Debug] Mode active: levels={debug_cfg['levels']}")
        if debug_cfg["max_steps"]:
            print(f"[Debug]   max_steps={debug_cfg['max_steps']}")
        if debug_cfg["max_epochs"]:
            print(f"[Debug]   max_epochs={debug_cfg['max_epochs']}")
        if debug_cfg["max_samples_per_task"]:
            print(f"[Debug]   max_samples_per_task={debug_cfg['max_samples_per_task']}")
        if debug_cfg["diagnostics"]:
            print(f"[Debug]   diagnostics=grad+act+perf")

    if stage == "stage1_seg":
        loaders = build_stage1_seg_loaders(cfg["data"])
    elif stage == "stage2_cls":
        loaders = build_stage2_cls_loaders(cfg["data"])
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    if not loaders["train"]:
        raise ValueError(f"No train loaders were built for stage={stage}.")
    require_validation = bool(cfg.get("train", {}).get("require_validation", True))
    if require_validation and not loaders["val"]:
        raise ValueError(
            f"No val loaders were built for stage={stage}. Set data.local_val_fraction > 0 "
            "or run with --full-train to skip validation."
        )

    print(f"[Info] Train loaders: {list(loaders['train'].keys())}")
    print(f"[Info] Val loaders: {list(loaders['val'].keys())}")
    if not loaders["val"]:
        print("[Info] Local validation disabled; training will save latest checkpoints only.")

    model = build_model(cfg).to(device)
    setup_log: List[str] = []
    resume_checkpoint = args.resume_checkpoint or cfg.get("train", {}).get("resume_checkpoint")
    resume_state: Optional[Dict[str, Any]] = None
    start_epoch = 1
    resume_best_metric = -1.0
    resume_best_epoch = -1

    init_checkpoint = None if resume_checkpoint else (
        args.init_checkpoint
        if args.init_checkpoint is not None
        else cfg.get("train", {}).get("init_checkpoint")
    )
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_state = torch.load(str(resume_path), map_location=device)
        state_dict = resume_state["model"] if isinstance(resume_state,
                                                         dict) and "model" in resume_state else resume_state
        strict_load = bool(cfg.get("train", {}).get("strict_load", True))
        model.load_state_dict(state_dict, strict=strict_load)
        resumed_epoch = int(resume_state.get("epoch", 0)) if isinstance(resume_state, dict) else 0
        start_epoch = resumed_epoch + 1
        resume_metrics = resume_state.get("metrics", {}) if isinstance(resume_state, dict) else {}
        if stage == "stage1_seg":
            resume_best_metric = float(resume_metrics.get("mean_score", resume_metrics.get("mean_dice", -1.0)))
        else:
            resume_best_metric = float(resume_metrics.get("mean_score", -1.0))
        resume_best_epoch = resumed_epoch if resume_best_metric >= 0 else -1
        setup_log.append(
            "[Setup] Resumed checkpoint: "
            f"{resume_path} (epoch={resumed_epoch}, strict={strict_load}, start_epoch={start_epoch})"
        )
    elif init_checkpoint:
        strict_load = bool(cfg.get("train", {}).get("strict_load", True))
        checkpoint_info = load_model_checkpoint(
            model=model,
            checkpoint_path=init_checkpoint,
            map_location=device,
            strict=strict_load,
        )
        setup_log.append(
            "[Setup] Loaded init checkpoint: "
            f"{checkpoint_info['path']} "
            f"(epoch={checkpoint_info['epoch']}, strict={checkpoint_info['strict']})"
        )
        if checkpoint_info["missing_keys"] or checkpoint_info["unexpected_keys"]:
            setup_log.append(
                "[Setup] Checkpoint key mismatch: "
                f"missing={len(checkpoint_info['missing_keys'])}, "
                f"unexpected={len(checkpoint_info['unexpected_keys'])}"
            )

    freeze_info = apply_freeze_config(model, cfg, stage)
    counts = freeze_info["parameter_counts"]
    setup_log.append(
        "[Setup] Parameter counts: "
        f"total={counts['total']:,}, trainable={counts['trainable']:,}, frozen={counts['frozen']:,}"
    )
    if freeze_info["freeze_keywords"]:
        setup_log.append(
            "[Setup] Freeze keywords: "
            + ", ".join(freeze_info["freeze_keywords"])
        )
        unmatched = freeze_info["freeze_result"]["unmatched_keywords"]
        if unmatched:
            setup_log.append("[Setup] Unmatched freeze keywords: " + ", ".join(unmatched))
    else:
        base_counts = count_parameters(model)
        setup_log.append(
            "[Setup] No freeze keywords configured; all parameters trainable: "
            f"{base_counts['trainable']:,}"
        )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        stage=stage,
        train_loaders=loaders["train"],
    )
    if resume_state is not None and isinstance(resume_state, dict):
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            setup_log.append("[Setup] Resumed optimizer state.")
        else:
            setup_log.append("[Setup] Resume checkpoint has no optimizer state.")
        if scheduler is not None and "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
            setup_log.append("[Setup] Resumed scheduler state.")
        elif scheduler is not None:
            setup_log.append("[Setup] Resume checkpoint has no scheduler state; scheduler restarts from config.")

    if stage == "stage1_seg":
        trainer = Stage1SegTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            save_dir=save_dir,
            max_epochs=cfg["train"]["max_epochs"],
            ce_weight=cfg["loss"]["ce_weight"],
            dice_weight=cfg["loss"]["dice_weight"],
            log_interval=cfg["trainer"].get("log_interval", 20),
            scheduler=scheduler,
            use_amp=cfg["train"].get("use_amp", False),
            task_sampling_ratio=cfg.get("sampling", {}).get("stage1_seg_ratio", {}),
            frozen_eval_keywords=freeze_info["frozen_eval_keywords"],
            start_epoch=start_epoch,
            best_metric=resume_best_metric,
            best_epoch=resume_best_epoch,
            monitor_metric=cfg["trainer"].get("monitor_metric", "mean_score"),
            keep_best=cfg["trainer"].get("keep_best", 3),
            nsd_tolerance=cfg.get("metrics", {}).get("nsd_tolerance", 1),
            boundary_weight=cfg.get("loss", {}).get("boundary_weight", 0.0),
            boundary_weight_step=cfg.get("loss", {}).get("boundary_weight_step", 0.0),
            boundary_weight_max=cfg.get("loss", {}).get("boundary_weight_max", 0.0),
            debug_cfg=debug_cfg,
        )
    else:
        trainer = Stage2ClsTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            save_dir=save_dir,
            max_epochs=cfg["train"]["max_epochs"],
            log_interval=cfg["trainer"].get("log_interval", 20),
            scheduler=scheduler,
            use_amp=cfg["train"].get("use_amp", False),
            task_sampling_ratio=cfg.get("sampling", {}).get("stage2_cls_ratio", {}),
            frozen_eval_keywords=freeze_info["frozen_eval_keywords"],
            start_epoch=start_epoch,
            best_metric=resume_best_metric,
            best_epoch=resume_best_epoch,
            monitor_metric=cfg["trainer"].get("monitor_metric", "mean_score"),
            keep_best=cfg["trainer"].get("keep_best", 3),
            label_smoothing=cfg.get("loss", {}).get("cls_label_smoothing", 0.0),
            debug_cfg=debug_cfg,
        )

    for line in setup_log:
        print(line)
        trainer.logger.info(line)

    trainer.fit(loaders)


if __name__ == "__main__":
    main()

