from typing import Any, Dict, List, Optional, Tuple
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

from models.losses.multi_task_loss import Stage1SegLoss
from utils.checkpoint import save_checkpoint, save_topk_best_checkpoint
from utils.diagnostics import DiagnosticsConfig, DiagnosticsHook
from utils.freeze import set_modules_eval_by_keywords
from utils.logger import get_logger
from utils.metrics import compute_binary_seg_score_from_logits, compute_ceus_official_score_from_logits
from utils.visualization import save_segmentation_visuals


def make_grad_scaler(enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled)


class Stage1SegTrainer:
    """
    Stage 1: segmentation only
    Supports:
    - image_seg
    - cardiac_video_seg
    - video_seg_ceus
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        save_dir: str,
        max_epochs: int,
        ce_weight: float = 0.4,
        dice_weight: float = 0.6,
        log_interval: int = 20,
        scheduler: Optional[Any] = None,
        use_amp: bool = False,
        task_sampling_ratio: Optional[Dict[str, int]] = None,
        frozen_eval_keywords: Optional[List[str]] = None,
        start_epoch: int = 1,
        best_metric: float = -1.0,
        best_epoch: int = -1,
        monitor_metric: str = "mean_score",
        keep_best: int = 3,
        debug_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.max_epochs = max_epochs
        self.log_interval = log_interval
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.task_sampling_ratio = task_sampling_ratio or {}
        self.frozen_eval_keywords = frozen_eval_keywords or []
        self.start_epoch = start_epoch

        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.save_dir, "train_stage1_seg.log"))

        self.criterion = Stage1SegLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        )

        self.scaler = make_grad_scaler(enabled=use_amp)
        self.best_metric = best_metric
        self.best_epoch = best_epoch
        self.monitor_metric = monitor_metric
        self.keep_best = max(int(keep_best), 0)
        self.best_records: List[Dict[str, Any]] = []

        # --- debug mode ---
        self._debug_cfg = debug_cfg or {}
        self._max_steps: Optional[int] = self._debug_cfg.get("max_steps")
        self._debug_diagnostics: Optional[Dict[str, Any]] = self._debug_cfg.get("diagnostics")
        self._global_step: int = 0
        self._early_stop: bool = False

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for k, v in batch.items():
            moved[k] = v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
        return moved

    @staticmethod
    def _loader_to_iterator_dict(loaders: Dict[str, Any]) -> Dict[str, Any]:
        return {name: iter(loader) for name, loader in loaders.items()}

    @staticmethod
    def _safe_next(loader_name: str, loader, iterator_dict):
        try:
            batch = next(iterator_dict[loader_name])
        except StopIteration:
            iterator_dict[loader_name] = iter(loader)
            batch = next(iterator_dict[loader_name])
        return batch

    def _build_train_schedule(self, train_loaders: Dict[str, Any]) -> List[str]:
        task_names = [
            name
            for name in ("image_seg", "cardiac_video_seg", "video_seg_ceus")
            if name in train_loaders
        ]
        if not task_names:
            raise ValueError("No train segmentation loaders found.")

        ratios = {
            name: max(int(self.task_sampling_ratio.get(name, 1)), 1)
            for name in task_names
        }
        loader_lengths = {name: len(train_loaders[name]) for name in task_names}
        scale = max(loader_lengths[name] / ratios[name] for name in task_names)

        schedule: List[str] = []
        for name in task_names:
            repeats = max(1, int(math.ceil(scale * ratios[name])))
            schedule.extend([name] * repeats)

        random.shuffle(schedule)
        return schedule

    def _should_save_topk(self, metric: float) -> bool:
        if self.keep_best <= 0:
            return False
        if len(self.best_records) < self.keep_best:
            return True
        worst_kept = min(float(record["metric"]) for record in self.best_records)
        return metric > worst_kept

    def _prune_best_epoch_outputs(self, removed_records: List[Dict[str, Any]]) -> None:
        for record in removed_records:
            epoch = int(record.get("epoch", 0))
            if epoch <= 0:
                continue
            output_dir = os.path.join(
                self.save_dir,
                "best_epoch_outputs",
                f"epoch_{epoch:03d}",
            )
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)

    @staticmethod
    def _extract_case_id(case_id_value: Any) -> Optional[str]:
        if isinstance(case_id_value, (list, tuple)):
            return str(case_id_value[0]) if case_id_value else None
        if case_id_value is None:
            return None
        return str(case_id_value)

    @staticmethod
    def _tensor_to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
        image_np = image_tensor.detach().cpu().numpy()
        if image_np.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dims, got {image_np.shape}")
        return np.transpose(image_np, (1, 2, 0))

    @staticmethod
    def _get_visual_reference(raw_batch: Dict[str, Any], sample_idx: int = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        visual_images = raw_batch.get("visual_image")
        visual_gt_masks = raw_batch.get("visual_gt_mask")
        if visual_images is None or visual_gt_masks is None:
            return None, None
        return visual_images[sample_idx], visual_gt_masks[sample_idx]

    @torch.no_grad()
    def _export_loader_visual(self, loader, loader_name: str, output_dir: str) -> bool:
        self.model.eval()
        for raw_batch in loader:
            batch = self._move_batch_to_device(raw_batch)
            outputs = self.model(batch)
            logits = outputs["seg_logits"]
            if logits is None:
                raise ValueError(f"seg_logits is None during seg visualization for loader={loader_name}")

            case_id = self._extract_case_id(raw_batch.get("case_id"))
            if logits.ndim == 4:
                image = self._tensor_to_numpy_image(batch["image"][0])
                gt_mask = batch["label_seg"][0].detach().cpu().numpy()
                pred_mask = torch.argmax(logits[0], dim=0).detach().cpu().numpy()
            elif logits.ndim == 5:
                frame_idx = int(logits.shape[1] // 2)
                image = self._tensor_to_numpy_image(batch["image"][0, frame_idx])
                gt_mask = batch["label_seg"][0, frame_idx].detach().cpu().numpy()
                pred_mask = torch.argmax(logits[0, frame_idx], dim=0).detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported seg_logits shape for visualization: {tuple(logits.shape)}")

            visual_image, visual_gt_mask = self._get_visual_reference(raw_batch)
            if visual_image is not None and visual_gt_mask is not None:
                image = visual_image
                gt_mask = visual_gt_mask

            save_segmentation_visuals(
                output_dir=output_dir,
                image=image,
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                case_id=case_id,
                loader_name=loader_name,
            )
            return True
        return False

    @torch.no_grad()
    def export_best_epoch_visuals(self, val_loaders: Dict[str, Any], epoch: int) -> None:
        output_dir = os.path.join(
            self.save_dir,
            "best_epoch_outputs",
            f"epoch_{epoch:03d}",
            "segmentation",
        )
        saved_loaders = []
        for loader_name, loader in val_loaders.items():
            if self._export_loader_visual(loader, loader_name, output_dir):
                saved_loaders.append(loader_name)
        if saved_loaders:
            self.logger.info(
                f"Saved best-epoch segmentation visuals for loaders={saved_loaders} to {output_dir}"
            )

    def train_one_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        batch = self._move_batch_to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            outputs = self.model(batch)
            loss_dict = self.criterion(batch, outputs)
            loss = loss_dict["loss_total"]

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss_total": float(loss_dict["loss_total"].detach().cpu().item()),
            "loss_ce": float(loss_dict["loss_ce"].detach().cpu().item()),
            "loss_dice": float(loss_dict["loss_dice"].detach().cpu().item()),
        }

    @torch.no_grad()
    def validate_one_loader(self, loader, loader_name: str) -> Dict[str, float]:
        self.model.eval()
        dsc_scores = []
        nsd_scores = []
        official_scores = []

        for raw_batch in tqdm(loader, desc=f"Val-{loader_name}", leave=False):
            batch = self._move_batch_to_device(raw_batch)
            outputs = self.model(batch)

            if outputs["seg_logits"] is None:
                raise ValueError(f"seg_logits is None during seg validation for loader={loader_name}")

            logits = outputs["seg_logits"]
            if raw_batch.get("official_gt_mask") is not None:
                score_dict = compute_ceus_official_score_from_logits(logits, raw_batch)
            else:
                target = batch["label_seg"]
                score_dict = compute_binary_seg_score_from_logits(logits, target)
            dsc_scores.append(score_dict["dsc"])
            nsd_scores.append(score_dict["nsd"])
            official_scores.append(score_dict["score"])

        mean_dsc = sum(dsc_scores) / max(len(dsc_scores), 1)
        mean_nsd = sum(nsd_scores) / max(len(nsd_scores), 1)
        mean_score = sum(official_scores) / max(len(official_scores), 1)
        return {"dsc": mean_dsc, "nsd": mean_nsd, "score": mean_score}

    @torch.no_grad()
    def validate(self, val_loaders: Dict[str, Any]) -> Dict[str, float]:
        self.model.eval()
        all_metrics = {}
        mean_score_list = []

        for loader_name, loader in val_loaders.items():
            metric = self.validate_one_loader(loader, loader_name)
            all_metrics[f"{loader_name}_dsc"] = metric["dsc"]
            all_metrics[f"{loader_name}_nsd"] = metric["nsd"]
            all_metrics[f"{loader_name}_score"] = metric["score"]
            mean_score_list.append(metric["score"])

        all_metrics["mean_score"] = sum(mean_score_list) / max(len(mean_score_list), 1)
        return all_metrics

    def train_one_epoch(self, epoch: int, train_loaders: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        if self.frozen_eval_keywords:
            set_modules_eval_by_keywords(self.model, self.frozen_eval_keywords)
        train_schedule = self._build_train_schedule(train_loaders)
        iterator_dict = self._loader_to_iterator_dict(train_loaders)
        steps_per_epoch = len(train_schedule)

        running_loss = 0.0
        running_ce = 0.0
        running_dice_loss = 0.0
        steps_done = 0

        # --- debug: diagnostics hook ---
        diag_hook: Optional[DiagnosticsHook] = None
        if self._debug_diagnostics:
            diag_cfg = DiagnosticsConfig(
                grad_monitor=bool(self._debug_diagnostics.get("grad_monitor")),
                act_monitor=bool(self._debug_diagnostics.get("act_monitor")),
                perf_monitor=bool(self._debug_diagnostics.get("perf_monitor")),
                log_interval=int(self._debug_diagnostics.get("log_interval", 1)),
            )
            diag_hook = DiagnosticsHook(self.model, diag_cfg, self.device)
            diag_hook.__enter__()

        pbar = tqdm(range(steps_per_epoch), desc=f"Train Epoch {epoch}", leave=False)
        for step_idx in pbar:
            # --- debug: early stop ---
            if self._max_steps is not None and self._global_step >= self._max_steps:
                self._early_stop = True
                break

            loader_name = train_schedule[step_idx]
            loader = train_loaders[loader_name]
            batch = self._safe_next(loader_name, loader, iterator_dict)

            # --- debug: perf timing start ---
            if diag_hook is not None:
                diag_hook.start_step_timer()
                diag_hook.record_data_loading_done()

            log_dict = self.train_one_step(batch)
            running_loss += log_dict["loss_total"]
            running_ce += log_dict["loss_ce"]
            running_dice_loss += log_dict["loss_dice"]
            steps_done += 1

            # --- debug: perf timing end ---
            if diag_hook is not None:
                # record_forward_done covers the full compute (forward+backward+optimizer)
                diag_hook.record_forward_done()

            self._global_step += 1

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = running_loss / steps_done
            avg_ce = running_ce / steps_done
            avg_dice_loss = running_dice_loss / steps_done

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                ce=f"{avg_ce:.4f}",
                dice_loss=f"{avg_dice_loss:.4f}",
                task=loader_name,
            )

            # --- debug: diagnostics output ---
            if diag_hook is not None and (self._global_step % diag_cfg.log_interval == 0):
                if diag_cfg.grad_monitor:
                    diag_hook.log_gradient_stats(self._global_step)
                if diag_cfg.act_monitor:
                    diag_hook.log_activation_stats(self._global_step)
                if diag_cfg.perf_monitor:
                    diag_hook.log_performance(self._global_step)

            if (step_idx + 1) % self.log_interval == 0:
                self.logger.info(
                    f"[Epoch {epoch} | Step {step_idx + 1}/{steps_per_epoch}] "
                    f"task={loader_name} "
                    f"loss={avg_loss:.4f} ce={avg_ce:.4f} dice_loss={avg_dice_loss:.4f}"
                )

            # --- debug: check early stop after step ---
            if self._max_steps is not None and self._global_step >= self._max_steps:
                self._early_stop = True
                break

        # --- debug: teardown diagnostics hook ---
        if diag_hook is not None:
            diag_hook.__exit__()

        denom = max(steps_done, 1)
        return {
            "train_loss": running_loss / denom,
            "train_ce": running_ce / denom,
            "train_dice_loss": running_dice_loss / denom,
        }

    def fit(self, loaders: Dict[str, Dict[str, Any]]):
        train_loaders = loaders["train"]
        val_loaders = loaders["val"]
        has_val = bool(val_loaders)

        self.logger.info("Start Stage1 Segmentation Training")
        self.logger.info(f"Train loaders: {list(train_loaders.keys())}")
        self.logger.info(f"Val loaders: {list(val_loaders.keys())}")
        if not has_val:
            self.logger.info("Local validation disabled; only latest checkpoints will be saved.")
        self.logger.info(f"Task sampling ratio: {self.task_sampling_ratio}")

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            start_time = time.time()
            train_log = self.train_one_epoch(epoch, train_loaders)

            # --- debug: early stop ---
            if self._early_stop:
                self.logger.info(
                    f"[Debug] Fast mode stopped after {self._global_step} steps. "
                    f"train_loss={train_log['train_loss']:.4f} "
                    f"train_ce={train_log['train_ce']:.4f} "
                    f"train_dice_loss={train_log['train_dice_loss']:.4f}"
                )
                print(
                    f"[Debug] Fast mode stopped after {self._global_step} steps. "
                    f"avg loss={train_log['train_loss']:.4f}"
                )
                break

            val_log = self.validate(val_loaders) if has_val else {}
            epoch_time = time.time() - start_time

            if has_val:
                self.logger.info(
                    f"[Epoch {epoch}/{self.max_epochs}] "
                    f"time={epoch_time:.1f}s "
                    f"train_loss={train_log['train_loss']:.4f} "
                    f"train_ce={train_log['train_ce']:.4f} "
                    f"train_dice_loss={train_log['train_dice_loss']:.4f} "
                    f"val_mean_score={val_log['mean_score']:.4f}"
                )
                for k, v in val_log.items():
                    self.logger.info(f"    {k}: {v:.4f}")
            else:
                self.logger.info(
                    f"[Epoch {epoch}/{self.max_epochs}] "
                    f"time={epoch_time:.1f}s "
                    f"train_loss={train_log['train_loss']:.4f} "
                    f"train_ce={train_log['train_ce']:.4f} "
                    f"train_dice_loss={train_log['train_dice_loss']:.4f}"
                )

            latest_path = os.path.join(self.save_dir, "latest_stage1_seg.pth")
            latest_saved_path = save_checkpoint(
                path=latest_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=val_log if has_val else train_log,
            )
            if os.path.normcase(latest_saved_path) != os.path.normcase(latest_path):
                self.logger.info(f"Latest checkpoint fallback path: {latest_saved_path}")

            if not has_val:
                continue

            if self.monitor_metric not in val_log:
                raise KeyError(f"Monitor metric '{self.monitor_metric}' not found in validation log: {val_log}")
            current_metric = val_log[self.monitor_metric]

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch

            if self._should_save_topk(current_metric):
                old_records = list(self.best_records)
                self.best_records, best_saved_path, removed_records = save_topk_best_checkpoint(
                    save_dir=self.save_dir,
                    prefix="best_stage1_seg",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_log,
                    monitor_metric=self.monitor_metric,
                    history=old_records,
                    keep_top_k=self.keep_best,
                )
                self._prune_best_epoch_outputs(removed_records)
                self.export_best_epoch_visuals(val_loaders, epoch)

                self.logger.info(
                    f"Top-{self.keep_best} model saved at epoch={epoch}, path={best_saved_path}, "
                    f"{self.monitor_metric}={current_metric:.4f}"
                )
                if removed_records:
                    removed_epochs = [int(record["epoch"]) for record in removed_records]
                    self.logger.info(f"Pruned old top-k best epochs: {removed_epochs}")

        if has_val:
            self.logger.info(
                f"Training finished. Best epoch={self.best_epoch}, "
                f"best {self.monitor_metric}={self.best_metric:.4f}"
            )
        else:
            self.logger.info("Training finished without local validation. Use the latest checkpoint.")