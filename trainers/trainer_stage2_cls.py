from typing import Any, Dict, List, Optional
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

from models.losses.auc_margin_loss import AUCMarginLoss
from utils.auc_utils import binary_accuracy_from_logits, binary_auc_from_logits, stack_numpy
from utils.checkpoint import save_checkpoint, save_topk_best_checkpoint
from utils.diagnostics import DiagnosticsConfig, DiagnosticsHook
from utils.freeze import set_modules_eval_by_keywords
from utils.logger import get_logger
from utils.visualization import save_classification_csv


def make_grad_scaler(enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled)


class Stage2ClsTrainer:
    """
    Stage 2: classification only
    Supports:
    - image_cls
    - standard_video_cls
    - video_cls_ceus
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        save_dir: str,
        max_epochs: int,
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
        label_smoothing: float = 0.0,
        auc_margin_weight: float = 0.0,
        auc_margin_margin: float = 1.0,
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
        self.logger = get_logger(os.path.join(self.save_dir, "train_stage2_cls.log"))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
        # AUC-margin 排序正则：weight>0 才实例化、才参与前向——weight=0 时行为与
        # 不接线的旧版本逐字节一致（baseline 可回退硬保证）。
        self.auc_margin_weight = float(auc_margin_weight)
        self.auc_margin: Optional[AUCMarginLoss] = (
            AUCMarginLoss(margin=auc_margin_margin) if self.auc_margin_weight > 0 else None
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

    def _build_train_schedule(self, train_loaders: Dict[str, Any]):
        task_names = [
            name
            for name in ("image_cls", "standard_video_cls", "video_cls_ceus")
            if name in train_loaders
        ]
        if not task_names:
            raise ValueError("No classification train loaders found.")

        ratios = {
            name: max(int(self.task_sampling_ratio.get(name, 1)), 1)
            for name in task_names
        }
        loader_lengths = {name: len(train_loaders[name]) for name in task_names}
        scale = max(loader_lengths[name] / ratios[name] for name in task_names)

        schedule = []
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
    def _extract_case_ids(case_id_value: Any, batch_size: int):
        if isinstance(case_id_value, (list, tuple)):
            case_ids = [str(x) for x in case_id_value]
            if len(case_ids) < batch_size:
                case_ids.extend([""] * (batch_size - len(case_ids)))
            return case_ids[:batch_size]
        if case_id_value is None:
            return [""] * batch_size
        return [str(case_id_value)] * batch_size

    @torch.no_grad()
    def export_best_epoch_predictions(self, val_loaders: Dict[str, Any], epoch: int) -> None:
        self.model.eval()
        rows = []
        for loader_name, loader in val_loaders.items():
            for raw_batch in tqdm(loader, desc=f"Export-{loader_name}", leave=False):
                batch = self._move_batch_to_device(raw_batch)
                outputs = self.model(batch)
                logits = outputs["cls_logits"]
                if logits is None:
                    raise ValueError(f"cls_logits is None during classification export for {loader_name}")

                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                preds = np.argmax(probs, axis=1)
                labels = batch["label_cls"].detach().cpu().numpy()
                case_ids = self._extract_case_ids(raw_batch.get("case_id"), probs.shape[0])

                for idx in range(probs.shape[0]):
                    rows.append(
                        {
                            "loader_name": loader_name,
                            "case_id": case_ids[idx],
                            "gt": int(labels[idx]),
                            "pred": int(preds[idx]),
                            "score_0": float(probs[idx, 0]),
                            "score_1": float(probs[idx, 1]),
                        }
                    )

        output_path = os.path.join(
            self.save_dir,
            "best_epoch_outputs",
            f"epoch_{epoch:03d}",
            "classification",
            "val_predictions.csv",
        )
        save_classification_csv(output_path, rows)
        self.logger.info(f"Saved best-epoch classification predictions to {output_path}")

    def train_one_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        batch = self._move_batch_to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            outputs = self.model(batch)
            logits = outputs["cls_logits"]
            if logits is None:
                raise ValueError("cls_logits is None for classification task.")
            loss = self.criterion(logits, batch["label_cls"])
            loss_aux: Dict[str, float] = {}
            if self.auc_margin is not None:
                loss_rank = self.auc_margin(logits, batch["label_cls"])
                loss = loss + self.auc_margin_weight * loss_rank
                loss_aux["loss_auc"] = float(loss_rank.detach().cpu().item())

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {"loss_total": float(loss.detach().cpu().item()), **loss_aux}

    @torch.no_grad()
    def validate_one_loader(self, loader, loader_name: str) -> Dict[str, float]:
        self.model.eval()
        logits_list = []
        labels_list = []

        for batch in tqdm(loader, desc=f"Val-{loader_name}", leave=False):
            batch = self._move_batch_to_device(batch)
            outputs = self.model(batch)
            logits = outputs["cls_logits"]
            if logits is None:
                raise ValueError(f"cls_logits is None during classification validation for {loader_name}")
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(batch["label_cls"].detach().cpu().numpy())

        logits_np = stack_numpy(logits_list)
        labels_np = stack_numpy(labels_list)
        acc = binary_accuracy_from_logits(logits_np, labels_np)
        auc = binary_auc_from_logits(logits_np, labels_np)
        score = 0.5 * (acc + auc)
        return {"acc": acc, "auc": auc, "score": score}

    @torch.no_grad()
    def validate(self, val_loaders: Dict[str, Any]) -> Dict[str, float]:
        self.model.eval()
        all_metrics = {}
        scores = []
        for loader_name, loader in val_loaders.items():
            metric = self.validate_one_loader(loader, loader_name)
            all_metrics[f"{loader_name}_acc"] = metric["acc"]
            all_metrics[f"{loader_name}_auc"] = metric["auc"]
            all_metrics[f"{loader_name}_score"] = metric["score"]
            scores.append(metric["score"])
        all_metrics["mean_score"] = float(np.mean(scores)) if scores else 0.0
        return all_metrics

    def train_one_epoch(self, epoch: int, train_loaders: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        if self.frozen_eval_keywords:
            set_modules_eval_by_keywords(self.model, self.frozen_eval_keywords)
        train_schedule = self._build_train_schedule(train_loaders)
        iterator_dict = self._loader_to_iterator_dict(train_loaders)
        steps_per_epoch = len(train_schedule)

        running_loss = 0.0
        running_loss_auc = 0.0
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
            running_loss_auc += log_dict.get("loss_auc", 0.0)
            steps_done += 1

            # --- debug: perf timing end ---
            if diag_hook is not None:
                # record_forward_done covers the full compute (forward+backward+optimizer)
                diag_hook.record_forward_done()

            self._global_step += 1

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = running_loss / steps_done
            pbar.set_postfix(loss=f"{avg_loss:.4f}", task=loader_name)

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
                    f"task={loader_name} loss={avg_loss:.4f}"
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
            "train_loss_auc": running_loss_auc / denom,
        }

    def fit(self, loaders: Dict[str, Dict[str, Any]]):
        train_loaders = loaders["train"]
        val_loaders = loaders["val"]
        has_val = bool(val_loaders)

        self.logger.info("Start Stage2 Classification Training")
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
                    f"train_loss={train_log['train_loss']:.4f}"
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
                    f"val_mean_score={val_log['mean_score']:.4f}"
                )
                for k, v in val_log.items():
                    self.logger.info(f"    {k}: {v:.4f}")
                if self.auc_margin is not None:
                    self.logger.info(f"    train_loss_auc: {train_log['train_loss_auc']:.4f}")
            else:
                self.logger.info(
                    f"[Epoch {epoch}/{self.max_epochs}] "
                    f"time={epoch_time:.1f}s "
                    f"train_loss={train_log['train_loss']:.4f}"
                )
                if self.auc_margin is not None:
                    self.logger.info(f"    train_loss_auc: {train_log['train_loss_auc']:.4f}")

            latest_path = os.path.join(self.save_dir, "latest_stage2_cls.pth")
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
                    prefix="best_stage2_cls",
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
                self.export_best_epoch_predictions(val_loaders, epoch)
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