"""Diagnostics utilities for --debug deep mode.

Provides gradient monitoring, activation statistics, and performance profiling
that can be injected into the training loop with minimal overhead when disabled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class DiagnosticsConfig:
    """Configuration for deep-diagnostic monitoring."""

    grad_monitor: bool = False
    act_monitor: bool = False
    perf_monitor: bool = False
    log_interval: int = 1


class DiagnosticsHook:
    """Attaches forward hooks and collects gradient/activation/performance stats."""

    def __init__(
        self,
        model: nn.Module,
        config: DiagnosticsConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._act_stats: Dict[str, torch.Tensor] = {}

        # performance state
        self._step_times: Dict[str, List[float]] = {
            "data": [],
            "forward": [],
            "backward": [],
            "optimizer": [],
            "total": [],
        }
        self._tick: Optional[float] = None
        self._cuda_events: Dict[str, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    _HOOK_POINTS: List[Tuple[str, str]] = [
        # (display_name, dotted attr path from model root)
        # Use top-level attributes that exist on the UnifiedModel / task models.
        # Hooks fire on the *output* of the named submodule.
    ]

    def _find_hook_targets(self) -> List[Tuple[str, nn.Module]]:
        """Discover submodules to monitor."""
        targets: List[Tuple[str, nn.Module]] = []
        # Try to locate key submodules common to this codebase.
        candidate_attrs = [
            "image_model.backbone.encoder",
            "image_model.backbone",
            "image_model.seg_head",
            "image_model.cls_head",
            "video_model.backbone.encoder",
            "video_model.backbone",
            "video_model.temporal_router",
            "video_model.temporal_router.simple_video_seg_head",
            "video_model.temporal_router.video_cls_head",
        ]
        for attr_path in candidate_attrs:
            try:
                mod = self.model.get_submodule(attr_path)
                targets.append((attr_path, mod))
            except AttributeError:
                continue
        return targets

    def _activation_hook(self, name: str) -> callable:
        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                self._act_stats[name] = output.detach()
            elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                self._act_stats[name] = output[0].detach()

        return hook

    # ------------------------------------------------------------------
    # Gradient stats
    # ------------------------------------------------------------------

    @staticmethod
    def _grad_summary(
        named_params: List[Tuple[str, nn.Parameter]],
        top_n: int = 5,
    ) -> str:
        entries = []
        nan_count = 0
        inf_count = 0

        for name, param in named_params:
            if param.grad is None:
                continue
            g = param.grad.detach()
            norm = float(g.norm().item())
            has_nan = bool(torch.isnan(g).any().item())
            has_inf = bool(torch.isinf(g).any().item())
            if has_nan:
                nan_count += 1
            if has_inf:
                inf_count += 1
            entries.append((name, norm, has_nan, has_inf))

        if not entries:
            return "  (no gradients available — all parameters may be frozen)"

        entries.sort(key=lambda x: x[1], reverse=True)
        lines = []

        if top_n > 0 and len(entries) > 2 * top_n:
            lines.append(f"  Top-{top_n} gradient norms:")
            for name, norm, has_nan, has_inf in entries[:top_n]:
                flags = ""
                if has_nan:
                    flags += " [NaN!]"
                if has_inf:
                    flags += " [Inf!]"
                lines.append(f"    {name}: norm={norm:.6f}{flags}")
            lines.append(f"  Bottom-{top_n} gradient norms:")
            for name, norm, has_nan, has_inf in entries[-top_n:]:
                flags = ""
                if has_nan:
                    flags += " [NaN!]"
                if has_inf:
                    flags += " [Inf!]"
                lines.append(f"    {name}: norm={norm:.6f}{flags}")
        else:
            for name, norm, has_nan, has_inf in entries:
                flags = ""
                if has_nan:
                    flags += " [NaN!]"
                if has_inf:
                    flags += " [Inf!]"
                lines.append(f"    {name}: norm={norm:.6f}{flags}")

        total = len(entries)
        lines.append(f"  Layers with grad: {total} | NaN: {nan_count} | Inf: {inf_count}")
        return "\n".join(lines)

    def log_gradient_stats(self, step: int) -> None:
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        header = f"=== [Deep Debug] Step {step} | Gradients ==="
        body = self._grad_summary(trainable)
        print(f"{header}\n{body}\n")

    # ------------------------------------------------------------------
    # Activation stats
    # ------------------------------------------------------------------

    def log_activation_stats(self, step: int) -> None:
        if not self._act_stats:
            print(f"=== [Deep Debug] Step {step} | Activations ===\n"
                  "  (no activations captured — hooks may not have fired)\n")
            return

        lines = [f"=== [Deep Debug] Step {step} | Activations ==="]
        for name, tensor in self._act_stats.items():
            t = tensor.float()
            lines.append(
                f"  {name} {list(tensor.shape)}: "
                f"mean={t.mean().item():.4f}, std={t.std().item():.4f}, "
                f"min={t.min().item():.4f}, max={t.max().item():.4f}"
            )
        print("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Performance profiling
    # ------------------------------------------------------------------

    def start_step_timer(self) -> None:
        self._tick = time.perf_counter()
        if self.device.type == "cuda":
            self._cuda_events["data_end"] = (torch.cuda.Event(enable_timing=True),
                                              torch.cuda.Event(enable_timing=True))

    def record_data_loading_done(self) -> None:
        if self._tick is None:
            return
        now = time.perf_counter()
        self._step_times["data"].append(now - self._tick)
        self._tick = now

    def record_forward_done(self) -> None:
        if self._tick is None:
            return
        now = time.perf_counter()
        self._step_times["forward"].append(now - self._tick)
        # compute total for this step (data + compute)
        total = sum(
            self._step_times[key][-1]
            for key in ("data", "forward")
            if self._step_times[key]
        )
        self._step_times["total"].append(total)
        self._tick = None

    def record_backward_done(self) -> None:
        if self._tick is None:
            return
        now = time.perf_counter()
        self._step_times["backward"].append(now - self._tick)
        self._tick = now

    def record_optimizer_done(self) -> None:
        if self._tick is None:
            return
        now = time.perf_counter()
        self._step_times["optimizer"].append(now - self._tick)
        total = sum(
            self._step_times[key][-1]
            for key in ("data", "forward", "backward", "optimizer")
            if self._step_times[key]
        )
        self._step_times["total"].append(total)
        self._tick = None

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds >= 1.0:
            return f"{seconds:.3f}s"
        if seconds >= 0.001:
            return f"{seconds*1000:.1f}ms"
        return f"{seconds*1_000_000:.0f}μs"

    def log_performance(self, step: int) -> None:
        lines = [f"=== [Deep Debug] Step {step} | Performance ==="]
        parts = []
        label_map = {"data": "data", "forward": "compute", "backward": "bwd", "optimizer": "opt", "total": "total"}
        for key, label in label_map.items():
            if self._step_times[key]:
                parts.append(f"{label}:{self._format_time(self._step_times[key][-1])}")
        lines.append("  " + " | ".join(parts))

        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            lines.append(f"  GPU: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
        print("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DiagnosticsHook":
        if self.config.act_monitor:
            for name, module in self._find_hook_targets():
                handle = module.register_forward_hook(self._activation_hook(name))
                self._handles.append(handle)
        return self

    def __exit__(self, *args: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._act_stats.clear()
