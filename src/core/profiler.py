from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torch.profiler import ProfilerActivity, profile, schedule as profiler_schedule

from omegaconf import DictConfig, OmegaConf

from .distributed import is_global_zero


def _as_list(value: Optional[Iterable[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _format_top_ops(events, attribute: str, topk: int) -> List[str]:
    metrics = []
    for evt in events:
        metric = getattr(evt, attribute, None)
        if metric is None or metric <= 0:
            continue
        metrics.append((evt.key, metric))
    metrics.sort(key=lambda item: item[1], reverse=True)
    formatted = []
    for idx, (name, value) in enumerate(metrics[:topk]):
        formatted.append(f"{idx + 1}. {name}: {value / 1000.0:.2f} ms")
    return formatted


class TrainingProfiler:
    """Light-weight wrapper around ``torch.profiler.profile`` for per-epoch summaries."""

    def __init__(self, cfg: Optional[DictConfig], logger=None):
        cfg = cfg or {}
        self.logger = logger
        self.enabled = bool(cfg.get("enabled", True))
        self.log_tensorboard = bool(cfg.get("tensorboard", True))
        self.topk = int(cfg.get("topk", 5))
        self.record_shapes = bool(cfg.get("record_shapes", False))
        self.profile_memory = bool(cfg.get("profile_memory", False))
        self.with_stack = bool(cfg.get("with_stack", False))
        self.rank_zero_only = bool(cfg.get("rank_zero_only", True))
        schedule_cfg = {}
        if hasattr(cfg, "get"):
            schedule_cfg = cfg.get("schedule", {})
        if isinstance(schedule_cfg, DictConfig):
            schedule_cfg = OmegaConf.to_container(schedule_cfg, resolve=True)
        elif schedule_cfg is None:
            schedule_cfg = {}
        elif not isinstance(schedule_cfg, dict):
            schedule_cfg = dict(schedule_cfg)
        if schedule_cfg:
            wait = int(schedule_cfg.get("wait", 1))
            warmup = int(schedule_cfg.get("warmup", 1))
            active = int(schedule_cfg.get("active", 4))
            repeat = int(schedule_cfg.get("repeat", 1))
            skip_first = int(schedule_cfg.get("skip_first", 0))
            self._schedule = profiler_schedule(
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat,
                skip_first=skip_first,
            )
        else:
            self._schedule = None

        if self.rank_zero_only and not is_global_zero():
            self.enabled = False

        self.activities: List[ProfilerActivity] = []
        self._profile = None
        self._step_count = 0
        self._current_epoch = None
        self._global_step_start = 0

        if not self.enabled:
            return

        requested_activities = _as_list(cfg.get("activities", ["cpu", "cuda"]))
        activities: List[ProfilerActivity] = []
        for name in requested_activities:
            lower = str(name).lower()
            if lower == "cpu":
                activities.append(ProfilerActivity.CPU)
            elif lower == "cuda":
                if torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)
                elif is_global_zero():
                    print("[Profiler] CUDA activity requested but CUDA is not available; skipping CUDA profiling.")
            else:
                if is_global_zero():
                    print(f"[Profiler] Unknown profiler activity '{name}', skipping.")
        if not activities:
            if is_global_zero():
                print("[Profiler] No valid activities configured; disabling profiler.")
            self.enabled = False
            return

        self.activities = activities

    def start_epoch(self, epoch_index: int, global_step_start: int):
        if not self.enabled or not self.activities:
            return
        if self._profile is not None:
            self._finalize(error=True)
        self._profile = profile(
            activities=self.activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            schedule=self._schedule,
        )
        self._profile.__enter__()
        self._step_count = 0
        self._current_epoch = epoch_index
        self._global_step_start = global_step_start

    def step(self):
        if not self.enabled or self._profile is None:
            return
        self._profile.step()
        self._step_count += 1

    def stop_epoch(self, epoch_index: int, global_step_end: int, error: Optional[BaseException] = None):
        if not self.enabled or self._profile is None:
            return
        profile_ctx = self._profile
        self._profile = None
        profile_ctx.__exit__(None, None, None)
        if error is not None:
            return

        events = profile_ctx.key_averages()
        cpu_total_us = sum(getattr(evt, "self_cpu_time_total", 0.0) for evt in events)
        cuda_total_us = sum(getattr(evt, "self_cuda_time_total", 0.0) for evt in events)

        scalars = {"profiler/epoch_steps": float(self._step_count)}
        if cpu_total_us > 0:
            scalars["profiler/epoch_cpu_time_ms"] = cpu_total_us / 1000.0
        if cuda_total_us > 0:
            scalars["profiler/epoch_cuda_time_ms"] = cuda_total_us / 1000.0

        epoch_display = epoch_index + 1
        if is_global_zero():
            summary = (
                f"[Profiler][Epoch {epoch_display}] steps={self._step_count}, "
                f"CPU={scalars.get('profiler/epoch_cpu_time_ms', 0.0):.2f}ms"
            )
            if "profiler/epoch_cuda_time_ms" in scalars:
                summary += f", CUDA={scalars['profiler/epoch_cuda_time_ms']:.2f}ms"
            print(summary)

        top_cpu_ops = _format_top_ops(events, "self_cpu_time_total", self.topk)
        top_cuda_ops = _format_top_ops(events, "self_cuda_time_total", self.topk)

        if is_global_zero():
            if top_cpu_ops:
                print("[Profiler] Top CPU ops:")
                for line in top_cpu_ops:
                    print(f"  {line}")
            if top_cuda_ops:
                print("[Profiler] Top CUDA ops:")
                for line in top_cuda_ops:
                    print(f"  {line}")

        if self.logger is not None and self.log_tensorboard and scalars:
            self.logger.log_scalars(scalars, global_step_end)
            text_sections = []
            if top_cpu_ops:
                text_sections.append("Top CPU ops:\n" + "\n".join(top_cpu_ops))
            if top_cuda_ops:
                text_sections.append("Top CUDA ops:\n" + "\n".join(top_cuda_ops))
            if text_sections and hasattr(self.logger, "log_text"):
                self.logger.log_text(
                    f"profiler/epoch_{epoch_display}",
                    "\n\n".join(text_sections),
                    global_step_end,
                )

    def _finalize(self, error: bool):
        if self._profile is None:
            return
        profile_ctx = self._profile
        self._profile = None
        profile_ctx.__exit__(None, None, None)
        if not error:
            profile_ctx.key_averages()

    def close(self):
        if self._profile is not None:
            self._finalize(error=False)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
