from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import threading
import time
from typing import Callable


class CancellationRequested(Exception):
    """Raised when one specific pipeline run is cancelled."""


class CancellationToken:
    """Per-job cancellation primitive; never shared implicitly between jobs."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise CancellationRequested("任务已被用户中断")


class PipelineStage(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    DOWNLOADING = "downloading"
    AUDIO = "audio"
    TRANSCRIBING = "transcribing"
    OCR = "ocr"
    TRANSLATING = "translating"
    RENDERING = "rendering"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


STAGE_LABELS = {
    PipelineStage.QUEUED: "等待开始",
    PipelineStage.PREPARING: "准备素材",
    PipelineStage.DOWNLOADING: "下载视频",
    PipelineStage.AUDIO: "提取音频",
    PipelineStage.TRANSCRIBING: "语音转写",
    PipelineStage.OCR: "画面文字识别",
    PipelineStage.TRANSLATING: "翻译字幕",
    PipelineStage.RENDERING: "渲染成片",
    PipelineStage.PUBLISHING: "投稿辅助",
    PipelineStage.COMPLETED: "处理完成",
    PipelineStage.CANCELLED: "已取消",
    PipelineStage.FAILED: "处理失败",
}


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    stage: PipelineStage
    progress: int
    message: str
    level: str = "info"
    timestamp: float = field(default_factory=time.time)


EventFn = Callable[[PipelineEvent], None]


@dataclass(slots=True)
class PipelineContext:
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    on_event: EventFn | None = None

    def check_cancelled(self) -> None:
        self.cancellation.check()

    def is_cancelled(self) -> bool:
        return self.cancellation.cancelled

    def emit(self, stage: PipelineStage, progress: int, message: str, level: str = "info") -> None:
        if self.on_event:
            self.on_event(PipelineEvent(stage, max(0, min(100, progress)), message, level))
