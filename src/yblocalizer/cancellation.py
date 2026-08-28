from __future__ import annotations

import threading

from .runtime import CancellationRequested


_cancellation_requested = threading.Event()


def is_cancellation_requested() -> bool:
    return _cancellation_requested.is_set()


def check_cancelled() -> None:
    if is_cancellation_requested():
        raise CancellationRequested("任务已被用户中断")
