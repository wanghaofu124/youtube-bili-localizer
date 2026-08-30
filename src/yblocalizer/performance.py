from __future__ import annotations

import os
import subprocess
from typing import Any


RESOURCE_PROFILES = {"background", "balanced", "maximum"}


def normalize_resource_profile(value: Any) -> str:
    profile = str(value or "balanced").strip().lower()
    if profile not in RESOURCE_PROFILES:
        raise ValueError("Resource profile must be background, balanced, or maximum.")
    return profile


def cpu_thread_limit(profile: str) -> int:
    cores = max(1, os.cpu_count() or 4)
    selected = normalize_resource_profile(profile)
    if selected == "background":
        return max(1, min(4, cores // 4))
    if selected == "balanced":
        return max(2, min(8, cores // 2))
    return cores


def ffmpeg_thread_args(profile: str) -> list[str]:
    selected = normalize_resource_profile(profile)
    return [] if selected == "maximum" else ["-threads", str(cpu_thread_limit(selected))]


def download_rate_limit(profile: str) -> int | None:
    """Return bytes/second, leaving bandwidth for the rest of the desktop."""
    override = os.getenv("YBLOCALIZER_DOWNLOAD_LIMIT_MIB", "").strip()
    if override:
        try:
            value = float(override)
        except ValueError:
            value = 0
        return int(value * 1024 * 1024) if value > 0 else None
    selected = normalize_resource_profile(profile)
    if selected == "background":
        return 2 * 1024 * 1024
    if selected == "balanced":
        return 6 * 1024 * 1024
    return None


def translation_worker_limit(profile: str) -> int:
    """Bound paid API concurrency without turning translation into a request storm."""
    return 1 if normalize_resource_profile(profile) == "background" else 2


def lower_process_priority(process: subprocess.Popen[Any], profile: str = "balanced") -> None:
    """Keep heavy child tools responsive to cancellation without starving the UI."""
    if os.name != "nt" or normalize_resource_profile(profile) == "maximum":
        return
    try:
        import ctypes

        below_normal_priority_class = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(int(process._handle), below_normal_priority_class)  # type: ignore[attr-defined]
    except (AttributeError, OSError, TypeError, ValueError):
        return
