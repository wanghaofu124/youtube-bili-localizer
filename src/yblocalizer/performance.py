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


class ReviewResourceBusy(RuntimeError):
    """A retryable resource warning, never a request to change model quality."""

    code = "cuda_resource_busy"

    def __init__(self, message: str, resources: dict[str, Any]):
        super().__init__(message)
        self.resources = resources


def review_cuda_memory(device: str, model: str, compute_type: str) -> dict[str, Any]:
    """Lightweight start-time estimate. Unknown probes must not deny a run.

    This is a conservative readiness hint, not a CUDA memory limit or a
    guarantee. On multiple cards the target ordinal is ambiguous, so only
    all-card pressure blocks the run; no other processes are touched.
    """
    if str(device).strip().lower() != "cuda":
        return {"status": "not_required"}
    from .dependencies import resolve_command

    executable = resolve_command("nvidia-smi")
    if executable is None:
        return {"status": "unknown"}
    try:
        probe = subprocess.run(
            [str(executable), "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        if probe.returncode:
            return {"status": "unknown"}
        cards: list[dict[str, int]] = []
        for line in probe.stdout.splitlines():
            if not line.strip():
                continue
            index, total, free = [int(part.strip()) for part in line.split(",")]
            if total <= 0 or free < 0 or free > total:
                return {"status": "unknown"}
            cards.append({"index": index, "total_mib": total, "free_mib": free})
        if not cards:
            return {"status": "unknown"}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"status": "unknown"}
    floors = {"tiny": 768, "base": 1024, "small": 1536, "medium": 3072, "large-v3": 4096}
    name = str(model).strip().lower().removesuffix(".en")
    floor = floors.get(name)
    precision = str(compute_type).strip().lower()
    scale = {"float16": 1.0, "float32": 2.0, "int8": 0.5, "int8_float16": 0.5}.get(precision)
    if floor is None or scale is None:
        return {"status": "unknown", "gpus": cards}
    required = int(floor * scale) + 512
    info: dict[str, Any] = {"status": "ok", "gpus": cards, "estimated_required_mib": required, "estimate_only": True}
    if all(card["free_mib"] < required for card in cards):
        info["status"] = "busy"
        free = max(card["free_mib"] for card in cards)
        raise ReviewResourceBusy(
            f"显存较紧张，本轮暂未启动：可用最多约 {free / 1024:.1f} GB，"
            f"{model} / {compute_type} 估算需预留约 {required / 1024:.1f} GB。"
            "这是按模型规模估算的提示，不是容量保证。释放显存后点击“开始复核”重试；已有进度保留。",
            info,
        )
    return info
