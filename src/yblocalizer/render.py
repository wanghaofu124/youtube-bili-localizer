from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

from .cancellation import is_cancellation_requested
from .util import require_command, run


def _probe_video_size(path: Path) -> tuple[int, int] | None:
    """Best-effort video display size via ffprobe."""
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        stream = ((json.loads(completed.stdout).get("streams") or [{}])[0])
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return None


def burn_subtitles(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
    font_name: str = "Microsoft YaHei",
    font_size: int = 24,
    primary_color: str = "&H00FFFFFF",
    outline_color: str = "&H00000000",
    outline: int = 1,
    shadow: int = 0,
    margin_v: int | None = None,
    raised_margin: bool = False,
    crf: int = 20,
    margin_ratio: float = 0.055,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    require_command("ffmpeg")
    video_path = video_path.resolve()
    subtitle_path = subtitle_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    size = _probe_video_size(video_path)
    if size is not None:
        # 以视频实际分辨率为 ASS 坐标空间，字号/边距按像素计算，
        # 避免 libass 默认 384x288 脚本空间导致的超大字号和位置错乱。
        width, height = size
        # 按短边适配：横屏以高度、竖屏(shorts 1080x1920)以宽度为基准，
        # 避免竖屏视频字号被 16:9 公式放大 3 倍以上导致字幕撑满/越界。
        base = min(width, height)
        actual_font = max(18, round(font_size * base / 540))  # 24 @1080p -> 48px
        ratio = max(0.01, min(0.4, margin_ratio))
        actual_margin = max(24, round(height * ratio)) if margin_v is None else margin_v
        if raised_margin:
            # OCR 模式：把中文字幕抬到原英文字幕上方
            actual_margin = max(actual_margin, round(height * 0.15))
        style = (
            f"PlayResX={width},PlayResY={height},FontName={font_name},FontSize={actual_font},"
            f"PrimaryColour={primary_color},OutlineColour={outline_color},"
            f"Outline={outline},Shadow={shadow},MarginV={actual_margin},Alignment=2,WrapStyle=2"
        )
    else:
        actual_font = max(14, font_size)
        actual_margin = margin_v if margin_v is not None else 24
        style = (
            f"FontName={font_name},FontSize={actual_font},"
            f"PrimaryColour={primary_color},OutlineColour={outline_color},"
            f"Outline={outline},Shadow={shadow},MarginV={actual_margin},Alignment=2,WrapStyle=2"
        )
    filter_expr = f"subtitles={subtitle_path.name}:force_style='{style}'"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            filter_expr,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(max(14, min(32, crf))),
            # 重新编码音频为aac，确保兼容性
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            # 如果原始视频没有音频流，也继续处理不报错
            "-map",
            "0:v:0?",
            "-map",
            "0:a:0?",
            str(output_path),
        ],
        cwd=subtitle_path.parent,
        cancel_check=cancel_check or is_cancellation_requested,
    )
    return output_path
