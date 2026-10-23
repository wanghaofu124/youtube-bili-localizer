from __future__ import annotations

from copy import copy
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

import pysubs2

from .cancellation import is_cancellation_requested
from .performance import ffmpeg_thread_args, normalize_resource_profile
from .util import require_command, run
from .dependencies import resolve_command


def _probe_video_size(path: Path) -> tuple[int, int] | None:
    """Best-effort video display size via ffprobe."""
    ffprobe = resolve_command("ffprobe")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [
                str(ffprobe), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "json", str(path),
            ],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        stream = ((json.loads(completed.stdout).get("streams") or [{}])[0])
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return None


def subtitle_line_capacity(
    video_path: Path,
    *,
    font_size: int,
    outline: int,
    shadow: int,
) -> int | None:
    """Estimate a conservative full-width glyph budget for one subtitle row.

    ASS font sizes are scaled to the smaller video dimension in
    ``_prepare_ass``. Using a fixed 18-character wrap therefore wastes about
    half the available width on ordinary landscape video. Keep a 20% side
    margin, account for outline/shadow, and cap rows at 32 full-width glyphs
    for readability. ``None`` asks the caller to use the conservative legacy
    width when video geometry cannot be probed.
    """
    size = _probe_video_size(video_path)
    if not size:
        return None
    width, height = size
    scaled_font = _scaled_ass_font_size(font_size, size)
    edge_effect = 2 * max(0, int(outline)) + 2 * abs(int(shadow))
    usable_width = max(1.0, width * 0.80 - edge_effect)
    return max(1, min(32, math.floor(usable_width / scaled_font)))


def _scaled_ass_font_size(font_size: int, size: tuple[int, int] | None) -> int:
    if size:
        return max(18, round(max(1, int(font_size)) * min(size) / 540))
    return max(14, int(font_size))


def _probe_video_duration(path: Path) -> float | None:
    ffprobe = resolve_command("ffprobe")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [str(ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        duration = float((json.loads(completed.stdout).get("format") or {}).get("duration") or 0)
        return duration if duration > 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return None


def render_encoder_status() -> dict[str, bool]:
    """Return encoders that are realistically usable on this machine."""
    ffmpeg = resolve_command("ffmpeg")
    if ffmpeg is None:
        return {"cpu": False, "nvidia": False}
    try:
        completed = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        encoders = completed.stdout + completed.stderr
    except (OSError, subprocess.SubprocessError):
        encoders = ""
    return {
        "cpu": "libx264" in encoders,
        "nvidia": "h264_nvenc" in encoders and resolve_command("nvidia-smi") is not None,
    }


def _video_encoder_args(encoder: str, crf: int) -> tuple[str, list[str]]:
    mode = str(encoder or "auto").strip().lower()
    if mode not in {"auto", "cpu", "nvidia"}:
        raise ValueError("渲染编码器必须是 auto、cpu 或 nvidia。")
    status = render_encoder_status()
    if mode == "nvidia" and not status["nvidia"]:
        raise RuntimeError("未检测到可用的 NVIDIA NVENC。请改用“自动”或“CPU”。")
    selected = "nvidia" if mode in {"auto", "nvidia"} and status["nvidia"] else "cpu"
    if selected == "nvidia":
        return selected, [
            "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
            "-cq", str(max(14, min(32, crf))), "-b:v", "0",
        ]
    return selected, [
        "-c:v", "libx264", "-preset", "veryfast",
        "-crf", str(max(14, min(32, crf))),
    ]


def _ass_color(value: str) -> pysubs2.Color:
    encoded = str(value).upper().removeprefix("&H").zfill(8)[-8:]
    alpha, blue, green, red = (int(encoded[index:index + 2], 16) for index in range(0, 8, 2))
    return pysubs2.Color(red, green, blue, alpha)


# Bilingual presets only style the source-language line; the translated line
# keeps the Default style, so single-language output never changes.
SUBTITLE_PRESETS: dict[str, dict[str, Any]] = {
    "soft": {"scale": 0.85, "color": "&H00C8C8C8"},     # 弱化原文（默认）：小一号、浅灰
    "compact": {"scale": 0.7, "color": "&H00A0A0A0"},   # 小字原文：更小、更灰
    "equal": {"scale": 1.0, "color": None},             # 两行一致：与译文同字号同色
}


def _style_bilingual_source_lines(text: str, display_mode: str, source_lines: int) -> str:
    """Switch the source lines of a bilingual cue to the Secondary style.

    ``source_lines`` comes from subtitle.cue_source_line_counts; the source block
    sits first or last depending on display_mode.
    """
    lines = text.split("\\N")
    source = range(source_lines) if display_mode == "bilingual-source-first" else range(len(lines) - source_lines, len(lines))
    return "\\N".join("{\\rSecondary}" + line + "{\\r}" if index in source else line for index, line in enumerate(lines))


def _prepare_ass(
    subtitle_path: Path,
    *,
    size: tuple[int, int] | None,
    font_name: str,
    font_size: int,
    primary_color: str,
    outline_color: str,
    outline: int,
    shadow: int,
    margin_v: int | None,
    raised_margin: bool,
    margin_ratio: float,
    cue_font_sizes: Mapping[int, int] | None = None,
    expected_cue_count: int | None = None,
    log: Callable[[str], None] | None = None,
    display_mode: str = "translated",
    subtitle_preset: str = "soft",
    cue_source_lines: Mapping[int, int] | None = None,
) -> Path:
    subtitles = pysubs2.load(subtitle_path, encoding="utf-8-sig")
    width, height = size or (384, 288)
    actual_font = _scaled_ass_font_size(font_size, size)
    if margin_v is None:
        actual_margin = max(24, round(height * max(0.01, min(0.4, margin_ratio)))) if size else 24
    else:
        actual_margin = margin_v
    if raised_margin and size:
        actual_margin = max(actual_margin, round(height * 0.15))

    subtitles.info["PlayResX"] = str(width)
    subtitles.info["PlayResY"] = str(height)
    subtitles.info["WrapStyle"] = "2"
    subtitles.info["ScaledBorderAndShadow"] = "yes"
    subtitles.styles["Default"] = pysubs2.SSAStyle(
        fontname=font_name,
        fontsize=actual_font,
        primarycolor=_ass_color(primary_color),
        outlinecolor=_ass_color(outline_color),
        outline=float(outline),
        shadow=float(shadow),
        alignment=pysubs2.Alignment.BOTTOM_CENTER,
        marginv=actual_margin,
    )
    bilingual = display_mode in {"bilingual-source-first", "bilingual-translation-first"}
    if bilingual:
        preset = SUBTITLE_PRESETS[subtitle_preset]
        secondary = copy(subtitles.styles["Default"])
        secondary.fontsize = max(1, round(actual_font * preset["scale"]))
        if preset["color"]:
            secondary.primarycolor = _ass_color(preset["color"])
        subtitles.styles["Secondary"] = secondary
    overrides = dict(cue_font_sizes or {})
    source_lines = dict(cue_source_lines or {})
    if expected_cue_count is not None and len(subtitles.events) != expected_cue_count:
        if log:
            log(
                f"字幕样式提醒：字幕文件解析出 {len(subtitles.events)} 条，"
                f"预期 {expected_cue_count} 条；已取消逐条字号调整，继续渲染。"
            )
        overrides = {}
        source_lines = {}
    elif any(
        not isinstance(index, int) or not 1 <= index <= len(subtitles.events)
        or not isinstance(size, int) or size < 1
        for index, size in overrides.items()
    ):
        if log:
            log("字幕样式提醒：逐条字号方案与字幕条目不一致；已使用默认字号继续渲染。")
        overrides = {}
    for index, event in enumerate(subtitles.events, start=1):
        if bilingual:
            event.text = _style_bilingual_source_lines(event.text, display_mode, source_lines.get(index, 0))
        requested_size = overrides.get(index)
        if requested_size is None:
            event.style = "Default"
            continue
        style_name = f"YBCueFont{requested_size}"
        if style_name not in subtitles.styles:
            style = copy(subtitles.styles["Default"])
            style.fontsize = _scaled_ass_font_size(requested_size, size)
            subtitles.styles[style_name] = style
        event.style = style_name

    ass_path = subtitle_path.with_suffix(".ass")
    subtitles.save(ass_path, encoding="utf-8-sig", format_="ass")
    return ass_path


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
    encoder: str = "auto",
    log: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    resource_profile: str = "balanced",
    progress: Callable[[float], None] | None = None,
    cue_font_sizes: Mapping[int, int] | None = None,
    expected_cue_count: int | None = None,
    display_mode: str = "translated",
    subtitle_preset: str = "soft",
    cue_source_lines: Mapping[int, int] | None = None,
) -> Path:
    require_command("ffmpeg")
    video_path = video_path.resolve()
    subtitle_path = subtitle_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = normalize_resource_profile(resource_profile)
    logger = log or (lambda _message: None)
    logger("正在读取视频参数并准备字幕样式。")
    duration = _probe_video_duration(video_path)

    ass_path = _prepare_ass(
        subtitle_path,
        size=_probe_video_size(video_path),
        font_name=font_name,
        font_size=font_size,
        primary_color=primary_color,
        outline_color=outline_color,
        outline=outline,
        shadow=shadow,
        margin_v=margin_v,
        raised_margin=raised_margin,
        margin_ratio=margin_ratio,
        cue_font_sizes=cue_font_sizes,
        expected_cue_count=expected_cue_count,
        log=logger,
        display_mode=display_mode,
        subtitle_preset=subtitle_preset,
        cue_source_lines=cue_source_lines,
    )
    filter_expr = f"ass={ass_path.name}"
    logger("正在检查可用视频编码器。")
    selected_encoder, encoder_args = _video_encoder_args(encoder, crf)
    logger("字幕渲染使用 NVIDIA NVENC。" if selected_encoder == "nvidia" else "字幕渲染使用 CPU libx264。")
    last_progress = 0.0

    def report_progress(fraction: float) -> None:
        nonlocal last_progress
        if progress:
            fraction = max(last_progress, min(1.0, max(0.0, fraction)))
            if fraction > last_progress:
                last_progress = fraction
                progress(fraction)

    def command(args: list[str]) -> list[str]:
        return [
            "ffmpeg",
            "-y",
            "-progress",
            "pipe:1",
            "-nostats",
            "-i",
            str(video_path),
            "-vf",
            filter_expr,
            *args,
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
            *ffmpeg_thread_args(profile),
            str(output_path),
        ]

    check = cancel_check or is_cancellation_requested
    try:
        logger("正在启动 FFmpeg 硬字幕编码。")
        run(
            command(encoder_args), cwd=ass_path.parent, cancel_check=check,
            resource_profile=profile, progress_callback=report_progress, progress_duration=duration,
        )
    except RuntimeError:
        if str(encoder).lower() != "auto" or selected_encoder != "nvidia":
            raise
        output_path.unlink(missing_ok=True)
        logger("NVIDIA NVENC 启动失败，已自动降级为 CPU libx264。")
        _, cpu_args = _video_encoder_args("cpu", crf)
        logger("正在启动 CPU 硬字幕编码。")
        run(
            command(cpu_args), cwd=ass_path.parent, cancel_check=check,
            resource_profile=profile, progress_callback=report_progress, progress_duration=duration,
        )
    return output_path
