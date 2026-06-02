from __future__ import annotations

from pathlib import Path

from .util import require_command, run


def burn_subtitles(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
    font_name: str = "Microsoft YaHei",
    font_size: int = 18,
    primary_color: str = "&H00FFFFFF",
    outline_color: str = "&H00000000",
    outline: int = 1,
    shadow: int = 0,
    margin_v: int = 24,
) -> Path:
    require_command("ffmpeg")
    video_path = video_path.resolve()
    subtitle_path = subtitle_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    style = (
        f"FontName={font_name},FontSize={font_size},PrimaryColour={primary_color},"
        f"OutlineColour={outline_color},Outline={outline},Shadow={shadow},"
        f"MarginV={margin_v},Alignment=2,WrapStyle=2"
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
    )
    return output_path
