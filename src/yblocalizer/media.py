from __future__ import annotations

from pathlib import Path

from .cancellation import is_cancellation_requested
from .util import require_command, run


def extract_audio(video_path: Path, output_path: Path | None = None) -> Path:
    require_command("ffmpeg")
    video_path = video_path.resolve()
    output_path = output_path or video_path.with_suffix(".wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        cancel_check=is_cancellation_requested,
    )
    return output_path
