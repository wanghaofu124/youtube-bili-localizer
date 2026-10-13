from __future__ import annotations

from pathlib import Path
import json
import subprocess

from collections.abc import Callable
from .cancellation import is_cancellation_requested
from .util import require_command, run


def audio_stream_channels(video_path: Path) -> int | None:
    """Return channels for the first selected audio stream, if probeable."""
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels", "-of", "json", str(video_path.resolve())],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(completed.stdout).get("streams", [])
        value = streams[0].get("channels") if streams else None
        return int(value) if value is not None else None
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None


def extract_audio(video_path: Path, output_path: Path | None = None, cancel_check: Callable[[], bool] | None = None, channel: str = "mix") -> Path:
    if channel not in {"mix", "left", "right"}:
        raise ValueError("audio channel must be mix, left, or right")
    require_command("ffmpeg")
    video_path = video_path.resolve()
    output_path = output_path or video_path.with_suffix(".wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
            "ffmpeg",
            "-y",
            "-copyts",
            "-start_at_zero",
            "-i",
            str(video_path),
            "-vn",
        ]
    if channel != "mix":
        command.extend(["-map", "0:a:0"])
    pan = "pan=mono|c0=FL" if channel == "left" else "pan=mono|c0=FR" if channel == "right" else None
    command.extend(["-af", f"{pan},aresample=async=1:first_pts=0" if pan else "aresample=async=1:first_pts=0", "-ac", "1"])
    command.extend([
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ])
    run(
        command,
        cancel_check=cancel_check or is_cancellation_requested,
    )
    return output_path
