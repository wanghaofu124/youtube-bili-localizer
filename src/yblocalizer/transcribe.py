from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

from .models import Segment, save_segments
from .subtitle import write_srt

LogFn = Callable[[str], None]


def transcribe_audio(
    audio_path: Path,
    segments_json: Path,
    srt_path: Path,
    model_size: str = "tiny",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    initial_prompt: str | None = None,
    log: LogFn | None = None,
) -> list[Segment]:
    if device.strip().lower() in {"cuda", "auto"}:
        _prepend_nvidia_cuda_dll_paths()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed. Run: pip install -r requirements.txt") from exc

    logger = log or (lambda _message: None)
    last_error: Exception | None = None
    for resolved_device, resolved_compute_type, note in _transcribe_attempts(device, compute_type):
        if note:
            logger(note)
        try:
            model = WhisperModel(model_size, device=resolved_device, compute_type=resolved_compute_type)
            whisper_segments, _info = model.transcribe(
                str(audio_path),
                language=language,
                initial_prompt=_clean_initial_prompt(initial_prompt),
                vad_filter=True,
                beam_size=5,
                condition_on_previous_text=True,
            )
            raw_segments = [
                Segment(start=float(item.start), end=float(item.end), text=item.text.strip())
                for item in whisper_segments
                if item.text.strip()
            ]
            segments = _split_long_segments(raw_segments)
            break
        except Exception as exc:
            last_error = exc
            if resolved_device == "cuda":
                logger(
                    "CUDA transcription failed; trying a safer fallback. "
                    f"device={resolved_device}, compute_type={resolved_compute_type}, error={exc}"
                )
                continue
            raise RuntimeError(
                "faster-whisper transcription failed on CPU fallback. "
                f"device={resolved_device}, compute_type={resolved_compute_type}, error={exc}"
            ) from exc
    else:
        raise RuntimeError(f"faster-whisper transcription failed: {last_error}") from last_error
    save_segments(segments_json, segments)
    write_srt(srt_path, segments, display_mode="source")
    return segments


def _clean_initial_prompt(initial_prompt: str | None) -> str | None:
    if not initial_prompt:
        return None
    cleaned = " ".join(initial_prompt.split())
    return cleaned[:1000] or None


def _split_long_segments(segments: list[Segment], max_duration: float = 4.8) -> list[Segment]:
    output: list[Segment] = []
    for segment in segments:
        output.extend(_split_long_segment(segment, max_duration=max_duration))
    return output


def _split_long_segment(segment: Segment, max_duration: float) -> list[Segment]:
    duration = segment.end - segment.start
    if duration <= max_duration:
        return [segment]
    parts = _sentence_parts(segment.text)
    if len(parts) <= 1:
        return [segment]
    weights = [max(1, len(re.sub(r"\s+", "", part))) for part in parts]
    total = sum(weights)
    output: list[Segment] = []
    cursor = segment.start
    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            end = segment.end
        else:
            end = cursor + duration * weights[index] / total
        if end - cursor >= 0.25:
            output.append(Segment(start=cursor, end=end, text=part))
        cursor = end
    return output or [segment]


def _sentence_parts(text: str) -> list[str]:
    parts = [part.strip() for part in re.findall(r"[^.!?]+[.!?]*", text) if part.strip()]
    if len(parts) <= 1:
        return [text.strip()]
    return parts


def _transcribe_attempts(device: str, compute_type: str) -> list[tuple[str, str, str]]:
    requested_device = (device or "cpu").strip().lower()
    requested_compute = (compute_type or "int8").strip().lower()
    if requested_compute == "default":
        requested_compute = "float16" if requested_device == "cuda" else "int8"

    if requested_device == "cuda":
        attempts = [(requested_device, requested_compute, "Using CUDA for faster-whisper transcription.")]
        if requested_compute not in {"float16", "int8_float16"}:
            attempts.append(("cuda", "float16", "Retrying CUDA with compute_type=float16."))
        attempts.append(("cpu", "int8", "Falling back to CPU transcription with compute_type=int8."))
        return _dedupe_attempts(attempts)

    if requested_device == "auto":
        return [
            ("cuda", "float16", "Auto device: trying CUDA first."),
            ("cpu", "int8", "Auto device: falling back to CPU if CUDA is unavailable."),
        ]

    return [(requested_device, requested_compute, "")]


def _dedupe_attempts(attempts: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    output: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for device, compute_type, note in attempts:
        key = (device, compute_type)
        if key in seen:
            continue
        seen.add(key)
        output.append((device, compute_type, note))
    return output


def _prepend_nvidia_cuda_dll_paths() -> None:
    try:
        import nvidia
    except ImportError:
        return
    root = Path(nvidia.__file__).parent
    candidates = [
        root / "cuda_runtime" / "bin",
        root / "cublas" / "bin",
        root / "cudnn" / "bin",
    ]
    existing = os.environ.get("PATH", "")
    parts = [str(path) for path in candidates if path.exists()]
    if not parts:
        return
    current_lower = {item.lower() for item in existing.split(os.pathsep) if item}
    missing = [item for item in parts if item.lower() not in current_lower]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + [existing])
