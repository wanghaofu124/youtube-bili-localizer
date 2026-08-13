from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

from .cancellation import check_cancelled as legacy_check_cancelled
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
    beam_size: int = 5,
    log: LogFn | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> list[Segment]:
    check = cancel_check or legacy_check_cancelled
    check()
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
            # word_timestamps=True：让 whisper 用词级对齐重算段边界。
            # 实测不开启时（仅 VAD 粗块），有 BGM 的视频会把数秒静默并入同一条段
            # （如 0-11s 一条字幕），开启后能精确切到真实语音边界（如 0.0-1.0 / 5.9-7.5）。
            whisper_segments, _info = model.transcribe(
                str(audio_path),
                language=language,
                initial_prompt=_clean_initial_prompt(initial_prompt),
                vad_filter=True,
                beam_size=max(1, min(10, beam_size)),
                condition_on_previous_text=True,
                word_timestamps=True,
            )
            raw_segments = []
            for item in whisper_segments:
                check()
                text = item.text.strip()
                if text:
                    raw_segments.append(Segment(start=float(item.start), end=float(item.end), text=text))
            segments = _split_long_segments(raw_segments)
            segments = _align_segment_starts(segments, audio_path)
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


def _align_segment_starts(segments: list[Segment], audio_path: Path) -> list[Segment]:
    """把每条字幕的开始时间对齐到音频里的真实语音起点。

    faster-whisper 的段时间戳（含词级对齐）实测系统性比真实语音早
    0.1~0.5 秒（尤其有 BGM 时），导致字幕在声音出现前就显示。
    这里用短窗 RMS 能量在每条段 start 前后找第一个语音帧；
    只后移不提前：若检测到更晚的语音起点则对齐过去，否则至少后移 0.25s，
    保证字幕不早于声音出现。
    """
    try:
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        return segments
    try:
        sample_rate, data = wavfile.read(str(audio_path))
        if data.ndim > 1:
            mono = data.mean(axis=1)
        else:
            mono = data
        mono = mono.astype(np.float32) / (np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0)
    except Exception:
        return segments
    window = max(80, sample_rate // 40)  # 25ms
    rms = np.sqrt(np.convolve(mono * mono, np.ones(window) / window, mode="same"))
    # 阈值：取 40 百分位的 3 倍，且不低于 -30dBFS
    threshold = max(0.0316, float(np.percentile(rms, 40)) * 3.0)

    def _first_voice_frame(start: float, end: float) -> float | None:
        begin = max(0, int((start - 0.6) * sample_rate))
        stop = int((end + 0.2) * sample_rate)
        segment = rms[begin:stop]
        hits = np.where(segment > threshold)[0]
        if len(hits) == 0:
            return None
        # 需要至少连续 3 帧（约 75ms）才算语音起点，避免瞬时噪声
        groups = np.split(hits, np.where(np.diff(hits) > sample_rate // 25)[0] + 1)
        for group in groups:
            if len(group) >= 3:
                return (begin + group[0]) / sample_rate
        return None

    aligned: list[Segment] = []
    for segment in segments:
        start = segment.start
        end = segment.end
        voice_at = _first_voice_frame(start, end)
        if voice_at is not None and voice_at > start + 0.1:
            start = voice_at
        else:
            # 能量检测不可靠（BGM 强时）也至少后移 0.25s，消除"字幕早于声音"
            start = start + 0.25
        shift = start - segment.start
        aligned.append(Segment(start=start, end=end + shift, text=segment.text))
    # 校正后可能出现重叠，规整一下
    for index, segment in enumerate(aligned):
        if index < len(aligned) - 1:
            next_start = aligned[index + 1].start
            if segment.end > next_start - 0.05:
                segment.end = max(segment.start + 0.1, next_start - 0.05)
    return aligned


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
