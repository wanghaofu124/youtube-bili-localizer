from __future__ import annotations

import os
import re
import json
import ctypes
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from .cancellation import check_cancelled as legacy_check_cancelled
from .dependencies import require_whisper_model, resolve_command
from .models import Segment, save_segments
from .performance import cpu_thread_limit, normalize_resource_profile
from .subtitle import write_srt
from .subtitle_speech_pack import (
    TimedWord,
    normalize_word_stream,
    pack_cues_from_words,
    save_words,
    timed_words_from_whisper_item,
    words_path_for_segments,
)

LogFn = Callable[[str], None]


class WhisperCudaOutOfMemory(RuntimeError):
    """A definite Whisper CUDA OOM that needs an explicit recovery choice."""

    code = "cuda_oom"


def _is_cuda_oom(error: BaseException | str) -> bool:
    text = str(error).lower()
    return (
        "cuda out of memory" in text
        or "cuda_error_out_of_memory" in text
        or "cuda error: out of memory" in text
        or "cublas_status_alloc_failed" in text
        or ("memory allocation failed" in text and ("cuda" in text or "cublas" in text))
    )


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
    resource_profile: str = "balanced",
) -> list[Segment]:
    if (getattr(sys, "frozen", False) or os.environ.get("YBLOCALIZER_FORCE_WHISPER_WORKER") == "1") and os.environ.get("YBLOCALIZER_WHISPER_WORKER") != "1":
        return _transcribe_audio_isolated(
            audio_path, segments_json, srt_path, model_size, language, device,
            compute_type, initial_prompt, beam_size, log, cancel_check, resource_profile,
        )
    return _transcribe_audio_in_process(
        audio_path, segments_json, srt_path, model_size, language, device,
        compute_type, initial_prompt, beam_size, log, cancel_check, resource_profile,
    )


def _transcribe_audio_in_process(
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
    resource_profile: str = "balanced",
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
    profile = normalize_resource_profile(resource_profile)
    thread_limit = cpu_thread_limit(profile)
    logger(f"资源模式：{profile}；Whisper CPU 线程上限 {thread_limit}。")
    last_error: Exception | None = None
    for resolved_device, resolved_compute_type, note in _transcribe_attempts(device, compute_type):
        if note:
            logger(note)
        try:
            model = WhisperModel(
                str(require_whisper_model(model_size)),
                device=resolved_device,
                compute_type=resolved_compute_type,
                cpu_threads=thread_limit,
                num_workers=1,
            )
            # word_timestamps=True：用词级起止覆盖 VAD 粗块。只读 segment.start/end
            # 时，有 BGM/空镜的视频会把一句台词拖到下一段说话前（实测可超过 50 秒）。
            transcribe_kwargs = {
                "language": language,
                "initial_prompt": _clean_initial_prompt(initial_prompt),
                "vad_filter": True,
                "beam_size": max(1, min(10, beam_size)),
                "condition_on_previous_text": False,
                "word_timestamps": True,
            }
            try:
                whisper_segments, _info = model.transcribe(
                    str(audio_path),
                    # The previous 120 ms pad clipped quiet leading words at
                    # speech onsets (verified at 07:17 in the current video).
                    vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 400},
                    **transcribe_kwargs,
                )
            except TypeError:
                whisper_segments, _info = model.transcribe(str(audio_path), **transcribe_kwargs)
            # Flatten every Whisper chunk into one word clock, then cut by speech
            # pauses. Per-chunk packing leaves mid-phrase breaks at VAD boundaries
            # (e.g. "handed me" | "this box") that no later merge reliably fixes.
            whisper_items = []
            for item in whisper_segments:
                check()
                whisper_items.append(item)
            all_words = _timed_words_with_fallback(whisper_items)
            if all_words:
                all_words = normalize_word_stream(all_words)
                all_words = _recover_vad_speech_gaps(
                    audio_path, model, all_words, transcribe_kwargs, check, logger,
                )
                save_words(words_path_for_segments(segments_json), all_words)
                segments = pack_cues_from_words(all_words)
                logger(f"按说话时间切分：{len(all_words)} 词 → {len(segments)} 条字幕。")
            else:
                segments = []
                for item in whisper_items:
                    segments.extend(_segments_from_whisper(item))
                words_path_for_segments(segments_json).unlink(missing_ok=True)
            segments = _coalesce_micro_segments(segments)
            break
        except Exception as exc:
            last_error = exc
            if resolved_device == "cuda":
                if device.strip().lower() == "cuda":
                    if _is_cuda_oom(exc):
                        raise WhisperCudaOutOfMemory(
                            f"Whisper CUDA 显存不足，已停止自动切换设备。原始错误：{exc}"
                        ) from exc
                    raise RuntimeError(f"Whisper CUDA 转写失败，未自动切换设备。原始错误：{exc}") from exc
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


def _uncovered_speech_gaps(
    speech_intervals: list[tuple[float, float]], words: list[TimedWord],
    *, min_duration: float = 1.25,
) -> list[tuple[float, float]]:
    """Find sustained VAD speech with no word-timed ASR result."""
    gaps: list[tuple[float, float]] = []
    for speech_start, speech_end in speech_intervals:
        cursor = speech_start
        for word in words:
            if word.end <= cursor or word.start >= speech_end:
                continue
            if word.start - cursor >= min_duration:
                gaps.append((cursor, word.start))
            cursor = max(cursor, word.end)
            if cursor >= speech_end:
                break
        if speech_end - cursor >= min_duration:
            gaps.append((cursor, speech_end))
    return gaps


def _recover_vad_speech_gaps(
    audio_path: Path,
    model: object,
    words: list[TimedWord],
    transcribe_kwargs: dict[str, object],
    check: Callable[[], None],
    logger: LogFn,
) -> list[TimedWord]:
    """Recheck only sustained uncovered speech; never invent word timings."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    check()
    audio = decode_audio(str(audio_path), sampling_rate=16000)
    speech = get_speech_timestamps(
        audio,
        VadOptions(min_silence_duration_ms=400, speech_pad_ms=400),
        sampling_rate=16000,
    )
    intervals = [(entry["start"] / 16000, entry["end"] / 16000) for entry in speech]
    candidates = _uncovered_speech_gaps(intervals, words)
    if not candidates:
        logger("人声漏检复核：未发现持续有人声却没有词级字幕的区间。")
        return words
    if len(candidates) > 24 or sum(end - start for start, end in candidates) > 120:
        raise RuntimeError(
            f"人声漏检复核发现 {len(candidates)} 处大范围空白；"
            "为避免把背景声音误转成字幕，已停止并要求人工检查源音频。"
        )
    logger(f"人声漏检复核：检查 {len(candidates)} 个有语音证据的空白区间。")
    recovered: list[TimedWord] = []
    for gap_start, gap_end in candidates:
        check()
        crop_start = max(0.0, gap_start - 2.0)
        crop_end = min(len(audio) / 16000, gap_end + 2.0)
        crop = audio[int(crop_start * 16000):int(crop_end * 16000)]
        retry_kwargs = dict(transcribe_kwargs)
        retry_kwargs["vad_filter"] = True
        segments, _ = model.transcribe(
            crop,
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 400},
            **retry_kwargs,
        )
        checked: list[TimedWord] = []
        for segment in segments:
            check()
            if float(getattr(segment, "avg_logprob", -1.0)) < -0.7:
                continue
            if float(getattr(segment, "no_speech_prob", 1.0)) > 0.3:
                continue
            checked.extend(
                TimedWord(crop_start + word.start, crop_start + word.end, word.word)
                for word in _timed_words_with_fallback([segment])
            )
        if not any(gap_start <= (word.start + word.end) / 2 <= gap_end for word in checked):
            continue
        added = 0
        for word in checked:
            midpoint = (word.start + word.end) / 2
            if not gap_start - 1.5 <= midpoint <= gap_end + 1.5:
                continue
            if any(word.start < old.end - 0.08 and word.end > old.start + 0.08 for old in words):
                continue
            recovered.append(word)
            added += 1
        if added:
            logger(f"复核区间 {gap_start:.1f}–{gap_end:.1f}s：补入 {added} 个候选词。")
    if not recovered:
        logger("人声漏检复核：没有足够可靠的新词；原有转写保持不变。")
        return words
    logger(f"人声漏检复核：共补入 {len(recovered)} 个有真实时间戳的词。")
    return normalize_word_stream(words + recovered)


def _transcribe_audio_isolated(
    audio_path: Path,
    segments_json: Path,
    srt_path: Path,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
    initial_prompt: str | None,
    beam_size: int,
    log: LogFn | None,
    cancel_check: Callable[[], None] | None,
    resource_profile: str,
) -> list[Segment]:
    logger = log or (lambda _message: None)
    check = cancel_check or legacy_check_cancelled
    segments_json.parent.mkdir(parents=True, exist_ok=True)
    control_dir = Path(tempfile.mkdtemp(prefix=".whisper-worker-", dir=segments_json.parent))
    try:
        attempts = [(device, compute_type)]
        if device.strip().lower() == "auto":
            attempts.append(("cpu", "int8"))
        last_error: RuntimeError | None = None
        for index, (attempt_device, attempt_compute) in enumerate(attempts):
            if index:
                logger("Whisper 子进程异常退出，主程序仍在运行；正在改用 CPU / int8 重试。")
            try:
                items, temporary_json, temporary_srt = _run_whisper_worker(
                    control_dir, audio_path, model_size, language, attempt_device,
                    attempt_compute, initial_prompt, beam_size, logger, check, resource_profile,
                )
            except WhisperCudaOutOfMemory as exc:
                if device.strip().lower() == "cuda":
                    raise
                last_error = exc
                continue
            except RuntimeError as exc:
                last_error = exc
                continue
            temporary_json.replace(segments_json)
            temporary_srt.replace(srt_path)
            temporary_words = words_path_for_segments(temporary_json)
            if temporary_words.is_file():
                temporary_words.replace(words_path_for_segments(segments_json))
            return items
        raise last_error or RuntimeError("Whisper 子进程未能启动。")
    finally:
        shutil.rmtree(control_dir, ignore_errors=True)


def _run_whisper_worker(
    control_dir: Path,
    audio_path: Path,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
    initial_prompt: str | None,
    beam_size: int,
    logger: LogFn,
    check: Callable[[], None],
    resource_profile: str,
) -> tuple[list[Segment], Path, Path]:
    request_path = control_dir / f"request-{device}.json"
    result_path = control_dir / f"result-{device}.json"
    events_path = control_dir / f"events-{device}.jsonl"
    diagnostic_path = control_dir / f"diagnostic-{device}.log"
    temporary_json = control_dir / f"segments-{device}.json"
    temporary_srt = control_dir / f"subtitles-{device}.srt"
    request_path.write_text(json.dumps({
        "audio_path": str(audio_path),
        "segments_json": str(temporary_json),
        "srt_path": str(temporary_srt),
        "result_path": str(result_path),
        "events_path": str(events_path),
        "model_size": model_size,
        "language": language,
        "device": device,
        "compute_type": compute_type,
        "initial_prompt": initial_prompt,
        "beam_size": beam_size,
        "resource_profile": resource_profile,
    }, ensure_ascii=False), encoding="utf-8")
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--whisper-worker", str(request_path)]
    else:
        command = [sys.executable, "-m", "yblocalizer.whisper_worker", str(request_path)]
    event_offset = 0
    with diagnostic_path.open("w", encoding="utf-8", errors="replace") as diagnostic:
        process = subprocess.Popen(command, stdout=diagnostic, stderr=subprocess.STDOUT, text=True)
        try:
            while process.poll() is None:
                check()
                event_offset = _forward_worker_events(events_path, event_offset, logger)
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        event_offset = _forward_worker_events(events_path, event_offset, logger)
    if process.returncode != 0:
        code = process.returncode & 0xFFFFFFFF
        detail = ""
        error_code = ""
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                detail = str(result.get("error") or "")
                error_code = str(result.get("error_code") or "")
            except (OSError, ValueError):
                pass
        if error_code == WhisperCudaOutOfMemory.code:
            raise WhisperCudaOutOfMemory(detail or "Whisper CUDA 显存不足。")
        if detail:
            raise RuntimeError(f"Whisper 转写失败：{detail}；主程序和已下载素材均已保留。")
        raise RuntimeError(f"Whisper 转写子进程异常退出（0x{code:08X}），主程序和已下载素材均已保留。")
    if not result_path.is_file() or not temporary_json.is_file() or not temporary_srt.is_file():
        raise RuntimeError("Whisper 转写子进程未生成完整结果，已保留原有检查点。")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        if result.get("error_code") == WhisperCudaOutOfMemory.code:
            raise WhisperCudaOutOfMemory(str(result.get("error") or "Whisper CUDA 显存不足。"))
        raise RuntimeError(f"Whisper 转写失败：{result.get('error') or '未知错误'}")
    from .models import load_segments
    return load_segments(temporary_json), temporary_json, temporary_srt


def _forward_worker_events(path: Path, offset: int, logger: LogFn) -> int:
    if not path.is_file():
        return offset
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            try:
                message = json.loads(line).get("message")
            except ValueError:
                message = None
            if message:
                logger(str(message))
        return handle.tell()


def _clean_initial_prompt(initial_prompt: str | None) -> str | None:
    if not initial_prompt:
        return None
    cleaned = " ".join(initial_prompt.split())
    return cleaned[:1000] or None


def _segment_from_whisper(item: object) -> Segment | None:
    text = str(getattr(item, "text", "") or "").strip()
    if not text:
        return None
    start = float(getattr(item, "start", 0.0) or 0.0)
    end = float(getattr(item, "end", start) or start)
    words = [
        word
        for word in (getattr(item, "words", None) or [])
        if getattr(word, "start", None) is not None and getattr(word, "end", None) is not None
    ]
    if words:
        start = float(words[0].start)
        end = float(words[-1].end)
    if end <= start:
        end = start + 0.1
    return Segment(start=start, end=end, text=text)


def _timed_words_with_fallback(items: list[object]) -> list[TimedWord]:
    words: list[TimedWord] = []
    for item in items:
        timed = timed_words_from_whisper_item(item)
        if timed:
            words.extend(timed)
            continue
        # Do not drop an entire spoken chunk merely because this one item has
        # no word timestamps while the rest of the recording does.
        fallback = _segment_from_whisper(item)
        if fallback is not None:
            words.append(TimedWord(fallback.start, fallback.end, fallback.text))
    return words


def _segments_from_whisper(item: object) -> list[Segment]:
    """Build subtitle cues from Whisper's word clock without guessing timings.

    Prefer the global packer when possible. A single Whisper chunk may still
    span several sentences separated by B-roll, so boundaries are chosen
    between words and retain those times.
    """
    words = timed_words_from_whisper_item(item)
    if words:
        return pack_cues_from_words(words)
    segment = _segment_from_whisper(item)
    return [segment] if segment is not None else []


def _coalesce_micro_segments(segments: list[Segment]) -> list[Segment]:
    """Remove word-timestamp fragments that make unusable subtitle cues.

    Whisper occasionally emits a short prefix (for example ``"va a"`` or
    ``"yo"``) immediately before the full cue. Keeping that prefix as its own
    subtitle gives the translator a one-word line and can also create a small
    overlap. Merge nearby micro-cues before translation; if the following cue
    already starts with the same words, keep the full cue and discard the
    duplicate prefix instead of repeating it.
    """
    if not segments:
        return []
    pending = [
        Segment(float(item.start), float(item.end), str(item.text or "").strip(), item.translated_text)
        for item in segments
        if str(item.text or "").strip() and float(item.end) > float(item.start)
    ]
    output: list[Segment] = []
    index = 0
    while index < len(pending):
        current = pending[index]
        if index + 1 < len(pending) and _is_micro_segment(current):
            following = pending[index + 1]
            if _segments_are_near(current, following):
                if _starts_with_same_words(following.text, current.text):
                    # Whisper repeated the prefix at the next boundary. Keep
                    # the complete following cue and extend it to the prefix's
                    # true start so its timing remains anchored to the audio.
                    pending[index + 1] = Segment(
                        min(current.start, following.start),
                        max(current.end, following.end),
                        following.text,
                        following.translated_text,
                    )
                    index += 1
                    continue
                # A short continuation such as "yo" belongs to the next
                # phrase. Join it before sending the text to the translator.
                pending[index + 1] = Segment(
                    min(current.start, following.start),
                    max(current.end, following.end),
                    _join_segment_text(current.text, following.text),
                    following.translated_text,
                )
                index += 1
                continue
        if output and _is_micro_segment(current) and _segments_are_near(output[-1], current):
            previous = output[-1]
            output[-1] = Segment(
                previous.start,
                max(previous.end, current.end),
                _join_segment_text(previous.text, current.text),
                previous.translated_text,
            )
        else:
            output.append(current)
        index += 1
    return output


def _is_micro_segment(segment: Segment) -> bool:
    tokens = re.findall(r"[\wÀ-ÿ]+|[\u3400-\u9fff]", segment.text or "", flags=re.UNICODE)
    return (segment.end - segment.start) <= 0.55 or len(tokens) <= 2


def _segments_are_near(left: Segment, right: Segment) -> bool:
    gap = right.start - left.end
    return gap <= 0.45 and gap >= -0.35


def _starts_with_same_words(text: str, prefix: str) -> bool:
    left = re.findall(r"[\wÀ-ÿ]+|[\u3400-\u9fff]", text.lower(), flags=re.UNICODE)
    right = re.findall(r"[\wÀ-ÿ]+|[\u3400-\u9fff]", prefix.lower(), flags=re.UNICODE)
    return bool(right) and len(left) >= len(right) and left[: len(right)] == right


def _join_segment_text(left: str, right: str) -> str:
    left, right = left.strip(), right.strip()
    if not left:
        return right
    if not right:
        return left
    if right[0] in ",.;:!?，。！？、；：)]}」』”’":
        return left + right
    return f"{left} {right}"


def _transcribe_attempts(device: str, compute_type: str) -> list[tuple[str, str, str]]:
    requested_device = (device or "cpu").strip().lower()
    requested_compute = (compute_type or "int8").strip().lower()
    if requested_compute == "default":
        requested_compute = "float16" if requested_device == "cuda" else "int8"

    if requested_device == "cuda":
        return [(requested_device, requested_compute, "Using CUDA for faster-whisper transcription.")]

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


def _nvidia_cuda_dll_directories() -> list[Path]:
    roots: list[Path] = []
    explicit_directories: list[Path] = []
    try:
        import nvidia
    except ImportError:
        pass
    else:
        roots.extend(Path(item) for item in getattr(nvidia, "__path__", ()))
        module_file = getattr(nvidia, "__file__", None)
        if module_file:
            roots.append(Path(module_file).parent)

    configured = os.environ.get("YBLOCALIZER_CUDA_DLL_DIR", "").strip()
    if configured:
        explicit_directories.append(Path(configured))

    # The onedir application intentionally does not bundle ~1.7 GB of optional
    # NVIDIA libraries. Reuse an existing Python CUDA runtime when present so a
    # developer or power user does not have to copy DLLs into the application.
    if os.name == "nt":
        try:
            import winreg

            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for access in (winreg.KEY_READ, winreg.KEY_READ | winreg.KEY_WOW64_32KEY, winreg.KEY_READ | winreg.KEY_WOW64_64KEY):
                    try:
                        with winreg.OpenKey(hive, r"Software\Python\PythonCore", 0, access) as python_core:
                            index = 0
                            while True:
                                try:
                                    version = winreg.EnumKey(python_core, index)
                                except OSError:
                                    break
                                index += 1
                                try:
                                    with winreg.OpenKey(python_core, version + r"\InstallPath", 0, access) as install_key:
                                        install_root = Path(str(winreg.QueryValue(install_key, None)))
                                except OSError:
                                    continue
                                roots.append(install_root / "Lib" / "site-packages" / "nvidia")
                    except OSError:
                        continue

        except ImportError:
            pass

        toolkit = Path(os.environ.get("ProgramFiles", "")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if toolkit.is_dir():
            roots.extend(path for path in toolkit.glob("v*/bin") if path.is_dir())

    candidates: list[Path] = list(explicit_directories)
    for root in roots:
        if root.name.lower() == "bin":
            candidates.append(root)
        else:
            candidates.extend([
                root / "cuda_runtime" / "bin",
                root / "cublas" / "bin",
                root / "cudnn" / "bin",
            ])
    return list(dict.fromkeys(path for path in candidates if path.is_dir()))


def _prepend_nvidia_cuda_dll_paths() -> None:
    candidates = _nvidia_cuda_dll_directories()
    existing = os.environ.get("PATH", "")
    parts = [str(path) for path in candidates]
    if not parts:
        return
    current_lower = {item.lower() for item in existing.split(os.pathsep) if item}
    missing = [item for item in parts if item.lower() not in current_lower]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + [existing])


def cuda_runtime_status() -> dict[str, object]:
    """Verify the CUDA libraries that faster-whisper actually loads.

    ``nvidia-smi`` only proves that a display driver sees the GPU.  It does not
    prove that cuBLAS/cuDNN are installed, which previously let preparation pass
    and then fail several minutes later in transcription.
    """
    if resolve_command("nvidia-smi") is None:
        return {
            "available": False,
            "missing": ["NVIDIA driver"],
            "message": "未检测到 NVIDIA 显卡驱动（nvidia-smi）。",
        }

    _prepend_nvidia_cuda_dll_paths()
    if os.name == "nt":
        required = (
            "cublas64_12.dll", "cublasLt64_12.dll",
            "cudnn64_9.dll", "cudnn_ops64_9.dll",
        )
        loader = ctypes.WinDLL
    else:
        required = ("libcublas.so.12", "libcudnn.so.9")
        loader = ctypes.CDLL

    search_dirs = _nvidia_cuda_dll_directories()
    missing: list[str] = []
    for name in required:
        resolved = shutil.which(name)
        if not resolved:
            resolved = next((str(path / name) for path in search_dirs if (path / name).is_file()), None)
        try:
            loader(resolved or name)
        except (OSError, TypeError):
            missing.append(name)

    if missing:
        return {
            "available": False,
            "missing": missing,
            "message": "显卡驱动可用，但缺少或无法加载 CUDA 转写运行库：" + "、".join(missing) + "。",
        }
    return {
        "available": True,
        "missing": [],
        "message": "NVIDIA 驱动、cuBLAS 和 cuDNN 均可加载。",
    }
