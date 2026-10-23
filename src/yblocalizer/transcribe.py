from __future__ import annotations

import os
import re
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Callable

from .cancellation import check_cancelled as legacy_check_cancelled
from .dependencies import (
    cuda_runtime_status,
    nvidia_cuda_dll_directories,
    prepend_nvidia_cuda_dll_paths,
    require_whisper_model,
    resolve_command,
)
from .models import Segment, save_segments
from .performance import cpu_thread_limit, normalize_resource_profile
from .subtitle import write_srt
from .subtitle_speech_pack import (
    TimedWord,
    normalize_word_stream,
    pack_cues_from_words,
    raw_words_path_for_segments,
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
    evidence_path: Path | None = None,
    vad_mode: str = "standard",
) -> list[Segment]:
    if os.environ.get("YBLOCALIZER_WHISPER_WORKER") != "1":
        return _transcribe_audio_isolated(
            audio_path, segments_json, srt_path, model_size, language, device,
            compute_type, initial_prompt, beam_size, log, cancel_check, resource_profile, evidence_path,
            vad_mode,
        )
    return _transcribe_audio_in_process(
        audio_path, segments_json, srt_path, model_size, language, device,
        compute_type, initial_prompt, beam_size, log, cancel_check, resource_profile, evidence_path,
        vad_mode,
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
    evidence_path: Path | None = None,
    vad_mode: str = "standard",
) -> list[Segment]:
    check = cancel_check or legacy_check_cancelled
    check()
    if device.strip().lower() in {"cuda", "auto"}:
        prepend_nvidia_cuda_dll_paths()
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
            vad_profile = _vad_profile(vad_mode)
            vad_parameters = {
                "min_silence_duration_ms": vad_profile["min_silence_duration_ms"],
                "speech_pad_ms": vad_profile["speech_pad_ms"],
            }
            if vad_profile["threshold"] is not None:
                vad_parameters["threshold"] = vad_profile["threshold"]
            transcribe_kwargs = {
                "language": language,
                "initial_prompt": _clean_initial_prompt(initial_prompt),
                "vad_filter": True,
                "beam_size": max(1, min(10, beam_size)),
                "condition_on_previous_text": False,
                "word_timestamps": True,
            }
            info = None
            try:
                whisper_segments, info = model.transcribe(
                    str(audio_path),
                    # The previous 120 ms pad clipped quiet leading words at
                    # speech onsets (verified at 07:17 in the current video).
                    vad_parameters=vad_parameters,
                    **transcribe_kwargs,
                )
            except TypeError:
                whisper_segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)
            # Flatten every Whisper chunk into one word clock, then cut by speech
            # pauses. Per-chunk packing leaves mid-phrase breaks at VAD boundaries
            # (e.g. "handed me" | "this box") that no later merge reliably fixes.
            whisper_items = []
            for item in whisper_segments:
                check()
                whisper_items.append(item)
            raw_words = _timed_words_with_fallback(whisper_items)
            raw_words_path = raw_words_path_for_segments(segments_json)
            save_words(raw_words_path, raw_words)
            normalization_stats: dict[str, int] = {}
            if raw_words:
                normalized_words = normalize_word_stream(raw_words, stats=normalization_stats)
                recovery_audit: dict[str, object] = {}
                detected_language = str(getattr(info, "language", "") or "").strip() or None
                all_words = _recover_vad_speech_gaps(
                    audio_path, model, normalized_words, transcribe_kwargs, check, logger,
                    normalization_stats=normalization_stats,
                    retry_language=language or detected_language,
                    recovery_audit=recovery_audit,
                    vad_profile=vad_profile,
                )
                save_words(words_path_for_segments(segments_json), all_words)
                segments = pack_cues_from_words(all_words)
                logger(f"按说话时间切分：{len(all_words)} 词 → {len(segments)} 条字幕。")
            else:
                normalized_words = []
                all_words = []
                recovery_audit = {}
                segments = []
                for item in whisper_items:
                    segments.extend(_segments_from_whisper(item))
                words_path_for_segments(segments_json).unlink(missing_ok=True)
            segments = _coalesce_micro_segments(segments)
            suspect_intervals: list[dict[str, object]] = [
                {"start": float(getattr(item, "start", 0.0)), "end": float(getattr(item, "end", 0.0)),
                 "avg_logprob": float(getattr(item, "avg_logprob", 0.0)),
                 "no_speech_prob": float(getattr(item, "no_speech_prob", 0.0))}
                for item in whisper_items
                if (float(getattr(item, "avg_logprob", 0.0)) < -1.0
                    or float(getattr(item, "no_speech_prob", 0.0)) > 0.6)
                and float(getattr(item, "end", 0.0)) > float(getattr(item, "start", 0.0))
            ]
            suspect_intervals.extend(list(recovery_audit.get("review_intervals", [])))
            audio_duration = _audio_duration_seconds(audio_path)
            audit = {
                "audio_duration_seconds": audio_duration,
                "raw": _word_clock_audit(raw_words, audio_duration=audio_duration),
                "normalized": _word_clock_audit(normalized_words, audio_duration=audio_duration),
                "recovered": _word_clock_audit(all_words, audio_duration=audio_duration),
                "packed": _word_clock_audit(segments, audio_duration=audio_duration),
            }
            _log_word_clock_audit(logger, audit)
            _write_asr_evidence(
                evidence_path, info, model_size=model_size, requested_language=language,
                suspect_intervals=suspect_intervals,
                transcription_audit=audit,
                normalization_stats=normalization_stats,
                recovery_audit=recovery_audit,
                initial_prompt=transcribe_kwargs.get("initial_prompt"),
                vad_mode=str(vad_profile["mode"]),
            )
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


def _recovery_windows(
    candidates: list[tuple[float, float]], *, max_core_seconds: float = 30.0,
) -> list[tuple[float, float, int]]:
    """Split long candidates and interleave them so one gap cannot consume all budget."""
    queues: list[tuple[int, list[tuple[float, float]]]] = []
    for candidate_index, (start, end) in enumerate(candidates):
        windows: list[tuple[float, float]] = []
        cursor = start
        while cursor < end:
            window_end = min(end, cursor + max_core_seconds)
            windows.append((cursor, window_end))
            cursor = window_end
        queues.append((candidate_index, windows))
    result: list[tuple[float, float, int]] = []
    while any(windows for _, windows in queues):
        for candidate_index, windows in queues:
            if windows:
                start, end = windows.pop(0)
                result.append((start, end, candidate_index))
    return result


def _recover_vad_speech_gaps(
    audio_path: Path,
    model: object,
    words: list[TimedWord],
    transcribe_kwargs: dict[str, object],
    check: Callable[[], None],
    logger: LogFn,
    normalization_stats: dict[str, int] | None = None,
    retry_language: str | None = None,
    recovery_audit: dict[str, object] | None = None,
    vad_profile: dict[str, object] | None = None,
) -> list[TimedWord]:
    """Recheck only sustained uncovered speech; never invent word timings."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    profile = vad_profile or _vad_profile("standard")
    recovery_threshold = float(profile["recovery_threshold"])
    check()
    audio = decode_audio(str(audio_path), sampling_rate=16000)
    speech = get_speech_timestamps(
        audio,
        VadOptions(
            threshold=recovery_threshold,
            min_silence_duration_ms=int(profile["recovery_silence_ms"]),
            speech_pad_ms=int(profile["recovery_pad_ms"]),
        ),
        sampling_rate=16000,
    )
    intervals = [(entry["start"] / 16000, entry["end"] / 16000) for entry in speech]
    candidates = _uncovered_speech_gaps(intervals, words)
    windows = _recovery_windows(candidates)
    audit = recovery_audit if recovery_audit is not None else {}
    audit.update({
        "vad_threshold": recovery_threshold,
        "candidate_count": len(candidates),
        "candidate_total_seconds": round(sum(end - start for start, end in candidates), 3),
        "window_count": len(windows),
        "window_total_seconds": round(sum(end - start for start, end, _ in windows), 3),
        "selected_windows": 0,
        "deferred_windows": 0,
        "actual_decode_seconds": 0.0,
        "deferred_decode_seconds": 0.0,
        "no_segments": 0,
        "rejected_logprob": 0,
        "rejected_no_speech": 0,
        "inserted_words": 0,
        "review_intervals": [],
        "review_intervals_total": 0,
        "review_intervals_truncated": False,
        "deferred_intervals": [],
        "deferred_intervals_total": 0,
        "deferred_intervals_truncated": False,
    })
    if not candidates:
        logger("人声漏检复核：未发现持续有人声却没有词级字幕的区间。")
        return words
    logger(
        f"人声漏检复核：发现 {len(candidates)} 个待复核区间，"
        f"拆分为 {len(windows)} 个窗口；使用复核阈值 {recovery_threshold}（不是本片实测最优值）。"
    )
    recovered: list[TimedWord] = []
    max_windows = 24
    max_decode_seconds = 120.0
    occupied_words = list(words)
    for core_start, core_end, candidate_index in windows:
        check()
        crop_start = max(0.0, core_start - 2.0)
        crop_end = min(len(audio) / 16000, core_end + 2.0)
        crop_duration = max(0.0, crop_end - crop_start)
        if audit["selected_windows"] >= max_windows or audit["actual_decode_seconds"] + crop_duration > max_decode_seconds:
            audit["deferred_windows"] += 1
            audit["deferred_intervals_total"] += 1
            audit["deferred_decode_seconds"] += crop_duration
            if len(audit["deferred_intervals"]) < 64:
                audit["deferred_intervals"].append({
                    "start": round(core_start, 3), "end": round(core_end, 3),
                    "candidate_index": candidate_index,
                })
            continue
        audit["selected_windows"] += 1
        audit["actual_decode_seconds"] += crop_duration
        crop = audio[int(crop_start * 16000):int(crop_end * 16000)]
        retry_kwargs = dict(transcribe_kwargs)
        retry_kwargs["language"] = retry_language
        retry_kwargs["vad_filter"] = False
        segments, _ = model.transcribe(crop, **retry_kwargs)
        segment_count = 0
        checked: list[TimedWord] = []
        accepted_batches: list[tuple[list[TimedWord], float, float]] = []
        for segment in segments:
            check()
            segment_count += 1
            avg_logprob = float(getattr(segment, "avg_logprob", -1.0))
            no_speech_prob = float(getattr(segment, "no_speech_prob", 1.0))
            if avg_logprob < -0.7:
                audit["rejected_logprob"] += 1
                continue
            if no_speech_prob > 0.3:
                audit["rejected_no_speech"] += 1
                continue
            shifted = [
                TimedWord(crop_start + word.start, crop_start + word.end, word.word)
                for word in _timed_words_with_fallback([segment])
            ]
            checked.extend(shifted)
            accepted_batches.append((shifted, avg_logprob, no_speech_prob))
        if segment_count == 0:
            audit["no_segments"] += 1
        inserted: list[TimedWord] = []
        for word in checked:
            midpoint = (word.start + word.end) / 2
            if not (
                core_start - 0.5 <= midpoint <= core_end + 0.5
                and word.start >= core_start - 0.5
                and word.end <= core_end + 0.5
            ):
                continue
            if any(word.start < old.end - 0.08 and word.end > old.start + 0.08 for old in occupied_words):
                continue
            recovered.append(word)
            inserted.append(word)
        if inserted:
            occupied_words.extend(inserted)
            audit["inserted_words"] += len(inserted)
            for batch, avg_logprob, no_speech_prob in accepted_batches:
                batch_inserted = [word for word in batch if word in inserted]
                if not batch_inserted:
                    continue
                audit["review_intervals_total"] += 1
                if len(audit["review_intervals"]) < 64:
                    audit["review_intervals"].append({
                        "start": round(min(word.start for word in batch_inserted), 3),
                        "end": round(max(word.end for word in batch_inserted), 3),
                        "avg_logprob": avg_logprob,
                        "no_speech_prob": no_speech_prob,
                        "recovery_review": True,
                        "reason": "recovery_inserted",
                    })
            logger(f"复核区间 {core_start:.1f}–{core_end:.1f}s：补入 {len(inserted)} 个候选词。")
    audit["deferred_intervals_truncated"] = audit["deferred_intervals_total"] > len(audit["deferred_intervals"])
    audit["review_intervals_truncated"] = audit["review_intervals_total"] > len(audit["review_intervals"])
    audit["actual_decode_seconds"] = round(audit["actual_decode_seconds"], 3)
    audit["deferred_decode_seconds"] = round(audit["deferred_decode_seconds"], 3)
    if audit["deferred_windows"]:
        logger(
            f"人声漏检复核：预算内检查 {audit['selected_windows']} 个窗口，"
            f"延后 {audit['deferred_windows']} 个窗口；原词流保留，延后区间待复核。"
        )
    if not recovered:
        logger("人声漏检复核：没有足够可靠的新词；原有转写保持不变。")
        return words
    logger(f"人声漏检复核：共补入 {len(recovered)} 个有真实时间戳的词。")
    return normalize_word_stream(words + recovered, stats=normalization_stats)


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
    evidence_path: Path | None = None,
    vad_mode: str = "standard",
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
                worker_result = _run_whisper_worker(
                    control_dir, audio_path, model_size, language, attempt_device,
                    attempt_compute, initial_prompt, beam_size, logger, check, resource_profile,
                    vad_mode,
                )
                items, temporary_json, temporary_srt = worker_result[:3]
                temporary_evidence = worker_result[3] if len(worker_result) > 3 else control_dir / "missing-evidence.json"
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
            temporary_raw_words = raw_words_path_for_segments(temporary_json)
            if temporary_raw_words.is_file():
                temporary_raw_words.replace(raw_words_path_for_segments(segments_json))
            if evidence_path and temporary_evidence.is_file():
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_evidence.replace(evidence_path)
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
    vad_mode: str = "standard",
) -> tuple[list[Segment], Path, Path, Path]:
    request_path = control_dir / f"request-{device}.json"
    result_path = control_dir / f"result-{device}.json"
    events_path = control_dir / f"events-{device}.jsonl"
    diagnostic_path = control_dir / f"diagnostic-{device}.log"
    temporary_json = control_dir / f"segments-{device}.json"
    temporary_srt = control_dir / f"subtitles-{device}.srt"
    temporary_evidence = control_dir / f"evidence-{device}.json"
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
        "vad_mode": vad_mode,
        "evidence_path": str(temporary_evidence),
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
    return load_segments(temporary_json), temporary_json, temporary_srt, temporary_evidence


def _write_asr_evidence(
    path: Path | None,
    info: object,
    *,
    model_size: str,
    requested_language: str | None,
    suspect_intervals: list[dict[str, object]] | None = None,
    transcription_audit: dict[str, object] | None = None,
    normalization_stats: dict[str, int] | None = None,
    recovery_audit: dict[str, object] | None = None,
    initial_prompt: object = None,
    vad_mode: str = "standard",
) -> None:
    if not path:
        return
    language = str(getattr(info, "language", "") or "").strip() or None
    probability_raw = getattr(info, "language_probability", None)
    try:
        probability = float(probability_raw) if probability_raw is not None else None
    except (TypeError, ValueError):
        probability = None
    if probability is not None and not math.isfinite(probability):
        probability = None
    evidence = {
        "version": 1,
        "model": model_size,
        "requested_language": requested_language or None,
        "detected_language": language,
        "language_probability": probability,
        "initial_prompt": initial_prompt or None,
        "vad_mode": vad_mode,
        "suspect_intervals": suspect_intervals or [],
        "transcription_audit": transcription_audit or {},
        "normalization_stats": normalization_stats or {},
        "recovery_audit": recovery_audit or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _audio_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (OSError, EOFError, wave.Error):
        return None


def _word_clock_audit(items: list[object], *, audio_duration: float | None = None) -> dict[str, object]:
    intervals = sorted(
        (float(getattr(item, "start", 0.0)), float(getattr(item, "end", 0.0)))
        for item in items
        if float(getattr(item, "end", 0.0)) > float(getattr(item, "start", 0.0))
    )
    coverage = 0.0
    gaps_ge_10_count = 0
    gaps_ge_10_seconds = 0.0
    gap_intervals: list[dict[str, float]] = []
    cursor: float | None = None
    if not intervals and audio_duration is not None and audio_duration >= 10.0:
        gaps_ge_10_count = 1
        gaps_ge_10_seconds = audio_duration
        gap_intervals.append({
            "start": 0.0,
            "end": round(audio_duration, 3),
            "duration": round(audio_duration, 3),
        })
    if intervals and audio_duration is not None and intervals[0][0] >= 10.0:
        gap = intervals[0][0]
        gaps_ge_10_count += 1
        gaps_ge_10_seconds += gap
        gap_intervals.append({"start": 0.0, "end": round(intervals[0][0], 3), "duration": round(gap, 3)})
    for start, end in intervals:
        if cursor is None:
            cursor = end
            continue
        if start > cursor:
            gap = start - cursor
            if gap >= 10.0:
                gaps_ge_10_count += 1
                gaps_ge_10_seconds += gap
                if len(gap_intervals) < 64:
                    gap_intervals.append({
                        "start": round(cursor, 3),
                        "end": round(start, 3),
                        "duration": round(gap, 3),
                    })
        if end > cursor:
            coverage += max(0.0, end - max(start, cursor)) if start < cursor else end - start
            cursor = end
    if cursor is not None and audio_duration is not None and audio_duration - cursor >= 10.0:
        gap = audio_duration - cursor
        gaps_ge_10_count += 1
        gaps_ge_10_seconds += gap
        if len(gap_intervals) < 64:
            gap_intervals.append({
                "start": round(cursor, 3),
                "end": round(audio_duration, 3),
                "duration": round(gap, 3),
            })
    if intervals and len(intervals) == 1:
        coverage = intervals[0][1] - intervals[0][0]
    elif intervals:
        # Recompute union length without depending on the gap loop above.
        coverage = 0.0
        union_start, union_end = intervals[0]
        for start, end in intervals[1:]:
            if start > union_end:
                coverage += union_end - union_start
                union_start, union_end = start, end
            else:
                union_end = max(union_end, end)
        coverage += union_end - union_start
    return {
        "count": len(items),
        "union_coverage_seconds": round(max(0.0, coverage), 3),
        "gaps_ge_10_seconds": {
            "count": gaps_ge_10_count,
            "total_seconds": round(gaps_ge_10_seconds, 3),
            "intervals": gap_intervals,
            "truncated": gaps_ge_10_count > len(gap_intervals),
        },
    }


def _log_word_clock_audit(logger: LogFn, audit: dict[str, object]) -> None:
    for stage in ("raw", "normalized", "recovered", "packed"):
        entry = audit[stage]
        gaps = entry["gaps_ge_10_seconds"]
        logger(
            f"ASR时间覆盖审计[{stage}]：{entry['count']}项，合并覆盖 "
            f"{entry['union_coverage_seconds']:.3f}s；≥10秒未覆盖/待复核区间 "
            f"{gaps['count']}段/{gaps['total_seconds']:.3f}s。"
        )
        if stage == "recovered" and gaps["intervals"]:
            points = ", ".join(
                f"{item['start']:.3f}–{item['end']:.3f}s"
                for item in gaps["intervals"][:8]
            )
            logger(f"ASR时间覆盖审计[recovered]：仍未覆盖/待复核起止点：{points}。")


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


def _vad_profile(mode: str | None) -> dict[str, object]:
    """Standard keeps the current VAD. Sensitive keeps more noisy outdoor speech."""
    if str(mode or "").strip().lower() == "sensitive":
        return {
            "mode": "sensitive",
            "threshold": 0.35,
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 500,
            "recovery_threshold": 0.30,
            "recovery_silence_ms": 400,
            "recovery_pad_ms": 500,
        }
    return {
        "mode": "standard",
        "threshold": None,
        "min_silence_duration_ms": 400,
        "speech_pad_ms": 400,
        "recovery_threshold": 0.35,
        "recovery_silence_ms": 400,
        "recovery_pad_ms": 400,
    }


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


_nvidia_cuda_dll_directories = nvidia_cuda_dll_directories
_prepend_nvidia_cuda_dll_paths = prepend_nvidia_cuda_dll_paths
