"""Persistent, non-destructive ASR gap-review checkpoints.

This module owns the durable queue.  Decoding is deliberately kept outside the
manifest code so a cancelled worker can be restarted without rebuilding the
primary transcription or changing committed subtitles.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import wave
from typing import Any, Callable

from .performance import ReviewResourceBusy, cpu_thread_limit, normalize_resource_profile, review_cuda_memory
from .util import read_json


SESSION_VERSION = 1
MAX_WINDOWS_PER_ROUND = 24
MAX_DECODE_SECONDS = 120.0
MIN_GAP_SECONDS = 1.25
MAX_GAP_SECONDS = 30.0


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:10]}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_signature(path: Path | None) -> str | None:
    if not path:
        return None
    try:
        stat = path.resolve().stat()
    except OSError:
        return None
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def _session_fingerprint(audio_path: Path, baseline_path: Path | None, model: str,
                         language: str | None, beam_size: int, channel: str, timeline_shift_seconds: float = 0.0,
                         source_video_path: Path | None = None) -> str:
    payload = {
        "audio": _file_fingerprint(audio_path),
        "baseline": _file_fingerprint(baseline_path) if baseline_path and baseline_path.is_file() else None,
        "model": model, "language": language or None, "beam_size": int(beam_size), "channel": channel,
        "timeline_shift_seconds": round(float(timeline_shift_seconds or 0.0), 3),
        "source_video": _path_signature(source_video_path),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None


def resolve_review_language(requested: Any, evidence_path: Path | None = None, *, allow_detected: bool = True) -> str | None:
    value = str(requested or "").strip()
    if value and value.lower() != "auto":
        return value
    if not allow_detected or not evidence_path:
        return None
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(evidence, dict):
        return None
    detected = str(evidence.get("detected_language") or "").strip()
    probability = _safe_float(evidence.get("language_probability"))
    if detected and probability is not None and probability >= 0.80:
        return detected
    return None


def _words_from_json(path: Path) -> list[dict[str, Any]]:
    raw = read_json(path, [])
    if not isinstance(raw, list):
        return []
    words: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start, end = _safe_float(item.get("start")), _safe_float(item.get("end"))
        if start is None or end is None or end <= start:
            continue
        words.append({"start": start, "end": end, "text": str(item.get("text", item.get("word", "")) or "")})
    return sorted(words, key=lambda item: (item["start"], item["end"]))


def _gaps(words: list[dict[str, Any]], duration: float | None) -> list[dict[str, Any]]:
    """Build a deterministic word-clock queue, independent of VAD."""
    if not words:
        return ([{"id": "gap-000001", "start": 0.0, "end": round(duration, 3), "status": "pending"}]
                if duration and duration >= MIN_GAP_SECONDS else [])
    result: list[dict[str, Any]] = []
    cursor = 0.0
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if start - cursor >= MIN_GAP_SECONDS:
            result.append({"id": f"gap-{len(result) + 1:06d}", "start": round(cursor, 3), "end": round(start, 3), "status": "pending"})
        cursor = max(cursor, end)
    if duration and duration - cursor >= MIN_GAP_SECONDS:
        result.append({"id": f"gap-{len(result) + 1:06d}", "start": round(cursor, 3), "end": round(duration, 3), "status": "pending"})
    expanded: list[dict[str, Any]] = []
    for gap in result:
        start, end = gap["start"], gap["end"]
        while end - start > MAX_GAP_SECONDS:
            expanded.append({"id": f"gap-{len(expanded) + 1:06d}", "parent_id": gap["id"], "start": start, "end": round(start + MAX_GAP_SECONDS, 3), "status": "pending"})
            start += MAX_GAP_SECONDS
        expanded.append({"id": f"gap-{len(expanded) + 1:06d}", "parent_id": gap["id"], "start": round(start, 3), "end": end, "status": "pending"})
    return expanded


@dataclass(slots=True)
class ReviewSession:
    version: int
    job_id: str
    status: str
    audio_path: str
    baseline_words_path: str | None
    baseline_kind: str
    audio_fingerprint: str
    model: str
    language: str | None
    beam_size: int
    channel: str
    subtitle_revision: int
    subtitle_fingerprint: str
    candidates: list[dict[str, Any]]
    windows: list[dict[str, Any]]
    recovered_words: list[dict[str, Any]]
    review_intervals: list[dict[str, Any]]
    cursor: int = 0
    actual_decode_seconds: float = 0.0
    error: str | None = None
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def session_path(work_dir: Path) -> Path:
    return work_dir / "asr-review" / "session.json"


def subtitle_commit_path(work_dir: Path, revision: int, commit_id: str | None = None) -> Path:
    suffix = f".{commit_id}" if commit_id else ""
    return work_dir / f"subtitle-edit.r{int(revision)}{suffix}.commit.json"


def read_session(work_dir: Path) -> dict[str, Any] | None:
    value = read_json(session_path(work_dir), None)
    return value if isinstance(value, dict) else None


def prepare_session(*, work_dir: Path, job_id: str, audio_path: Path, baseline_words_path: Path | None,
                    baseline_kind: str, model: str, language: str | None, beam_size: int,
                    channel: str, subtitle_revision: int, subtitle_fingerprint: str,
                    duration: float | None = None, timeline_shift_seconds: float = 0.0,
                    source_video_path: Path | None = None) -> dict[str, Any]:
    existing = read_session(work_dir)
    fingerprint = _session_fingerprint(audio_path, baseline_words_path, model, language, beam_size, channel, timeline_shift_seconds, source_video_path)
    if existing and existing.get("session_fingerprint") == fingerprint and existing.get("job_id") == job_id:
        if existing.get("status") == "running":
            existing["status"] = "pending"
        # Subtitle edits rebase the CAS metadata only; completed/pending review
        # windows remain stable and are never silently recomputed.
        existing["subtitle_revision"] = int(subtitle_revision)
        existing["subtitle_fingerprint"] = subtitle_fingerprint
        existing["channel"] = channel
        existing["timeline_shift_seconds"] = float(timeline_shift_seconds or 0.0)
        _atomic_json(session_path(work_dir), existing)
        return existing
    if existing:
        # Keep an immutable recovery point when the audio/model/channel changes.
        # Uncommitted candidates from the old session must remain inspectable.
        old_fp = str(existing.get("session_fingerprint") or "legacy")[:12]
        backup = session_path(work_dir).with_name(f"session-{old_fp}.json")
        if not backup.exists():
            shutil.copyfile(session_path(work_dir), backup)
    words = _words_from_json(baseline_words_path) if baseline_words_path and baseline_words_path.is_file() else []
    if duration is None:
        try:
            with wave.open(str(audio_path), "rb") as stream:
                duration = stream.getnframes() / max(1, stream.getframerate())
        except (OSError, EOFError, wave.Error):
            duration = None
    session = ReviewSession(
        version=SESSION_VERSION, job_id=job_id, status="pending", audio_path=str(audio_path),
        baseline_words_path=str(baseline_words_path) if baseline_words_path else None,
        baseline_kind=baseline_kind, audio_fingerprint=_file_fingerprint(audio_path), model=model,
        language=language, beam_size=int(beam_size), channel=channel,
        subtitle_revision=int(subtitle_revision), subtitle_fingerprint=subtitle_fingerprint,
        candidates=_gaps(words, duration), windows=[], recovered_words=[], review_intervals=[],
        updated_at=time.time(),
    )
    payload = session.to_dict()
    payload["session_fingerprint"] = fingerprint
    payload["round_id"] = None
    payload["owner_token"] = None
    payload["result_candidates"] = []
    payload["diagnostics"] = []
    payload["audio_duration"] = duration
    payload["timeline_shift_seconds"] = float(timeline_shift_seconds or 0.0)
    payload["source_video_signature"] = _path_signature(source_video_path)
    _atomic_json(session_path(work_dir), payload)
    return payload


def update_session(work_dir: Path, **changes: Any) -> dict[str, Any]:
    session = read_session(work_dir)
    if session is None:
        raise FileNotFoundError("ASR review session does not exist")
    session.update(changes)
    session["updated_at"] = time.time()
    _atomic_json(session_path(work_dir), session)
    return session


def reset_interrupted_round(work_dir: Path, *, status: str = "paused", error: str | None = None) -> dict[str, Any] | None:
    """Fence a stale/terminated owner and return all unfinished windows to pending."""
    session = read_session(work_dir)
    if not session:
        return None
    candidates = session.get("candidates") if isinstance(session.get("candidates"), list) else []
    for item in candidates:
        if item.get("status") == "running":
            item["status"] = "pending"
    session.update({"status": status, "candidates": candidates, "round_id": None, "owner_token": None, "selected_window_ids": [], "error": error})
    _atomic_json(session_path(work_dir), session)
    return session


def _crop_bounds(start: float, end: float, duration: float, shift: float) -> tuple[float, float]:
    """Use the same audio clock and padding for budgeting and reading."""
    return (
        min(duration, max(0.0, start + shift - 2.0)),
        min(duration, max(0.0, end + shift + 2.0)),
    )


def _review_audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getcomptype()) != (1, 16000, 2, "NONE") or audio.getnframes() <= 0:
            raise RuntimeError("复核音频必须是有效的 16 kHz 单声道 PCM WAV，请重新准备区间。")
        return audio.getnframes() / 16000.0


def _read_review_crop(path: Path, start: float, end: float) -> tuple[Any, float]:
    """Seek directly to one bounded PCM range; never decode the whole file."""
    import numpy as np

    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getcomptype()) != (1, 16000, 2, "NONE"):
            raise RuntimeError("复核音频格式已变化，请重新准备区间。")
        first = min(audio.getnframes(), max(0, int(start * 16000)))
        last = min(audio.getnframes(), max(first, int(end * 16000)))
        if last <= first:
            raise RuntimeError("复核区间不在当前音频范围内，请重新准备区间。")
        audio.setpos(first)
        pcm = audio.readframes(last - first)
        if len(pcm) != (last - first) * 2:
            raise RuntimeError("复核音频不完整，请重新准备区间；已有字幕和复核进度已保留。")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    samples *= np.float32(1.0 / 32768.0)
    return samples, first / 16000.0


def next_round(work_dir: Path) -> list[dict[str, Any]]:
    session = read_session(work_dir) or {}
    if session.get("status") in {"running", "cancelled", "completed"}:
        return []
    candidates = [item for item in session.get("candidates", []) if item.get("status") == "pending"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        groups.setdefault(str(item.get("parent_id") or item["id"]), []).append(item)
    candidates = [group[index] for index in range(max((len(group) for group in groups.values()), default=0)) for group in groups.values() if index < len(group)]
    selected: list[dict[str, Any]] = []
    budget = 0.0
    shift = _safe_float(session.get("timeline_shift_seconds")) or 0.0
    for item in candidates:
        audio_duration = _safe_float(session.get("audio_duration"))
        crop_start, crop_end = _crop_bounds(float(item["start"]), float(item["end"]), audio_duration if audio_duration is not None else float(item["end"]) + abs(shift) + 2.0, shift)
        crop_duration = max(0.0, crop_end - crop_start)
        if len(selected) >= MAX_WINDOWS_PER_ROUND:
            break
        if budget + crop_duration > MAX_DECODE_SECONDS:
            continue
        selected.append(item)
        budget += crop_duration
    return selected


def start_round(work_dir: Path) -> dict[str, Any]:
    session = read_session(work_dir)
    if not session:
        raise RuntimeError("ASR review session does not exist")
    selected = next_round(work_dir)
    if not selected:
        return update_session(work_dir, status="completed")
    round_id, owner = uuid.uuid4().hex, uuid.uuid4().hex
    selected_ids = {item["id"] for item in selected}
    candidates = list(session.get("candidates") or [])
    for item in candidates:
        if item.get("id") in selected_ids:
            item["status"] = "running"
            item["round_id"] = round_id
    return update_session(
        work_dir, status="running", round_id=round_id, owner_token=owner,
        selected_window_ids=list(selected_ids), windows=selected, candidates=candidates, error=None, error_code=None,
    )


def _pack_candidate_words(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        if current and float(word["start"]) - float(current[-1]["end"]) > 1.25:
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def decode_round(*, work_dir: Path, model_size: str, device: str, compute_type: str,
                 user_data_root: Path, cancel_check: Callable[[], None],
                 round_id: str, owner_token: str,
                 resource_profile: str = "balanced",
                 log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Decode one persisted round with one model load.

    This function only writes the review manifest and recovered-word sidecar;
    committed source/translation files are deliberately untouched.
    """
    session = read_session(work_dir)
    if not session:
        raise RuntimeError("ASR review session does not exist")
    if session.get("status") != "running" or session.get("round_id") != round_id or session.get("owner_token") != owner_token:
        raise RuntimeError("ASR review round is stale or fenced")
    selected_ids = set(session.get("selected_window_ids") or [])
    selected = [item for item in session.get("windows", []) if item.get("id") in selected_ids]
    if not selected:
        return update_session(work_dir, status="completed")
    try:
        from faster_whisper import WhisperModel
        from .dependencies import prepend_nvidia_cuda_dll_paths, require_whisper_model
        cancel_check()
        session_model = str(session.get("model") or model_size)
        if model_size and str(model_size) != session_model:
            raise RuntimeError("ASR review model does not match the prepared session")
        profile = normalize_resource_profile(resource_profile)
        thread_limit = cpu_thread_limit(profile)
        audio_path = Path(session["audio_path"])
        audio_duration = _review_audio_duration(audio_path)
        if log:
            log(f"语音复核资源模式：{profile}；CPU 线程上限 {thread_limit}；逐区间读取音频。")
        session["resource_info"] = review_cuda_memory(device, session_model, compute_type)
        if device.strip().lower() == "cuda":
            prepend_nvidia_cuda_dll_paths()
        model = WhisperModel(str(require_whisper_model(session_model, user_data_root)), device=device, compute_type=compute_type, cpu_threads=thread_limit, num_workers=1)
        recovered = list(session.get("recovered_words") or [])
        occupied = recovered + (_words_from_json(Path(session["baseline_words_path"])) if session.get("baseline_words_path") else [])
        review = list(session.get("review_intervals") or [])
        result_candidates = list(session.get("result_candidates") or [])
        diagnostics = list(session.get("diagnostics") or [])
        by_id = {item.get("id"): item for item in session.get("candidates", [])}
        for item in selected:
            cancel_check()
            start, end = float(item["start"]), float(item["end"])
            shift = _safe_float(session.get("timeline_shift_seconds")) or 0.0
            crop_start, crop_end = _crop_bounds(start, end, audio_duration, shift)
            crop, crop_start = _read_review_crop(audio_path, crop_start, crop_end)
            segments, _info = model.transcribe(
                crop, language=session.get("language") or None,
                beam_size=int(session.get("beam_size") or 5), vad_filter=False,
                word_timestamps=True, condition_on_previous_text=False,
            )
            accepted: list[dict[str, Any]] = []
            for segment in segments:
                cancel_check()
                avg = float(getattr(segment, "avg_logprob", -99.0))
                no_speech = float(getattr(segment, "no_speech_prob", 1.0))
                if avg >= -0.7 and no_speech <= 0.3:
                    tier, adoptable = "strong", True
                elif not (no_speech > 0.6 and avg <= -1.0):
                    tier, adoptable = "weak", False
                else:
                    diagnostics.append({"window_id": item["id"], "text": str(getattr(segment, "text", "") or ""), "start": round(crop_start + float(segment.start) - shift, 3), "end": round(crop_start + float(segment.end) - shift, 3), "avg_logprob": avg, "no_speech_prob": no_speech, "reason": "diagnostic-only"})
                    continue
                for word in getattr(segment, "words", None) or []:
                    ws, we = crop_start + float(word.start) - shift, crop_start + float(word.end) - shift
                    overlaps = any(ws < float(old.get("end", 0)) and we > float(old.get("start", 0)) for old in occupied)
                    if ws >= start - 0.5 and we <= end + 0.5 and we > ws and not overlaps and str(word.word or "").strip():
                        accepted.append({"start": round(ws, 3), "end": round(we, 3), "text": str(word.word).strip(), "review": True, "source": "deep-review", "candidate_id": item["id"], "avg_logprob": avg, "no_speech_prob": no_speech, "tier": tier, "weak": tier == "weak", "adoptable": adoptable})
                        occupied.append({"start": ws, "end": we})
            recovered.extend(accepted)
            if accepted:
                review.append({"start": start, "end": end, "reason": "deep_review", "needs_review": True, "candidate_id": item["id"]})
                for ordinal, group in enumerate(_pack_candidate_words(accepted), 1):
                    result_candidates.append({"id": f"{item['id']}-{ordinal:03d}", "window_id": item["id"], "start": group[0]["start"], "end": group[-1]["end"], "words": group, "tier": "weak" if any(word.get("tier") == "weak" for word in group) else "strong", "adoptable": all(word.get("adoptable") is True for word in group), "confirmed_weak": False, "needs_review": True})
            by_id[item["id"]]["status"] = "done"
            by_id[item["id"]]["accepted_words"] = len(accepted)
            session.update({"recovered_words": recovered, "review_intervals": review, "result_candidates": result_candidates, "diagnostics": diagnostics[-512:], "candidates": list(by_id.values()), "actual_decode_seconds": float(session.get("actual_decode_seconds") or 0) + crop_end - crop_start})
            session["status"] = "running"
            current = read_session(work_dir) or {}
            if current.get("status") != "running" or current.get("round_id") != round_id or current.get("owner_token") != owner_token:
                raise RuntimeError("ASR review round was cancelled or fenced before checkpoint")
            _atomic_json(session_path(work_dir), session)
            if log:
                log(f"语音复核完成 {item['id']}：采用 {len(accepted)} 个词，保留人工复核标记。")
        pending = any(item.get("status") == "pending" for item in session["candidates"])
        session["status"] = "paused" if pending else "completed"
        session["cursor"] = int(session.get("cursor") or 0) + len(selected)
        session["round_id"] = None
        session["owner_token"] = None
        session["selected_window_ids"] = []
        return _atomic_update_payload(work_dir, session)
    except Exception as exc:
        from .transcribe import WhisperCudaOutOfMemory, _is_cuda_oom

        if _is_cuda_oom(exc) or (str(device).strip().lower() == "cuda" and "out of memory" in str(exc).lower()):
            exc = WhisperCudaOutOfMemory("语音复核运行时显存不足，已暂停并保留完成进度。释放显存后重新开始复核。")
        resource_busy = isinstance(exc, (ReviewResourceBusy, WhisperCudaOutOfMemory))
        current = read_session(work_dir) or {}
        if current.get("round_id") == round_id and current.get("owner_token") == owner_token:
            selected_ids = set(current.get("selected_window_ids") or [])
            for candidate in current.get("candidates", []):
                if candidate.get("id") in selected_ids and candidate.get("status") == "running":
                    candidate["status"] = "pending"
            current.update({"status": "paused" if resource_busy or type(exc).__name__ in {"CancellationRequested", "CancelledError"} else "failed", "error": "已取消；未完成区间保留待续跑。" if type(exc).__name__ in {"CancellationRequested", "CancelledError"} else str(exc), "error_code": getattr(exc, "code", None), "resource_info": getattr(exc, "resources", session.get("resource_info")), "round_id": None, "owner_token": None, "selected_window_ids": []})
            _atomic_json(session_path(work_dir), current)
            if type(exc).__name__ in {"CancellationRequested", "CancelledError"}:
                return current
        raise exc


def _atomic_update_payload(work_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload["updated_at"] = time.time()
    _atomic_json(session_path(work_dir), payload)
    return payload
