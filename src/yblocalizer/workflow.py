"""Checkpointed business stages shared by the desktop workbench and CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import uuid
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from . import dependencies
from .asr_prompt import build_initial_prompt
from .glossary import load_glossary
from .asr_review import prepare_session, resolve_review_language, session_path
from .download import download_with_ytdlp, import_local_video
from .media import extract_audio
from .models import Segment, VideoJob, load_segments, save_segments
from .ocr_subtitle import extract_ocr_subtitles
from .performance import translation_worker_limit
from .publish_bili import assist_publish
from .publish_text import ensure_source_link
from .render import burn_subtitles, subtitle_line_capacity
from .runtime import CancellationRequested, PipelineContext, PipelineStage
from .subtitle import cue_source_line_counts, format_srt_timestamp, read_srt_segments, translated_layout_issues, write_srt
from .subtitle_continuity import coalesce_incomplete_source_cues, segments_content_equal
from .subtitle_merge import merge_audio_ocr_segments
from .subtitle_resegment import resegment_source_segments
from .subtitle_speech_pack import (
    group_speech_translation_units, load_words, long_gap_fragment_warnings, pack_cues_from_words_detailed,
    split_corrected_punctuation_cues, words_path_for_segments,
)
from .subtitle_sentence_plan import (
    subtitle_density_overflow_count,
    subtitle_high_reading_speed_count,
)
from .transcribe import transcribe_audio
from .translate import (
    compact_overflowing_subtitles,
    correct_source_segments,
    generate_publish_metadata,
    instruction_like_content_warning,
    save_publish_metadata,
    translate_segments_file,
)
from .util import ensure_rights_confirmed


WORKFLOW_VERSION = 2
# Stages that must all complete for a finished job; dub and publish are optional.
CORE_STAGES = ("acquire", "extract", "translate", "render")
WORKFLOW_STAGES = (*CORE_STAGES, "dub", "publish")
STAGE_LABELS = {
    "acquire": "获取素材", "extract": "字幕提取", "translate": "翻译",
    "render": "渲染", "dub": "配音", "publish": "发布辅助",
}
STAGE_PROGRESS = {"acquire": 8, "extract": 28, "translate": 62, "render": 84, "dub": 90, "publish": 96}
CONFIG_INVALIDATION: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"source", "source_kind", "output_dir", "cookies_from_browser", "cookies_file", "youtube_po_token_mode", "youtube_proxy", "download_quality", "max_seconds", "require_reuse_allowed", "prefer_platform_subtitles"}), "acquire"),
    (frozenset({"subtitle_source", "ocr_fallback_to_audio", "whisper_model_size", "source_language", "beam_size", "device", "compute_type", "ocr_interval", "ocr_crop_ratio", "ocr_min_chars", "ocr_language", "audio_channel", "asr_hint", "asr_vad_mode"}), "extract"),
    (frozenset({"translator", "target_lang", "translate_model", "batch_size", "smart_translation", "smart_subtitle_layout"}), "translate"),
    (frozenset({"font_name", "font_size", "subtitle_display_mode", "subtitle_color", "subtitle_outline_color", "subtitle_outline", "subtitle_shadow", "subtitle_margin_ratio", "subtitle_preset", "render_crf", "render_encoder"}), "render"),
    (frozenset({"dub_enabled", "dub_mix"}), "dub"),
    (frozenset({"title", "description", "tags", "publish_to_bilibili", "include_source_link", "bilibili_browser", "close_after_fill"}), "publish"),
)


@dataclass(slots=True)
class WorkflowArtifacts:
    original_video: str | None = None
    raw_video: str | None = None
    audio: str | None = None
    asr_evidence: str | None = None
    asr_review: str | None = None
    subtitle_commit: str | None = None
    asr_timeline_shift_seconds: float = 0.0
    platform_subtitles: str | None = None
    source_segments: str | None = None
    source_srt: str | None = None
    translated_segments: str | None = None
    translated_srt: str | None = None
    publish_metadata: str | None = None
    llm_usage_path: str | None = None
    llm_usage_index_path: str | None = None
    llm_usage_total: dict[str, Any] | None = None
    rendered_video: str | None = None
    dub_video: str | None = None
    dub_report: str | None = None
    title: str | None = None
    description: str = ""
    source: str = ""
    source_kind: str = ""
    license: str | None = None
    view_count: int | None = None
    webpage_url: str | None = None
    thumbnail_url: str | None = None
    thumbnail_path: str | None = None
    used_ocr_subtitles: bool = False
    subtitle_extraction_mode: str = ""
    ocr_status: str = "not-requested"
    ocr_message: str | None = None
    content_warnings: list[str] = field(default_factory=list)
    revision: int = 0
    last_edit: str | None = None
    edit_state: str = "clean"

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "WorkflowArtifacts":
        source = value if isinstance(value, dict) else {}
        return cls(**{key: source.get(key) for key in cls.__dataclass_fields__ if key in source})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def path(self, name: str) -> Path | None:
        value = getattr(self, name, None)
        return Path(value) if value else None

    def public_summary(self) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for name in ("original_video", "raw_video", "audio", "asr_evidence", "asr_review", "subtitle_commit", "platform_subtitles", "source_segments", "source_srt", "translated_segments", "translated_srt", "publish_metadata", "rendered_video", "dub_video", "dub_report", "thumbnail_path"):
            path = self.path(name)
            output[name] = {
                "available": bool(path and path.is_file()),
                "name": path.name if path else None,
                "bytes": path.stat().st_size if path and path.is_file() else 0,
            }
        return output


def stage_state() -> dict[str, Any]:
    return {
        "status": "pending", "progress": 0, "error": None,
        "error_code": None, "recovery": None,
        "started_at": None, "finished_at": None, "config_fingerprint": None,
    }


def new_stage_states(include_publish: bool = False) -> dict[str, dict[str, Any]]:
    del include_publish
    return {name: stage_state() for name in WORKFLOW_STAGES}


def invalidation_stage(changed_fields: set[str]) -> str | None:
    candidates = [stage for fields, stage in CONFIG_INVALIDATION if fields & changed_fields]
    return min(candidates, key=WORKFLOW_STAGES.index) if candidates else None


def invalidate_downstream(states: dict[str, dict[str, Any]], from_stage: str) -> list[str]:
    changed: list[str] = []
    start = WORKFLOW_STAGES.index(from_stage)
    for stage in WORKFLOW_STAGES[start:]:
        row = states.setdefault(stage, stage_state())
        if row.get("status") not in {"pending", "stale"}:
            row.update({
                "status": "stale", "progress": 0, "error": None,
                "error_code": None, "recovery": None,
                "started_at": None, "finished_at": None,
            })
            changed.append(stage)
    return changed


def atomic_write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _emit(ctx: PipelineContext, stage: PipelineStage, percent: int, message: str, log: Callable[[str], None]) -> None:
    log(message)
    ctx.emit(stage, percent, message)



def _discard_audio_artifact(artifacts: WorkflowArtifacts, work_dir: Path) -> None:
    """Clear only the new snapshot's audio pointer.

    Audio is a versioned checkpoint just like video/subtitles.  Removing the
    previous WAV during a successful acquire made an otherwise safe retry
    destructive and could invalidate a restored job that still referenced it.
    The next extract stage always regenerates ``audio.wav`` as needed.
    """
    artifacts.audio = None


def _versioned_target(work_dir: Path, stem: str, suffix: str) -> Path:
    """Choose a new artifact path without overwriting an active checkpoint."""
    first = work_dir / f"{stem}{suffix}"
    if not first.exists():
        return first
    index = 2
    while True:
        candidate = work_dir / f"{stem}.v{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _attempt_dir(work_dir: Path, stage: str) -> Path:
    """Create an isolated immutable output directory for one stage attempt."""
    path = work_dir / f".attempt-{stage}-{uuid.uuid4().hex[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _seed_attempt_checkpoint(work_dir: Path, name: str, stage: str = "translate") -> Path:
    """Copy the newest prior checkpoint; its consumer revalidates its fingerprint."""
    target = work_dir / name
    candidates = sorted(
        (path / name for path in work_dir.parent.glob(f".attempt-{stage}-*")),
        key=lambda path: path.stat().st_mtime_ns if path.is_file() else 0,
        reverse=True,
    )
    for source in candidates:
        if source.is_file():
            try:
                shutil.copyfile(source, target)
                break
            except OSError:
                continue
    return target


def _commit_acquired_file(source: Path, work_dir: Path, stem: str) -> Path:
    target = _versioned_target(work_dir, stem, source.suffix.lower() or ".mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return target


def _youtube_video_id(source: str) -> str | None:
    parsed = urlparse(str(source).strip())
    host = (parsed.hostname or "").lower()
    segments = [segment for segment in parsed.path.split("/") if segment]
    if host == "youtu.be" or host.endswith(".youtu.be"):
        return segments[0] if segments else None
    if (
        host == "youtube.com"
        or host.endswith(".youtube.com")
        or host == "youtube-nocookie.com"
        or host.endswith(".youtube-nocookie.com")
    ):
        value = (parse_qs(parsed.query).get("v") or [None])[0]
        if value:
            return value
        if len(segments) >= 2 and segments[0] in {"embed", "live", "shorts", "v"}:
            return segments[1]
    return None


def _acquire_source_identity(source: str) -> str:
    video_id = _youtube_video_id(source)
    if video_id:
        return f"youtube:{video_id}"
    return str(source).strip()


def _acquire_cache_contract(options: Any) -> dict[str, Any]:
    proposal = getattr(options, "quality_proposal", None) or {}
    return {
        "schema_version": 2,
        "source_identity_sha256": hashlib.sha256(
            _acquire_source_identity(str(options.source)).encode("utf-8")
        ).hexdigest(),
        "youtube_video_id": _youtube_video_id(str(options.source)),
        "quality": str(options.download_quality),
        "video_format_id": str(proposal.get("video_format_id") or ""),
        "audio_format_id": str(proposal.get("audio_format_id") or ""),
        "width": proposal.get("width"),
        "height": proposal.get("height"),
        "max_seconds": options.max_seconds,
        "output_template_version": 1,
    }


def _canonical_acquire_cache_dir(work_dir: Path, options: Any) -> Path:
    contract = _acquire_cache_contract(options)
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return work_dir / f".acquire-cache-{fingerprint}"


def _cache_child_is_safe(candidate: Path, work_dir: Path) -> bool:
    try:
        return (
            candidate.name.startswith(".acquire-cache-")
            and candidate.is_dir()
            and not candidate.is_symlink()
            and candidate.resolve().parent == work_dir.resolve()
        )
    except OSError:
        return False


def _read_acquire_cache_manifest(candidate: Path) -> dict[str, Any] | None:
    path = candidate / ".acquire-cache.json"
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _legacy_acquire_cache_matches(candidate: Path, options: Any) -> bool:
    """Adopt only a legacy cache whose source, formats, and partial names agree."""
    proposal = getattr(options, "quality_proposal", None) or {}
    video_format_id = str(proposal.get("video_format_id") or "")
    audio_format_id = str(proposal.get("audio_format_id") or "")
    try:
        width, height = int(proposal.get("width") or 0), int(proposal.get("height") or 0)
        if (
            not video_format_id
            or width <= 0
            or height <= 0
            or options.max_seconds not in (None, 0)
        ):
            return False
        info_files = list(candidate.glob("*.info.json"))
        partials = list(candidate.glob("*.part"))
        if len(info_files) != 1 or not partials:
            return False
        info_path = info_files[0]
        if info_path.is_symlink() or not info_path.is_file() or info_path.stat().st_size > 8 * 1024 * 1024:
            return False
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            return False

        expected_source = _acquire_source_identity(str(options.source))
        saved_source = str(info.get("webpage_url") or info.get("original_url") or "")
        if not saved_source or _acquire_source_identity(saved_source) != expected_source:
            return False
        video_id = str(info.get("id") or "")
        expected_video_id = _youtube_video_id(str(options.source))
        if expected_video_id and video_id != expected_video_id:
            return False
        if (
            str(info.get("format_id") or "").split("+") != [video_format_id, audio_format_id]
            and set(str(info.get("format_id") or "").split("+"))
            != {value for value in (video_format_id, audio_format_id) if value}
        ):
            return False
        if int(info.get("width") or 0) != width or int(info.get("height") or 0) != height:
            return False
        if str(info.get("vcodec") or "none").lower() == "none":
            return False
        if audio_format_id and str(info.get("acodec") or "none").lower() == "none":
            return False

        video_prefix = f"{video_id}.f{video_format_id}."
        audio_prefix = f"{video_id}.f{audio_format_id}." if audio_format_id else None
        matching_partials = 0
        for partial in partials:
            if partial.is_symlink() or not partial.is_file() or partial.stat().st_size <= 0:
                return False
            if partial.name.startswith(video_prefix) or (audio_prefix and partial.name.startswith(audio_prefix)):
                matching_partials += 1
            else:
                return False
        return matching_partials > 0
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _acquire_cache_dir(work_dir: Path, options: Any) -> Path:
    """Reuse a cache only when its media-byte contract matches this request."""
    canonical = _canonical_acquire_cache_dir(work_dir, options)
    matches: list[Path] = []
    if canonical.exists() or canonical.is_symlink():
        if not _cache_child_is_safe(canonical, work_dir):
            raise RuntimeError("素材缓存路径不安全，未触碰现有未完成文件。")
        manifest = _read_acquire_cache_manifest(canonical)
        if manifest is not None:
            if manifest != _acquire_cache_contract(options):
                raise RuntimeError("素材缓存与当前视频流方案不一致，未触碰现有未完成文件。")
            matches.append(canonical)
        elif any(canonical.glob("*.part")):
            if not _legacy_acquire_cache_matches(canonical, options):
                raise RuntimeError("无法核实现有部分下载与当前视频流一致；原文件已保留，未开始续传。")
            matches.append(canonical)
        elif any(canonical.iterdir()):
            raise RuntimeError("无法核实现有素材缓存与当前视频流方案一致；现有文件已保留，未开始下载。")

    for candidate in work_dir.iterdir() if work_dir.is_dir() else ():
        if candidate == canonical:
            continue
        if not _cache_child_is_safe(candidate, work_dir):
            continue
        manifest = _read_acquire_cache_manifest(candidate)
        if manifest is not None:
            if manifest == _acquire_cache_contract(options):
                matches.append(candidate)
            continue
        if _legacy_acquire_cache_matches(candidate, options):
            matches.append(candidate)

    if len(matches) > 1:
        raise RuntimeError("发现多个符合当前画质的旧素材缓存，无法安全决定续传来源；所有未完成文件均已保留。")
    return matches[0] if matches else canonical


def _write_acquire_cache_manifest(candidate: Path, options: Any) -> None:
    contract = _acquire_cache_contract(options)
    path = candidate / ".acquire-cache.json"
    existing = _read_acquire_cache_manifest(candidate)
    if existing is not None:
        if existing != contract:
            raise RuntimeError("素材缓存与当前视频流方案不一致，未触碰现有未完成文件。")
        return
    temporary = candidate / f".acquire-cache-{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(contract, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def commit_rendered_candidate(candidate: Path, work_dir: Path) -> Path:
    """Atomically publish a validated render while retaining prior renders."""
    target = _versioned_target(work_dir, "rendered", ".mp4")
    candidate.replace(target)
    return target


def run_acquire(options: Any, work_dir: Path, artifacts: WorkflowArtifacts, ctx: PipelineContext, log: Callable[[str], None], progress: Callable[[float], None] | None = None) -> WorkflowArtifacts:
    ensure_rights_confirmed(bool(options.i_have_rights))
    ctx.check_cancelled()
    _emit(ctx, PipelineStage.DOWNLOADING, 5, "正在获取素材", log)
    canonical_cache_dir = _canonical_acquire_cache_dir(work_dir, options) if options.source_kind == "url" else None
    candidate_dir = (_acquire_cache_dir(work_dir, options) if options.source_kind == "url"
                     else work_dir / f".acquire-candidate-{uuid.uuid4().hex[:10]}")
    candidate_dir.mkdir(parents=True, exist_ok=options.source_kind == "url")
    if options.source_kind == "url":
        _write_acquire_cache_manifest(candidate_dir, options)
        if canonical_cache_dir is not None and candidate_dir != canonical_cache_dir:
            proposal = getattr(options, "quality_proposal", None) or {}
            width, height = proposal.get("width"), proposal.get("height")
            resolution = f"{width}×{height}" if width and height else "已确认方案"
            log(f"已核对现有部分下载的来源、视频格式与目标画质（{resolution}）；将沿用该缓存续传。")
    try:
        if options.source_kind == "url":
            job = download_with_ytdlp(
                options.source, work_dir=candidate_dir, title=options.title, max_seconds=options.max_seconds,
                require_reuse_allowed=options.require_reuse_allowed, cookies_from_browser=options.cookies_from_browser,
                cookies_file=options.cookies_file, po_token_mode=options.youtube_po_token_mode,
                proxy=options.youtube_proxy, download_quality=options.download_quality,
                resource_profile=options.resource_profile,
                source_language=options.source_language,
                prefer_platform_subtitles=getattr(options, "prefer_platform_subtitles", True),
                progress=progress, cancel_check=ctx.check_cancelled, log=log,
                quality_proposal=getattr(options, "quality_proposal", None),
            )
        elif options.source_kind == "file":
            source = Path(options.source)
            # Re-running acquire for the same registered local material is a
            # no-op: preserve its original path and hash. New local imports
            # still go through the isolated candidate directory above.
            if artifacts.original_video and source.resolve() == Path(artifacts.original_video).resolve():
                job = VideoJob(work_dir.name, str(source), "file", work_dir, title=options.title or source.stem, raw_video=source)
            else:
                job = import_local_video(source, work_dir=candidate_dir, title=options.title)
        else:
            raise RuntimeError(f"Unknown source kind: {options.source_kind}")
        if not job.raw_video or not job.raw_video.is_file():
            raise RuntimeError("No raw video was prepared.")
        ctx.check_cancelled()
        if not validate_media(job.raw_video, "video"):
            raise RuntimeError("获取的原视频无法通过 ffprobe 校验，旧素材与成片保持不变。")
        # Commit only after ffprobe validation performed by download/import.
        committed_video = job.raw_video if (options.source_kind == "file" and artifacts.original_video and Path(job.raw_video).resolve() == Path(artifacts.original_video).resolve()) else _commit_acquired_file(job.raw_video, work_dir, "raw")
        committed_subtitles = None
        if job.source_subtitles and job.source_subtitles.is_file():
            committed_subtitles = _commit_acquired_file(job.source_subtitles, work_dir, "platform")
        committed_thumbnail = None
        if job.thumbnail_path and job.thumbnail_path.is_file():
            committed_thumbnail = _commit_acquired_file(job.thumbnail_path, work_dir, "cover")
    except BaseException:
        # The active pointers remain untouched because assignment happens below.
        raise
    finally:
        # A failed URL acquire may contain gigabytes of resumable .part data.
        # Local imports never need resume and retain their prior cleanup rule.
        if options.source_kind == "file":
            shutil.rmtree(candidate_dir, ignore_errors=True)
    artifacts.original_video = str(committed_video)
    artifacts.raw_video = str(committed_video)
    _discard_audio_artifact(artifacts, work_dir)
    artifacts.platform_subtitles = str(committed_subtitles) if committed_subtitles else None
    artifacts.revision = 0
    artifacts.last_edit = None
    artifacts.edit_state = "clean"
    artifacts.title = job.title
    artifacts.description = job.description or ""
    artifacts.source = job.source
    artifacts.source_kind = job.source_kind
    artifacts.license = job.license
    artifacts.view_count = job.view_count
    artifacts.webpage_url = job.webpage_url
    artifacts.thumbnail_url = job.thumbnail_url
    artifacts.thumbnail_path = str(committed_thumbnail) if committed_thumbnail else None
    log(f"Raw video: {committed_video}")
    if options.source_kind == "url":
        has_partial = any(
            path.is_file() and not path.is_symlink()
            for path in candidate_dir.glob("*.part")
        )
        protection_root = candidate_dir / ".resume-protection"
        has_recovery_data = protection_root.is_dir() and any(
            path.is_file() or path.is_symlink()
            for path in protection_root.rglob("*")
        )
        if has_partial or has_recovery_data:
            log("素材已校验并提交；缓存中仍有未完成或恢复保护文件，因此保留该下载缓存。")
        else:
            shutil.rmtree(candidate_dir, ignore_errors=True)
    return artifacts


def _extract_audio(video: Path, output: Path, ctx: PipelineContext) -> Path:
    import inspect
    signature = inspect.signature(extract_audio)
    supports_cancel = "cancel_check" in signature.parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    return extract_audio(video, output, cancel_check=ctx.is_cancelled) if supports_cancel else extract_audio(video, output)


def _estimate_audio_offset(audio_segments: list[Any], ocr_segments: list[Any]) -> float | None:
    from difflib import SequenceMatcher
    clean = lambda text: re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()
    diffs: list[float] = []
    for ocr in ocr_segments:
        if len(clean(ocr.text)) < 6:
            continue
        scored = [(SequenceMatcher(None, clean(ocr.text), clean(audio.text)).ratio(), audio) for audio in audio_segments]
        if scored:
            score, best = max(scored, key=lambda item: item[0])
            if score >= 0.75:
                diffs.append(best.start - ocr.start)
    if len(diffs) < 2:
        return None
    diffs.sort(); median = diffs[len(diffs) // 2]
    core = sorted(value for value in diffs if abs(value - median) <= 1.5)
    return core[len(core) // 2] if len(core) >= 2 else None


def _shift_segments(segments: list[Any], offset: float) -> None:
    for segment in segments:
        segment.start = max(0.0, round(segment.start - offset, 2))
        segment.end = max(0.1, round(segment.end - offset, 2))


def run_extract(options: Any, work_dir: Path, artifacts: WorkflowArtifacts, ctx: PipelineContext, log: Callable[[str], None], progress: Callable[[float], None] | None = None) -> WorkflowArtifacts:
    # Extraction can be retried with a different subtitle source; never carry
    # language evidence from the previous attempt into the new translation.
    artifacts.asr_evidence = None
    artifacts.asr_review = None
    artifacts.asr_timeline_shift_seconds = 0.0
    raw_video = artifacts.path("raw_video")
    if not raw_video or not validate_media(raw_video, "video"):
        raise RuntimeError("原视频检查点无效，请先重新获取素材。")
    glossary = load_glossary(work_dir)
    work_dir = _attempt_dir(work_dir, "extract")
    source_segments, source_srt = work_dir / "segments.source.json", work_dir / "source.srt"
    audio: Path | None = artifacts.path("audio")
    used_ocr = False
    items: list[Any] = []
    mode = options.subtitle_source
    platform_items: list[Any] = []
    platform_path = artifacts.path("platform_subtitles")
    if getattr(options, "prefer_platform_subtitles", True) and mode in {"auto", "merged"} and platform_path and platform_path.is_file():
        try:
            platform_items = read_srt_segments(platform_path, max_seconds=options.max_seconds)
        except (OSError, ValueError) as exc:
            log(f"平台字幕无法读取，已回退 Whisper：{exc}")
        else:
            if platform_items:
                log(f"检测到平台字幕（{len(platform_items)} 条），本次跳过 Whisper 语音转写。")
            else:
                log("平台字幕为空，已回退 Whisper 语音转写。")

    def ensure_audio() -> Path:
        nonlocal audio
        # Always refresh. Cached WAV may predate timestamp-preserving extract_audio
        # fixes, or belong to a previous raw_video after acquire.
        _emit(ctx, PipelineStage.AUDIO, 16, "正在提取音频", log)
        audio = extract_audio(
            raw_video, work_dir / "audio.wav", cancel_check=ctx.is_cancelled,
            channel=getattr(options, "audio_channel", "mix"),
        )
        artifacts.audio = str(audio)
        return audio

    def transcribe(target_json: Path, target_srt: Path) -> list[Any]:
        if platform_items:
            copied = [Segment(item.start, item.end, item.text) for item in platform_items]
            save_segments(target_json, copied)
            write_srt(target_srt, copied, display_mode="source")
            return copied
        source_audio = ensure_audio()
        evidence_path = target_json.with_name(target_json.stem + ".asr-evidence.json")
        vad_mode = str(getattr(options, "asr_vad_mode", "standard") or "standard")
        initial_prompt = build_initial_prompt(
            getattr(options, "title", None),
            getattr(options, "description", None),
            getattr(options, "tags", None),
            getattr(options, "asr_hint", None),
            glossary=glossary,
        )
        if initial_prompt:
            log(f"识别提示：{initial_prompt}")
        else:
            log("识别提示：未使用。标题、简介和标签里没有可用专名，也没有手写提示。")
        log("人声检测：敏感。更早保留风声和水声里的说话，句首多留一点。" if vad_mode == "sensitive" else "人声检测：标准。")
        result = transcribe_audio(
            source_audio, segments_json=target_json, srt_path=target_srt,
            model_size=options.whisper_model_size, language=options.source_language,
            device=options.device, compute_type=options.compute_type, initial_prompt=initial_prompt,
            beam_size=options.beam_size, log=log, cancel_check=ctx.check_cancelled,
            resource_profile=options.resource_profile,
            evidence_path=evidence_path,
            vad_mode=vad_mode,
        )
        artifacts.asr_evidence = str(evidence_path)
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            detected = str(evidence.get("detected_language") or "").strip()
            probability = float(evidence.get("language_probability"))
        except (OSError, ValueError, TypeError):
            detected, probability = "", 0.0
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            probability = 0.0
        if detected:
            log(f"Whisper 检测语言：{detected}（置信度 {probability:.2f}）。")
        return result

    def ocr(target_json: Path, target_srt: Path) -> list[Any]:
        return extract_ocr_subtitles(
            raw_video, work_dir=work_dir, segments_json=target_json, srt_path=target_srt,
            interval_seconds=max(0.2, options.ocr_interval), crop_bottom_ratio=max(0.05, min(1.0, options.ocr_crop_ratio)),
            min_chars=max(1, options.ocr_min_chars), cancel_check=ctx.is_cancelled,
            ocr_language=options.ocr_language, log=log, resource_profile=options.resource_profile,
            progress=progress,
        )

    if mode == "merged":
        _emit(ctx, PipelineStage.TRANSCRIBING, 28, "正在整理平台字幕并读取画面文字" if platform_items else "正在转写并读取画面文字", log)
        audio_items = transcribe(work_dir / "segments.audio.json", work_dir / "audio.srt")
        try:
            ocr_items = ocr(work_dir / "segments.ocr.json", work_dir / "ocr.srt")
        except CancellationRequested:
            raise
        except Exception as exc:
            artifacts.ocr_status, artifacts.ocr_message = "fallback", str(exc)
            log(f"OCR 识别失败，本次合并模式将只使用 Whisper 音频字幕。原因：{exc}")
            ocr_items = []
        else:
            artifacts.ocr_status, artifacts.ocr_message = "completed", None
        used_ocr = bool(ocr_items)
        offset = _estimate_audio_offset(audio_items, ocr_items)
        if offset is not None and 0.12 <= abs(offset) <= 3.0:
            _shift_segments(audio_items, offset); log(f"Audio timeline corrected by {offset:+.2f}s using OCR anchors.")
            artifacts.asr_timeline_shift_seconds = offset
        items = merge_audio_ocr_segments(audio_items, ocr_items)
        if platform_items:
            artifacts.subtitle_extraction_mode = "platform+ocr" if used_ocr else "platform"
        else:
            artifacts.subtitle_extraction_mode = "merged" if used_ocr else "audio-fallback"
        save_segments(source_segments, items); write_srt(source_srt, items, display_mode="source")
    elif mode == "ocr":
        _emit(ctx, PipelineStage.OCR, 28, "正在读取画面字幕", log)
        try:
            items = ocr(source_segments, source_srt); used_ocr = True
        except CancellationRequested:
            raise
        except Exception as exc:
            artifacts.ocr_status, artifacts.ocr_message = "fallback", str(exc)
            if not options.ocr_fallback_to_audio:
                raise
            log(f"OCR 识别失败，已明确切换为 Whisper 音频转写。原因：{exc}")
            items = transcribe(source_segments, source_srt)
            artifacts.subtitle_extraction_mode = "platform-fallback" if platform_items else "audio-fallback"
        else:
            artifacts.ocr_status, artifacts.ocr_message = "completed", None
            artifacts.subtitle_extraction_mode = "ocr"
    else:
        _emit(ctx, PipelineStage.TRANSCRIBING, 28, "正在整理平台字幕" if platform_items else "正在语音转写", log)
        items = transcribe(source_segments, source_srt)
        artifacts.subtitle_extraction_mode = "platform" if platform_items else "audio"
        artifacts.ocr_status, artifacts.ocr_message = "not-requested", None
        if not items and (mode == "auto" or options.ocr_fallback_to_audio):
            artifacts.asr_evidence = None
            try:
                items = ocr(source_segments, source_srt); used_ocr = True
            except CancellationRequested:
                raise
            except Exception as exc:
                artifacts.ocr_status, artifacts.ocr_message = "failed", str(exc)
                log(f"OCR fallback also failed: {exc}")
            else:
                artifacts.ocr_status, artifacts.ocr_message = "completed", None
                artifacts.subtitle_extraction_mode = "ocr-fallback"
    if not items:
        # An empty result is different from a model load/runtime failure.  The
        # latter is raised by transcribe_audio with the concrete CUDA/model
        # error; reaching this branch means the selected extraction path ran
        # successfully but produced no usable segments.  Keep the message
        # tied to the selected source so a user does not mistake a silent or
        # music-only video for a broken Whisper installation.
        if mode == "audio":
            raise RuntimeError(
                f"Whisper {options.whisper_model_size} 已正常运行，但没有识别到可用人声字幕。"
                "这通常表示视频没有旁白/人声，或人声过于短促；请确认视频包含语音，"
                "或将“字幕来源”改为“自动/画面字幕 OCR”。模型文件和设备检查已通过。"
            )
        if mode == "ocr":
            raise RuntimeError(
                "OCR 已运行，但没有在指定区域识别到有效字幕。请检查字幕来源、截取区域和最小字符数，"
                "或改用“自动/音频转写”。"
            )
        if mode == "merged":
            raise RuntimeError(
                "音频转写和画面字幕 OCR 都未产生可用字幕。请确认视频含有人声或平台字幕，"
                "并检查 OCR 截取区域。"
            )
        raise RuntimeError(
            "自动字幕提取未得到可用结果：音频转写没有识别到人声，画面字幕 OCR 也未识别到有效文字。"
            "请确认视频内容，或在处理设置中明确选择“音频转写”或“画面字幕 OCR”并调整参数。"
        )
    artifacts.source_segments, artifacts.source_srt = str(source_segments), str(source_srt)
    artifacts.used_ocr_subtitles = used_ocr
    # Prepare a durable review queue without loading a model or running VAD.
    # The queue is advisory until the user explicitly starts a review round.
    if audio and audio.is_file() and mode in {"audio", "auto", "merged"}:
        baseline_words = words_path_for_segments(source_segments)
        review = prepare_session(
            work_dir=work_dir.parent, job_id=work_dir.parent.name, audio_path=audio,
            baseline_words_path=baseline_words if baseline_words.is_file() else source_segments,
            baseline_kind="canonical" if baseline_words.is_file() else "legacy_partial",
            model=str(options.whisper_model_size),
            language=resolve_review_language(
                options.source_language,
                Path(artifacts.asr_evidence) if artifacts.asr_evidence else None,
                allow_detected=mode in {"audio", "audio-fallback", "merged"},
            ),
            beam_size=int(options.beam_size), channel=getattr(options, "audio_channel", "mix"),
            subtitle_revision=int(artifacts.revision),
            subtitle_fingerprint=hashlib.sha256(source_segments.read_bytes()).hexdigest(),
            timeline_shift_seconds=float(artifacts.asr_timeline_shift_seconds or 0.0),
            source_video_path=raw_video,
        )
        artifacts.asr_review = str(session_path(work_dir.parent))
        log(f"语音复核队列已保存：{len(review.get('candidates', []))} 个待复核区间；未自动启动模型。")
    return artifacts


def run_translate(options: Any, work_dir: Path, artifacts: WorkflowArtifacts, ctx: PipelineContext, log: Callable[[str], None], progress: Callable[[float], None] | None = None) -> WorkflowArtifacts:
    source_segments = artifacts.path("source_segments")
    if not source_segments or not validate_segments(source_segments):
        raise RuntimeError("源字幕检查点无效，请先重新提取字幕。")
    glossary = load_glossary(work_dir)
    log(f"术语表：{len(glossary)} 条。" if glossary else "术语表：未使用。")
    work_dir = _attempt_dir(work_dir, "translate")
    extracted_items = load_segments(source_segments)
    configured_source_language = str(options.source_language or "").strip() or None
    effective_source_language = resolve_review_language(
        configured_source_language,
        Path(artifacts.asr_evidence) if artifacts.asr_evidence else None,
        allow_detected=artifacts.subtitle_extraction_mode in {"audio", "audio-fallback"},
    )
    if effective_source_language is None and configured_source_language and configured_source_language.lower() != "auto":
        effective_source_language = configured_source_language
    if effective_source_language is None and artifacts.asr_evidence and artifacts.subtitle_extraction_mode in {"audio", "audio-fallback"}:
        try:
            evidence = json.loads(Path(artifacts.asr_evidence).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            evidence = {}
        if not isinstance(evidence, dict):
            evidence = {}
        detected = str(evidence.get("detected_language") or "").strip()
        try:
            probability = float(evidence.get("language_probability"))
        except (TypeError, ValueError):
            probability = 0.0
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            probability = 0.0
        if detected and probability >= 0.80:
            effective_source_language = detected
            log(f"ASR 检测语言：{detected}（置信度 {probability:.2f}）；翻译将使用该语言。")
        elif detected:
            log(f"ASR 检测语言 {detected} 的置信度为 {probability:.2f}，低于 0.80；翻译保持自动识别。")
    translation_source = work_dir / "segments.translation-source.json"
    translation_source_srt = work_dir / "translation-source.srt"

    def prepare_progress(fraction: float) -> None:
        if progress:
            progress(max(0.0, min(0.2, fraction * 0.2)))

    source_items: list[Any]
    strict_speech_alignment = False
    source_corrected_before_grouping = False
    speech_fragment_reviews: list[str] = []
    detached_speech_reviews: list[str] = []
    boundary_warnings: list[str] = []

    def record_boundary_warning(message: str) -> None:
        boundary_warnings.append(message)
        log(message)

    words_path = words_path_for_segments(source_segments)
    source_mode = artifacts.subtitle_extraction_mode
    source_has_word_timing = (
        source_mode == "audio"
        or (source_mode == "audio-fallback" and options.subtitle_source == "ocr")
        or (not source_mode and options.subtitle_source == "audio")
    )
    specialized_translation_only = (
        options.translator.lower() == "ollama"
        and options.translate_model in dependencies.OLLAMA_SPECIALIZED_TRANSLATION_MODELS
    )
    if source_has_word_timing and words_path.is_file():
        # Establish punctuation on speech-timed atoms before choosing translation
        # boundaries. Every resulting unit keeps its outer word timestamps.
        ctx.check_cancelled()
        _emit(ctx, PipelineStage.TRANSLATING, 58, "正在按语音时间整理字幕", log)
        timed_words = load_words(words_path)
        packed_speech = pack_cues_from_words_detailed(timed_words)
        speech_cues = packed_speech.cues
        review_speech_source = options.translator.lower() in {"openai", "deepseek"}
        if (options.smart_translation and options.translator.lower() in {"openai", "deepseek", "ollama"}
                and not specialized_translation_only):
            if review_speech_source:
                log("正在先校对转写并确定句末标点，再划分翻译字幕；这会增加翻译耗时和 API 用量。")
            try:
                corrected_cues = correct_source_segments(
                    speech_cues, provider=options.translator, model=options.translate_model,
                    batch_size=options.batch_size, source_title=options.title or artifacts.title,
                    source_description=artifacts.description,
                    cancel_check=ctx.check_cancelled, force_review=review_speech_source,
                    strict_alignment=True,
                    checkpoint_path=_seed_attempt_checkpoint(work_dir, "source_correction_checkpoint.json"),
                    log=log,
                    deepseek_thinking=getattr(options, "deepseek_thinking", "disabled"),
                    glossary=glossary,
                )
            except CancellationRequested:
                raise
            except Exception as exc:
                if review_speech_source:
                    raise RuntimeError(
                        "源文标点校对中断；已保留完成的批次，修正 API 问题后重试翻译阶段即可续跑。"
                    ) from exc
                log(f"源文标点校对未完成，继续使用原始转写标点；原因：{exc}")
                corrected_cues = speech_cues
            if corrected_cues:
                speech_cues = split_corrected_punctuation_cues(
                    packed_speech.cues, corrected_cues, timed_words, log=record_boundary_warning,
                )
                source_corrected_before_grouping = True
        source_items = group_speech_translation_units(speech_cues, log=record_boundary_warning)
        if not source_items:
            raise RuntimeError("词级转写记录为空，请重新提取字幕。")
        if packed_speech.detached_fragments:
            detached_path = work_dir / "translation-source.detached.json"
            atomic_write_manifest(detached_path, {
                "version": 1,
                "fragments": [asdict(item) for item in packed_speech.detached_fragments],
            })
            detached_speech_reviews = [
                f"{item.start:.1f}s “{item.text}”"
                for item in packed_speech.detached_fragments
            ]
        speech_fragment_reviews = long_gap_fragment_warnings(packed_speech.cues)
        strict_speech_alignment = True
        log(
            f"语音时间整理：{len(packed_speech.cues)} 个时间片经标点整理得到 {len(source_items)} 个翻译单元；"
            f"{len(detached_speech_reviews)} 处停顿残词已单独留档，长停顿不会被译文跨越。"
        )
        prepare_progress(1.0)
    elif options.smart_translation and options.translator.lower() in {"openai", "deepseek", "ollama"} and not specialized_translation_only:
        ctx.check_cancelled()
        _emit(ctx, PipelineStage.TRANSLATING, 58, "正在按语义重切字幕", log)
        source_items = resegment_source_segments(
            extracted_items,
            provider=options.translator,
            model=options.translate_model or None,
            batch_size=options.batch_size,
            checkpoint_path=_seed_attempt_checkpoint(work_dir, "semantic_resegmentation_checkpoint.json"),
            cancel_check=ctx.check_cancelled,
            progress=prepare_progress,
            log=log,
            deepseek_thinking=getattr(options, "deepseek_thinking", "disabled"),
        )
    else:
        source_items = coalesce_incomplete_source_cues(extracted_items)
        if not segments_content_equal(extracted_items, source_items):
            log("已按规则合并未完成字幕切分。")
        if specialized_translation_only:
            log("TranslateGemma 专用于翻译；本次跳过通用模型语义切句与源文校对，保留规则切句结果。")

    if artifacts.asr_evidence:
        try:
            evidence = json.loads(Path(artifacts.asr_evidence).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            evidence = {}
        suspect_intervals = evidence.get("suspect_intervals") if isinstance(evidence, dict) else []
        if isinstance(suspect_intervals, list):
            valid_intervals: list[tuple[float, float]] = []
            try:
                clock_offset = float(artifacts.asr_timeline_shift_seconds or 0.0)
            except (TypeError, ValueError):
                clock_offset = 0.0
            if not math.isfinite(clock_offset):
                clock_offset = 0.0
            for interval in suspect_intervals:
                if not isinstance(interval, dict):
                    continue
                try:
                    start, end = float(interval.get("start")), float(interval.get("end"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(start) and math.isfinite(end) and end > start:
                    valid_intervals.append((start - clock_offset, end - clock_offset))
            for unit in source_items:
                flagged = any(
                    start < unit.end and end > unit.start
                    for start, end in valid_intervals
                )
                if flagged:
                    unit.needs_review = True
            flagged_count = sum(1 for unit in source_items if unit.needs_review)
            if flagged_count:
                log(f"ASR 低置信度区间与 {flagged_count} 个翻译单元重叠，已标记待复核。")
    save_segments(translation_source, source_items)
    write_srt(translation_source_srt, source_items, display_mode="source")
    content_warning = instruction_like_content_warning(source_items)
    artifacts.content_warnings = [content_warning] if content_warning else []
    artifacts.content_warnings.extend(boundary_warnings)
    if content_warning:
        log(f"人工复核提醒：{content_warning}")
    if detached_speech_reviews:
        detached_warning = (
            f"有 {len(detached_speech_reviews)} 处词句在停顿边界前被截断；"
            "原文已保存到 translation-source.detached.json，未作为独立字幕翻译或烧录。请复核："
            + "；".join(detached_speech_reviews[:3])
        )
        artifacts.content_warnings.append(detached_warning)
        log(f"人工复核提醒：{detached_warning}")
    if speech_fragment_reviews:
        fragment_warning = (
            f"有 {len(speech_fragment_reviews)} 处词句在长停顿前后被剪开；"
            "为避免字幕跨越无声画面，未自动合并。请复核："
            + "；".join(speech_fragment_reviews[:3])
        )
        artifacts.content_warnings.append(fragment_warning)
        log(f"人工复核提醒：{fragment_warning}")
    if (options.smart_translation and options.translator.lower() in {"openai", "deepseek", "ollama"}
            and not specialized_translation_only and not source_corrected_before_grouping):
        try:
            review_speech_source = strict_speech_alignment and options.translator.lower() in {"openai", "deepseek"}
            if review_speech_source:
                log("正在校对语音转写中可能听错的词；这会增加翻译耗时和 API 用量。")
            corrected = correct_source_segments(
                source_items, provider=options.translator, model=options.translate_model,
                batch_size=options.batch_size, source_title=options.title or artifacts.title,
                source_description=artifacts.description,
                cancel_check=ctx.check_cancelled,
                force_review=review_speech_source,
                strict_alignment=strict_speech_alignment,
                checkpoint_path=_seed_attempt_checkpoint(work_dir, "source_correction_checkpoint.json"),
                log=log,
                deepseek_thinking=getattr(options, "deepseek_thinking", "disabled"),
                glossary=glossary,
            )
            if corrected:
                source_items = corrected
                save_segments(translation_source, source_items)
                write_srt(translation_source_srt, source_items, display_mode="source")
        except CancellationRequested:
            raise
        except Exception as exc:
            warning = f"源文校对未完成，后续译文可能保留识别错词；请人工复核。原因：{exc}"
            artifacts.content_warnings.append(warning)
            log(warning)
            if review_speech_source:
                raise RuntimeError(
                    "源文校对中断；已保留已完成批次，未用未校对的文字继续翻译。"
                    "修正 API 问题后重试翻译阶段即可续跑。"
                ) from exc
    ctx.check_cancelled(); _emit(ctx, PipelineStage.TRANSLATING, 62, "正在翻译字幕", log)

    def translate_progress(fraction: float) -> None:
        if progress:
            progress(0.2 + max(0.0, min(0.8, fraction * 0.8)))

    translated_json, translated_srt = work_dir / "segments.translated.json", work_dir / "zh.srt"
    translated_items = translate_segments_file(
        translation_source, output_json=translated_json, output_srt=translated_srt,
        provider=options.translator, target_lang=options.target_lang, model=options.translate_model,
        batch_size=options.batch_size, validate_language=True,
        display_mode=options.subtitle_display_mode, smart_translation=options.smart_translation,
        smart_layout=options.smart_subtitle_layout,
        checkpoint_path=_seed_attempt_checkpoint(work_dir, "translation_checkpoint.json"),
        cancel_check=ctx.check_cancelled,
        max_workers=translation_worker_limit(options.resource_profile),
        progress=translate_progress,
        source_timing_slots=extracted_items if not strict_speech_alignment else None,
        strict_alignment=strict_speech_alignment,
        log=log,
        source_lang=effective_source_language,
        deepseek_thinking=getattr(options, "deepseek_thinking", "disabled"),
        glossary=glossary,
    )
    dense_count = subtitle_density_overflow_count(translated_items)
    if dense_count:
        log(
            f"字幕排版提醒：仍有 {dense_count} 条译文超过两行建议容量。"
            "渲染时会尝试用当前翻译模型精简超长译文，并保留原时间轴；"
            "无法在保留主要含义的前提下压缩的字幕再提示人工复核。"
        )
    fast_count = subtitle_high_reading_speed_count(translated_items)
    if fast_count:
        log(
            f"阅读速度提醒：{fast_count} 条字幕超过每秒 12 个中文显示单位。"
            "系统保留了原语音时间，不会擅自延长字幕或错开声音；建议人工精简译文。"
        )
    ctx.check_cancelled()
    # The dedicated 发布 page is the primary publish path. Generate localized
    # title/tags whenever a translator is selected, even if Playwright is not
    # set to open automatically after render.
    # TranslateGemma is translation-specialized. Avoid sending unrelated
    # source-correction, segmentation, and publish-metadata prompts to it.
    metadata_provider = "none" if specialized_translation_only else options.translator
    try:
        metadata = generate_publish_metadata(
            source_title=options.title or artifacts.title, source_description=artifacts.description,
            source_segments=source_items, translated_segments=translated_items,
            provider=metadata_provider, target_lang=options.target_lang, model=options.translate_model,
            cancel_check=ctx.check_cancelled,
            deepseek_thinking=getattr(options, "deepseek_thinking", "disabled"),
        )
        ctx.check_cancelled()
    except CancellationRequested:
        raise
    except Exception as exc:
        log(f"Smart publish metadata failed, using fallback: {exc}")
        metadata = generate_publish_metadata(
            source_title=options.title or artifacts.title, source_description=artifacts.description,
            source_segments=source_items, translated_segments=translated_items,
            provider="none", target_lang=options.target_lang, model=None,
        )
    metadata_path = save_publish_metadata(work_dir / "publish_metadata.json", metadata)
    artifacts.translated_segments, artifacts.translated_srt = str(translated_json), str(translated_srt)
    artifacts.publish_metadata = str(metadata_path)
    return artifacts


def _plan_cue_font_sizes(
    segments: list[Segment],
    video: Path,
    *,
    font_size: int,
    outline: int,
    shadow: int,
    smart_layout: bool,
    line_width: int,
    display_limit: float,
) -> tuple[dict[int, int], list[tuple[int, float, str]]]:
    """Keep the configured font except where a cue needs a small local reduction.

    The SRT is still wrapped at ``line_width``. Check those exact lines against
    the smaller font's physical capacity, so the preflight matches the render.
    Cue numbers are one-based to match SRT and ASS event order.
    """
    issues = translated_layout_issues(
        segments,
        smart_layout=smart_layout,
        max_chars_per_line=line_width,
        max_display_units_per_line=display_limit,
    )
    selected_size = max(1, int(font_size))
    minimum_size = min(selected_size, max(18, math.ceil(selected_size * 0.75)))
    capacities: dict[int, int | None] = {}
    overrides: dict[int, int] = {}
    unresolved: list[tuple[int, float, str]] = []
    for cue_number, start, formatted in issues:
        for candidate_size in range(selected_size - 1, minimum_size - 1, -1):
            if candidate_size not in capacities:
                capacities[candidate_size] = subtitle_line_capacity(
                    video, font_size=candidate_size, outline=outline, shadow=shadow,
                )
            capacity = capacities[candidate_size]
            if capacity is not None and not translated_layout_issues(
                [segments[cue_number - 1]],
                smart_layout=smart_layout,
                max_chars_per_line=line_width,
                max_display_units_per_line=capacity + 0.5,
            ):
                overrides[cue_number] = candidate_size
                break
        else:
            unresolved.append((cue_number, start, formatted))
    return overrides, unresolved


def _record_render_layout_warning(
    artifacts: WorkflowArtifacts, log: Callable[[str], None], message: str,
) -> None:
    log(message)
    if not isinstance(artifacts.content_warnings, list):
        artifacts.content_warnings = []
    if message not in artifacts.content_warnings:
        artifacts.content_warnings.append(message)


def run_render(options: Any, work_dir: Path, artifacts: WorkflowArtifacts, ctx: PipelineContext, log: Callable[[str], None], progress: Callable[[float], None] | None = None) -> WorkflowArtifacts:
    video, subtitle = artifacts.path("raw_video"), artifacts.path("translated_srt")
    translated_json = artifacts.path("translated_segments")
    regenerated_subtitle: Path | None = None
    rendered_candidate: Path | None = None
    cue_font_sizes: dict[int, int] = {}
    translated_items: list[Segment] = []
    changed_count = 0
    render_line_width = 18
    render_line_display_limit = 36.5

    def report_progress(fraction: float) -> None:
        if progress:
            progress(max(0.0, min(1.0, fraction)))

    try:
        log("正在检查原视频媒体流。")
        report_progress(0.03)
        if not video or not validate_media(video, "video"):
            raise RuntimeError("原视频检查点无效，请重新获取素材。")
        if translated_json and translated_json.is_file():
            # The JSON checkpoint is authoritative. Rebuild SRT with the current
            # display mode; a previous SRT may still use another mode.
            log("正在读取并检查译文字幕。")
            report_progress(0.12)
            translated_items = load_segments(translated_json)
            if options.subtitle_display_mode == "translated":
                try:
                    measured_width = subtitle_line_capacity(
                        video,
                        font_size=options.font_size,
                        outline=options.subtitle_outline,
                        shadow=options.subtitle_shadow,
                    )
                except CancellationRequested:
                    raise
                except Exception as exc:
                    measured_width = None
                    _record_render_layout_warning(
                        artifacts, log,
                        f"字幕排版测量未完成，按默认行宽继续渲染：{exc}",
                    )
                if measured_width is None:
                    log("无法读取原片尺寸；字幕行宽按保守默认值检查。")
                else:
                    render_line_width = measured_width
                    render_line_display_limit = measured_width + 0.5
                    log(
                        f"字幕排版已按原片尺寸和当前字号自适配：每行最多 {measured_width} 个全角字，最多两行。"
                    )
                try:
                    layout_issues = translated_layout_issues(
                        translated_items,
                        smart_layout=options.smart_subtitle_layout,
                        max_chars_per_line=render_line_width,
                        max_display_units_per_line=render_line_display_limit,
                    )
                except CancellationRequested:
                    raise
                except Exception as exc:
                    layout_issues = []
                    _record_render_layout_warning(
                        artifacts, log,
                        f"字幕排版检查未完成，保留完整译文继续渲染：{exc}",
                    )
                if layout_issues:
                    try:
                        cue_font_sizes, layout_issues = _plan_cue_font_sizes(
                            translated_items, video,
                            font_size=options.font_size,
                            outline=options.subtitle_outline,
                            shadow=options.subtitle_shadow,
                            smart_layout=options.smart_subtitle_layout,
                            line_width=render_line_width,
                            display_limit=render_line_display_limit,
                        )
                    except CancellationRequested:
                        raise
                    except Exception as exc:
                        cue_font_sizes = {}
                        _record_render_layout_warning(
                            artifacts, log,
                            f"字幕局部字号调整未完成，保留完整译文继续渲染：{exc}",
                        )
                    if cue_font_sizes:
                        log(
                            f"渲染排版自修复：{len(cue_font_sizes)} 条字幕单独调整字号，"
                            f"从 {options.font_size} 调至最低 {min(cue_font_sizes.values())}；"
                            "译文和时间轴保持不变。"
                        )
                if layout_issues:
                    repair_indexes = {
                        index - 1 for index, _start, _text in layout_issues
                    }
                    repair_indexes = sorted(repair_indexes)
                    model_label = options.translate_model or options.translator
                    model_usage_hint = (
                        "会产生额外 API 调用。"
                        if options.translator.lower() in {"openai", "deepseek"}
                        else ""
                    )
                    log(
                        f"渲染排版自修复：发现 {len(repair_indexes)} 条字幕超出安全容量，"
                        f"将用当前翻译模型（{model_label}）逐条压缩；不会移动时间轴。"
                        f"{model_usage_hint}"
                    )
                    try:
                        repaired_items, _unresolved = compact_overflowing_subtitles(
                            translated_items,
                            repair_indexes,
                            provider=options.translator,
                            target_lang=options.target_lang,
                            model=options.translate_model,
                            max_chars_per_line=render_line_width,
                            max_display_units_per_line=render_line_display_limit,
                            smart_layout=options.smart_subtitle_layout,
                            checkpoint_path=work_dir / "subtitle-layout-repair.json",
                            cancel_check=ctx.check_cancelled,
                            log=log,
                            deepseek_thinking=getattr(options, "deepseek_thinking", "disabled"),
                        )
                    except CancellationRequested:
                        raise
                    except Exception as exc:
                        _record_render_layout_warning(
                            artifacts, log,
                            f"字幕自动精简未完成，保留原译文继续渲染：{exc}",
                        )
                        repaired_items = translated_items
                    try:
                        repair_aligned = len(repaired_items) == len(translated_items) and all(
                            isinstance(after, Segment)
                            and before.start == after.start
                            and before.end == after.end
                            and before.text == after.text
                            for before, after in zip(translated_items, repaired_items)
                        )
                    except (TypeError, ValueError):
                        repair_aligned = False
                    if not repair_aligned:
                        _record_render_layout_warning(
                            artifacts, log,
                            "字幕自动精简返回的条数或时间轴不一致，已保留原译文继续渲染。",
                        )
                        repaired_items = translated_items

                    changed_count = sum(
                        1
                        for before, after in zip(translated_items, repaired_items)
                        if before.translated_text != after.translated_text
                    )
                    if changed_count:
                        translated_items = repaired_items
                        log(
                            f"字幕自动精简生成 {changed_count} 条新译文；"
                            "成片验证通过后才会保存，原译文会保留。"
                        )
                    try:
                        cue_font_sizes, layout_issues = _plan_cue_font_sizes(
                            translated_items,
                            video,
                            font_size=options.font_size,
                            outline=options.subtitle_outline,
                            shadow=options.subtitle_shadow,
                            smart_layout=options.smart_subtitle_layout,
                            line_width=render_line_width,
                            display_limit=render_line_display_limit,
                        )
                    except CancellationRequested:
                        raise
                    except Exception as exc:
                        cue_font_sizes = {}
                        _record_render_layout_warning(
                            artifacts, log,
                            f"字幕排版复查未完成，保留完整译文继续渲染：{exc}",
                        )
                if layout_issues:
                    examples = ", ".join(
                        format_srt_timestamp(start) for _, start, _ in layout_issues[:5]
                    )
                    _record_render_layout_warning(
                        artifacts, log,
                        f"排版自修复后仍有 {len(layout_issues)} 条中文字幕可能超出画面"
                        f"（首批时间点：{examples}）；保留完整文字和时间轴继续渲染，"
                        "成片后可在字幕页复核。",
                    )
            log("正在生成本次渲染使用的字幕文件。")
            report_progress(0.24)
            regenerated_subtitle = work_dir / f"subtitle.next-{uuid.uuid4().hex[:8]}.srt"
            write_srt(
                regenerated_subtitle, translated_items,
                display_mode=options.subtitle_display_mode,
                smart_layout=options.smart_subtitle_layout,
                max_chars_per_line=render_line_width if options.subtitle_display_mode == "translated" else 18,
            )
            subtitle = regenerated_subtitle
        log("正在验证字幕文件。")
        report_progress(0.34)
        if not subtitle or not validate_srt(subtitle):
            raise RuntimeError("中文字幕检查点无效，请重新翻译。")
        ctx.check_cancelled()
        _emit(ctx, PipelineStage.RENDERING, 84, "字幕已准备，正在启动硬字幕渲染", log)
        rendered_candidate = work_dir / f"rendered.next-{uuid.uuid4().hex[:8]}.mp4"

        last_progress_event = 84

        def encoding_progress(fraction: float) -> None:
            nonlocal last_progress_event
            report_progress(0.84 + max(0.0, min(1.0, fraction)) * 0.14)
            percent = min(98, 84 + int(max(0.0, min(1.0, fraction)) * 14))
            if percent >= last_progress_event + 5 or fraction >= 1.0:
                last_progress_event = percent
                ctx.emit(PipelineStage.RENDERING, percent, f"硬字幕编码进度：{int(fraction * 100)}%")

        log("正在准备字幕样式和视频编码器。")
        report_progress(0.40)

        def render_log(message: str) -> None:
            if message.startswith("字幕样式提醒："):
                _record_render_layout_warning(artifacts, log, message)
            else:
                log(message)

        rendered = burn_subtitles(
            video, subtitle, rendered_candidate, font_name=options.font_name,
            font_size=options.font_size, primary_color=options.subtitle_color,
            outline_color=options.subtitle_outline_color, outline=options.subtitle_outline,
            shadow=options.subtitle_shadow, raised_margin=artifacts.used_ocr_subtitles,
            crf=options.render_crf, encoder=options.render_encoder, log=render_log,
            resource_profile=options.resource_profile,
            margin_ratio=options.subtitle_margin_ratio, cancel_check=ctx.is_cancelled,
            progress=encoding_progress,
            cue_font_sizes=cue_font_sizes,
            expected_cue_count=len(translated_items) if translated_items else None,
            display_mode=options.subtitle_display_mode,
            subtitle_preset=getattr(options, "subtitle_preset", "soft"),
            cue_source_lines=cue_source_line_counts(
                translated_items, options.subtitle_display_mode, options.smart_subtitle_layout,
            ) if translated_items else None,
        )
        ctx.check_cancelled()
        log("编码完成，正在验证成片文件。")
        report_progress(0.99)
        ctx.emit(PipelineStage.RENDERING, 99, "正在验证渲染结果")
        if not validate_media(rendered, "video"):
            raise RuntimeError("新成片验证失败，已保留上一版成片。")
    except BaseException:
        if rendered_candidate is not None:
            rendered_candidate.unlink(missing_ok=True)
        if regenerated_subtitle is not None:
            regenerated_subtitle.unlink(missing_ok=True)
        raise
    finally:
        if regenerated_subtitle is not None:
            regenerated_subtitle.with_suffix(".ass").unlink(missing_ok=True)
    log("成片验证通过，正在提交新的渲染结果。")
    translated_target: Path | None = None
    subtitle_target: Path | None = None
    rendered_target: Path | None = None
    try:
        if changed_count and translated_json is not None:
            translated_target = _versioned_target(translated_json.parent, "segments.translated", ".json")
            save_segments(translated_target, translated_items)
        if regenerated_subtitle is not None:
            subtitle_target = _versioned_target(work_dir, "zh", ".srt")
            regenerated_subtitle.replace(subtitle_target)
        rendered_target = commit_rendered_candidate(rendered, work_dir)
    except BaseException:
        if translated_target is not None:
            translated_target.unlink(missing_ok=True)
        if subtitle_target is not None:
            subtitle_target.unlink(missing_ok=True)
        if rendered_target is not None:
            rendered_target.unlink(missing_ok=True)
        rendered.unlink(missing_ok=True)
        raise
    if translated_target is not None:
        artifacts.translated_segments = str(translated_target)
    if subtitle_target is not None:
        artifacts.translated_srt = str(subtitle_target)
    artifacts.rendered_video = str(rendered_target)
    return artifacts


def _merge_tags(*groups: list[str]) -> list[str]:
    output: list[str] = []
    for group in groups:
        for raw in group:
            tag = raw.strip().lstrip("#")[:16]
            if tag and tag.lower() not in {item.lower() for item in output}:
                output.append(tag)
            if len(output) >= 10:
                return output
    return output


def run_publish(options: Any, work_dir: Path, artifacts: WorkflowArtifacts, ctx: PipelineContext, log: Callable[[str], None], progress: Callable[[float], None] | None = None) -> WorkflowArtifacts:
    del progress
    if not options.publish_to_bilibili:
        log("发布辅助未启用，阶段已跳过。")
        return artifacts
    rendered = artifacts.path("rendered_video")
    if not rendered or not validate_media(rendered, "video"):
        raise RuntimeError("成片检查点无效，请先重新渲染。")
    metadata: dict[str, Any] = {}
    metadata_path = artifacts.path("publish_metadata")
    if metadata_path and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ctx.check_cancelled(); _emit(ctx, PipelineStage.PUBLISHING, 96, "正在打开投稿辅助", log)
    source_url = artifacts.webpage_url or (artifacts.source if artifacts.source.startswith(("http://", "https://")) else "")
    assist_publish(
        rendered, title=str(metadata.get("title") or options.title or artifacts.title or rendered.stem),
        description=ensure_source_link(options.description or "已获授权转载/本地化。", source_url, include_source_link=options.include_source_link_in_description),
        tags=_merge_tags([str(tag) for tag in metadata.get("tags") or []], options.tags or []),
        cover_path=artifacts.path("thumbnail_path"), profile_dir=options.bilibili_profile_dir,
        browser=options.bilibili_browser, screenshot_path=work_dir / "bilibili-upload-page.png",
        wait_for_review=options.bilibili_wait_for_review, log=log,
    )
    return artifacts


def run_dub(options: Any, work_dir: Path, artifacts: WorkflowArtifacts, ctx: PipelineContext, log: Callable[[str], None], progress: Callable[[float], None] | None = None) -> WorkflowArtifacts:
    """Optional VoxCPM dub after render; the subtitled render is left untouched."""
    del progress
    if not getattr(options, "dub_enabled", False):
        log("配音未开启，阶段已跳过。")
        return artifacts
    from .dub import dub_video, llm_rewrite
    status = dependencies.voxcpm_status(options.voxcpm_python, options.voxcpm_model_dir)
    if not status["voxcpm_available"]:
        raise RuntimeError(f"配音不可用：{status['voxcpm_reason']}")
    rendered = artifacts.path("rendered_video")
    segments_path = artifacts.path("translated_segments")
    if not rendered or not segments_path:
        raise RuntimeError("配音需要先完成翻译和渲染。")
    ctx.check_cancelled(); _emit(ctx, PipelineStage.DUBBING, 90, "正在生成配音", log)
    output, report = dub_video(
        rendered, load_segments(segments_path), work_dir,
        python=options.voxcpm_python, model_dir=options.voxcpm_model_dir, mix_mode=options.dub_mix,
        log=log, cancel_check=ctx.is_cancelled, rewrite=llm_rewrite(options),
    )
    artifacts.dub_video, artifacts.dub_report = str(output) if output else None, str(report)
    return artifacts


STAGE_RUNNERS = {"acquire": run_acquire, "extract": run_extract, "translate": run_translate, "render": run_render, "dub": run_dub, "publish": run_publish}


def run_stage(name: str, options: Any, work_dir: Path, artifacts: WorkflowArtifacts, ctx: PipelineContext, log: Callable[[str], None], progress: Callable[[float], None] | None = None) -> WorkflowArtifacts:
    try:
        runner = STAGE_RUNNERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown workflow stage: {name}") from exc
    work_dir.mkdir(parents=True, exist_ok=True)
    result = runner(options, work_dir, artifacts, ctx, log, progress)
    if name in CORE_STAGES and not validate_stage(name, result):
        raise RuntimeError(
            f"{STAGE_LABELS[name]}产物校验失败，未保存为完成检查点；"
            "已完成的其他阶段与原有文件均保留。"
        )
    return result


def validate_media(path: Path, kind: str = "video") -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    ffprobe = dependencies.resolve_command("ffprobe")
    if ffprobe is None:
        return False
    selector = "v:0" if kind == "video" else "a:0"
    try:
        result = subprocess.run(
            [str(ffprobe), "-v", "error", "-select_streams", selector, "-show_entries", "stream=index", "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        return result.returncode == 0 and bool((json.loads(result.stdout).get("streams") or []))
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return False


def validate_segments(path: Path) -> bool:
    try:
        rows = load_segments(path)
        return bool(rows) and all(row.start >= 0 and row.end > row.start and bool(row.text.strip()) for row in rows)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def validate_srt(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return False
    return bool(re.search(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,.]\d{3}", text))


def validate_stage(stage: str, artifacts: WorkflowArtifacts) -> bool:
    if stage == "acquire":
        path = artifacts.path("raw_video"); return bool(path and validate_media(path, "video"))
    if stage == "extract":
        path = artifacts.path("source_segments"); srt = artifacts.path("source_srt")
        return bool(path and srt and validate_segments(path) and validate_srt(srt))
    if stage == "translate":
        path = artifacts.path("translated_segments"); srt = artifacts.path("translated_srt")
        return bool(path and srt and validate_segments(path) and validate_srt(srt))
    if stage == "render":
        path = artifacts.path("rendered_video"); return bool(path and validate_media(path, "video"))
    return True


def artifacts_to_video_job(options: Any, work_dir: Path, artifacts: WorkflowArtifacts) -> VideoJob:
    return VideoJob(
        job_id=work_dir.name, source=artifacts.source or options.source,
        source_kind=artifacts.source_kind or options.source_kind, work_dir=work_dir,
        title=artifacts.title, description=artifacts.description,
        raw_video=artifacts.path("raw_video"), audio=artifacts.path("audio"),
        source_subtitles=artifacts.path("source_srt"), translated_subtitles=artifacts.path("translated_srt"),
        rendered_video=artifacts.path("rendered_video"), license=artifacts.license,
        view_count=artifacts.view_count, webpage_url=artifacts.webpage_url,
        thumbnail_url=artifacts.thumbnail_url, thumbnail_path=artifacts.path("thumbnail_path"),
    )
