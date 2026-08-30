"""Checkpointed business stages shared by the desktop workbench and CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import subprocess
import uuid
from typing import Any, Callable

from . import dependencies
from .download import download_with_ytdlp, import_local_video
from .media import extract_audio
from .models import Segment, VideoJob, load_segments, save_segments
from .ocr_subtitle import extract_ocr_subtitles
from .performance import translation_worker_limit
from .publish_bili import assist_publish
from .publish_text import ensure_source_link
from .render import burn_subtitles
from .runtime import CancellationRequested, PipelineContext, PipelineStage
from .subtitle import read_srt_segments, write_srt
from .subtitle_merge import merge_audio_ocr_segments
from .transcribe import transcribe_audio
from .translate import (
    correct_source_segments,
    generate_publish_metadata,
    instruction_like_content_warning,
    save_publish_metadata,
    translate_segments_file,
)
from .util import ensure_rights_confirmed


WORKFLOW_VERSION = 2
WORKFLOW_STAGES = ("acquire", "extract", "translate", "render", "publish")
STAGE_LABELS = {
    "acquire": "获取素材", "extract": "字幕提取", "translate": "翻译",
    "render": "渲染", "publish": "发布辅助",
}
STAGE_PROGRESS = {"acquire": 8, "extract": 28, "translate": 62, "render": 84, "publish": 96}
CONFIG_INVALIDATION: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"source", "source_kind", "output_dir", "cookies_from_browser", "cookies_file", "youtube_po_token_mode", "youtube_proxy", "download_quality", "max_seconds", "require_reuse_allowed", "prefer_platform_subtitles"}), "acquire"),
    (frozenset({"subtitle_source", "ocr_fallback_to_audio", "whisper_model_size", "source_language", "beam_size", "device", "compute_type", "ocr_interval", "ocr_crop_ratio", "ocr_min_chars", "ocr_language"}), "extract"),
    (frozenset({"translator", "target_lang", "translate_model", "batch_size", "smart_translation", "smart_subtitle_layout"}), "translate"),
    (frozenset({"font_name", "font_size", "subtitle_display_mode", "subtitle_color", "subtitle_outline_color", "subtitle_outline", "subtitle_shadow", "subtitle_margin_ratio", "render_crf", "render_encoder"}), "render"),
    (frozenset({"title", "description", "tags", "publish_to_bilibili", "include_source_link", "bilibili_browser", "close_after_fill"}), "publish"),
)


@dataclass(slots=True)
class WorkflowArtifacts:
    original_video: str | None = None
    raw_video: str | None = None
    audio: str | None = None
    platform_subtitles: str | None = None
    source_segments: str | None = None
    source_srt: str | None = None
    translated_segments: str | None = None
    translated_srt: str | None = None
    publish_metadata: str | None = None
    rendered_video: str | None = None
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
        for name in ("original_video", "raw_video", "audio", "platform_subtitles", "source_segments", "source_srt", "translated_segments", "translated_srt", "publish_metadata", "rendered_video", "thumbnail_path"):
            path = self.path(name)
            output[name] = {
                "available": bool(path and path.is_file()),
                "name": path.name if path else None,
                "bytes": path.stat().st_size if path and path.is_file() else 0,
            }
        return output


def stage_state() -> dict[str, Any]:
    return {"status": "pending", "progress": 0, "error": None, "started_at": None, "finished_at": None, "config_fingerprint": None}


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
            row.update({"status": "stale", "progress": 0, "error": None, "started_at": None, "finished_at": None})
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


def run_acquire(options: Any, work_dir: Path, artifacts: WorkflowArtifacts, ctx: PipelineContext, log: Callable[[str], None], progress: Callable[[float], None] | None = None) -> WorkflowArtifacts:
    ensure_rights_confirmed(bool(options.i_have_rights))
    ctx.check_cancelled()
    _emit(ctx, PipelineStage.DOWNLOADING, 5, "正在获取素材", log)
    if options.source_kind == "url":
        job = download_with_ytdlp(
            options.source, work_dir=work_dir, title=options.title, max_seconds=options.max_seconds,
            require_reuse_allowed=options.require_reuse_allowed, cookies_from_browser=options.cookies_from_browser,
            cookies_file=options.cookies_file, po_token_mode=options.youtube_po_token_mode,
            proxy=options.youtube_proxy, download_quality=options.download_quality,
            resource_profile=options.resource_profile,
            source_language=options.source_language,
            prefer_platform_subtitles=getattr(options, "prefer_platform_subtitles", True),
            progress=progress, cancel_check=ctx.check_cancelled, log=log,
        )
    elif options.source_kind == "file":
        job = import_local_video(Path(options.source), work_dir=work_dir, title=options.title)
    else:
        raise RuntimeError(f"Unknown source kind: {options.source_kind}")
    if not job.raw_video or not job.raw_video.is_file():
        raise RuntimeError("No raw video was prepared.")
    artifacts.original_video = str(job.raw_video)
    artifacts.raw_video = str(job.raw_video)
    artifacts.platform_subtitles = str(job.source_subtitles) if job.source_subtitles else None
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
    artifacts.thumbnail_path = str(job.thumbnail_path) if job.thumbnail_path else None
    log(f"Raw video: {job.raw_video}")
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
    raw_video = artifacts.path("raw_video")
    if not raw_video or not validate_media(raw_video, "video"):
        raise RuntimeError("原视频检查点无效，请先重新获取素材。")
    source_segments, source_srt = work_dir / "segments.source.json", work_dir / "source.srt"
    audio: Path | None = artifacts.path("audio")
    used_ocr = False
    items: list[Any] = []
    mode = options.subtitle_source
    platform_items: list[Any] = []
    platform_path = artifacts.path("platform_subtitles")
    if getattr(options, "prefer_platform_subtitles", True) and mode in {"audio", "auto", "merged"} and platform_path and platform_path.is_file():
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
        if not audio or not audio.is_file():
            _emit(ctx, PipelineStage.AUDIO, 16, "正在提取音频", log)
            audio = _extract_audio(raw_video, work_dir / "audio.wav", ctx)
            artifacts.audio = str(audio)
        return audio

    def transcribe(target_json: Path, target_srt: Path) -> list[Any]:
        if platform_items:
            copied = [Segment(item.start, item.end, item.text) for item in platform_items]
            save_segments(target_json, copied)
            write_srt(target_srt, copied, display_mode="source")
            return copied
        source_audio = ensure_audio()
        return transcribe_audio(
            source_audio, segments_json=target_json, srt_path=target_srt,
            model_size=options.whisper_model_size, language=options.source_language,
            device=options.device, compute_type=options.compute_type, initial_prompt=None,
            beam_size=options.beam_size, log=log, cancel_check=ctx.check_cancelled,
            resource_profile=options.resource_profile,
        )

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
        raise RuntimeError("没有找到可用字幕。请检查语音、字幕来源或 OCR 截取区域。")
    artifacts.source_segments, artifacts.source_srt = str(source_segments), str(source_srt)
    artifacts.used_ocr_subtitles = used_ocr
    return artifacts


def run_translate(options: Any, work_dir: Path, artifacts: WorkflowArtifacts, ctx: PipelineContext, log: Callable[[str], None], progress: Callable[[float], None] | None = None) -> WorkflowArtifacts:
    source_segments = artifacts.path("source_segments")
    if not source_segments or not validate_segments(source_segments):
        raise RuntimeError("源字幕检查点无效，请先重新提取字幕。")
    source_items = load_segments(source_segments)
    content_warning = instruction_like_content_warning(source_items)
    artifacts.content_warnings = [content_warning] if content_warning else []
    if content_warning:
        log(f"人工复核提醒：{content_warning}")
    source_srt = artifacts.path("source_srt") or work_dir / "source.srt"
    if options.smart_translation and options.translator.lower() in {"openai", "deepseek"}:
        try:
            corrected = correct_source_segments(
                source_items, provider=options.translator, model=options.translate_model,
                batch_size=options.batch_size, source_title=options.title or artifacts.title,
                source_description=artifacts.description,
                cancel_check=ctx.check_cancelled,
            )
            if corrected:
                source_items = corrected; save_segments(source_segments, source_items)
                write_srt(source_srt, source_items, display_mode="source")
        except Exception as exc:
            log(f"Source subtitle review failed; continuing: {exc}")
    ctx.check_cancelled(); _emit(ctx, PipelineStage.TRANSLATING, 62, "正在翻译字幕", log)
    translated_json, translated_srt = work_dir / "segments.translated.json", work_dir / "zh.srt"
    translated_items = translate_segments_file(
        source_segments, output_json=translated_json, output_srt=translated_srt,
        provider=options.translator, target_lang=options.target_lang, model=options.translate_model,
        batch_size=options.batch_size, validate_language=True,
        display_mode=options.subtitle_display_mode, smart_translation=options.smart_translation,
        smart_layout=options.smart_subtitle_layout,
        checkpoint_path=work_dir / "translation_checkpoint.json",
        cancel_check=ctx.check_cancelled,
        max_workers=translation_worker_limit(options.resource_profile),
        progress=progress,
    )
    ctx.check_cancelled()
    # Publishing copy is not part of subtitle translation.  Avoid a second LLM
    # round trip unless the user explicitly enabled the publish stage.
    metadata_provider = options.translator if options.publish_to_bilibili else "none"
    try:
        metadata = generate_publish_metadata(
            source_title=options.title or artifacts.title, source_description=artifacts.description,
            source_segments=source_items, translated_segments=translated_items,
            provider=metadata_provider, target_lang=options.target_lang, model=options.translate_model,
        )
        ctx.check_cancelled()
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


def run_render(options: Any, work_dir: Path, artifacts: WorkflowArtifacts, ctx: PipelineContext, log: Callable[[str], None], progress: Callable[[float], None] | None = None) -> WorkflowArtifacts:
    del progress
    video, subtitle = artifacts.path("raw_video"), artifacts.path("translated_srt")
    if not video or not validate_media(video, "video"):
        raise RuntimeError("原视频检查点无效，请重新获取素材。")
    if not subtitle or not validate_srt(subtitle):
        raise RuntimeError("中文字幕检查点无效，请重新翻译。")
    ctx.check_cancelled(); _emit(ctx, PipelineStage.RENDERING, 84, "正在渲染硬字幕成片", log)
    rendered_target = work_dir / "rendered.mp4"
    rendered_candidate = work_dir / f"rendered.next-{uuid.uuid4().hex[:8]}.mp4"
    rendered = burn_subtitles(
        video, subtitle, rendered_candidate, font_name=options.font_name,
        font_size=options.font_size, primary_color=options.subtitle_color,
        outline_color=options.subtitle_outline_color, outline=options.subtitle_outline,
        shadow=options.subtitle_shadow, raised_margin=artifacts.used_ocr_subtitles,
        crf=options.render_crf, encoder=options.render_encoder, log=log,
        resource_profile=options.resource_profile,
        margin_ratio=options.subtitle_margin_ratio, cancel_check=ctx.is_cancelled,
    )
    ctx.check_cancelled()
    if not validate_media(rendered, "video"):
        raise RuntimeError("新成片验证失败，已保留上一版成片。")
    rendered.replace(rendered_target)
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


STAGE_RUNNERS = {"acquire": run_acquire, "extract": run_extract, "translate": run_translate, "render": run_render, "publish": run_publish}


def run_stage(name: str, options: Any, work_dir: Path, artifacts: WorkflowArtifacts, ctx: PipelineContext, log: Callable[[str], None], progress: Callable[[float], None] | None = None) -> WorkflowArtifacts:
    try:
        runner = STAGE_RUNNERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown workflow stage: {name}") from exc
    work_dir.mkdir(parents=True, exist_ok=True)
    return runner(options, work_dir, artifacts, ctx, log, progress)


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
            capture_output=True, text=True, timeout=15,
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
