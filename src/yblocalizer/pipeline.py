from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from .models import VideoJob
from .util import allocate_job_folder_name, ensure_rights_confirmed
from .runtime import CancellationRequested, PipelineContext, PipelineStage
from .cancellation import _cancellation_requested as _deprecated_global_cancellation


LogFn = Callable[[str], None]

_legacy_context = PipelineContext()


def request_cancellation() -> None:
    """Deprecated adapter for the maintenance-only Tk GUI."""
    _legacy_context.cancellation.cancel()
    _deprecated_global_cancellation.set()


def reset_cancellation() -> None:
    """重置中断标志。"""
    _legacy_context.cancellation.reset()
    _deprecated_global_cancellation.clear()


def is_cancellation_requested() -> bool:
    """检查是否已请求中断。"""
    return _legacy_context.cancellation.cancelled


def check_cancelled() -> None:
    """检查中断标志，如果已请求中断则抛出异常。"""
    try:
        _legacy_context.check_cancelled()
    except CancellationRequested as exc:
        raise CancellationError(str(exc)) from exc


class CancellationError(Exception):
    """任务被用户中断时抛出的异常。"""
    pass


@dataclass(slots=True)
class PipelineOptions:
    source: str
    source_kind: str
    output_dir: Path = Path("outputs")
    title: str | None = None
    description: str = ""
    tags: list[str] | None = None
    i_have_rights: bool = False
    require_reuse_allowed: bool = False
    cookies_from_browser: str | None = None
    cookies_file: str | None = None
    youtube_po_token_mode: str = "auto"
    youtube_proxy: str | None = None
    download_quality: str = "1080p"
    quality_proposal: dict | None = None
    resource_profile: str = "balanced"
    max_seconds: int | None = None
    subtitle_source: str = "audio"
    prefer_platform_subtitles: bool = True
    ocr_fallback_to_audio: bool = True
    whisper_model_size: str = "small"
    source_language: str | None = None
    beam_size: int = 5
    asr_hint: str = ""
    asr_vad_mode: str = "standard"
    ocr_interval: float = 0.5
    ocr_crop_ratio: float = 0.30
    ocr_min_chars: int = 3
    ocr_language: str = "eng"
    audio_channel: str = "mix"
    subtitle_margin_ratio: float = 0.055
    render_crf: int = 20
    render_encoder: str = "auto"
    device: str = "cpu"
    compute_type: str = "int8"
    translator: str = "deepseek"
    target_lang: str = "zh-Hans"
    translate_model: str | None = None
    deepseek_thinking: str = "disabled"
    batch_size: int = 40
    smart_translation: bool = True
    smart_subtitle_layout: bool = True
    font_name: str = "Microsoft YaHei"
    font_size: int = 24
    subtitle_display_mode: str = "translated"
    subtitle_preset: str = "soft"
    subtitle_color: str = "&H00FFFFFF"
    subtitle_outline_color: str = "&H00000000"
    subtitle_outline: int = 1
    subtitle_shadow: int = 0
    subtitle_margin_v: int = 24
    publish_to_bilibili: bool = False
    include_source_link_in_description: bool = True
    bilibili_browser: str = "chromium"
    bilibili_profile_dir: Path | None = None
    bilibili_wait_for_review: bool = True
    dub_enabled: bool = False
    dub_mix: str = "duck"
    voxcpm_python: str = ""
    voxcpm_model_dir: str = ""


@dataclass(slots=True)
class PipelineResult:
    job: VideoJob
    work_dir: Path
    source_srt: Path
    translated_srt: Path
    rendered_video: Path


def _estimate_audio_offset(audio_segments: list, ocr_segments: list) -> float | None:
    """Estimate the global audio-subtitle time offset using OCR anchors.

    OCR timestamps come from video frames (accurate), Whisper timestamps may
    drift. Text-match each OCR cue to the closest audio cue, keep only
    high-confidence matches, then use the median of the inlier time
    differences as the correction offset.
    """
    if not audio_segments or not ocr_segments:
        return None
    from difflib import SequenceMatcher

    def clean(text: str) -> str:
        import re
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

    diffs: list[float] = []
    for ocr in ocr_segments:
        ocr_text = clean(ocr.text)
        if len(ocr_text) < 6:
            continue
        best = None
        best_score = 0.0
        for audio in audio_segments:
            score = SequenceMatcher(None, ocr_text, clean(audio.text)).ratio()
            if score > best_score:
                best_score = score
                best = audio
        if best is not None and best_score >= 0.75:
            diffs.append(best.start - ocr.start)
    if len(diffs) < 2:
        return None
    diffs.sort()
    median = diffs[len(diffs) // 2]
    # 剔除与中位数相差过大的离群匹配（OCR 残片误配），只保留一致的核心样本
    core = [d for d in diffs if abs(d - median) <= 1.5]
    if len(core) < 2:
        return None
    core.sort()
    return core[len(core) // 2]


def _shift_segments(segments: list, offset: float) -> None:
    """Shift all segment times by ``offset`` seconds (in place)."""
    for segment in segments:
        segment.start = max(0.0, round(segment.start - offset, 2))
        segment.end = max(0.1, round(segment.end - offset, 2))


def run_pipeline(
    options: PipelineOptions,
    log: LogFn | None = None,
    progress: Callable[[float], None] | None = None,
    context: PipelineContext | None = None,
) -> PipelineResult:
    """Run the CLI-compatible full flow through the checkpointable stages.

    The desktop calls the same stage functions individually.  Keeping this
    small scheduler here preserves the original one-command CLI without
    maintaining a second implementation of the media workflow.
    """
    from .workflow import CORE_STAGES, WorkflowArtifacts, artifacts_to_video_job, run_stage

    ctx = context or _legacy_context
    logger = log or (lambda message: print(message, flush=True))
    ensure_rights_confirmed(options.i_have_rights)
    work_dir = options.output_dir / allocate_job_folder_name(
        options.output_dir,
        title=getattr(options, "title", None),
        source_url=options.source if getattr(options, "source_kind", "") == "url" else None,
        source_name=Path(options.source).name if options.source and getattr(options, "source_kind", "") != "url" else None,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts = WorkflowArtifacts()
    if options.source_kind == "url" and not options.quality_proposal:
        # Legacy/Tk callers share this scheduler; never let them bypass the
        # server-side proposal gate by calling the downloader directly.
        from .download import get_video_metadata
        metadata = get_video_metadata(
            options.source, options.cookies_from_browser, options.cookies_file,
            proxy=options.youtube_proxy, po_token_mode=options.youtube_po_token_mode,
            download_quality=options.download_quality,
        )
        proposal = metadata.quality_proposal or {}
        if not proposal.get("width") or not proposal.get("height"):
            raise RuntimeError("无法确定服务器拟下载的视频流，请重新准备素材。")
        if proposal.get("requires_consent"):
            raise RuntimeError("当前下载画质低于档位上限，需要先明确确认画质方案。")
        options.quality_proposal = proposal
    stages = [*CORE_STAGES, "dub"]
    if options.publish_to_bilibili:
        stages.append("publish")
    try:
        for stage in stages:
            artifacts = run_stage(stage, options, work_dir, artifacts, ctx, logger, progress)
        ctx.emit(PipelineStage.COMPLETED, 100, "处理完成")
        job = artifacts_to_video_job(options, work_dir, artifacts)
        source_srt = artifacts.path("source_srt")
        translated_srt = artifacts.path("translated_srt")
        rendered_video = artifacts.path("rendered_video")
        if not source_srt or not translated_srt or not rendered_video:
            raise RuntimeError("Pipeline completed without all required artifacts.")
        return PipelineResult(job, work_dir, source_srt, translated_srt, rendered_video)
    except (CancellationError, CancellationRequested) as exc:
        logger("\n任务已被用户中断。")
        if isinstance(exc, CancellationError):
            raise
        raise CancellationError(str(exc)) from exc
    finally:
        if context is None:
            reset_cancellation()
