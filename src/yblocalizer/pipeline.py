from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import threading
from .cancellation import CancellationRequested, _cancellation_requested as _shared_cancellation_requested
from .download import download_with_ytdlp, import_local_video
from .media import extract_audio
from .models import VideoJob, save_segments
from .ocr_subtitle import extract_ocr_subtitles
from .publish_bili import assist_publish
from .publish_text import ensure_source_link
from .render import burn_subtitles
from .subtitle import write_srt
from .subtitle_merge import merge_audio_ocr_segments
from .transcribe import transcribe_audio
from .translate import correct_source_segments, generate_publish_metadata, save_publish_metadata, translate_segments_file
from .util import ensure_rights_confirmed, timestamp_id


LogFn = Callable[[str], None]

# 全局中断标志
_cancellation_requested = _shared_cancellation_requested


def request_cancellation() -> None:
    """请求中断当前运行的任务。"""
    _cancellation_requested.set()


def reset_cancellation() -> None:
    """重置中断标志。"""
    _cancellation_requested.clear()


def is_cancellation_requested() -> bool:
    """检查是否已请求中断。"""
    return _cancellation_requested.is_set()


def check_cancelled() -> None:
    """检查中断标志，如果已请求中断则抛出异常。"""
    if _cancellation_requested.is_set():
        raise CancellationError("任务已被用户中断")


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
    max_seconds: int | None = None
    subtitle_source: str = "auto"
    ocr_fallback_to_audio: bool = True
    whisper_model_size: str = "small"
    source_language: str | None = None
    beam_size: int = 5
    ocr_interval: float = 1.0
    ocr_crop_ratio: float = 0.30
    ocr_min_chars: int = 3
    subtitle_margin_ratio: float = 0.055
    render_crf: int = 20
    device: str = "cpu"
    compute_type: str = "int8"
    translator: str = "deepseek"
    target_lang: str = "zh-Hans"
    translate_model: str | None = None
    batch_size: int = 25
    smart_translation: bool = True
    smart_subtitle_layout: bool = True
    font_name: str = "Microsoft YaHei"
    font_size: int = 24
    subtitle_display_mode: str = "translated"
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
) -> PipelineResult:
    logger = log or (lambda message: print(message, flush=True))
    ensure_rights_confirmed(options.i_have_rights)
    work_dir = options.output_dir / timestamp_id()
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if options.subtitle_source not in {"audio", "ocr", "auto", "merged"}:
            raise RuntimeError(f"Unknown subtitle source: {options.subtitle_source}")
        logger("1/5 Preparing source video...")
        check_cancelled()
        if options.source_kind == "url":
            job = download_with_ytdlp(
                options.source,
                work_dir=work_dir,
                title=options.title,
                max_seconds=options.max_seconds,
                require_reuse_allowed=options.require_reuse_allowed,
                cookies_from_browser=options.cookies_from_browser,
                cookies_file=options.cookies_file,
                progress=progress,
            )
            if job.license:
                logger(f"YouTube license: {job.license}")
            if job.view_count is not None:
                logger(f"YouTube views: {job.view_count}")
        elif options.source_kind == "file":
            job = import_local_video(Path(options.source), work_dir=work_dir, title=options.title)
        else:
            raise RuntimeError(f"Unknown source kind: {options.source_kind}")
        if job.raw_video is None:
            raise RuntimeError("No raw video was prepared.")
        logger(f"Raw video: {job.raw_video}")
        if job.thumbnail_path:
            logger(f"YouTube cover: {job.thumbnail_path}")
        elif job.thumbnail_url:
            logger("YouTube cover URL was found, but the cover image could not be downloaded.")
        # 注意：不要给 whisper 传 initial_prompt！实测 initial_prompt 会导致
        # faster-whisper 段时间戳系统性提前（最多 10-15 秒）且长段粘连吞掉静音，
        # 使字幕在无语音的片段里提前出现。正确时间轴优先于专名提示。
        transcription_prompt = None

        if options.subtitle_source == "ocr":
            logger("2/5 Skipping audio extraction; OCR subtitle mode is enabled.")
        else:
            logger("2/5 Extracting audio...")
            check_cancelled()
            job.audio = extract_audio(job.raw_video, work_dir / "audio.wav")
            logger(f"Audio: {job.audio}")

        check_cancelled()
        source_segments = work_dir / "segments.source.json"
        source_srt = work_dir / "source.srt"
        used_ocr_subtitles = False
        source_segment_items = []
        if options.subtitle_source == "merged":
            logger("3/5 Merged subtitle mode: transcribing audio and reading on-screen text with OCR...")
            logger("首次使用会自动下载 Whisper 模型（约 500MB），需要联网，请耐心等待；后续会使用本地缓存。")
            if job.audio is None:
                raise RuntimeError("Audio was not extracted.")
            audio_segments_json = work_dir / "segments.audio.json"
            audio_srt = work_dir / "audio.srt"
            ocr_segments_json = work_dir / "segments.ocr.json"
            ocr_srt = work_dir / "ocr.srt"
            audio_segment_items = transcribe_audio(
                job.audio,
                segments_json=audio_segments_json,
                srt_path=audio_srt,
                model_size=options.whisper_model_size,
                language=options.source_language,
                device=options.device,
                compute_type=options.compute_type,
                initial_prompt=transcription_prompt,
                beam_size=options.beam_size,
                log=logger,
            )
            logger(f"Audio subtitle lines: {len(audio_segment_items)}")
            try:
                ocr_segment_items = extract_ocr_subtitles(
                    job.raw_video,
                    work_dir=work_dir,
                    segments_json=ocr_segments_json,
                    srt_path=ocr_srt,
                    interval_seconds=max(0.2, options.ocr_interval),
                    crop_bottom_ratio=max(0.05, min(1.0, options.ocr_crop_ratio)),
                    min_chars=max(1, options.ocr_min_chars),
                )
                logger(f"OCR subtitle lines: {len(ocr_segment_items)}")
            except Exception as exc:
                logger(f"OCR subtitle reading failed: {exc}")
                ocr_segment_items = []
            used_ocr_subtitles = bool(ocr_segment_items)
            if audio_segment_items and ocr_segment_items:
                # 用画面 OCR 时间戳作为锚点，自动校正音频字幕的整体时间偏移
                # 注意：仅校正「较小的系统性误差」（Whisper 常见 0.1~3s）。
                # 偏移过大说明素材本身音画不同步，跟随画面字幕反而会出错，跳过并提示。
                offset = _estimate_audio_offset(audio_segment_items, ocr_segment_items)
                if offset is not None and 0.12 <= abs(offset) <= 3.0:
                    _shift_segments(audio_segment_items, offset)
                    logger(f"Detected audio-subtitle time offset {offset:+.2f}s; auto-corrected using OCR anchors.")
                elif offset is not None and abs(offset) > 3.0:
                    logger(
                        f"Detected a large audio/OCR time mismatch ({offset:+.2f}s); "
                        "the source video itself may be out of sync. Skipping auto-correction; "
                        "use manual alignment in the subtitle page if needed."
                    )
                else:
                    logger("OCR anchors found; audio-subtitle timing is within tolerance, no shift needed.")
            source_segment_items = merge_audio_ocr_segments(audio_segment_items, ocr_segment_items)
            save_segments(source_segments, source_segment_items)
            write_srt(source_srt, source_segment_items, display_mode="source")
            logger(f"Merged subtitle lines: {len(source_segment_items)}")
        elif options.subtitle_source == "ocr":
            logger("3/5 Reading burned-in English subtitles from video frames with OCR...")
            try:
                source_segment_items = extract_ocr_subtitles(
                    job.raw_video,
                    work_dir=work_dir,
                    segments_json=source_segments,
                    srt_path=source_srt,
                    interval_seconds=max(0.2, options.ocr_interval),
                    crop_bottom_ratio=max(0.05, min(1.0, options.ocr_crop_ratio)),
                    min_chars=max(1, options.ocr_min_chars),
                )
                used_ocr_subtitles = True
                logger(f"OCR subtitle lines: {len(source_segment_items)}")
            except Exception as exc:
                if not options.ocr_fallback_to_audio:
                    raise
                logger(f"OCR subtitle reading failed: {exc}")
                logger("Falling back to audio transcription with faster-whisper.")
                check_cancelled()
                if job.audio is None:
                    job.audio = extract_audio(job.raw_video, work_dir / "audio.wav")
                    logger(f"Audio: {job.audio}")
                source_segment_items = transcribe_audio(
                    job.audio,
                    segments_json=source_segments,
                    srt_path=source_srt,
                    model_size=options.whisper_model_size,
                    language=options.source_language,
                    device=options.device,
                    compute_type=options.compute_type,
                    initial_prompt=transcription_prompt,
                    log=logger,
                )
        else:
            if options.subtitle_source == "auto":
                logger("3/5 Auto subtitle mode: trying audio transcription first...")
            else:
                logger("3/5 Transcribing audio with faster-whisper...")
            if job.audio is None:
                raise RuntimeError("Audio was not extracted.")
            source_segment_items = transcribe_audio(
                job.audio,
                segments_json=source_segments,
                srt_path=source_srt,
                model_size=options.whisper_model_size,
                language=options.source_language,
                device=options.device,
                compute_type=options.compute_type,
                initial_prompt=transcription_prompt,
                log=logger,
                )
            if not source_segment_items:
                if options.subtitle_source == "audio" and not options.ocr_fallback_to_audio:
                    logger("Audio transcription produced no speech segments, and OCR fallback is disabled.")
                else:
                    logger("Audio transcription produced no speech segments. Falling back to on-screen OCR.")
                    try:
                        source_segment_items = extract_ocr_subtitles(
                            job.raw_video,
                            work_dir=work_dir,
                            segments_json=source_segments,
                            srt_path=source_srt,
                            interval_seconds=max(0.2, options.ocr_interval),
                            crop_bottom_ratio=max(0.05, min(1.0, options.ocr_crop_ratio)),
                            min_chars=max(1, options.ocr_min_chars),
                        )
                        used_ocr_subtitles = True
                        logger(f"OCR subtitle lines: {len(source_segment_items)}")
                    except Exception as exc:
                        logger(f"OCR fallback also failed: {exc}")
        if not source_segment_items:
            raise RuntimeError(
                "No readable subtitle text was found. Audio transcription produced no speech segments, "
                "and OCR could not extract usable on-screen text. Try a clearer video/caption area, "
                "or add subtitles manually before publishing."
            )
        if options.smart_translation and options.translator.lower() in {"openai", "deepseek"}:
            logger("3/5 Reviewing source subtitles with the translation model before Chinese translation...")
            try:
                corrected_source_items = correct_source_segments(
                    source_segment_items,
                    provider=options.translator,
                    model=options.translate_model,
                    batch_size=options.batch_size,
                    source_title=options.title or job.title,
                    source_description=job.description,
                )
                if corrected_source_items:
                    source_segment_items = corrected_source_items
                    save_segments(source_segments, source_segment_items)
                    write_srt(source_srt, source_segment_items, display_mode="source")
                    logger("Source subtitle review finished; corrected source SRT will be used for translation.")
            except Exception as exc:
                logger(f"Source subtitle review failed; continuing with original source subtitles: {exc}")
        logger(f"Source SRT: {source_srt}")

        if options.smart_translation:
            logger(f"4/5 Translating subtitles with {options.translator} using context-aware mode...")
        else:
            logger(f"4/5 Translating subtitles with {options.translator}...")
        check_cancelled()
        translated_segments = work_dir / "segments.translated.json"
        translated_srt = work_dir / "zh.srt"
        translated_segment_items = translate_segments_file(
            source_segments,
            output_json=translated_segments,
            output_srt=translated_srt,
            provider=options.translator,
            target_lang=options.target_lang,
            model=options.translate_model,
            batch_size=options.batch_size,
            validate_language=True,
            display_mode=options.subtitle_display_mode,
            smart_translation=options.smart_translation,
            smart_layout=options.smart_subtitle_layout,
        )
        job.translated_subtitles = translated_srt
        logger(f"Chinese SRT: {translated_srt}")
        try:
            publish_metadata = generate_publish_metadata(
                source_title=options.title or job.title,
                source_description=job.description,
                source_segments=source_segment_items,
                translated_segments=translated_segment_items,
                provider=options.translator,
                target_lang=options.target_lang,
                model=options.translate_model,
            )
        except Exception as exc:
            logger(f"Smart publish metadata failed, using fallback metadata: {exc}")
            publish_metadata = generate_publish_metadata(
                source_title=options.title or job.title,
                source_description=job.description,
                source_segments=source_segment_items,
                translated_segments=translated_segment_items,
                provider="none",
                target_lang=options.target_lang,
                model=None,
            )
        metadata_path = save_publish_metadata(work_dir / "publish_metadata.json", publish_metadata)
        logger(f"Smart Bilibili title: {publish_metadata.title}")
        logger(f"Smart Bilibili tags: {', '.join(publish_metadata.tags)}")
        logger(f"Publish metadata: {metadata_path}")

        logger("5/5 Rendering hard subtitles with ffmpeg...")
        check_cancelled()
        if used_ocr_subtitles:
            logger("OCR mode: raising Chinese subtitles above the original English captions.")
        rendered = burn_subtitles(
            job.raw_video,
            translated_srt,
            work_dir / "rendered.mp4",
            font_name=options.font_name,
            font_size=options.font_size,
            primary_color=options.subtitle_color,
            outline_color=options.subtitle_outline_color,
            outline=options.subtitle_outline,
            shadow=options.subtitle_shadow,
            raised_margin=used_ocr_subtitles,
            crf=options.render_crf,
            margin_ratio=options.subtitle_margin_ratio,
        )
        job.rendered_video = rendered
        logger(f"Rendered video: {rendered}")

        if options.publish_to_bilibili:
            check_cancelled()
            logger("Opening Bilibili Creator Center for assisted publishing...")
            source_for_description = job.webpage_url or (job.source if job.source.startswith(("http://", "https://")) else "")
            publish_description = ensure_source_link(
                options.description or f"已获授权转载/本地化。",
                source_for_description,
                include_source_link=options.include_source_link_in_description,
            )
            publish_tags = _merge_tags(publish_metadata.tags, options.tags or [])
            assist_publish(
                rendered,
                title=publish_metadata.title or options.title or job.title or rendered.stem,
                description=publish_description,
                tags=publish_tags,
                cover_path=job.thumbnail_path,
                profile_dir=options.bilibili_profile_dir,
                browser=options.bilibili_browser,
                screenshot_path=work_dir / "bilibili-upload-page.png",
                wait_for_review=options.bilibili_wait_for_review,
                log=logger,
            )

        return PipelineResult(
            job=job,
            work_dir=work_dir,
            source_srt=source_srt,
            translated_srt=translated_srt,
            rendered_video=rendered,
        )
    except (CancellationError, CancellationRequested) as exc:
        logger("\n任务已被用户中断。")
        if isinstance(exc, CancellationError):
            raise
        raise CancellationError(str(exc)) from exc
    finally:
        # 无论成功还是中断，都重置标志
        reset_cancellation()


def _merge_tags(*groups: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for tag in group:
            cleaned = tag.strip().lstrip("#")
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(cleaned[:16])
            if len(output) >= 10:
                return output
    return output


def _build_transcription_prompt(title: str | None, description: str | None) -> str | None:
    parts = [
        "Transcribe accurately. Preserve proper nouns, names, brands, numbers, racing classes, and technical terms.",
    ]
    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description[:600]}")
    text = " ".join(part for part in parts if part)
    return text if text.strip() else None
