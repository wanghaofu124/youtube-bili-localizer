from __future__ import annotations

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from .media import extract_audio
from .download import download_with_ytdlp
from .pipeline import PipelineOptions, run_pipeline
from .publish_bili import assist_publish
from .publish_text import ensure_source_link
from .render import burn_subtitles
from .transcribe import transcribe_audio
from .translate import translate_segments_file
from .util import ensure_rights_confirmed, timestamp_id


def main() -> None:
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)
    if load_dotenv:
        load_dotenv(Path(".env"))
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="yblocalizer",
        description="Authorized video localization pipeline for Chinese subtitles and Bilibili assisted publishing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download an authorized URL with yt-dlp")
    download.add_argument("--url", required=True)
    download.add_argument("--title")
    download.add_argument("--output-dir", default="outputs")
    download.add_argument("--max-seconds", type=int, default=None, help="Download only the first N seconds")
    download.add_argument("--require-reuse-allowed", action="store_true", help="Require YouTube metadata to say reuse is allowed")
    download.add_argument("--cookies-from-browser", choices=["chrome", "edge", "firefox", "brave", "chromium"], help="Use browser cookies for YouTube bot/login checks")
    download.add_argument("--i-have-rights", action="store_true", help="Confirm you have the rights/license to process this video")
    download.set_defaults(func=cmd_download)

    transcribe = subparsers.add_parser("transcribe", help="Extract audio and transcribe it with faster-whisper")
    transcribe.add_argument("--video", required=True)
    transcribe.add_argument("--model-size", default="small")
    transcribe.add_argument("--language", default=None)
    transcribe.add_argument("--initial-prompt", default=None)
    transcribe.add_argument("--device", default="cpu")
    transcribe.add_argument("--compute-type", default="int8")
    transcribe.add_argument("--output-dir")
    transcribe.set_defaults(func=cmd_transcribe)

    translate = subparsers.add_parser("translate", help="Translate a segments JSON file and write SRT")
    translate.add_argument("--segments", required=True)
    translate.add_argument("--translator", default="none", choices=["none", "openai", "deepseek"])
    translate.add_argument("--target-lang", default="zh-Hans")
    translate.add_argument("--model", default=None, help="Translator model. Defaults depend on provider.")
    translate.add_argument("--batch-size", type=int, default=25)
    translate.add_argument("--no-smart-translation", action="store_true", help="Disable context-aware subtitle translation prompts")
    translate.add_argument("--no-smart-layout", action="store_true", help="Disable automatic subtitle line wrapping")
    translate.add_argument("--allow-untranslated", action="store_true", help="Allow translator=none for debugging")
    translate.add_argument("--output-dir")
    translate.set_defaults(func=cmd_translate)

    render = subparsers.add_parser("render", help="Burn SRT subtitles into a video")
    render.add_argument("--video", required=True)
    render.add_argument("--subtitle", required=True)
    render.add_argument("--output")
    render.add_argument("--font-name", default="Microsoft YaHei")
    render.add_argument("--font-size", type=int, default=18)
    render.add_argument("--subtitle-color", default="&H00FFFFFF")
    render.add_argument("--outline-color", default="&H00000000")
    render.add_argument("--outline", type=int, default=1)
    render.add_argument("--shadow", type=int, default=0)
    render.set_defaults(func=cmd_render)

    publish = subparsers.add_parser("publish", help="Assist publishing a prepared video through Bilibili Creator Center")
    publish.add_argument("--video", required=True)
    publish.add_argument("--title", required=True)
    publish.add_argument("--description", default="")
    publish.add_argument("--source-url", default="", help="Original source URL to include in the Bilibili description")
    publish.add_argument("--no-source-link", action="store_true", help="Do not append the original source URL to the description")
    publish.add_argument("--tags", default="", help="Comma-separated tags")
    publish.add_argument("--cover", default=None, help="Optional cover image to upload to Bilibili")
    publish.add_argument("--browser", default="chromium", choices=["chromium", "edge", "msedge"], help="Browser used for Bilibili publishing")
    publish.add_argument("--profile-dir", default=None)
    publish.add_argument("--headless", action="store_true")
    publish.add_argument(
        "--no-review-wait",
        action="store_true",
        help="Close the browser after saving the upload-page screenshot. Do not use this for real Bilibili uploads because closing the browser stops unfinished uploads.",
    )
    publish.set_defaults(func=cmd_publish)

    process = subparsers.add_parser("process", help="Run download/import, transcription, translation, and subtitle burn")
    source_group = process.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url")
    source_group.add_argument("--file")
    process.add_argument("--title")
    process.add_argument("--output-dir", default="outputs")
    process.add_argument("--max-seconds", type=int, default=None, help="Download only the first N seconds when source is a URL")
    process.add_argument("--require-reuse-allowed", action="store_true", help="Require YouTube metadata to say reuse is allowed")
    process.add_argument("--cookies-from-browser", choices=["chrome", "edge", "firefox", "brave", "chromium"], help="Use browser cookies for YouTube bot/login checks")
    process.add_argument("--i-have-rights", action="store_true", help="Confirm you have the rights/license to process this video")
    process.add_argument(
        "--subtitle-source",
        default="auto",
        choices=["auto", "audio", "ocr", "merged"],
        help="Use auto detection, audio transcription, OCR, or merged audio+OCR subtitles",
    )
    process.add_argument("--no-ocr-audio-fallback", action="store_true", help="Disable subtitle-source fallback between OCR and audio transcription")
    process.add_argument("--whisper-model-size", default="small")
    process.add_argument("--source-language", default=None)
    process.add_argument("--device", default="cpu")
    process.add_argument("--compute-type", default="int8")
    process.add_argument("--translator", default="none", choices=["none", "openai", "deepseek"])
    process.add_argument("--target-lang", default="zh-Hans")
    process.add_argument("--translate-model", default=None, help="Translator model. Defaults depend on provider.")
    process.add_argument("--batch-size", type=int, default=25)
    process.add_argument("--no-smart-translation", action="store_true", help="Disable context-aware subtitle translation prompts")
    process.add_argument("--no-smart-layout", action="store_true", help="Disable automatic subtitle line wrapping")
    process.add_argument("--font-name", default="Microsoft YaHei")
    process.add_argument("--font-size", type=int, default=18)
    process.add_argument("--subtitle-mode", default="translated", choices=["translated", "source", "bilingual-source-first", "bilingual-translation-first"])
    process.add_argument("--subtitle-color", default="&H00FFFFFF")
    process.add_argument("--outline-color", default="&H00000000")
    process.add_argument("--outline", type=int, default=1)
    process.add_argument("--shadow", type=int, default=0)
    process.add_argument("--publish", action="store_true")
    process.add_argument("--description", default="")
    process.add_argument("--no-source-link", action="store_true", help="Do not append the original source URL to the Bilibili description")
    process.add_argument("--tags", default="")
    process.add_argument("--bilibili-browser", default="chromium", choices=["chromium", "edge", "msedge"], help="Browser used for Bilibili publishing")
    process.add_argument(
        "--close-browser-after-fill",
        action="store_true",
        help="Close the Bilibili browser after filling the upload form. This stops any unfinished browser upload.",
    )
    process.set_defaults(func=cmd_process)

    return parser


def cmd_download(args: Namespace) -> None:
    ensure_rights_confirmed(args.i_have_rights)
    work_dir = Path(args.output_dir) / timestamp_id()
    job = download_with_ytdlp(
        args.url,
        work_dir=work_dir,
        title=args.title,
        max_seconds=args.max_seconds,
        require_reuse_allowed=args.require_reuse_allowed,
        cookies_from_browser=args.cookies_from_browser,
    )
    print(f"Downloaded: {job.raw_video}")


def cmd_transcribe(args: Namespace) -> None:
    video = Path(args.video)
    work_dir = Path(args.output_dir) if args.output_dir else video.resolve().parent
    audio = extract_audio(video, work_dir / "audio.wav")
    segments_json = work_dir / "segments.source.json"
    source_srt = work_dir / "source.srt"
    transcribe_audio(
        audio,
        segments_json=segments_json,
        srt_path=source_srt,
        model_size=args.model_size,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        initial_prompt=args.initial_prompt,
        log=lambda message: print(message, flush=True),
    )
    print(f"Segments: {segments_json}")
    print(f"Source SRT: {source_srt}")


def cmd_translate(args: Namespace) -> None:
    segments = Path(args.segments)
    work_dir = Path(args.output_dir) if args.output_dir else segments.resolve().parent
    output_json = work_dir / "segments.translated.json"
    output_srt = work_dir / "zh.srt"
    translate_segments_file(
        segments,
        output_json=output_json,
        output_srt=output_srt,
        provider=args.translator,
        target_lang=args.target_lang,
        model=args.model,
        batch_size=args.batch_size,
        validate_language=not args.allow_untranslated,
        smart_translation=not args.no_smart_translation,
        smart_layout=not args.no_smart_layout,
    )
    print(f"Translated segments: {output_json}")
    print(f"Translated SRT: {output_srt}")


def cmd_render(args: Namespace) -> None:
    video = Path(args.video)
    subtitle = Path(args.subtitle)
    output = Path(args.output) if args.output else video.resolve().parent / "rendered.mp4"
    rendered = burn_subtitles(
        video,
        subtitle,
        output,
        font_name=args.font_name,
        font_size=args.font_size,
        primary_color=args.subtitle_color,
        outline_color=args.outline_color,
        outline=args.outline,
        shadow=args.shadow,
    )
    print(f"Rendered video: {rendered}")


def cmd_publish(args: Namespace) -> None:
    tags = _split_tags(args.tags)
    description = ensure_source_link(args.description, args.source_url, include_source_link=not args.no_source_link)
    assist_publish(
        Path(args.video),
        title=args.title,
        description=description,
        tags=tags,
        cover_path=Path(args.cover) if args.cover else None,
        profile_dir=Path(args.profile_dir) if args.profile_dir else None,
        browser=args.browser,
        screenshot_path=Path(args.video).resolve().parent / "bilibili-upload-page.png",
        wait_for_review=not args.no_review_wait,
        headless=args.headless,
    )


def cmd_process(args: Namespace) -> None:
    source = args.url or args.file
    if not source:
        raise RuntimeError("No source was provided.")
    run_pipeline(
        PipelineOptions(
            source=source,
            source_kind="url" if args.url else "file",
            output_dir=Path(args.output_dir),
            title=args.title,
            description=args.description,
            tags=_split_tags(args.tags),
            i_have_rights=args.i_have_rights,
            require_reuse_allowed=args.require_reuse_allowed,
            cookies_from_browser=args.cookies_from_browser,
            max_seconds=args.max_seconds,
            subtitle_source=args.subtitle_source,
            ocr_fallback_to_audio=not args.no_ocr_audio_fallback,
            whisper_model_size=args.whisper_model_size,
            source_language=args.source_language,
            device=args.device,
            compute_type=args.compute_type,
            translator=args.translator,
            target_lang=args.target_lang,
            translate_model=args.translate_model,
            batch_size=args.batch_size,
            smart_translation=not args.no_smart_translation,
            smart_subtitle_layout=not args.no_smart_layout,
            font_name=args.font_name,
            font_size=args.font_size,
            subtitle_display_mode=args.subtitle_mode,
            subtitle_color=args.subtitle_color,
            subtitle_outline_color=args.outline_color,
            subtitle_outline=args.outline,
            subtitle_shadow=args.shadow,
            publish_to_bilibili=args.publish,
            include_source_link_in_description=not args.no_source_link,
            bilibili_browser=args.bilibili_browser,
            bilibili_wait_for_review=not args.close_browser_after_fill,
        )
    )


def _split_tags(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _default_description(source: str) -> str:
    return f"已获授权转载/本地化。来源：{source}"


if __name__ == "__main__":
    main()
