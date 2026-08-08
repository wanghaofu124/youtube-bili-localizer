"""Run and record a real, reproducible benchmark for the authorized demo.

This script deliberately wraps the existing pipeline instead of reimplementing
it. The wrappers only measure stages; yt-dlp, Whisper, the configured LLM, and
FFmpeg remain the production implementations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any, Callable
from unittest.mock import patch

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yblocalizer.pipeline import PipelineOptions, run_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the authorized 10-second localization demo.")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--compute-type", required=True)
    parser.add_argument("--whisper-model-size", default="small")
    parser.add_argument("--translator", choices=["openai", "deepseek"], default="deepseek")
    parser.add_argument("--translate-model", default=None)
    parser.add_argument("--subtitle-source", choices=["audio", "auto", "ocr", "merged"], default="audio")
    parser.add_argument("--input", type=Path, default=ROOT / "demo" / "authorized-demo-10s.mp4")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "benchmarks" / "runs")
    parser.add_argument(
        "--export-demo",
        action="store_true",
        help="Copy the reviewed source SRT, Chinese SRT, and rendered video to demo/artifacts after a successful run.",
    )
    return parser


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _video_duration(path: Path) -> float | None:
    output = _command_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    try:
        return round(float(output), 3) if output else None
    except ValueError:
        return None


def _gpu_info() -> list[str]:
    output = _command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return output.splitlines() if output else []


def _environment() -> dict[str, Any]:
    ffmpeg_output = _command_output(["ffmpeg", "-version"])
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "gpu": _gpu_info(),
        "ffmpeg": ffmpeg_output.splitlines()[0] if ffmpeg_output else None,
    }


def _output_summary(result: Any) -> dict[str, dict[str, Any]]:
    paths = {
        "source_srt": result.source_srt,
        "translated_srt": result.translated_srt,
        "rendered_video": result.rendered_video,
    }
    return {
        name: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "exists": path.exists()}
        for name, path in paths.items()
    }


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(ROOT / ".env")
    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"Demo input does not exist: {source}")
    key_name = "OPENAI_API_KEY" if args.translator == "openai" else "DEEPSEEK_API_KEY"
    if not os.getenv(key_name):
        raise SystemExit(f"{key_name} is required for a real benchmark. Configure it in .env.")

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    original_functions: dict[str, Callable[..., Any]] = {}
    import yblocalizer.pipeline as pipeline

    stage_functions = {
        "prepare_source": "import_local_video",
        "extract_audio": "extract_audio",
        "transcribe": "transcribe_audio",
        "review_source": "correct_source_segments",
        "translate": "translate_segments_file",
        "publish_metadata": "generate_publish_metadata",
        "render": "burn_subtitles",
    }
    for stage, name in stage_functions.items():
        original_functions[name] = getattr(pipeline, name)

    def measure(stage: str, func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            stage_started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                timings[stage] = round(time.perf_counter() - stage_started, 3)
        return wrapped

    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "environment": _environment(),
        "input": {"path": str(source.relative_to(ROOT)), "bytes": source.stat().st_size, "duration_seconds": _video_duration(source)},
        "configuration": {
            "device": args.device,
            "compute_type": args.compute_type,
            "whisper_model_size": args.whisper_model_size,
            "translator": args.translator,
            "translate_model": args.translate_model,
            "subtitle_source": args.subtitle_source,
        },
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = args.results_dir / f"{args.device}-{stamp}.json"
    try:
        with ExitStack() as stack:
            for stage, name in stage_functions.items():
                stack.enter_context(patch.object(pipeline, name, measure(stage, original_functions[name])))
            result = run_pipeline(
                PipelineOptions(
                    source=str(source), source_kind="file", output_dir=ROOT / "outputs" / "benchmarks",
                    title="Authorized 10-second benchmark demo", i_have_rights=True,
                    subtitle_source=args.subtitle_source, whisper_model_size=args.whisper_model_size,
                    device=args.device, compute_type=args.compute_type, translator=args.translator,
                    translate_model=args.translate_model, smart_translation=True,
                ),
                log=lambda message: print(message, flush=True),
            )
        report["status"] = "success"
        report["outputs"] = _output_summary(result)
        if args.export_demo:
            artifact_dir = ROOT / "demo" / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for source_path, name in (
                (result.source_srt, "source.srt"),
                (result.translated_srt, "zh.srt"),
                (result.rendered_video, "rendered.mp4"),
            ):
                shutil.copy2(source_path, artifact_dir / name)
            report["demo_artifacts"] = [str((artifact_dir / name).relative_to(ROOT)) for name in ("source.srt", "zh.srt", "rendered.mp4")]
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["stage_seconds"] = timings
        report["total_seconds"] = round(time.perf_counter() - started, 3)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Benchmark report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
