"""Local HTTP bridge used by the React workbench and packaged EXE.

The bridge owns filesystem access. The browser receives only local API routes
and never sees API keys, cookies, or arbitrary absolute paths.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import parse_qs, unquote, urlparse

from dotenv import load_dotenv

# This must happen before importing modules which may read the template store.
# A packaged GUI can be launched with a protected system directory as its
# current working directory, so relative "data" paths are never safe here.
_DEFAULT_USER_DATA_ROOT = Path(os.environ.get("APPDATA") or Path.home()) / "YouTubeBiliLocalizer"
os.environ.setdefault("YBLOCALIZER_DATA_DIR", str(_DEFAULT_USER_DATA_ROOT))
_DEFAULT_USER_DATA_ROOT.mkdir(parents=True, exist_ok=True)
load_dotenv(_DEFAULT_USER_DATA_ROOT / ".env")

from .download import get_video_metadata
from .models import Segment, load_segments, save_segments
from .pipeline import (
    CancellationError,
    PipelineOptions,
    _estimate_audio_offset,
    _shift_segments,
    is_cancellation_requested,
    request_cancellation,
    run_pipeline,
)
from .publish_bili import assist_publish, open_upload_page
from .publish_text import build_bilibili_description, delete_custom_template, get_all_templates, save_custom_template
from .render import burn_subtitles
from .storage import delete_paths, format_bytes, scan_outputs
from .subtitle import write_srt
from .util import run as run_command
from . import db as job_db


def _runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _asset_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return _runtime_root()


PROJECT_ROOT = _runtime_root()
ASSET_ROOT = _asset_root()
# 用户数据目录（独立于 EXE 安装目录）：重建/升级 EXE 不会删除用户数据
USER_DATA_ROOT = _DEFAULT_USER_DATA_ROOT
DEMO_VIDEO = ASSET_ROOT / "demo" / "authorized-demo-10s.mp4"
OUTPUT_ROOT = USER_DATA_ROOT / "outputs" / "workbench_demo"
UPLOAD_ROOT = USER_DATA_ROOT / "outputs" / "workbench_uploads"
MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
COLOR_MAP = {
    "白色": "&H00FFFFFF", "黄色": "&H0000FFFF", "青色": "&H00FFFF00",
    "绿色": "&H0000FF00", "黑色": "&H00000000", "灰色": "&H00808080", "蓝色": "&H00FF0000",
}


def _safe_options(value: Any) -> dict[str, Any]:
    """Normalize the workbench form into PipelineOptions-compatible values."""
    raw = value if isinstance(value, dict) else {}
    translator = str(raw.get("translator", "deepseek")).lower()
    if translator not in {"deepseek", "openai", "none"}:
        raise ValueError("translator must be deepseek, openai, or none.")
    subtitle_source = str(raw.get("subtitle_source", "merged"))
    if subtitle_source not in {"auto", "audio", "ocr", "merged"}:
        raise ValueError("Unknown subtitle source.")
    display_mode = str(raw.get("subtitle_display_mode", "translated"))
    if display_mode not in {"translated", "bilingual-source-first", "bilingual-translation-first"}:
        raise ValueError("Unknown subtitle display mode.")
    try:
        font_size = max(8, min(96, int(raw.get("font_size", 24))))
        max_seconds = raw.get("max_seconds")
        max_seconds = None if max_seconds in {None, "", 0, "0"} else max(1, int(max_seconds))
    except (TypeError, ValueError) as exc:
        raise ValueError("Font size and URL duration must be valid numbers.") from exc
    effect = str(raw.get("subtitle_effect", "描边"))
    output_dir = str(raw.get("output_dir") or OUTPUT_ROOT)
    cookies_file = str(raw.get("cookies_file", "")).strip() or os.getenv("YBLOCALIZER_COOKIES_FILE", "").strip() or None
    cookies_from_browser = str(raw.get("cookies_from_browser", "")).strip() or None
    if cookies_file:
        # 已配置 cookies.txt 时忽略浏览器来源，避免 UI 状态混乱导致读取失败
        cookies_from_browser = None
    return {
        "title": str(raw.get("title", "")).strip() or None,
        "require_reuse_allowed": bool(raw.get("require_reuse_allowed", False)),
        "cookies_from_browser": cookies_from_browser,
        "cookies_file": cookies_file,
        "max_seconds": max_seconds,
        "subtitle_source": subtitle_source,
        "whisper_model_size": str(raw.get("whisper_model_size", "small")),
        "source_language": str(raw.get("source_language", "")).strip() or None,
        "beam_size": max(1, min(10, int(raw.get("beam_size", 5)))),
        "ocr_interval": max(0.2, float(raw.get("ocr_interval", 1.0))),
        "ocr_crop_ratio": max(0.05, min(1.0, float(raw.get("ocr_crop_ratio", 0.30)))),
        "ocr_min_chars": max(1, int(raw.get("ocr_min_chars", 3))),
        "subtitle_margin_ratio": max(0.01, min(0.4, float(raw.get("subtitle_margin_ratio", 0.055)))),
        "render_crf": max(14, min(32, int(raw.get("render_crf", 20)))),
        "translator": translator,
        "target_lang": str(raw.get("target_lang", "zh-Hans")).strip() or "zh-Hans",
        "translate_model": str(raw.get("translate_model", "")).strip() or None,
        "smart_translation": bool(raw.get("smart_translation", True)),
        "smart_subtitle_layout": bool(raw.get("smart_subtitle_layout", True)),
        "font_name": str(raw.get("font_name", "Microsoft YaHei")).strip() or "Microsoft YaHei",
        "font_size": font_size,
        "subtitle_display_mode": display_mode,
        "subtitle_color": COLOR_MAP.get(str(raw.get("subtitle_color", "白色")), "&H00FFFFFF"),
        "subtitle_outline_color": COLOR_MAP.get(str(raw.get("subtitle_outline_color", "黑色")), "&H00000000"),
        "subtitle_outline": 0 if effect in {"阴影", "无"} else 1,
        "subtitle_shadow": 1 if effect in {"阴影", "描边+阴影"} else 0,
        "output_dir": output_dir,
        "description": str(raw.get("description", "")).strip(),
        "tags": [str(item).strip() for item in raw.get("tags", []) if str(item).strip()] if isinstance(raw.get("tags", []), list) else [],
        "publish_to_bilibili": bool(raw.get("publish_to_bilibili", False)),
        "include_source_link": bool(raw.get("include_source_link", True)),
        "bilibili_browser": "msedge" if str(raw.get("bilibili_browser", "chromium")).lower() in {"edge", "msedge"} else "chromium",
        "close_after_fill": bool(raw.get("close_after_fill", False)),
    }


def _resolve_output_root(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _public_options(options: dict[str, Any]) -> dict[str, Any]:
    return dict(options)


def _cookies_file_has_login(path: Path) -> bool:
    """cookies.txt 是否包含主登录凭据（SID/HSID/LOGIN_INFO）。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "\tSID\t" in text or "\tHSID\t" in text or "LOGIN_INFO" in text


def _upsert_env(path: Path, updates: dict[str, str]) -> None:
    """Upsert .env lines by key. Handles UTF-8 BOM written by other tools."""
    existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    normalized = [line.lstrip("\ufeff") for line in existing.splitlines()]
    remaining = [line for line in normalized if line.partition("=")[0].strip() not in updates]
    remaining.extend(f"{key}={value}" for key, value in updates.items())
    path.write_text("\n".join(remaining).rstrip() + "\n", encoding="utf-8")


def _normalize_cut_ranges(value: Any, duration: float) -> list[tuple[float, float]]:
    """Validate and merge cut ranges; clip to [0, duration]."""
    if not isinstance(value, list):
        raise RuntimeError("cut_ranges must be a list of [start, end] pairs.")
    raw: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RuntimeError("Each cut range must be [start, end].")
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            raise RuntimeError("Cut range values must be numbers (seconds).") from None
        if start < 0 or end > duration + 0.5:
            raise RuntimeError(f"Cut range {start}-{end} is outside the video (0-{duration:.1f}s).")
        if end - start < 0.1:
            raise RuntimeError(f"Cut range {start}-{end} is too short.")
        raw.append((start, end))
    raw.sort(key=lambda pair: pair[0])
    merged: list[tuple[float, float]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1] + 0.01:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _remap_segments_after_cut(segments: list[Segment], ranges: list[tuple[float, float]]) -> list[Segment]:
    """Shift subtitle times after cutting ranges out; drop cues that overlap a cut."""
    kept: list[Segment] = []
    for segment in segments:
        if any(a < segment.end - 1e-6 and segment.start + 1e-6 < b for a, b in ranges):
            continue
        removed_before_start = sum(b - a for a, b in ranges if b <= segment.start + 1e-6)
        removed_before_end = sum(b - a for a, b in ranges if b <= segment.end + 1e-6)
        new_start = segment.start - removed_before_start
        new_end = segment.end - removed_before_end
        if new_end - new_start >= 0.1:
            segment.start = round(max(0.0, new_start), 2)
            segment.end = round(max(0.1, new_end), 2)
            kept.append(segment)
    return kept


def _remap_segments_for_keep(
    segments: list[Segment],
    keep_ranges: list[tuple[float, float]],
) -> list[Segment]:
    """Keep cues overlapping the kept ranges (clipped to the range) and shift them."""
    kept: list[Segment] = []
    prev_kept = 0.0
    for start, end in keep_ranges:
        for segment in segments:
            overlap_start = max(segment.start, start)
            overlap_end = min(segment.end, end)
            if overlap_end - overlap_start < 0.2:
                continue
            shift = start - prev_kept
            segment.start = round(max(0.0, overlap_start - shift), 2)
            segment.end = round(max(0.1, overlap_end - shift), 2)
            kept.append(segment)
        prev_kept += end - start
    return kept


def _remap_segments_for_reorder(
    segments: list[Segment],
    boundaries: list[float],
    order: list[int],
) -> list[Segment]:
    """Remap subtitle times after segment reordering.

    Segments that span a boundary are dropped; the rest are moved to their
    new position on the timeline.
    """
    ranges = list(zip(boundaries, boundaries[1:]))
    new_durations = [b - a for a, b in ranges]
    new_starts: list[float] = []
    cursor = 0.0
    for index in order:
        new_starts.append(cursor)
        cursor += new_durations[index]
    kept: list[Segment] = []
    for segment in segments:
        owner = None
        for index, (start, end) in enumerate(ranges):
            if start - 1e-6 <= segment.start and segment.end <= end + 1e-6:
                owner = index
                break
        if owner is None:
            continue
        new_start = new_starts[order.index(owner)] + (segment.start - ranges[owner][0])
        new_end = new_start + (segment.end - segment.start)
        segment.start = round(max(0.0, new_start), 2)
        segment.end = round(max(0.1, new_end), 2)
        kept.append(segment)
    return kept


def stage_from_log(message: str) -> tuple[str, int]:
    normalized = message.lower()
    if "1/5" in normalized:
        return "准备素材", 12
    if "2/5" in normalized:
        return "提取音频", 28
    if "3/5" in normalized or "transcrib" in normalized or "ocr" in normalized:
        return "转写字幕", 48
    if "4/5" in normalized or "translat" in normalized:
        return "翻译字幕", 72
    if "5/5" in normalized or "render" in normalized:
        return "渲染成片", 90
    return "处理中", 8


def public_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _safe_filename(value: str) -> str:
    name = Path(value).name
    cleaned = re.sub(r"[^\w.() -]", "_", name, flags=re.UNICODE).strip(" .")
    return cleaned or "video.mp4"


def _probe_media(path: Path) -> tuple[float | None, int | None, int | None]:
    """Best-effort local metadata; importing remains usable without ffprobe."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration:stream=width,height", "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        data = json.loads(completed.stdout)
        stream = (data.get("streams") or [{}])[0]
        duration = float((data.get("format") or {}).get("duration"))
        return round(duration, 2), int(stream.get("width") or 0) or None, int(stream.get("height") or 0) or None
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return None, None, None


@dataclass(slots=True)
class Material:
    id: str
    path: Path | None
    name: str
    duration_seconds: float | None
    width: int | None
    height: int | None
    authorized: bool
    is_demo: bool = False
    source_url: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
            "authorized": self.authorized,
            "is_demo": self.is_demo,
        }


@dataclass
class DemoJob:
    id: str
    material: Material
    device: str
    compute_type: str
    options: dict[str, Any] = field(default_factory=dict)
    output_root: Path = field(default_factory=lambda: OUTPUT_ROOT)
    status: str = "queued"
    stage: str = "等待开始"
    progress: int = 0
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    result: dict[str, str] | None = None
    work_dir: Path | None = None
    raw_video: Path | None = None
    rendered_video: Path | None = None
    edited_segments: Path | None = None
    duration_override: float | None = None
    started_at: float | None = None
    finished_at: float | None = None

    def add_log(self, message: str) -> None:
        self.logs.append(message)
        match = re.search(r"\[p=(\d+)\]", message)
        if match:
            self.progress = max(self.progress, min(100, int(match.group(1))))
            return
        stage, progress = stage_from_log(message)
        if stage != "处理中":
            self.stage = stage
        self.progress = max(self.progress, progress)

    def snapshot(self) -> dict[str, Any]:
        material = self.material.snapshot()
        if self.duration_override is not None:
            material["duration_seconds"] = self.duration_override
        return {
            "id": self.id, "status": self.status, "stage": self.stage, "progress": self.progress,
            "logs": self.logs, "error": self.error, "result": self.result,
            "device": self.device, "compute_type": self.compute_type, "material": material,
            "options": _public_options(self.options),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round((self.finished_at or time.time()) - self.started_at, 1) if self.started_at else 0,
        }


@dataclass
class PublishSession:
    """Live status of a manual Bilibili publish-assist run."""

    status: str = "idle"  # idle | running | waiting_review | finished | failed
    message: str = ""
    logs: list[str] = field(default_factory=list)
    error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "logs": self.logs,
            "error": self.error,
            "active": self.status in {"running", "waiting_review"},
        }


class WorkbenchJobs:
    """One media job at a time because pipeline cancellation is process-global."""

    def __init__(self) -> None:
        demo_duration, demo_width, demo_height = _probe_media(DEMO_VIDEO)
        self._lock = threading.Lock()
        self._jobs: dict[str, DemoJob] = {}
        self._materials: dict[str, Material] = {
            "demo": Material("demo", DEMO_VIDEO, "authorized-demo-10s.mp4", demo_duration or 10, demo_width or 1280, demo_height or 720, True, True)
        }
        self._running_id: str | None = None
        self._native_publish_files: dict[str, Path] = {}
        self._publish_session = PublishSession()

    def materials(self) -> list[dict[str, Any]]:
        with self._lock:
            return [material.snapshot() for material in self._materials.values()]

    def material_path(self, material_id: str) -> Path | None:
        with self._lock:
            material = self._materials.get(material_id)
            return material.path if material and material.path else None

    def add_upload(self, filename: str, source: BinaryIO, authorized: bool) -> Material:
        safe_name = _safe_filename(filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            raise ValueError("Supported video formats: mp4, mov, mkv, webm, avi, m4v.")
        if not authorized:
            raise ValueError("请先确认拥有处理该本地视频的授权，再导入。")
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_ROOT / f"{uuid.uuid4().hex[:10]}-{safe_name}"
        with target.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        duration, width, height = _probe_media(target)
        material = Material(uuid.uuid4().hex[:12], target, safe_name, duration, width, height, True)
        with self._lock:
            self._materials[material.id] = material
        return material

    def add_local_file(self, path: Path, authorized: bool) -> Material:
        """Register an already-existing local video by its original path (no copy)."""
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        suffix = resolved.suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            raise ValueError("Supported video formats: mp4, mov, mkv, webm, avi, m4v.")
        if not authorized:
            raise ValueError("请先确认拥有处理该本地视频的授权，再导入。")
        duration, width, height = _probe_media(resolved)
        material = Material(uuid.uuid4().hex[:12], resolved, resolved.name, duration, width, height, True)
        with self._lock:
            self._materials[material.id] = material
        return material

    def _persist(self, job: DemoJob) -> None:
        """Write job history to SQLite. Failures are logged, never fatal."""
        try:
            with self._lock:
                snapshot = {
                    "job_id": job.id,
                    "material_id": job.material.id,
                    "source_url": job.material.source_url,
                    "title": str(job.options.get("title") or "") or job.material.name,
                    "status": job.status,
                    "stage": job.stage,
                    "progress": job.progress,
                    "error": job.error,
                    "output_dir": str(job.work_dir) if job.work_dir else None,
                    "rendered_video": str(job.rendered_video) if job.rendered_video else None,
                    "device": job.device,
                    "compute_type": job.compute_type,
                    "options": job.options,
                    "created_at": job.started_at,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                }
            job_db.record_job(**snapshot)
            try:
                (USER_DATA_ROOT / "history_debug.log").write_text(
                    "{} persisted ({} -> {})\n".format(job.id[:8], job.status, snapshot["created_at"]),
                    encoding="utf-8", append=True,
                )
            except Exception:
                pass
        except Exception as exc:  # pragma: no cover - storage must never break tasks
            try:
                (USER_DATA_ROOT / "history_errors.log").write_text(
                    "{}: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), exc),
                    encoding="utf-8", append=True,
                )
            except Exception:
                pass

    def publish_session(self) -> PublishSession:
        with self._lock:
            return self._publish_session

    def begin_publish_session(self) -> PublishSession:
        with self._lock:
            session = self._publish_session
            session.status = "running"
            session.message = "正在打开 B 站投稿页…"
            session.logs = [session.message]
            session.error = None
            return session

    def _publish_log(self, message: str) -> None:
        with self._lock:
            session = self._publish_session
            session.logs.append(message)
            if session.status in {"idle", "finished"}:
                session.status = "running"
            lowered = message.lower()
            if "pre-submit" in lowered or "reached the pre-submit step" in lowered:
                session.status = "waiting_review"
                session.message = "已到人工提交前：请检查分区、声明、封面与简介，确认无误后手动投稿。"
            elif "already uploaded" in lowered or "upload complete" in lowered:
                session.status = "waiting_review"
                session.message = "视频已上传，等待你在页面中确认后手动投稿。"
            else:
                session.message = message

    def get(self, job_id: str) -> DemoJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def create(self, material_id: str, device: str, compute_type: str, options: dict[str, Any] | None = None) -> DemoJob:
        options = _safe_options(options)
        with self._lock:
            if self._running_id:
                raise RuntimeError("A task is already running. Wait for it to finish or cancel it first.")
            material = self._materials.get(material_id)
            if material is None or material.path is None or not material.path.exists():
                raise RuntimeError("所选素材不存在或已被删除，请重新选择素材。")
            job = DemoJob(uuid.uuid4().hex[:12], material, device, compute_type, options, _resolve_output_root(options["output_dir"]))
            self._jobs[job.id] = job
            self._running_id = job.id
        self._persist(job)
        threading.Thread(target=self._run_pipeline, args=(job,), daemon=True, name=f"workbench-{job.id}").start()
        return job

    def create_url(self, source_url: str, device: str, compute_type: str, authorized: bool, options: dict[str, Any] | None = None) -> DemoJob:
        """Create a real yt-dlp task directly from an authorized HTTP(S) URL."""
        options = _safe_options(options)
        parsed = urlparse(source_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Enter a valid HTTP or HTTPS video URL.")
        if not authorized:
            raise ValueError("请先确认拥有处理或转载授权，再开始下载。")
        with self._lock:
            if self._running_id:
                raise RuntimeError("A task is already running. Wait for it to finish or cancel it first.")
            host = parsed.netloc.removeprefix("www.")
            material = Material(
                id=f"url-{uuid.uuid4().hex[:10]}", path=None, name=options["title"] or f"{host} video",
                duration_seconds=None, width=None, height=None, authorized=True,
                source_url=source_url.strip(),
            )
            job = DemoJob(uuid.uuid4().hex[:12], material, device, compute_type, options, _resolve_output_root(options["output_dir"]))
            self._jobs[job.id] = job
            self._running_id = job.id
        self._persist(job)
        threading.Thread(target=self._run_pipeline, args=(job,), daemon=True, name=f"workbench-{job.id}").start()
        return job

    def cancel(self, job_id: str) -> DemoJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in {"queued", "running"}:
                job.status = "cancelling"
                job.stage = "正在取消"
                job.add_log("Cancellation requested from workbench.")
                request_cancellation()
        if job:
            self._persist(job)
        return job

    def register_native_publish_file(self, path: Path) -> str:
        """Register a path picked by the native shell without exposing it via HTTP."""
        resolved = path.resolve()
        if not resolved.is_file() or resolved.suffix.lower() not in MEDIA_EXTENSIONS:
            raise ValueError("Choose a supported rendered video file.")
        token = uuid.uuid4().hex
        with self._lock:
            self._native_publish_files[token] = resolved
        return token

    def publish_file_metadata(self, token: str) -> dict[str, Any]:
        with self._lock:
            video = self._native_publish_files.get(token)
        if video is None or not video.exists():
            raise ValueError("The selected video is no longer available. Choose it again.")
        title, tags = video.stem, []
        metadata_path = video.parent / "publish_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                title = str(metadata.get("title") or title).strip() or title
                tags = [str(tag).strip() for tag in metadata.get("tags") or [] if str(tag).strip()]
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return {"token": token, "name": video.name, "title": title, "tags": tags, "has_metadata": metadata_path.exists()}

    def publish_file_from_token(self, token: str) -> Path | None:
        with self._lock:
            video = self._native_publish_files.get(token)
        return video if video and video.exists() else None

    def cues(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None or job.work_dir is None:
            raise RuntimeError("Subtitle data is not ready yet.")
        translated_file = job.edited_segments or job.work_dir / "segments.translated.json"
        translated_ready = translated_file.exists()
        if translated_ready:
            translated = load_segments(translated_file)
        else:
            source_file = job.work_dir / "segments.source.json"
            if not source_file.exists():
                # 转写/OCR 尚未完成
                raise RuntimeError("Subtitle data is not ready yet.")
            translated = load_segments(source_file)  # 翻译中：原文预览
        # 以当前数据为准：时间轴随编辑/裁剪/保留/重排变化，text 始终是原文
        audio_texts: set[str] = set()
        ocr_texts: set[str] = set()
        for name in ("segments.audio.json", "segments.ocr.json"):
            path = job.work_dir / name
            if path.exists():
                for segment in load_segments(path):
                    (ocr_texts if name.startswith("segments.ocr") else audio_texts).add(segment.text.strip())
        output: list[dict[str, Any]] = []
        for index, item in enumerate(translated):
            raw_source = item.text
            stripped = raw_source.strip()
            if stripped in ocr_texts and stripped not in audio_texts:
                kind = "ocr"
            elif stripped in audio_texts and stripped not in ocr_texts:
                kind = "audio"
            else:
                kind = "merged"
            output.append({
                "id": index, "start": item.start, "end": item.end,
                "source": raw_source,
                "translated": item.translated_text or item.text,
                "kind": kind,
            })
        return {"cues": output, "translation_ready": translated_ready}

    def save_cues(self, job_id: str, cues: list[dict[str, Any]]) -> DemoJob:
        """Save subtitle edits: text, timing and deletions.

        ``cues`` is a full list aligned with the current segments; each item:
        ``{start, end, translated, deleted}``.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.work_dir is None or job.status != "completed":
                raise RuntimeError("Finish the initial task before editing subtitles.")
            source_path = job.edited_segments or job.work_dir / "segments.translated.json"
            segments = load_segments(source_path)
            if len(cues) != len(segments):
                raise RuntimeError("Subtitle changed while editing; reload the latest cues.")
            kept: list[Segment] = []
            for segment, item in zip(segments, cues):
                if not isinstance(item, dict):
                    raise RuntimeError("Invalid subtitle edit payload.")
                deleted = bool(item.get("deleted"))
                if deleted:
                    continue
                cleaned = str(item.get("translated", "")).strip()
                if not cleaned:
                    raise RuntimeError("Subtitle text cannot be empty.")
                try:
                    start = float(item.get("start", segment.start))
                    end = float(item.get("end", segment.end))
                except (TypeError, ValueError):
                    raise RuntimeError("Subtitle time must be a number of seconds.") from None
                if not (0 <= start < end):
                    raise RuntimeError("Subtitle time is invalid: start must be >= 0 and end must be greater than start.")
                segment.start = round(start, 2)
                segment.end = round(end, 2)
                segment.translated_text = cleaned
                kept.append(segment)
            if not kept:
                raise RuntimeError("At least one subtitle must remain.")
            edited_segments = job.work_dir / "segments.translated.edited.json"
            edited_srt = job.work_dir / "zh.edited.srt"
            save_segments(edited_segments, kept)
            write_srt(edited_srt, kept, display_mode=job.options["subtitle_display_mode"], smart_layout=job.options["smart_subtitle_layout"])
            job.edited_segments = edited_segments
            if job.result:
                job.result["translated_srt"] = public_path(edited_srt) or edited_srt.name
            job.add_log(f"Subtitle edits saved: {len(kept)} cues, {len(segments) - len(kept)} deleted.")
            return job

    def align(self, job_id: str) -> DemoJob:
        """Re-align the finished subtitles using OCR anchors (manual fallback)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.work_dir is None or job.raw_video is None or job.status != "completed":
                raise RuntimeError("完成初始任务后才能对齐字幕。")
            if self._running_id:
                raise RuntimeError("已有任务在运行。")
        work_dir = job.work_dir
        audio_path = work_dir / "segments.audio.json"
        ocr_path = work_dir / "segments.ocr.json"
        if not audio_path.exists() or not ocr_path.exists():
            raise RuntimeError("该任务没有同时包含音频与画面字幕，无法自动对齐。")
        with self._lock:
            self._running_id = job.id
        try:
            audio_segments = load_segments(audio_path)
            ocr_segments = load_segments(ocr_path)
            offset = _estimate_audio_offset(audio_segments, ocr_segments)
            if offset is None or abs(offset) < 0.12:
                with self._lock:
                    job.add_log("Alignment: no meaningful offset detected; nothing to shift.")
                    job.status, job.stage, job.progress = "completed", "对齐完成", 100
                    job.finished_at = time.time()
                return job
            current_path = job.edited_segments or work_dir / "segments.translated.json"
            if not current_path.exists():
                raise RuntimeError("没有可对齐的字幕数据。")
            segments = load_segments(current_path)
            _shift_segments(segments, offset)
            new_segments = work_dir / "segments.aligned.json"
            new_srt = work_dir / "zh.aligned.srt"
            save_segments(new_segments, segments)
            write_srt(new_srt, segments, display_mode=job.options["subtitle_display_mode"], smart_layout=job.options["smart_subtitle_layout"])
            with self._lock:
                job.status, job.stage, job.progress = "running", "对齐字幕", 80
                job.error = None
                job.started_at = time.time()
                job.finished_at = None
                job.add_log(f"Alignment: shifting subtitles by {offset:+.2f}s.")
            self._finish_modify(job, job.raw_video, f"Aligned subtitles by {offset:+.2f}s.", subtitle=new_srt, segments_file=new_segments)
        except Exception as exc:
            traceback.print_exc()
            with self._lock:
                job.status, job.stage, job.error = "failed", "对齐失败", str(exc)
                job.finished_at = time.time()
                job.add_log(f"Alignment failed: {exc}")
        finally:
            with self._lock:
                if self._running_id == job.id:
                    self._running_id = None
            self._persist(job)
        return job

    def modify(self, job_id: str, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Video modifications: keep / mute / export / reorder.

        ``keep``/``mute``/``reorder`` update the finished clip (running ->
        completed). ``export`` is synchronous and only writes a new file.
        """
        op = str(op).strip().lower()
        if op == "export":
            return self._export_segment(job_id, payload)
        if op not in ("keep", "mute", "reorder"):
            raise RuntimeError(f"未知的视频修改操作：{op}")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.work_dir is None or job.raw_video is None or job.status != "completed":
                raise RuntimeError("完成初始任务后才能修改视频。")
            if self._running_id:
                raise RuntimeError("已有任务在运行。")
        # 参数校验（在进入任务状态机之前，参数错误直接报错而不创建失败任务）
        duration = _probe_media(job.raw_video)[0] or 0
        if op in ("keep", "mute"):
            ranges = _normalize_cut_ranges(payload.get("ranges", []), duration)
            if not ranges:
                raise RuntimeError("请至少添加一段时间范围。")
            if op == "keep":
                covered = sum(b - a for a, b in ranges)
                if covered >= duration - 0.2:
                    raise RuntimeError("保留范围覆盖了整个视频，无需修改。")
        else:
            raw_points = payload.get("points", [])
            raw_order = payload.get("order", [])
            if not isinstance(raw_points, list) or not isinstance(raw_order, list):
                raise RuntimeError("重排参数无效。")
            try:
                times = [float(p) for p in raw_points]
                indexes = [int(i) for i in raw_order]
            except (TypeError, ValueError):
                raise RuntimeError("重排参数必须是数字。") from None
            boundaries = [0.0] + sorted(t for t in times if 0 < t < duration) + [duration]
            boundaries = [boundaries[0]] + [b for a, b in zip(boundaries, boundaries[1:]) if b > a + 0.1]
            count = len(boundaries) - 1
            if count < 2:
                raise RuntimeError("至少需要两个片段才能重排。")
            if sorted(indexes) != list(range(count)):
                raise RuntimeError("新顺序必须包含每一段且不重复（如 3,1,2）。")
        with self._lock:
            self._running_id = job.id
        try:
            if op == "keep":
                with self._lock:
                    job.status, job.stage, job.progress = "running", "截取保留片段", 50
                    job.error = None
                    job.started_at = time.time()
                    job.finished_at = None
                    job.add_log(f"Keeping {len(ranges)} segment(s).")
                self._run_keep(job, ranges)
            elif op == "mute":
                with self._lock:
                    job.status, job.stage, job.progress = "running", "片段静音", 50
                    job.error = None
                    job.started_at = time.time()
                    job.finished_at = None
                    job.add_log(f"Muting {len(ranges)} segment(s).")
                self._run_mute(job, ranges)
            else:
                with self._lock:
                    job.status, job.stage, job.progress = "running", "片段重排", 50
                    job.error = None
                    job.started_at = time.time()
                    job.finished_at = None
                    job.add_log("Reordering segments.")
                self._run_reorder(job, boundaries, indexes)
        except Exception as exc:
            traceback.print_exc()
            with self._lock:
                job.status, job.stage, job.error = "failed", "修改失败", str(exc)
                job.finished_at = time.time()
                job.add_log(f"Modify failed: {exc}")
        finally:
            with self._lock:
                if self._running_id == job.id:
                    self._running_id = None
            self._persist(job)
        return job.snapshot()

    def _run_keep(self, job: DemoJob, keep_ranges: list[tuple[float, float]]) -> None:
        """Keep only the given ranges (subtitle-aware)."""
        work_dir = job.work_dir
        raw = job.raw_video
        kept_video = work_dir / f"kept-{uuid.uuid4().hex[:8]}.mp4"
        # 保留区间用 +（或）连接：任一保留区间内的帧保留
        select_v = "+".join(f"between(t,{a + 0.05:.3f},{b - 0.05:.3f})" for a, b in keep_ranges)
        select_a = "+".join(f"between(t,{a + 0.05:.3f},{b - 0.05:.3f})" for a, b in keep_ranges)
        run_command(
            [
                "ffmpeg", "-y", "-i", str(raw),
                "-vf", f"select='{select_v}',setpts=N/FRAME_RATE/TB",
                "-af", f"aselect='{select_a}',asetpts=N/SR/TB",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-map", "0:v:0?", "-map", "0:a:0?",
                str(kept_video),
            ],
            cancel_check=is_cancellation_requested,
        )
        if not kept_video.exists():
            raise RuntimeError("截取保留没有生成输出文件。")
        segments_path = job.edited_segments or work_dir / "segments.translated.json"
        segments = load_segments(segments_path)
        kept_segments = _remap_segments_for_keep(segments, keep_ranges)
        new_segments = work_dir / "segments.kept.json"
        new_srt = work_dir / "zh.kept.srt"
        save_segments(new_segments, kept_segments)
        write_srt(new_srt, kept_segments, display_mode=job.options["subtitle_display_mode"], smart_layout=job.options["smart_subtitle_layout"])
        self._finish_modify(
            job, kept_video, f"Kept {len(keep_ranges)} segment(s).",
            subtitle=new_srt if kept_segments else None,
            segments_file=new_segments,
        )

    def _run_mute(self, job: DemoJob, ranges: list[tuple[float, float]]) -> None:
        work_dir = job.work_dir
        raw = job.raw_video
        muted = work_dir / f"muted-{uuid.uuid4().hex[:8]}.mp4"
        volume_expr = "+".join(f"volume=enable='between(t,{a + 0.05:.3f},{b - 0.05:.3f})':volume=0" for a, b in ranges)
        run_command(
            [
                "ffmpeg", "-y", "-i", str(raw),
                "-af", volume_expr,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-map", "0:v:0?", "-map", "0:a:0?",
                str(muted),
            ],
            cancel_check=is_cancellation_requested,
        )
        if not muted.exists():
            raise RuntimeError("静音处理没有生成输出文件。")
        self._finish_modify(job, muted, f"Muted {len(ranges)} segment(s).")

    def _run_reorder(self, job: DemoJob, boundaries: list[float], indexes: list[int]) -> None:
        work_dir = job.work_dir
        raw = job.raw_video
        count = len(boundaries) - 1
        parts: list[Path] = []
        try:
            for i in range(count):
                part = work_dir / f"part-{i}.mp4"
                run_command(
                    [
                        "ffmpeg", "-y", "-ss", f"{boundaries[i]:.3f}", "-to", f"{boundaries[i + 1]:.3f}",
                        "-i", str(raw),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "128k",
                        "-map", "0:v:0?", "-map", "0:a:0?",
                        str(part),
                    ],
                    cancel_check=is_cancellation_requested,
                )
                parts.append(part)
            reordered = work_dir / f"reordered-{uuid.uuid4().hex[:8]}.mp4"
            list_file = work_dir / "concat.txt"
            list_file.write_text(
                "\n".join(f"file '{parts[i].as_posix()}'" for i in indexes) + "\n", encoding="utf-8")
            run_command(
                [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c", "copy", str(reordered),
                ],
                cancel_check=is_cancellation_requested,
            )
        finally:
            for part in parts:
                part.unlink(missing_ok=True)
            (work_dir / "concat.txt").unlink(missing_ok=True)
        if not reordered.exists():
            raise RuntimeError("重排没有生成输出文件。")

        segments_path = job.edited_segments or work_dir / "segments.translated.json"
        segments = load_segments(segments_path)
        remapped = _remap_segments_for_reorder(segments, boundaries, indexes)
        new_segments = work_dir / "segments.reordered.json"
        new_srt = work_dir / "zh.reordered.srt"
        save_segments(new_segments, remapped)
        write_srt(new_srt, remapped, display_mode=job.options["subtitle_display_mode"], smart_layout=job.options["smart_subtitle_layout"])
        self._finish_modify(
            job, reordered, f"Reordered {count} segments.",
            subtitle=new_srt if remapped else None,
            segments_file=new_segments,
        )

    def _finish_modify(
        self,
        job: DemoJob,
        new_raw: Path,
        log_message: str,
        subtitle: Path | None = None,
        segments_file: Path | None = None,
    ) -> None:
        work_dir = job.work_dir
        if subtitle is None or not subtitle.exists():
            # 无字幕可用：直接以修改后的视频作为成片（避免空 SRT 渲染失败）
            rendered = work_dir / "rendered.mp4"
            shutil.copy2(new_raw, rendered)
            with self._lock:
                job.add_log("No subtitles for the modified clip; the result contains no burned subtitles.")
            new_duration = _probe_media(new_raw)[0]
            with self._lock:
                job.raw_video = new_raw
                job.rendered_video = rendered
                if segments_file:
                    job.edited_segments = segments_file
                job.duration_override = round(new_duration, 2) if new_duration else None
                job.result = {
                    "output_dir": public_path(work_dir) or "",
                    "source_srt": None,
                    "translated_srt": None,
                    "rendered_video": public_path(rendered) or rendered.name,
                }
                job.add_log(log_message)
                job.status, job.stage, job.progress = "completed", "修改完成", 100
                job.finished_at = time.time()
            return
        rendered = burn_subtitles(
            new_raw, subtitle, work_dir / "rendered.mp4",
            font_name=job.options["font_name"], font_size=job.options["font_size"],
            primary_color=job.options["subtitle_color"], outline_color=job.options["subtitle_outline_color"],
            outline=job.options["subtitle_outline"], shadow=job.options["subtitle_shadow"],
            raised_margin=True, crf=job.options["render_crf"], margin_ratio=job.options["subtitle_margin_ratio"],
        )
        new_duration = _probe_media(new_raw)[0]
        with self._lock:
            job.raw_video = new_raw
            job.rendered_video = rendered
            if segments_file:
                job.edited_segments = segments_file
            job.duration_override = round(new_duration, 2) if new_duration else None
            job.result = {
                "output_dir": public_path(work_dir) or "",
                "source_srt": public_path(subtitle) or subtitle.name,
                "translated_srt": public_path(subtitle) or subtitle.name,
                "rendered_video": public_path(rendered) or rendered.name,
            }
            job.add_log(log_message)
            job.status, job.stage, job.progress = "completed", "修改完成", 100
            job.finished_at = time.time()

    def _export_segment(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.rendered_video is None:
                raise RuntimeError("先完成任务，才能导出片段。")
            rendered = job.rendered_video
            work_dir = job.work_dir
        ranges = _normalize_cut_ranges(payload.get("ranges", []), _probe_media(rendered)[0] or 0)
        if len(ranges) != 1:
            raise RuntimeError("导出片段请填写一段开始/结束时间。")
        start, end = ranges[0]
        out = work_dir / f"export-{uuid.uuid4().hex[:8]}.mp4"
        run_command(
            [
                "ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(rendered),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-map", "0:v:0?", "-map", "0:a:0?",
                str(out),
            ],
            cancel_check=is_cancellation_requested,
        )
        if not out.exists():
            raise RuntimeError("导出片段失败。")
        return {"exported": str(out), "name": out.name}

    def trim(self, job_id: str, cut_ranges: list[list[float]]) -> DemoJob:
        """Cut video segments out of the finished clip and re-render subtitles."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.work_dir is None or job.raw_video is None or job.status != "completed":
                raise RuntimeError("Finish the initial task before trimming the video.")
            if self._running_id:
                raise RuntimeError("A task is already running.")
            work_dir = job.work_dir
            raw_video = job.raw_video
        duration = _probe_media(raw_video)[0]
        if not duration or duration <= 0:
            raise RuntimeError("Could not read the video duration for trimming.")
        ranges = _normalize_cut_ranges(cut_ranges, duration)
        if not ranges:
            raise RuntimeError("Choose at least one segment to cut.")
        kept_total = duration - sum(b - a for a, b in ranges)
        if kept_total < 0.5:
            raise RuntimeError("The cut segments cover the whole video; nothing would remain.")

        with self._lock:
            job.status, job.stage, job.progress = "running", "裁剪视频片段", 60
            job.error = None
            job.started_at = time.time()
            job.finished_at = None
            job.add_log(f"Trimming video: removing {len(ranges)} segment(s) ({sum(b - a for a, b in ranges):.1f}s).")
            self._running_id = job.id
        try:
            self._run_trim(job, ranges)
        except Exception as exc:
            traceback.print_exc()
            with self._lock:
                job.status, job.stage, job.error = "failed", "裁剪失败", str(exc)
                job.finished_at = time.time()
                job.add_log(f"Trim failed: {exc}")
        finally:
            with self._lock:
                if self._running_id == job.id:
                    self._running_id = None
            self._persist(job)
        return job

    def _run_trim(self, job: DemoJob, ranges: list[tuple[float, float]]) -> None:
        work_dir = job.work_dir
        raw_video = job.raw_video
        # 唯一输出名：避免连续裁剪时输入输出同名（ffmpeg 拒绝原地覆盖）
        trimmed = work_dir / f"trimmed-{uuid.uuid4().hex[:8]}.mp4"
        # 多个区间用 *（布尔与）连接：不在任何删除区间内的帧才保留
        select_v = "*".join(f"not(between(t,{a + 0.05:.3f},{b - 0.05:.3f}))" for a, b in ranges)
        select_a = "*".join(f"not(between(t,{a + 0.05:.3f},{b - 0.05:.3f}))" for a, b in ranges)
        run_command(
            [
                "ffmpeg", "-y", "-i", str(raw_video),
                "-vf", f"select='{select_v}',setpts=N/FRAME_RATE/TB",
                "-af", f"aselect='{select_a}',asetpts=N/SR/TB",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-map", "0:v:0?", "-map", "0:a:0?",
                str(trimmed),
            ],
            cancel_check=is_cancellation_requested,
        )
        if not trimmed.exists():
            raise RuntimeError("Trimming produced no output video.")

        segments_path = job.edited_segments or work_dir / "segments.translated.json"
        segments = load_segments(segments_path)
        kept = _remap_segments_after_cut(segments, ranges)
        new_segments = work_dir / "segments.trimmed.json"
        new_srt = work_dir / "zh.trimmed.srt"
        save_segments(new_segments, kept)
        write_srt(new_srt, kept, display_mode=job.options["subtitle_display_mode"], smart_layout=job.options["smart_subtitle_layout"])

        if kept:
            rendered = burn_subtitles(
                trimmed, new_srt, work_dir / "rendered.mp4",
                font_name=job.options["font_name"], font_size=job.options["font_size"],
                primary_color=job.options["subtitle_color"], outline_color=job.options["subtitle_outline_color"],
                outline=job.options["subtitle_outline"], shadow=job.options["subtitle_shadow"],
                raised_margin=True, crf=job.options["render_crf"], margin_ratio=job.options["subtitle_margin_ratio"],
            )
        else:
            # 裁剪后没有剩余字幕：直接以裁剪视频作为成片（避免空 SRT 导致渲染失败）
            rendered = work_dir / "rendered.mp4"
            shutil.copy2(trimmed, rendered)
            with self._lock:
                job.add_log("No subtitles remain after trimming; the result contains no burned subtitles.")
        new_duration = _probe_media(trimmed)[0]

        with self._lock:
            job.raw_video = trimmed
            job.rendered_video = rendered
            job.edited_segments = new_segments
            job.duration_override = round(new_duration, 2) if new_duration else None
            job.result = {
                "output_dir": public_path(work_dir) or "",
                "source_srt": public_path(new_srt) or new_srt.name,
                "translated_srt": public_path(new_srt) or new_srt.name,
                "rendered_video": public_path(rendered) or rendered.name,
            }
            job.add_log(f"Trim finished: kept {len(kept)} cues, new duration {job.duration_override}s.")
            job.status, job.stage, job.progress = "completed", "裁剪完成", 100
            job.finished_at = time.time()

    def rerender(self, job_id: str) -> DemoJob:
        with self._lock:
            if self._running_id:
                raise RuntimeError("A task is already running.")
            job = self._jobs.get(job_id)
            if job is None or job.work_dir is None or job.raw_video is None or job.status != "completed":
                raise RuntimeError("Finish the initial task before re-rendering.")
            subtitle = next(
                (job.work_dir / name for name in
                 ("zh.edited.srt", "zh.kept.srt", "zh.trimmed.srt", "zh.reordered.srt", "zh.srt")
                 if (job.work_dir / name).exists()),
                None,
            )
            if subtitle is None:
                raise RuntimeError("No subtitle file is available for re-rendering.")
            job.status = "running"
            job.stage = "重新渲染字幕"
            job.progress = 92
            job.error = None
            job.started_at = time.time()
            job.finished_at = None
            job.add_log("Re-rendering the edited Chinese subtitles.")
            self._running_id = job.id
        threading.Thread(target=self._run_rerender, args=(job, subtitle), daemon=True, name=f"rerender-{job.id}").start()
        return job

    def media(self, job_id: str, kind: str) -> Path | None:
        job = self.get(job_id)
        if job is None:
            return None
        if kind == "source":
            return job.raw_video or job.material.path
        if kind == "rendered":
            return job.rendered_video
        return None

    def _run_pipeline(self, job: DemoJob) -> None:
        load_dotenv(USER_DATA_ROOT / ".env")
        with self._lock:
            job.status = "running"
            job.started_at = time.time()
            job.stage = "准备素材"
            job.progress = 4
            job.add_log(f"Started {job.material.name}.")

        def log(message: str) -> None:
            with self._lock:
                job.add_log(message)

        def download_progress(fraction: float) -> None:
            # 准备素材阶段：4% -> 12% 反映真实下载字节进度
            with self._lock:
                job.add_log(f"[p={4 + int(8 * max(0.0, min(1.0, fraction)))}] 正在下载视频 {int(fraction * 100)}%…")

        try:
            source_url = job.material.source_url
            options = job.options
            result = run_pipeline(PipelineOptions(
                source=source_url or str(job.material.path), source_kind="url" if source_url else "file", output_dir=job.output_root,
                title=options["title"] or job.material.name, description=options["description"], tags=options["tags"], i_have_rights=job.material.authorized,
                require_reuse_allowed=options["require_reuse_allowed"] and bool(source_url), cookies_from_browser=options["cookies_from_browser"],
                cookies_file=options["cookies_file"],
                max_seconds=options["max_seconds"] if source_url else None, subtitle_source=options["subtitle_source"],
                whisper_model_size=options["whisper_model_size"], source_language=options["source_language"],
                beam_size=options["beam_size"], ocr_interval=options["ocr_interval"],
                ocr_crop_ratio=options["ocr_crop_ratio"], ocr_min_chars=options["ocr_min_chars"],
                subtitle_margin_ratio=options["subtitle_margin_ratio"], render_crf=options["render_crf"],
                device=job.device, compute_type=job.compute_type, translator=options["translator"], target_lang=options["target_lang"],
                translate_model=options["translate_model"], smart_translation=options["smart_translation"],
                smart_subtitle_layout=options["smart_subtitle_layout"], font_name=options["font_name"], font_size=options["font_size"],
                subtitle_display_mode=options["subtitle_display_mode"], subtitle_color=options["subtitle_color"],
                subtitle_outline_color=options["subtitle_outline_color"], subtitle_outline=options["subtitle_outline"], subtitle_shadow=options["subtitle_shadow"],
                publish_to_bilibili=options["publish_to_bilibili"], include_source_link_in_description=options["include_source_link"],
                bilibili_browser=options["bilibili_browser"], bilibili_profile_dir=USER_DATA_ROOT / "data" / ("bilibili-edge-profile" if options["bilibili_browser"] == "msedge" else "bilibili-profile"),
                bilibili_wait_for_review=not options["close_after_fill"],
            ), log=log, progress=download_progress)
        except CancellationError:
            with self._lock:
                job.status, job.stage = "cancelled", "已取消"
                job.finished_at = time.time()
                job.add_log("Task was cancelled.")
        except Exception as exc:
            traceback.print_exc()
            with self._lock:
                job.status, job.stage, job.error = "failed", "处理失败", str(exc)
                job.finished_at = time.time()
                job.add_log(f"Task failed: {exc}")
        else:
            with self._lock:
                job.work_dir, job.raw_video = result.work_dir, result.job.raw_video
                job.rendered_video = result.rendered_video
                job.result = {
                    "output_dir": public_path(result.work_dir) or "", "source_srt": public_path(result.source_srt) or "",
                    "translated_srt": public_path(result.translated_srt) or "", "rendered_video": public_path(result.rendered_video) or "",
                }
                job.add_log("Task completed successfully.")
                job.status, job.stage, job.progress = "completed", "处理完成", 100
                job.finished_at = time.time()
        finally:
            with self._lock:
                if self._running_id == job.id:
                    self._running_id = None
            self._persist(job)

    def _run_rerender(self, job: DemoJob, subtitle: Path) -> None:
        try:
            rendered = burn_subtitles(
                job.raw_video, subtitle, job.work_dir / "rendered.edited.mp4",  # type: ignore[arg-type]
                font_name=job.options["font_name"], font_size=job.options["font_size"],
                primary_color=job.options["subtitle_color"], outline_color=job.options["subtitle_outline_color"],
                outline=job.options["subtitle_outline"], shadow=job.options["subtitle_shadow"],
                crf=job.options["render_crf"], margin_ratio=job.options["subtitle_margin_ratio"],
            )
        except Exception as exc:
            traceback.print_exc()
            with self._lock:
                job.status, job.stage, job.error = "failed", "重新渲染失败", str(exc)
                job.finished_at = time.time()
                job.add_log(f"Re-render failed: {exc}")
        else:
            with self._lock:
                if job.result:
                    job.result["rendered_video"] = public_path(rendered) or rendered.name
                job.rendered_video = rendered
                job.add_log("Edited rendered.mp4 created successfully.")
                job.status, job.stage, job.progress = "completed", "重新渲染完成", 100
                job.finished_at = time.time()
        finally:
            with self._lock:
                if self._running_id == job.id:
                    self._running_id = None
            self._persist(job)


class WorkbenchHandler(BaseHTTPRequestHandler):
    frontend_dir: Path
    jobs: WorkbenchJobs
    server_version = "YBLocalizerWorkbench/2.0"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT); self._cors_headers(); self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json({"ok": True, "demo_source": "demo/authorized-demo-10s.mp4"}); return
        if path == "/api/settings":
            cookies_file = os.getenv("YBLOCALIZER_COOKIES_FILE", "")
            try:
                default_output_dir = str(OUTPUT_ROOT.relative_to(PROJECT_ROOT))
            except ValueError:
                default_output_dir = str(OUTPUT_ROOT)
            self._json({
                "deepseek_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
                "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
                "default_output_dir": default_output_dir,
                "cookies_from_browser": os.getenv("YBLOCALIZER_COOKIES_FROM_BROWSER", ""),
                "cookies_file": cookies_file,
                "cookies_file_valid": _cookies_file_has_login(Path(cookies_file)) if cookies_file else None,
            }); return
        if path == "/api/templates":
            self._json({"templates": [{"name": name, "body": body} for name, body in get_all_templates().items()]}); return
        if path == "/api/outputs":
            self._json(self._outputs_snapshot(OUTPUT_ROOT)); return
        if path == "/api/publish/status":
            browser = "msedge" if str(parse_qs(urlparse(self.path).query).get("browser", ["chromium"])[0]).lower() in {"edge", "msedge"} else "chromium"
            pids = self._publish_profile_process_ids(browser)
            self._json({"profile_busy": bool(pids), "process_count": len(pids), "message": "B 站自动化浏览器正在使用中。" if pids else "B 站自动化 Profile 可用。"}); return
        if path == "/api/publish/assist/status":
            self._json(self.jobs.publish_session().snapshot()); return
        if path == "/api/materials":
            self._json({"materials": self.jobs.materials()}); return
        if path == "/api/history/jobs":
            try:
                limit = int(parse_qs(urlparse(self.path).query).get("limit", ["50"])[0])
            except ValueError:
                limit = 50
            self._json({"jobs": job_db.list_jobs(max(1, min(200, limit)))}); return
        match = re.fullmatch(r"/api/materials/([\w-]+)/media", path)
        if match:
            media = self.jobs.material_path(match.group(1))
            if media is None or not media.exists(): self._json({"error": "Material is not available."}, HTTPStatus.NOT_FOUND)
            else: self._serve_media(media)
            return
        match = re.fullmatch(r"/api/jobs/([\w-]+)/media/(source|rendered)", path)
        if match:
            media = self.jobs.media(*match.groups())
            if media is None or not media.exists(): self._json({"error": "Requested media is not ready."}, HTTPStatus.NOT_FOUND)
            else: self._serve_media(media)
            return
        match = re.fullmatch(r"/api/jobs/([\w-]+)/cues", path)
        if match:
            try: self._json(self.jobs.cues(match.group(1)))
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        match = re.fullmatch(r"/api/jobs/([\w-]+)", path)
        if match:
            job = self.jobs.get(match.group(1))
            self._json(job.snapshot() if job else {"error": "Task not found."}, HTTPStatus.OK if job else HTTPStatus.NOT_FOUND)
            return
        self._serve_frontend(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/materials":
            self._upload_material(); return
        if path == "/api/publish/upload":
            self._upload_publish_video(); return
        payload = self._payload()
        if path == "/api/history/open":
            target = Path(str(payload.get("path", ""))).expanduser()
            if not target.exists():
                self._json({"error": "该任务输出已不存在（可能已被删除）。"}, HTTPStatus.BAD_REQUEST); return
            try:
                if os.name != "nt": raise OSError("Only available in Windows.")
                os.startfile(target)  # type: ignore[attr-defined]
            except OSError as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR); return
            self._json({"opened": str(target)})
            return
        if path == "/api/metadata":
            try:
                url = str(payload.get("url", "")).strip()
                cookies_file = str(payload.get("cookies_file", "")).strip() or os.getenv("YBLOCALIZER_COOKIES_FILE", "").strip() or None
                cookies_from_browser = str(payload.get("cookies_from_browser", "")).strip() or None
                if cookies_file:
                    cookies_from_browser = None
                metadata = get_video_metadata(url, cookies_from_browser, cookies_file)
                self._json({"title": metadata.title, "duration": metadata.duration, "license": metadata.license, "view_count": metadata.view_count, "webpage_url": metadata.webpage_url, "thumbnail_url": metadata.thumbnail_url})
            except Exception as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/readiness":
            self._json(self._readiness(payload)); return
        if path == "/api/publish/native-video":
            try: self._json(self.jobs.publish_file_metadata(str(payload.get("token", ""))))
            except ValueError as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/settings":
            try: self._save_settings(payload)
            except ValueError as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            else: self._json({"saved": True})
            return
        if path == "/api/templates":
            try:
                name, body = str(payload.get("name", "")).strip(), str(payload.get("body", ""))
                if not name or not body: raise ValueError("Template name and body are required.")
                save_custom_template(name, body)
            except ValueError as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            else: self._json({"saved": True})
            return
        if path == "/api/publish/description":
            try:
                body = build_bilibili_description(
                    str(payload.get("template", "授权本地化")),
                    str(payload.get("source_url", "")),
                    bool(payload.get("include_source_link", True)),
                    str(payload.get("custom_text", "")),
                    [str(line) for line in payload.get("extra_lines", []) if str(line).strip()],
                    template_body=str(payload.get("template_body", "") or None),
                )
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            else:
                self._json({"description": body})
            return
        if path == "/api/publish/check":
            self._start_publish_check(payload); return
        if path == "/api/publish/assist":
            self._start_publish_assist(payload); return
        if path == "/api/publish/close-browser":
            self._close_publish_browser(payload); return
        if path == "/api/outputs/scan":
            try: self._json(self._outputs_snapshot(_resolve_output_root(str(payload.get("output_dir") or OUTPUT_ROOT))))
            except ValueError as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/outputs/open":
            self._open_output_root(payload); return
        if path == "/api/outputs/open-path":
            self._open_output_path(payload); return
        if path == "/api/outputs/delete":
            self._delete_output_paths(payload); return
        if path == "/api/jobs":
            device, compute_type = str(payload.get("device", "cuda")).lower(), str(payload.get("compute_type", "float16")).lower()
            if device not in {"cuda", "cpu", "auto"}: self._json({"error": "device must be cuda, cpu, or auto."}, HTTPStatus.BAD_REQUEST); return
            source_url = payload.get("source_url")
            try:
                if isinstance(source_url, str) and source_url.strip():
                    job = self.jobs.create_url(source_url, device, compute_type, bool(payload.get("authorized", False)), payload.get("options"))
                else:
                    job = self.jobs.create(str(payload.get("material_id", "demo")), device, compute_type, payload.get("options"))
            except ValueError as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            else: self._json(job.snapshot(), HTTPStatus.ACCEPTED)
            return
        match = re.fullmatch(r"/api/jobs/([\w-]+)/(cancel|open-output|rerender)", path)
        if match:
            job_id, action = match.groups()
            if action == "cancel": job = self.jobs.cancel(job_id)
            elif action == "rerender":
                try: job = self.jobs.rerender(job_id)
                except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            else:
                job = self.jobs.get(job_id)
                if job and job.result and job.work_dir:
                    try:
                        if os.name != "nt": raise OSError("Only available in Windows.")
                        os.startfile(job.work_dir)  # type: ignore[attr-defined]
                    except OSError as exc: self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR); return
            self._json(job.snapshot() if job else {"error": "Task not found."}, HTTPStatus.OK if job else HTTPStatus.NOT_FOUND); return
        match = re.fullmatch(r"/api/jobs/([\w-]+)/trim", path)
        if match:
            try: job = self.jobs.trim(match.group(1), payload.get("cut_ranges", []))
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(job.snapshot())
            return
        match = re.fullmatch(r"/api/jobs/([\w-]+)/modify", path)
        if match:
            try: result = self.jobs.modify(match.group(1), str(payload.get("op", "")), payload)
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(result)
            return
        match = re.fullmatch(r"/api/jobs/([\w-]+)/align", path)
        if match:
            try: job = self.jobs.align(match.group(1))
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(job.snapshot())
            return
        self._json({"error": "Route not found."}, HTTPStatus.NOT_FOUND)

    def do_PUT(self) -> None:  # noqa: N802
        match = re.fullmatch(r"/api/jobs/([\w-]+)/cues", urlparse(self.path).path)
        if not match: self._json({"error": "Route not found."}, HTTPStatus.NOT_FOUND); return
        payload = self._payload(); cues = payload.get("cues")
        if not isinstance(cues, list):
            self._json({"error": "cues must be a list of {start, end, translated, deleted}."}, HTTPStatus.BAD_REQUEST); return
        try: job = self.jobs.save_cues(match.group(1), cues)
        except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        else: self._json(job.snapshot())

    def do_DELETE(self) -> None:  # noqa: N802
        match = re.fullmatch(r"/api/templates/(.+)", urlparse(self.path).path)
        if not match: self._json({"error": "Route not found."}, HTTPStatus.NOT_FOUND); return
        name = unquote(match.group(1))
        if delete_custom_template(name): self._json({"deleted": True})
        else: self._json({"error": "Only custom templates can be deleted."}, HTTPStatus.NOT_FOUND)

    def _save_settings(self, payload: dict[str, Any]) -> None:
        translator = str(payload.get("translator", "deepseek")).lower()
        key = str(payload.get("api_key", "")).strip()
        cookies_from_browser = str(payload.get("cookies_from_browser", "")).strip().lower()
        cookies_file = str(payload.get("cookies_file", "")).strip()
        if cookies_from_browser not in {"", "chrome", "edge", "firefox", "brave", "chromium"}:
            raise ValueError("Choose a supported browser for YouTube Cookies.")
        _upsert_env(USER_DATA_ROOT / ".env", {"YBLOCALIZER_COOKIES_FROM_BROWSER": cookies_from_browser})
        os.environ["YBLOCALIZER_COOKIES_FROM_BROWSER"] = cookies_from_browser
        if cookies_file:
            resolved = Path(cookies_file).expanduser()
            if not resolved.is_file():
                raise ValueError(f"Cookies 文件不存在：{resolved}")
            if not _cookies_file_has_login(resolved):
                raise ValueError(
                    f"Cookies 文件缺少登录凭据（未找到 SID/LOGIN_INFO）：{resolved}\n"
                    "这通常是未登录状态下导出的文件，下载会被 YouTube 拦截。"
                    "请确认已登录 YouTube 后重新导出。"
                )
        _upsert_env(USER_DATA_ROOT / ".env", {"YBLOCALIZER_COOKIES_FILE": str(Path(cookies_file).expanduser()) if cookies_file else ""})
        os.environ["YBLOCALIZER_COOKIES_FILE"] = str(Path(cookies_file).expanduser()) if cookies_file else ""
        if key:
            if translator not in {"deepseek", "openai"}:
                raise ValueError("Choose DeepSeek or OpenAI before saving an API key.")
            env_key = "DEEPSEEK_API_KEY" if translator == "deepseek" else "OPENAI_API_KEY"
            _upsert_env(USER_DATA_ROOT / ".env", {env_key: key})
            os.environ[env_key] = key

    def _outputs_snapshot(self, root: Path) -> dict[str, Any]:
        root = root.resolve()
        tasks = scan_outputs(root)
        category_totals: dict[str, int] = {}
        for task in tasks:
            for file in task.files:
                category_totals[file.kind] = category_totals.get(file.kind, 0) + file.size
        return {
            "root": str(root),
            "task_count": len(tasks),
            "total_bytes": sum(task.size for task in tasks),
            "total_size": format_bytes(sum(task.size for task in tasks)),
            "category_totals": {kind: {"bytes": size, "label": format_bytes(size)} for kind, size in category_totals.items()},
            "tasks": [{
                "id": str(task.path.relative_to(root)).replace("\\", "/"), "name": task.name,
                "size": task.size, "size_label": format_bytes(task.size), "modified_at": task.modified_at,
                "files": [{"id": str(file.path.relative_to(root)).replace("\\", "/"), "name": file.path.name, "kind": file.kind, "size": file.size, "size_label": format_bytes(file.size)} for file in task.files],
            } for task in tasks],
        }

    def _open_output_root(self, payload: dict[str, Any]) -> None:
        try:
            root = _resolve_output_root(str(payload.get("output_dir") or OUTPUT_ROOT)); root.mkdir(parents=True, exist_ok=True)
            if os.name != "nt": raise OSError("Only available in Windows.")
            os.startfile(root)  # type: ignore[attr-defined]
        except OSError as exc: self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        else: self._json({"opened": str(root)})

    def _open_output_path(self, payload: dict[str, Any]) -> None:
        try:
            root = _resolve_output_root(str(payload.get("output_dir") or OUTPUT_ROOT))
            relative = str(payload.get("path", ""))
            if not relative:
                raise ValueError("Choose a task or file first.")
            target = (root / relative).resolve()
            target.relative_to(root.resolve())
            if not target.exists():
                raise ValueError("The selected output no longer exists.")
            if os.name != "nt": raise OSError("Only available in Windows.")
            os.startfile(target)  # type: ignore[attr-defined]
        except (ValueError, OSError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        else:
            self._json({"opened": relative})

    def _delete_output_paths(self, payload: dict[str, Any]) -> None:
        if payload.get("confirmed") is not True:
            self._json({"error": "Deletion requires explicit confirmation."}, HTTPStatus.BAD_REQUEST); return
        try:
            root = _resolve_output_root(str(payload.get("output_dir") or OUTPUT_ROOT))
            values = payload.get("paths")
            if not isinstance(values, list) or not values: raise ValueError("Choose at least one task or file to delete.")
            paths = [(root / str(value)).resolve() for value in values]
            deleted = delete_paths(paths, root)
        except (ValueError, RuntimeError, OSError) as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        else: self._json({"deleted_bytes": deleted, "deleted_size": format_bytes(deleted)})

    def _publish_profile_dir(self, browser: str) -> Path:
        return USER_DATA_ROOT / "data" / ("bilibili-edge-profile" if browser == "msedge" else "bilibili-profile")

    def _publish_profile_process_ids(self, browser: str) -> list[str]:
        profile = str(self._publish_profile_dir(browser).resolve()).replace("'", "''")
        command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -notin @('powershell.exe','pwsh.exe') -and $_.CommandLine -and $_.CommandLine -like '*" + profile + "*' } | Select-Object -ExpandProperty ProcessId"
        try:
            completed = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=5)
            return [line.strip() for line in completed.stdout.splitlines() if line.strip().isdigit()]
        except (OSError, subprocess.SubprocessError):
            return []

    def _readiness(self, payload: dict[str, Any]) -> dict[str, Any]:
        options = _safe_options(payload.get("options"))
        source_url = str(payload.get("source_url", "")).strip()
        has_material = bool(payload.get("material_id"))
        authorized = bool(payload.get("authorized", False))
        device = str(payload.get("device", "cpu")).lower()
        compute_type = str(payload.get("compute_type", "int8")).lower()
        issues: list[str] = []
        if not source_url and not has_material: issues.append("还没有选择素材。")
        if not authorized: issues.append("还没有确认处理授权。")
        if source_url and options["max_seconds"] is not None and options["max_seconds"] < 1: issues.append("URL 读取长度必须大于 0。")
        if options["target_lang"].lower().startswith("zh") and options["translator"] == "none": issues.append("中文字幕不能使用 none 翻译器。")
        if options["translator"] == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"): issues.append("DeepSeek API Key 尚未配置。")
        if options["translator"] == "openai" and not os.getenv("OPENAI_API_KEY"): issues.append("OpenAI API Key 尚未配置。")
        if options["cookies_file"] and not Path(options["cookies_file"]).expanduser().is_file(): issues.append("Cookies 文件不存在，请重新选择。")
        if device == "cpu" and compute_type in {"float16", "int8_float16"}: issues.append("CPU 不支持当前 float16 精度，请使用 int8 或 float32。")
        if options["publish_to_bilibili"] and self._publish_profile_process_ids(options["bilibili_browser"]): issues.append("B 站自动化浏览器正在使用中，请关闭后再开始全流程。")
        return {"ready": not issues, "issues": issues, "message": "基础配置可以运行。" if not issues else "；".join(issues)}

    def _start_publish_check(self, payload: dict[str, Any]) -> None:
        browser = "msedge" if str(payload.get("browser", "chromium")).lower() in {"edge", "msedge"} else "chromium"
        session = self.jobs.begin_publish_session()
        session.status, session.message = "running", "正在打开 B 站登录检查页…"
        session.logs = [session.message]

        def runner() -> None:
            try:
                open_upload_page(profile_dir=self._publish_profile_dir(browser), browser=browser, log=self.jobs._publish_log)
            except Exception as exc:
                traceback.print_exc()
                session.status, session.error, session.message = "failed", str(exc), f"打开登录检查页失败：{exc}"
            else:
                session.status, session.message = "finished", "B 站登录检查页已关闭。"
        threading.Thread(target=runner, daemon=True, name="bilibili-check").start()
        self._json({"started": True, "message": "B 站登录检查页已打开；不会上传或投稿。"}, HTTPStatus.ACCEPTED)

    def _start_publish_assist(self, payload: dict[str, Any]) -> None:
        if payload.get("confirmed") is not True:
            self._json({"error": "Confirm you have rights and will manually review before submitting."}, HTTPStatus.BAD_REQUEST); return
        try:
            browser = "msedge" if str(payload.get("browser", "chromium")).lower() in {"edge", "msedge"} else "chromium"
            if self._publish_profile_process_ids(browser):
                raise ValueError("B 站自动化浏览器正在使用中。请先关闭后台浏览器，再开始投稿辅助。")
            job_id = str(payload.get("job_id", ""))
            job = self.jobs.get(job_id) if job_id else None
            video = self.jobs.media(job_id, "rendered") if job else None
            if video is None:
                native_token = str(payload.get("native_video_token", ""))
                video = self.jobs.publish_file_from_token(native_token) if native_token else None
            if video is None:
                root = _resolve_output_root(str(payload.get("output_dir") or OUTPUT_ROOT))
                video = (root / str(payload.get("video_id", ""))).resolve()
                video.relative_to(root)
            if not video.exists() or video.suffix.lower() not in MEDIA_EXTENSIONS: raise ValueError("Choose a rendered video from the output list.")
            title = str(payload.get("title", "")).strip() or video.stem
            description = str(payload.get("description", "")).strip()
            tags = [str(item).strip() for item in payload.get("tags", []) if str(item).strip()]
            cover = next((video.parent / name for name in ("cover.jpg", "cover.png", "thumbnail.jpg", "thumbnail.png") if (video.parent / name).exists()), None)
            session = self.jobs.begin_publish_session()

            def assist_runner() -> None:
                try:
                    assist_publish(
                        video_path=video, title=title, description=description, tags=tags, cover_path=cover,
                        profile_dir=self._publish_profile_dir(browser), browser=browser,
                        screenshot_path=video.parent / "bilibili-upload-page.png",
                        wait_for_review=not bool(payload.get("close_after_fill", False)),
                        log=self.jobs._publish_log,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    session.status, session.error = "failed", str(exc)
                    session.message = f"B 站投稿辅助失败：{exc}"
                else:
                    if session.status != "failed":
                        session.status, session.message = "finished", "B 站投稿辅助已结束（浏览器已关闭）。"
            threading.Thread(target=assist_runner, daemon=True, name="bilibili-assist").start()
        except (ValueError, OSError) as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        else: self._json({"started": True, "message": "正在打开 B 站投稿辅助。填写完成后请人工检查并手动投稿。"}, HTTPStatus.ACCEPTED)

    def _close_publish_browser(self, payload: dict[str, Any]) -> None:
        if payload.get("confirmed") is not True:
            self._json({"error": "Closing a browser requires explicit confirmation."}, HTTPStatus.BAD_REQUEST); return
        profiles = [str((USER_DATA_ROOT / "data" / name).resolve()) for name in ("bilibili-profile", "bilibili-edge-profile")]
        patterns = " -or ".join("$_.CommandLine -like '*{}*'".format(value.replace("'", "''")) for value in profiles)
        command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -notin @('powershell.exe','pwsh.exe') -and $_.CommandLine -and (" + patterns + ") } | Select-Object -ExpandProperty ProcessId"
        try:
            found = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=5)
            pids = [line.strip() for line in found.stdout.splitlines() if line.strip().isdigit()]
            if pids: subprocess.run(["powershell", "-NoProfile", "-Command", "Stop-Process -Id " + ",".join(pids) + " -Force"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc: self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        else: self._json({"closed": len(pids)})

    def _upload_material(self) -> None:
        temporary: Path | None = None
        try:
            filename, authorized, temporary = self._receive_multipart_video()
            with temporary.open("rb") as source:
                material = self.jobs.add_upload(filename, source, authorized)
        except (ValueError, OSError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        else:
            self._json(material.snapshot(), HTTPStatus.CREATED)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _upload_publish_video(self) -> None:
        """Development-browser fallback. The packaged desktop app uses native paths."""
        temporary: Path | None = None
        try:
            filename, _authorized, temporary = self._receive_multipart_video()
            token = self.jobs.register_native_publish_file(temporary)
            details = self.jobs.publish_file_metadata(token)
            details["name"] = filename
        except (ValueError, OSError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        else:
            self._json(details, HTTPStatus.CREATED)

    def _receive_multipart_video(self) -> tuple[str, bool, Path]:
        """Small streaming multipart reader; Python 3.14 no longer ships cgi."""
        content_type = self.headers.get("Content-Type", "")
        match = re.search(r"boundary=([^;]+)", content_type)
        if not match:
            raise ValueError("Expected multipart/form-data.")
        boundary = b"--" + match.group(1).strip().strip('"').encode("utf-8")
        try:
            remaining = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Missing upload size.") from None
        if remaining <= 0 or remaining > 2 * 1024 * 1024 * 1024:
            raise ValueError("Video upload must be between 1 byte and 2 GB.")
        first = self.rfile.readline()
        remaining -= len(first)
        if not first.rstrip(b"\r\n") == boundary:
            raise ValueError("Malformed multipart upload.")
        filename = ""
        authorized = False
        temporary: Path | None = None
        while remaining > 0:
            headers: dict[str, str] = {}
            while True:
                line = self.rfile.readline(); remaining -= len(line)
                if not line or line in {b"\r\n", b"\n"}: break
                key, _, value = line.decode("utf-8", "replace").partition(":")
                headers[key.lower().strip()] = value.strip()
            disposition = headers.get("content-disposition", "")
            name_match = re.search(r'name="([^"]+)"', disposition)
            field_name = name_match.group(1) if name_match else ""
            file_match = re.search(r'filename="([^"]*)"', disposition)
            if file_match:
                filename = file_match.group(1)
                UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
                temporary = UPLOAD_ROOT / f".upload-{uuid.uuid4().hex}.part"
                with temporary.open("wb") as destination:
                    previous = b""
                    while remaining > 0:
                        line = self.rfile.readline(); remaining -= len(line)
                        if line.startswith(boundary):
                            if previous.endswith(b"\r\n"): previous = previous[:-2]
                            destination.write(previous)
                            break
                        if previous: destination.write(previous)
                        previous = line
            else:
                value = b""
                while remaining > 0:
                    line = self.rfile.readline(); remaining -= len(line)
                    if line.startswith(boundary): break
                    value += line
                if field_name == "authorized": authorized = value.decode("utf-8", "replace").strip().lower() == "true"
            if line.rstrip(b"\r\n").endswith(b"--"): break
        if temporary is None or not filename:
            raise ValueError("Choose a video file first.")
        return filename, authorized, temporary

    def _payload(self) -> dict[str, Any]:
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")) or 0)
            value = json.loads(raw.decode("utf-8") or "{}")
            return value if isinstance(value, dict) else {}
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError): return {}

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self._cors_headers(); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5173")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")

    def _serve_media(self, path: Path) -> None:
        size = path.stat().st_size; start, end = 0, size - 1; status = HTTPStatus.OK
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match:
            start = int(match.group(1) or 0); end = min(int(match.group(2) or end), end); status = HTTPStatus.PARTIAL_CONTENT
        if start > end or start >= size: self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE); return
        length = end - start + 1
        self.send_response(status); self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4"); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as file:
            file.seek(start); remaining = length
            while remaining:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk: break
                self.wfile.write(chunk); remaining -= len(chunk)

    def _serve_frontend(self, request_path: str) -> None:
        candidate = (self.frontend_dir / (request_path.lstrip("/") or "index.html")).resolve()
        try: candidate.relative_to(self.frontend_dir.resolve())
        except ValueError: self.send_error(HTTPStatus.FORBIDDEN); return
        if not candidate.is_file(): candidate = self.frontend_dir / "index.html"
        if not candidate.is_file(): self.send_error(HTTPStatus.NOT_FOUND, "Build the frontend first: cd frontend; npm run build"); return
        content = candidate.read_bytes(); self.send_response(HTTPStatus.OK); self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"); self.send_header("Content-Length", str(len(content))); self.send_header("Cache-Control", "no-store, no-cache, must-revalidate"); self.send_header("Pragma", "no-cache"); self.send_header("Expires", "0"); self.end_headers(); self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        # A PyInstaller ``--windowed`` executable has no reliable console
        # stream.  The default access logger may therefore raise before a
        # response is sent.  Pipeline events are exposed through the app UI;
        # HTTP access logging is intentionally suppressed here.
        return


def build_server(frontend_dir: Path, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(WorkbenchHandler): pass
    job_db.init_db()
    Handler.frontend_dir, Handler.jobs = frontend_dir.resolve(), WorkbenchJobs()

    class WorkbenchHTTPServer(ThreadingHTTPServer):
        # 提高连接等待队列（listen 时即生效）：默认 5 在浏览器多连接+轮询并发时会溢出导致偶发断连
        request_queue_size = 128

    return WorkbenchHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the local workbench and Python pipeline bridge.")
    parser.add_argument("--frontend", type=Path, default=ASSET_ROOT / "frontend" / "dist"); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv); server = build_server(args.frontend, args.host, args.port)
    print(f"Workbench: http://{args.host}:{args.port}", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: return 0
    finally: server.server_close()


if __name__ == "__main__": raise SystemExit(main())
