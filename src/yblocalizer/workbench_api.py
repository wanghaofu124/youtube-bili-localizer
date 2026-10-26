"""Local HTTP bridge used by the React workbench and packaged EXE.

The bridge owns filesystem access. The browser receives only local API routes
and never sees API keys, cookies, or arbitrary absolute paths.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import mimetypes
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
import webbrowser
from contextlib import contextmanager
from dataclasses import dataclass, field
from .llm_usage import LlmUsageTracker, usage_session, update_usage_index
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import parse_qs, unquote, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

# This must happen before importing modules which may read the template store.
# A packaged GUI can be launched with a protected system directory as its
# current working directory, so relative "data" paths are never safe here.
_DEFAULT_USER_DATA_ROOT = Path(os.environ.get("APPDATA") or Path.home()) / "YouTubeBiliLocalizer"
os.environ.setdefault("YBLOCALIZER_DATA_DIR", str(_DEFAULT_USER_DATA_ROOT))
_DEFAULT_USER_DATA_ROOT.mkdir(parents=True, exist_ok=True)
load_dotenv(_DEFAULT_USER_DATA_ROOT / ".env")

from .download import DownloadProposalMismatch, DownloadResolutionMismatch, YouTubeAccessError, get_video_metadata, list_playlist_entries, po_token_provider_status
from .subscriptions import add_channel, refresh_weekly_hot, remove_channel, set_subscriptions_enabled, subscription_view, weekly_snapshot
from .dub import DUB_MIX_MODES
from .dependencies import (
    voxcpm_status,
    OLLAMA_MODEL_TAGS,
    DependencyManager,
    clear_dependency_cache,
    dependency_statuses,
    ollama_local_status,
    require_whisper_model,
    resolve_command,
)
from .models import Segment, load_segments, save_segments
from .ocr_subtitle import available_ocr_languages, ocr_language_ready
from .pipeline import (
    CancellationError,
    PipelineOptions,
    _estimate_audio_offset,
    _shift_segments,
)
from . import __version__
from .diagnostics import configure_logging, export_diagnostics_zip, install_excepthooks, log_frontend_error
from .workbench_config import (
    InputValidationError,
    MAX_DESCRIPTION_LENGTH,
    MAX_TAG_LENGTH,
    MAX_TAGS,
    MAX_TITLE_LENGTH,
    capabilities,
    default_options,
    normalize_options,
    options_for_public,
    options_for_storage,
    options_from_storage,
    preflight,
    require_boolean,
    require_text,
)
from .runtime import CancellationRequested, CancellationToken, PipelineContext, PipelineEvent, STAGE_LABELS
from .job_service import transition_job
from .publish_bili import assist_publish, open_upload_page
from .publish_douyin import assist_publish as assist_douyin_publish
from .publish_text import (
    build_publish_description,
    delete_custom_template,
    get_all_templates,
    normalize_publish_platform,
    save_custom_template,
)
from .translate import ai_provider_request_info, compact_overflowing_subtitles, generate_publish_metadata, save_publish_metadata, translate_single_cue
from .render import burn_subtitles, subtitle_line_capacity
from .subtitle import translated_layout_issues
from .subtitle_sentence_plan import (
    MIN_ESTIMATED_SLOT_DURATION_SEC,
    subtitle_density_overflow_indices,
    subtitle_high_reading_speed_indices,
)
from .render import render_encoder_status
from .media import audio_stream_channels, extract_audio
from .performance import ReviewResourceBusy, cpu_thread_limit, ffmpeg_thread_args, lower_process_priority, normalize_resource_profile, review_cuda_memory
from .storage import delete_paths, format_bytes, resolve_task_title, scan_outputs
from .subtitle import write_srt
from .transcribe import WhisperCudaOutOfMemory, cuda_runtime_status
from .util import allocate_job_folder_name, run as run_command
from . import db as job_db
from .workflow import (
    CORE_STAGES,
    WORKFLOW_STAGES,
    WORKFLOW_VERSION,
    STAGE_LABELS as WORKFLOW_STAGE_LABELS,
    WorkflowArtifacts,
    atomic_write_manifest,
    commit_rendered_candidate,
    invalidate_downstream,
    invalidation_stage,
    new_stage_states,
    run_stage as run_workflow_stage,
    validate_media,
    validate_segments,
    validate_srt,
    validate_stage,
)
from . import glossary as glossary_module
from .glossary import DEFAULT_GLOSSARY_PATH, global_glossary_path, glossary_hash, load_glossary, read_glossary_file, save_glossary
from .asr_review import _atomic_json, _file_fingerprint, _path_signature, next_round, prepare_session, read_session, reset_interrupted_round, resolve_review_language, session_path, start_round, subtitle_commit_path, update_session

LOGGER = logging.getLogger("yblocalizer.workbench")
TUTORIAL_VERSION = 1
TUTORIAL_PREFERENCES_MAX_BYTES = 4096
JOB_ID_PATTERN = r"(?=[^/]{1,128}(?:/|$))[\w-]+(?:\.[\w-]+)*"


def _safe_job_id(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 128 or not re.fullmatch(JOB_ID_PATTERN, value):
        return False
    stem = value.split(".", 1)[0].upper()
    return stem not in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


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
LOG_ROOT = USER_DATA_ROOT / "logs"
LOG_ROOT.mkdir(parents=True, exist_ok=True)
if not logging.getLogger().handlers:
    configure_logging(LOG_ROOT)
install_excepthooks()
DEMO_VIDEO = ASSET_ROOT / "demo" / "authorized-demo-10s.mp4"
OUTPUT_ROOT = USER_DATA_ROOT / "outputs" / "workbench_demo"
glossary_module.GLOBAL_ROOT = OUTPUT_ROOT
UPLOAD_ROOT = USER_DATA_ROOT / "outputs" / "workbench_uploads"
MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
DOUYIN_UPLOAD_URL = "https://creator.douyin.com/creator-micro/content/upload"


def _read_publish_metadata(path: Path | None) -> dict[str, Any]:
    title, tags = "", []
    if path and path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            title = str(data.get("title") or "").strip()
            tags = [str(tag).strip() for tag in (data.get("tags") or []) if str(tag).strip()]
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return {"title": title, "tags": tags, "has_metadata": bool(title or tags)}
MAX_JSON_BODY = 1024 * 1024
MAX_URL_LENGTH = 2048
MAX_HTTP_CONNECTIONS = 32
UPLOAD_RESERVE_BYTES = 64 * 1024 * 1024


class PayloadReadError(InputValidationError):
    def __init__(self, message: str, *, status: HTTPStatus, code: str) -> None:
        super().__init__(message, code=code)
        self.status = status


class ActiveTaskDeletionError(RuntimeError):
    """Deletion conflicts with a currently running task."""


def _validate_public_http_url(value: Any, field: str = "source_url") -> str:
    url = require_text(value, field, MAX_URL_LENGTH, allow_empty=False)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise InputValidationError(
            "请输入有效的公开 HTTP 或 HTTPS 视频链接。", code="invalid_url", field=field,
        )
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        raise InputValidationError(
            "视频链接不能指向本机或局域网地址。", code="private_url", field=field,
        )
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise InputValidationError(
            "视频链接不能指向本机或局域网地址。", code="private_url", field=field,
        )
    return url


def _validate_imported_media(path: Path) -> tuple[float, int | None, int | None]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("视频文件为空或不存在。")
    duration, width, height = _probe_media(path)
    if duration is None or duration <= 0 or width is None or height is None:
        raise ValueError("文件不是有效视频，或 FFprobe 无法读取其中的视频流。")
    return duration, width, height


def _ffconcat_path(path: Path) -> str:
    """Escape a path for FFmpeg's concat-demuxer single-quoted syntax."""
    return path.resolve().as_posix().replace("'", "'\\''")


def _validate_tags(value: Any, field: str = "tags") -> list[str]:
    if not isinstance(value, list):
        raise InputValidationError(f"{field} 必须是文本数组。", code="invalid_type", field=field)
    if len(value) > MAX_TAGS:
        raise InputValidationError(
            f"标签最多允许 {MAX_TAGS} 个。", code="too_many_items", field=field,
            limits={"max_items": MAX_TAGS},
        )
    result: list[str] = []
    for index, item in enumerate(value):
        text = require_text(item, f"{field}[{index}]", MAX_TAG_LENGTH)
        if text:
            result.append(text)
    return result


def _validate_device_precision(device_value: Any, compute_value: Any) -> tuple[str, str]:
    device = require_text(device_value, "device", 16, allow_empty=False).lower()
    compute = require_text(compute_value, "compute_type", 32, allow_empty=False).lower()
    if device not in {"cuda", "cpu", "auto"}:
        raise InputValidationError("推理设备必须是 cuda、cpu 或 auto。", code="invalid_choice", field="device")
    if compute not in {"int8", "int8_float16", "default", "float16", "float32"}:
        raise InputValidationError("计算精度选项无效。", code="invalid_choice", field="compute_type")
    return device, compute
def _safe_options(value: Any) -> dict[str, Any]:
    """Compatibility adapter; defaults and validation live in one module."""
    return normalize_options(value, str(OUTPUT_ROOT))


def _resolve_output_root(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _path_contains_reparse(path: Path) -> bool:
    """Return true for symlinks/junctions anywhere in a deletion target."""
    try:
        candidates = [path]
        current = path.parent
        while current != current.parent:
            candidates.append(current)
            current = current.parent
        if path.is_dir():
            candidates.extend(path.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink():
                return True
            try:
                if int(candidate.stat(follow_symlinks=False).st_file_attributes) & 0x400:
                    return True
            except (AttributeError, FileNotFoundError):
                pass
        return False
    except OSError:
        return True


def _public_options(options: dict[str, Any]) -> dict[str, Any]:
    return options_for_public(options)


def _cookies_file_has_login(path: Path) -> bool:
    """cookies.txt 是否包含主登录凭据（SID/HSID/LOGIN_INFO）。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "\tSID\t" in text or "\tHSID\t" in text or "LOGIN_INFO" in text


def _probe_translation_service(provider: str, model: str | None = None) -> tuple[bool, str]:
    """Perform a small authenticated models request before media download."""
    normalized = provider.strip().lower()
    if normalized == "none":
        return True, "未启用在线翻译，不需要 API 连接。"
    if normalized == "ollama":
        local = ollama_local_status()
        if not local["runtime_available"]:
            return False, str(local["runtime_message"])
        selected_model = model or "qwen3.5:4b"
        model_id = next((key for key, tag in OLLAMA_MODEL_TAGS.items() if tag == selected_model), None)
        if not model_id or not local["models"].get(model_id, False):
            return False, f"本地模型 {selected_model} 尚未下载；请在依赖中心下载后重新检查。"
        return True, f"Ollama 与本地模型 {selected_model} 可用；翻译请求只发送到本机。"
    key_name = "DEEPSEEK_API_KEY" if normalized == "deepseek" else "OPENAI_API_KEY"
    api_key = os.getenv(key_name, "").strip()
    if not api_key:
        return False, f"缺少 {provider} API Key。"
    default_model = os.getenv(
        "DEEPSEEK_TRANSLATE_MODEL" if normalized == "deepseek" else "OPENAI_TRANSLATE_MODEL",
        "deepseek-v4-flash" if normalized == "deepseek" else "gpt-4.1-mini",
    )
    expected_model = model or default_model
    endpoint = "https://api.deepseek.com/models" if normalized == "deepseek" else "https://api.openai.com/v1/models"
    request = Request(endpoint, headers={"Authorization": f"Bearer {api_key}", "User-Agent": "YouTubeBiliLocalizer/preflight"})
    try:
        with urlopen(request, timeout=8) as response:
            if 200 <= int(response.status) < 300:
                try:
                    payload = json.loads(response.read().decode("utf-8"))
                    ids = {str(item.get("id")) for item in (payload.get("data") or []) if isinstance(item, dict)}
                except (ValueError, UnicodeDecodeError, TypeError):
                    ids = set()
                compatible_ids = {expected_model}
                if normalized == "deepseek":
                    compatible_ids.add({"deepseek-v4-flash": "deepseek-flash"}.get(expected_model, expected_model))
                if expected_model and ids and not compatible_ids.intersection(ids):
                    return False, f"翻译服务已连接，但模型 {expected_model} 不在 /models 返回列表中。"
                if expected_model and not ids:
                    return False, "翻译服务 /models 返回无效，无法核验所选模型。"
                return True, "API Key、模型与网络连接验证通过。"
            return False, f"翻译服务返回 HTTP {response.status}。"
    except HTTPError as exc:
        if exc.code in {401, 403}:
            return False, "API Key 无效或没有访问权限，请在设置中更新。"
        return False, f"翻译服务暂时不可用（HTTP {exc.code}）。"
    except (URLError, OSError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        return False, f"无法连接翻译服务：{reason}"


def _read_srt_rows(path: Path) -> list[tuple[float, float, str]]:
    if not path.is_file():
        return []
    rows: list[tuple[float, float, str]] = []
    for block in re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig")):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timing_index = next((index for index, line in enumerate(lines) if "-->" in line), -1)
        if timing_index < 0:
            continue
        try:
            start_raw, end_raw = (part.strip() for part in lines[timing_index].split("-->", 1))
            def seconds(value: str) -> float:
                hours, minutes, rest = value.replace(",", ".").split(":")
                return int(hours) * 3600 + int(minutes) * 60 + float(rest)
            rows.append((seconds(start_raw), seconds(end_raw), "\n".join(lines[timing_index + 1:])))
        except (ValueError, IndexError):
            continue
    return rows


def _demo_cues(source_path: Path, translated_path: Path) -> list[dict[str, Any]]:
    source_rows, translated_rows = _read_srt_rows(source_path), _read_srt_rows(translated_path)
    return [
        {"id": index, "start": start, "end": end, "source": text,
         "translated": translated_rows[index][2] if index < len(translated_rows) else text, "kind": "demo"}
        for index, (start, end, text) in enumerate(source_rows)
    ]


def _upsert_env(path: Path, updates: dict[str, str]) -> None:
    """Upsert .env lines by key. Handles UTF-8 BOM written by other tools."""
    existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    normalized = [line.lstrip("\ufeff") for line in existing.splitlines()]
    remaining = [line for line in normalized if line.partition("=")[0].strip() not in updates]
    remaining.extend(f"{key}={value}" for key, value in updates.items())
    path.write_text("\n".join(remaining).rstrip() + "\n", encoding="utf-8")


_tutorial_preferences_lock = threading.Lock()


def _tutorial_preferences_path() -> Path:
    # Resolve at request time so isolated preview/test data roots are honored.
    return USER_DATA_ROOT / "tutorial-preferences.json"


def _read_tutorial_preferences() -> dict[str, int]:
    path = _tutorial_preferences_path()
    try:
        if not path.is_file() or path.stat().st_size > TUTORIAL_PREFERENCES_MAX_BYTES:
            if path.is_file():
                LOGGER.warning("Ignoring missing or oversized tutorial preferences file.")
            return {"dismissedVersion": 0}
        value = json.loads(path.read_text(encoding="utf-8"))
        version = value.get("dismissedVersion") if isinstance(value, dict) else None
        if type(version) is not int or version < 0:
            LOGGER.warning("Ignoring invalid tutorial preferences file.")
            return {"dismissedVersion": 0}
        return {"dismissedVersion": version}
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        LOGGER.warning("Ignoring unreadable tutorial preferences file.")
        return {"dismissedVersion": 0}


def _write_tutorial_preferences(payload: dict[str, Any]) -> None:
    if set(payload) != {"dismissedVersion"}:
        raise InputValidationError("教程偏好字段无效。", code="invalid_payload")
    version = payload.get("dismissedVersion")
    if type(version) is not int or version != TUTORIAL_VERSION:
        raise InputValidationError("教程偏好版本无效。", code="invalid_value", field="dismissedVersion")
    path = _tutorial_preferences_path()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:10]}.tmp")
    with _tutorial_preferences_lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps({"dismissedVersion": version}, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise


def _normalize_cut_ranges(value: Any, duration: float) -> list[tuple[float, float]]:
    """Validate and merge cut ranges; clip to [0, duration]."""
    if not isinstance(value, list):
        raise RuntimeError("cut_ranges must be a list of [start, end] pairs.")
    raw: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RuntimeError("Each cut range must be [start, end].")
        try:
            if any(type(part) not in {int, float} for part in item):
                raise TypeError
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            raise RuntimeError("Cut range values must be numbers (seconds).") from None
        if not math.isfinite(start) or not math.isfinite(end):
            raise RuntimeError("Cut range values must be finite numbers.")
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


def _classify_job_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, DownloadResolutionMismatch):
        return exc.code, "实际文件画质与下载器选择不一致，应用已停止，不会继续处理低清素材；请重试获取素材。"
    if isinstance(exc, DownloadProposalMismatch):
        return exc.code, str(exc)
    if isinstance(exc, WhisperCudaOutOfMemory):
        return exc.code, "CUDA 显存不足，未自动切换设备。可确认后改用 CPU / int8，或选择已安装的较小模型，仅重新提取字幕。"
    if isinstance(exc, YouTubeAccessError):
        return exc.code, exc.suggested_action
    text = str(exc).lower()
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "translator_rate_limited", "翻译服务的有界自动重试已用尽。等待一会儿后，只重试翻译阶段；不会重复下载或转写。"
    if "产物校验失败" in text:
        return "stage_output_invalid", "本阶段结果未保存为完成检查点。查看完整日志后重试该阶段；之前有效的阶段成果会保留。"
    if "ffmpeg" in text:
        return "ffmpeg_unavailable", "安装 FFmpeg 并加入 PATH，然后重新运行配置检查。"
    if "api key" in text or "authentication" in text:
        return "translator_auth_failed", "在设置中重新保存翻译 API Key。"
    if "cookie" in text or "not a bot" in text:
        return "youtube_auth_required", "需要登录的视频请更新 cookies.txt；公开内容可先不使用 Cookies 并开启自动浏览器验证。"
    if "cuda" in text or "out of memory" in text:
        return "compute_failed", "切换到 CPU + int8，或选择更小的 Whisper 模型。"
    return "pipeline_failed", "打开完整日志查看失败阶段，修正配置后重试。"


def _carry_quality_baseline(proposal: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    """A repeated preparation must not silently forget an earlier 4K promise."""
    updated = dict(proposal)
    old = previous or {}
    previous_width = int(old.get("baseline_width") or old.get("width") or 0)
    previous_height = int(old.get("baseline_height") or old.get("height") or 0)
    width, height = int(updated.get("width") or 0), int(updated.get("height") or 0)
    updated["baseline_width"] = max(previous_width, width)
    updated["baseline_height"] = max(previous_height, height)
    if width < updated["baseline_width"] or height < updated["baseline_height"]:
        updated["requires_consent"] = True
    return updated


def _recovery_for_error(exc: Exception, stage: str | None, job: "DemoJob") -> dict[str, Any] | None:
    if stage == "extract" and job.device == "cuda" and isinstance(exc, WhisperCudaOutOfMemory):
        return {
            "id": "retry_extract_with_lower_resource",
            "stage": "extract",
            "requires_confirmation": True,
            "impact": "可切换为 CPU / int8（速度可能变慢），或改用已安装的较小 Whisper 模型；只重新提取字幕，不重新下载视频。",
        }
    return None


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
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True, timeout=15)
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
            "source_url": self.source_url,
        }


@dataclass
class DemoJob:
    id: str
    material: Material
    device: str
    compute_type: str
    options: dict[str, Any] = field(default_factory=dict)
    output_root: Path = field(default_factory=lambda: OUTPUT_ROOT)
    status: str = "draft"
    stage: str = "等待准备检查"
    progress: int = 0
    logs: list[str] = field(default_factory=list)
    log_sequence: int = 0
    error: str | None = None
    error_code: str | None = None
    suggested_action: str | None = None
    result: dict[str, str] | None = None
    work_dir: Path | None = None
    raw_video: Path | None = None
    rendered_video: Path | None = None
    edited_segments: Path | None = None
    duration_override: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    created_at: float = field(default_factory=time.time)
    workflow_version: int = WORKFLOW_VERSION
    current_stage: str | None = None
    next_stage: str | None = "acquire"
    auto_run: bool = False
    stages: dict[str, dict[str, Any]] = field(default_factory=new_stage_states)
    checks: list[dict[str, Any]] = field(default_factory=list)
    artifacts: WorkflowArtifacts = field(default_factory=WorkflowArtifacts)
    checkpoint_validation: str = "pending"
    llm_usage: dict[str, Any] | None = None
    llm_usage_index: dict[str, Any] | None = field(default=None, repr=False)
    llm_run_token: str | None = field(default=None, repr=False)
    cancellation: CancellationToken = field(default_factory=CancellationToken, repr=False)

    def add_log(self, message: str) -> None:
        self._append_log(message)
        self._write_log(message)

    def apply_event(self, event: PipelineEvent) -> None:
        self.stage = STAGE_LABELS[event.stage]
        self.progress = max(self.progress, event.progress)
        if not self.logs or self.logs[-1] != event.message:
            self._append_log(event.message)
            self._write_log(event.message)

    def _append_log(self, message: str) -> None:
        self.logs.append(message)
        self.log_sequence += 1
        if len(self.logs) > 2000:
            del self.logs[:-2000]

    def _write_log(self, message: str) -> None:
        try:
            with (LOG_ROOT / f"{self.id}.log").open("a", encoding="utf-8") as stream:
                stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
        except OSError:
            LOGGER.warning("Could not append per-job log for %s", self.id)

    def refresh_llm_usage(self) -> None:
        """Read the persisted attempt pointer once when restoring a job."""
        index_path = self.artifacts.path("llm_usage_index_path")
        if index_path:
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(index, dict) and index.get("schema_version") == 1 and isinstance(index.get("sessions"), dict):
                    self.llm_usage_index = index
                    self.artifacts.llm_usage_total = copy.deepcopy(index.get("total"))
            except (OSError, ValueError, TypeError):
                LOGGER.warning("Could not restore cumulative LLM usage for %s", self.id)
        path = self.artifacts.path("llm_usage_path")
        if not path:
            return
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("schema_version") == 1:
                self.llm_usage = value
        except (OSError, ValueError, TypeError):
            return

    def snapshot(self, *, log_after: int | None = None, log_limit: int = 200) -> dict[str, Any]:
        material = self.material.snapshot()
        if self.duration_override is not None:
            material["duration_seconds"] = self.duration_override
        limit = max(0, min(2000, int(log_limit)))
        if log_after is None:
            public_logs = self.logs[-limit:] if limit else []
        else:
            first_sequence = max(1, self.log_sequence - len(self.logs) + 1)
            first_requested = max(first_sequence, int(log_after) + 1)
            offset = max(0, first_requested - first_sequence)
            public_logs = self.logs[offset:offset + limit] if limit else []
        recovery = next((
            self.stages.get(stage, {}).get("recovery")
            for stage in WORKFLOW_STAGES
            if self.stages.get(stage, {}).get("status") == "failed"
            and self.stages.get(stage, {}).get("recovery")
        ), None)
        asr_summary: dict[str, Any] = {}
        if self.artifacts.asr_evidence:
            try:
                evidence = json.loads(Path(self.artifacts.asr_evidence).read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                evidence = {}
            if isinstance(evidence, dict):
                asr_summary = {
                    "detected_language": evidence.get("detected_language"),
                    "language_probability": (
                        evidence.get("language_probability")
                        if isinstance(evidence.get("language_probability"), (int, float))
                        and math.isfinite(float(evidence.get("language_probability")))
                        and 0.0 <= float(evidence.get("language_probability")) <= 1.0
                        else None
                    ),
                }
        usage_summary = copy.deepcopy(self.llm_usage)
        return {
            "id": self.id, "status": self.status, "stage": self.stage, "progress": self.progress,
            "logs": public_logs, "log_cursor": self.log_sequence, "log_total": self.log_sequence,
            "error": self.error, "error_code": self.error_code,
            "suggested_action": self.suggested_action, "recovery": recovery, "result": self.result,
            "device": self.device, "compute_type": self.compute_type, "material": material,
            "options": _public_options(self.options),
            "started_at": self.started_at,
            "finished_at": self.finished_at, "created_at": self.created_at,
            "elapsed_seconds": round((self.finished_at or time.time()) - self.started_at, 1) if self.started_at else 0,
            "workflow_version": self.workflow_version,
            "stages": self.stages if self.options.get("dub_enabled") else {name: row for name, row in self.stages.items() if name != "dub"},
            "checks": self.checks,
            "current_stage": self.current_stage,
            "next_stage": self.next_stage,
            "can_resume": any(self.stages[name].get("status") == "interrupted" for name in CORE_STAGES),
            "auto_run": self.auto_run,
            "artifacts": {**self.artifacts.public_summary(), "dub_video": {
                "available": bool(self.artifacts.dub_video and Path(self.artifacts.dub_video).is_file()),
                "path": public_path(self.artifacts.path("dub_video")) if self.artifacts.dub_video else None,
            }},
            "subtitle_extraction": {
                "mode": self.artifacts.subtitle_extraction_mode,
                "ocr_status": self.artifacts.ocr_status,
                "message": self.artifacts.ocr_message,
                **asr_summary,
            },
            "content_warnings": list(self.artifacts.content_warnings or []),
            "artifact_revision": self.artifacts.revision,
            "edit_state": self.artifacts.edit_state,
            "checkpoint_validation": self.checkpoint_validation,
            "publish_copy": _read_publish_metadata(self.artifacts.path("publish_metadata")),
            "quality_proposal": self.options.get("quality_proposal"),
            "quality_consent": bool(self.options.get("quality_consent")),
            "llm_usage": usage_summary,
            "llm_usage_total": copy.deepcopy(self.artifacts.llm_usage_total),
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


def _restored_created_at(record: dict[str, Any]) -> float:
    for key in ("created_at", "started_at", "updated_at"):
        try:
            value = float(record.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return 0.0


class WorkbenchJobs:
    """Application service for live workbench jobs and persisted snapshots."""

    def __init__(self, *, restore: bool = False) -> None:
        demo_duration, demo_width, demo_height = _probe_media(DEMO_VIDEO)
        self._lock = threading.RLock()
        self._jobs: dict[str, DemoJob] = {}
        self._materials: dict[str, Material] = {
            "demo": Material("demo", DEMO_VIDEO, "authorized-demo-10s.mp4", demo_duration or 10, demo_width or 1280, demo_height or 720, True, True)
        }
        self._running_id: str | None = None
        self._subtitle_model_action_id: str | None = None
        self._preparing_ids: set[str] = set()
        self._worker: threading.Thread | None = None
        self._voice_review_worker: threading.Thread | None = None
        self._voice_review_process: subprocess.Popen[str] | None = None
        # IDs removed from history remain tombstoned for this process.  This
        # prevents a late worker persist from resurrecting a just-deleted row.
        self._deleted_history_ids: set[str] = set()
        self._native_publish_files: dict[str, Path] = {}
        self._publish_session = PublishSession()
        self._batch_items: list[dict[str, Any]] = []
        self._batch_thread: threading.Thread | None = None
        self._batch_job_id: str | None = None
        if restore:
            self._restore_jobs()
            self._load_batch()

    def _restore_jobs(self) -> None:
        """Restore staged tasks without automatically resuming heavy work."""
        for record in reversed(job_db.list_jobs(None)):
            artifacts = WorkflowArtifacts.from_dict(record.get("artifacts"))
            source_url = str(record.get("source_url") or "") or None
            source_path = artifacts.path("raw_video") if not source_url else None
            if artifacts.source_kind == "file" and artifacts.source:
                candidate = Path(artifacts.source)
                source_path = candidate if candidate.is_file() else source_path
            material = Material(
                str(record.get("material_id") or f"restored-{record['id']}"), source_path,
                str(record.get("title") or (source_path.name if source_path else "历史任务")),
                None, None, None, True, False, source_url,
            )
            options = options_from_storage(record.get("options"), str(record.get("output_dir") or OUTPUT_ROOT))
            states = record.get("stage_states") if isinstance(record.get("stage_states"), dict) else {}
            merged_states = new_stage_states(bool(options.get("publish_to_bilibili")))
            for name, value in states.items():
                if name in merged_states and isinstance(value, dict):
                    merged_states[name].update(value)
            current = str(record.get("current_stage") or "") or None
            if str(record.get("status")) == "interrupted" and current in merged_states:
                merged_states[current]["status"] = "interrupted"
            job = DemoJob(
                id=str(record["id"]), material=material,
                device=str(record.get("device") or "cpu"), compute_type=str(record.get("compute_type") or "int8"),
                options=options, output_root=Path(str(record.get("output_dir") or OUTPUT_ROOT)).parent,
                status=str(record.get("status") or "draft"), stage=str(record.get("stage") or "等待准备检查"),
                progress=int(record.get("progress") or 0), error=record.get("error"),
                work_dir=Path(str(record["output_dir"])) if record.get("output_dir") else None,
                raw_video=artifacts.path("raw_video"), rendered_video=artifacts.path("rendered_video"),
                edited_segments=artifacts.path("translated_segments") if (artifacts.path("translated_segments") and ".edited." in artifacts.path("translated_segments").name) else None,
                started_at=record.get("started_at"), finished_at=record.get("finished_at"),
                # Undated legacy/test rows must not masquerade as newly-created
                # tasks and push the user's real recent jobs out of the first
                # page of the recovery picker.
                created_at=_restored_created_at(record),
                workflow_version=int(record.get("workflow_version") if record.get("workflow_version") is not None else 0),
                current_stage=current, next_stage=str(record.get("next_stage") or "") or None,
                auto_run=bool(record.get("auto_run")), stages=merged_states,
                # Preparation checks describe a previous process/environment;
                # never trust them after restart.  Keep the persisted snapshot
                # intact in storage, but require an explicit prepare call.
                checks=[], artifacts=artifacts,
            )
            job.refresh_llm_usage()
            if job.work_dir and any((artifacts.source_srt, artifacts.translated_srt, artifacts.rendered_video)):
                job.result = {
                    "output_dir": public_path(job.work_dir) or "",
                    "source_srt": public_path(artifacts.path("source_srt")) or "" if artifacts.path("source_srt") else "",
                    "translated_srt": public_path(artifacts.path("translated_srt")) or "" if artifacts.path("translated_srt") else "",
                    "rendered_video": public_path(artifacts.path("rendered_video")) or "" if artifacts.path("rendered_video") else "",
                }
            log_path = LOG_ROOT / f"{job.id}.log"
            if log_path.is_file():
                try:
                    job.logs = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
                    job.log_sequence = len(job.logs)
                except OSError:
                    pass
            self._jobs[job.id] = job
            if job.work_dir and job.work_dir.is_dir():
                # Unfinished subtitle/render candidates from an interrupted run.
                keep = {job.rendered_video, artifacts.path("source_srt"), artifacts.path("translated_srt")}
                for leftover in job.work_dir.glob("*.next-*"):
                    if leftover.name.startswith(("subtitle.next-", "rendered.next-")) and leftover not in keep:
                        leftover.unlink(missing_ok=True)
            if job.work_dir:
                session = read_session(job.work_dir)
                # A published subtitle commit is authoritative even if the
                # process stopped between DB persistence and the final review
                # session checkpoint.  Reconcile only the same immutable ASR
                # session; never write candidates from an unrelated session.
                commit_file = artifacts.path("subtitle_commit")
                if commit_file and commit_file.is_file() and session:
                    try:
                        receipt = json.loads(commit_file.read_text(encoding="utf-8"))
                        translated_file = artifacts.path("translated_segments")
                        receipt_ok = (
                            isinstance(receipt, dict)
                            and str(receipt.get("jobId")) == job.id
                            and int(receipt.get("toRevision")) == int(artifacts.revision)
                            and translated_file is not None and translated_file.is_file()
                            and hmac.compare_digest(str(receipt.get("newFingerprint") or ""), hashlib.sha256(translated_file.read_bytes()).hexdigest())
                            and str(receipt.get("sessionFingerprint") or "") == str(session.get("session_fingerprint") or "")
                        )
                        if receipt_ok and int(session.get("subtitle_revision") or 0) < int(receipt["toRevision"]):
                            committed_ids = {str(item.get("candidateId")) for item in receipt.get("committed", []) if isinstance(item, dict)}
                            for item in session.get("result_candidates", []):
                                if str(item.get("id")) in committed_ids: item["status"] = "committed"
                            update_session(job.work_dir, candidates=session.get("candidates", []), result_candidates=session.get("result_candidates", []), subtitle_revision=int(receipt["toRevision"]), subtitle_fingerprint=str(receipt.get("newFingerprint")))
                            session = read_session(job.work_dir) or session
                    except (OSError, TypeError, ValueError, KeyError):
                        pass
                if session and (session.get("status") == "running" or any(item.get("status") == "running" for item in session.get("candidates", []) if isinstance(item, dict))):
                    try:
                        reset_interrupted_round(job.work_dir, status="pending", error="上次进程已结束，当前复核窗口可继续。")
                    except OSError:
                        LOGGER.warning("Could not reconcile stale ASR review session for %s", job.id)

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
        try:
            with target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            duration, width, height = _validate_imported_media(target)
        except Exception:
            target.unlink(missing_ok=True)
            raise
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
        duration, width, height = _validate_imported_media(resolved)
        material = Material(uuid.uuid4().hex[:12], resolved, resolved.name, duration, width, height, True)
        with self._lock:
            self._materials[material.id] = material
        return material

    def _persist(self, job: DemoJob) -> None:
        """Write SQLite authoritatively; manifest is only a rebuildable mirror."""
        with self._lock:
                if job.id in self._deleted_history_ids:
                    return
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
                    "options": options_for_storage(job.options),
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                    "workflow_version": job.workflow_version,
                    "current_stage": job.current_stage,
                    "next_stage": job.next_stage,
                    "auto_run": job.auto_run,
                    "stage_states": job.stages,
                    "checks": job.checks,
                    "artifacts": job.artifacts.to_dict(),
                    "updated_at": time.time(),
                }
                # Keep the DB write under the same lock as the tombstone
                # check; otherwise a worker could upsert after deletion.
                job_db.record_job(**snapshot)
                if job.work_dir:
                    try:
                        atomic_write_manifest(job.work_dir / "job_manifest.json", {
                        "workflow_version": job.workflow_version,
                        "job_id": job.id,
                        "status": job.status,
                        "current_stage": job.current_stage,
                        "next_stage": job.next_stage,
                        "auto_run": job.auto_run,
                        "stages": job.stages,
                        "checks": job.checks,
                        "llm_usage": job.llm_usage,
                        "artifacts": job.artifacts.to_dict(),
                        "options": options_for_storage(job.options),
                        "updated_at": time.time(),
                        })
                    except Exception as exc:
                        LOGGER.warning("Could not mirror manifest for %s: %s", job.id, exc)

    def _reconcile_subtitle_commit(self, job: DemoJob) -> dict[str, Any] | None:
        """Advance only a matching review session after a published save."""
        if not job.work_dir:
            return None
        session = read_session(job.work_dir)
        receipt_path = job.artifacts.path("subtitle_commit")
        if not session or not receipt_path or not receipt_path.is_file():
            return session
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            translated = job.artifacts.path("translated_segments")
            valid = (
                isinstance(receipt, dict) and str(receipt.get("jobId")) == job.id
                and int(receipt.get("toRevision")) == int(job.artifacts.revision)
                and translated is not None and translated.is_file()
                and hmac.compare_digest(str(receipt.get("newFingerprint") or ""), hashlib.sha256(translated.read_bytes()).hexdigest())
                and str(receipt.get("sessionFingerprint") or "") == str(session.get("session_fingerprint") or "")
            )
            if not valid:
                return session
            if int(session.get("subtitle_revision") or 0) < int(receipt["toRevision"]):
                committed = {str(item.get("candidateId")) for item in receipt.get("committed", []) if isinstance(item, dict)}
                for item in session.get("result_candidates", []):
                    if str(item.get("id")) in committed: item["status"] = "committed"
                update_session(job.work_dir, result_candidates=session.get("result_candidates", []), subtitle_revision=int(receipt["toRevision"]), subtitle_fingerprint=str(receipt.get("newFingerprint")))
                return read_session(job.work_dir) or session
        except (OSError, TypeError, ValueError, KeyError):
            return session
        return session

    def _review_media_is_current(self, job: DemoJob, session: dict[str, Any]) -> bool:
        expected = session.get("source_video_signature")
        current = _path_signature(job.artifacts.path("raw_video") or job.raw_video)
        return bool(expected and current and hmac.compare_digest(str(expected), str(current)))

    @staticmethod
    def _valid_review_wav(path: Path) -> bool:
        try:
            with wave.open(str(path), "rb") as handle:
                return (
                    handle.getnchannels() == 1
                    and handle.getframerate() == 16000
                    and handle.getsampwidth() == 2
                    and handle.getnframes() > 0
                )
        except (OSError, EOFError, wave.Error):
            return False

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
            # Only flip to waiting_review on confirmed pre-submit readiness.
            # Progress heartbeats may contain "上传完成" while the form is still filling.
            if (
                "pre-submit" in lowered
                or "reached the pre-submit step" in lowered
                or "尚未正式投稿" in message
                or "已尽量填好" in message
            ):
                session.status = "waiting_review"
                session.message = "已到人工提交前：请检查分区、声明、封面与简介，确认无误后手动投稿。"
            else:
                session.message = message

    def get(self, job_id: str) -> DemoJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def restorable(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return light-weight summaries; checkpoint probing is intentionally lazy."""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            rows = [
                job for job in jobs
                if (job.status in {"draft", "ready", "failed", "cancelled", "interrupted"} or job.next_stage)
                and self._job_is_restorable(job)
            ]
            return [{
                "id": job.id,
                "title": str(job.options.get("title") or job.material.name),
                "status": job.status,
                "next_stage": job.next_stage,
                "updated_at": job.finished_at or job.started_at or job.created_at,
                "checkpoint_validation": job.checkpoint_validation,
            } for job in rows[:max(1, min(100, limit))]]

    @staticmethod
    def _job_is_restorable(job: DemoJob) -> bool:
        """Hide library restore actions for jobs whose outputs were already wiped."""
        if job.material.source_url:
            return True
        if job.work_dir and job.work_dir.exists():
            return True
        if job.raw_video and Path(job.raw_video).expanduser().is_file():
            return True
        raw = job.artifacts.path("raw_video")
        return bool(raw and raw.is_file())

    def load(self, job_id: str) -> DemoJob:
        job = self.get(job_id)
        if not job:
            raise RuntimeError("Task not found.")
        if job.status in {"running", "preparing", "cancelling"}:
            raise RuntimeError("任务仍在运行，不能重复载入。")
        self._verify_checkpoints(job)
        self._persist(job)
        return job

    def forget_history(self, job_ids: set[str]) -> None:
        with self._lock:
            for job_id in job_ids:
                job = self._jobs.get(job_id)
                if job and job.status not in {"running", "preparing", "cancelling"}:
                    self._jobs.pop(job_id, None)
                self._deleted_history_ids.add(job_id)

    @staticmethod
    def _history_fingerprint(plan: dict[str, Any]) -> str:
        payload = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def history_delete_plan(self, job_ids: list[str], mode: str, *, allow_many: bool = False) -> dict[str, Any]:
        """Build a validated, side-effect-free plan for history deletion."""
        mode = str(mode).strip().lower()
        if mode not in {"record_only", "record_and_outputs"}:
            raise ValueError("Deletion mode must be record_only or record_and_outputs.")
        if not isinstance(job_ids, list) or not job_ids or (not allow_many and len(job_ids) > 200):
            raise ValueError("Choose between 1 and 200 task IDs.")
        if any(
            not _safe_job_id(value)
            for value in job_ids
        ):
            raise ValueError("Task IDs contain an invalid value.")
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("Task IDs must not contain duplicates.")
        ids = list(job_ids)
        with self._lock:
            records = {str(row["id"]): row for row in job_db.list_jobs(None)}
            missing = [job_id for job_id in ids if job_id not in records]
            if missing:
                raise RuntimeError("Some selected tasks no longer exist; refresh the task library.")
            active = []
            for job_id in ids:
                row = records[job_id]
                live = self._jobs.get(job_id)
                if row.get("status") in {"queued", "running", "preparing", "cancelling"} or (live and live.status in {"queued", "running", "preparing", "cancelling"}) or self._running_id == job_id or self._subtitle_model_action_id == job_id:
                    active.append(job_id)
            if active:
                raise RuntimeError("运行中的任务不能删除，请先取消并等待当前阶段结束。")
            rows: list[dict[str, Any]] = []
            for job_id in ids:
                row = records[job_id]
                item: dict[str, Any] = {"id": job_id, "title": row.get("title") or "未命名任务", "status": row.get("status") or "unknown", "output_dir": str(row.get("output_dir") or ""), "files": [], "bytes": 0, "disposition": "record_only_required", "reason": "未选择删除输出。"}
                if mode == "record_and_outputs":
                    output_text = item["output_dir"]
                    stored_options = row.get("options")
                    configured_value = stored_options.get("output_dir") if isinstance(stored_options, dict) else None
                    reason = None
                    if not output_text:
                        reason = "历史任务没有输出目录，仅删除记录。"
                    elif not isinstance(configured_value, str) or not configured_value.strip():
                        reason = "缺少可信输出根配置，仅删除记录。"
                    else:
                        output = Path(output_text).expanduser()
                        resolved = output.resolve()
                        configured_root = _resolve_output_root(configured_value)
                        expected = (configured_root / job_id).resolve()
                        if resolved == configured_root or resolved != expected:
                            reason = "输出路径与当前配置根或任务 ID 不匹配，仅删除记录。"
                        elif self._history_output_conflict(resolved, job_id, records.values()):
                            reason = "输出目录与其他任务共享或嵌套，仅删除记录。"
                        elif self._local_material_is_protected(row, resolved):
                            reason = "输出包含受保护的本地素材，仅删除记录。"
                        elif _path_contains_reparse(output):
                            reason = "输出目录包含符号链接或 reparse point，仅删除记录。"
                        else:
                            manifest = resolved / "job_manifest.json"
                            if manifest.exists():
                                try:
                                    data = json.loads(manifest.read_text(encoding="utf-8"))
                                except (OSError, ValueError, json.JSONDecodeError):
                                    reason = "任务清单无法验证，仅删除记录。"
                                else:
                                    if str(data.get("job_id") or "") != job_id:
                                        reason = "任务清单与任务 ID 不一致，仅删除记录。"
                            if reason is None:
                                item["disposition"] = "verified_files"
                                item["reason"] = "已验证任务目录、清单和所有权，可删除输出。"
                                if resolved.is_dir():
                                    for file in sorted(resolved.rglob("*")):
                                        if file.is_file():
                                            size = file.stat().st_size
                                            item["files"].append({"path": file.relative_to(resolved).as_posix(), "name": file.name, "size": size, "mtime_ns": file.stat().st_mtime_ns})
                                            item["bytes"] += size
                    if reason is not None:
                        item["reason"] = reason
                rows.append(item)
            plan: dict[str, Any] = {"mode": mode, "job_ids": ids, "tasks": rows, "task_count": len(rows), "file_count": sum(len(row["files"]) for row in rows), "bytes": sum(row["bytes"] for row in rows), "verified_files_count": sum(row["disposition"] == "verified_files" for row in rows), "record_only_required_count": sum(row["disposition"] == "record_only_required" for row in rows)}
            plan["fingerprint"] = self._history_fingerprint(plan)
            return plan

    @staticmethod
    def _history_output_conflict(path: Path, job_id: str, records: Any) -> bool:
        for row in records:
            other_id = str(row.get("id") or "")
            if other_id == job_id or not row.get("output_dir"):
                continue
            try:
                other = Path(str(row["output_dir"])).expanduser().resolve()
                if other == path or path in other.parents or other in path.parents:
                    return True
            except OSError:
                return True
        return False

    def _local_material_is_protected(self, record: dict[str, Any], output: Path) -> bool:
        """Never recursively delete a directory that may contain an imported source."""
        if str(record.get("source_url") or "").strip():
            return False
        values: list[str] = []
        artifacts = record.get("artifacts")
        if isinstance(artifacts, dict):
            for key in ("original_video", "source"):
                value = artifacts.get(key)
                if isinstance(value, str) and value.strip():
                    values.append(value)
        live = self._jobs.get(str(record.get("id") or ""))
        if live and live.material.path:
            values.append(str(live.material.path))
        for value in values:
            try:
                candidate = Path(value).expanduser().resolve()
                if candidate == output or output in candidate.parents:
                    return True
            except OSError:
                return True
        return False

    def output_paths_active(self, paths: list[Path]) -> bool:
        """Check output targets against persisted and live active task state."""
        targets = [path.expanduser().resolve() for path in paths]
        with self._lock:
            active_dirs: list[Path] = []
            for record in job_db.list_jobs(None):
                if record.get("status") not in {"queued", "running", "preparing", "cancelling"} or not record.get("output_dir"):
                    continue
                active_dirs.append(Path(str(record["output_dir"])).expanduser().resolve())
            for job in self._jobs.values():
                if job.status in {"queued", "running", "preparing", "cancelling"} and job.work_dir:
                    active_dirs.append(job.work_dir.expanduser().resolve())
            if self._running_id and self._running_id in self._jobs and self._jobs[self._running_id].work_dir:
                active_dirs.append(self._jobs[self._running_id].work_dir.expanduser().resolve())
            if self._subtitle_model_action_id and self._subtitle_model_action_id in self._jobs and self._jobs[self._subtitle_model_action_id].work_dir:
                active_dirs.append(self._jobs[self._subtitle_model_action_id].work_dir.expanduser().resolve())
        return any(target == active or active in target.parents for target in targets for active in active_dirs)

    def delete_history(self, job_ids: list[str], mode: str, fingerprint: str | None = None, *, allow_many: bool = False) -> dict[str, Any]:
        with self._lock:
            plan = self.history_delete_plan(job_ids, mode, allow_many=allow_many)
            if mode == "record_and_outputs" and not fingerprint:
                raise ValueError("Output deletion requires a non-empty preview fingerprint.")
            if fingerprint and not hmac.compare_digest(str(fingerprint), str(plan["fingerprint"])):
                raise RuntimeError("删除预览已过期，请刷新后重新确认。")
            results: list[dict[str, Any]] = []
            deleted_bytes = 0
            for row in plan["tasks"]:
                job_id = row["id"]
                output_removed = False
                try:
                    if mode == "record_and_outputs" and row.get("disposition") == "verified_files":
                        output = Path(row["output_dir"]).expanduser()
                        if output.exists():
                            delete_paths([output], output.parent)
                        output_removed = True
                    if job_db.delete_jobs([job_id]) != 1:
                        raise RuntimeError("任务记录已发生变化，请刷新后重试。")
                except Exception as exc:
                    LOGGER.exception("Failed to delete history item %s", job_id)
                    results.append({
                        "id": job_id, "title": row["title"],
                        "status": "partial" if output_removed or mode == "record_and_outputs" else "failed",
                        "record_removed": False, "bytes": 0, "error": str(exc),
                    })
                    continue
                self._deleted_history_ids.add(job_id)
                self._jobs.pop(job_id, None)
                bytes_removed = row["bytes"] if mode == "record_and_outputs" else 0
                deleted_bytes += bytes_removed
                results.append({"id": job_id, "title": row["title"], "status": "deleted", "record_removed": True, "bytes": bytes_removed, "error": None})
            deleted_ids = [item["id"] for item in results if item["record_removed"]]
            return {**plan, "deleted": len(deleted_ids), "deleted_ids": deleted_ids, "deleted_bytes": deleted_bytes, "results": results}

    def full_logs(self, job_id: str, limit: int = 5000) -> dict[str, Any]:
        job = self.get(job_id)
        if not job:
            raise RuntimeError("Task not found.")
        safe_limit = max(1, min(20000, int(limit)))
        log_path = LOG_ROOT / f"{job.id}.log"
        if log_path.is_file():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                return {"logs": lines[-safe_limit:], "total": len(lines), "truncated": len(lines) > safe_limit}
            except OSError:
                pass
        return {"logs": job.logs[-safe_limit:], "total": job.log_sequence, "truncated": job.log_sequence > safe_limit}

    def create(self, material_id: str, device: str, compute_type: str, options: dict[str, Any] | None = None) -> DemoJob:
        options = _safe_options(options)
        with self._lock:
            material = self._materials.get(material_id)
            if material is None or material.path is None or not material.path.exists():
                raise RuntimeError("所选素材不存在或已被删除，请重新选择素材。")
            output_root = _resolve_output_root(options["output_dir"])
            job_id = allocate_job_folder_name(
                output_root,
                title=str(options.get("title") or material.name or ""),
                source_url=getattr(material, "source_url", None),
                source_name=material.name,
            )
            job = DemoJob(job_id, material, device, compute_type, options, output_root, work_dir=output_root / job_id)
            job.work_dir.mkdir(parents=True, exist_ok=True)
            job.artifacts.source, job.artifacts.source_kind = str(material.path), "file"
            job.artifacts.original_video = str(material.path)
            job.artifacts.raw_video = str(material.path)
            job.raw_video = material.path
            job.stages["acquire"].update({
                "status": "completed", "progress": 100, "finished_at": time.time(),
                "config_fingerprint": self._config_fingerprint(job),
            })
            job.next_stage = "extract"
            self._jobs[job.id] = job
        self._persist(job)
        return job

    def create_url(self, source_url: str, device: str, compute_type: str, authorized: bool, options: dict[str, Any] | None = None) -> DemoJob:
        """Create a real yt-dlp task directly from an authorized HTTP(S) URL."""
        options = _safe_options(options)
        source_url = _validate_public_http_url(source_url)
        parsed = urlparse(source_url)
        if not authorized:
            raise ValueError("请先确认拥有处理或转载授权，再开始下载。")
        with self._lock:
            host = parsed.netloc.removeprefix("www.")
            material = Material(
                id=f"url-{uuid.uuid4().hex[:10]}", path=None, name=options["title"] or f"{host} video",
                duration_seconds=None, width=None, height=None, authorized=True,
                source_url=source_url,
            )
            output_root = _resolve_output_root(options["output_dir"])
            job_id = allocate_job_folder_name(
                output_root,
                title=str(options.get("title") or ""),
                source_url=source_url,
                source_name=host,
            )
            job = DemoJob(job_id, material, device, compute_type, options, output_root, work_dir=output_root / job_id)
            job.work_dir.mkdir(parents=True, exist_ok=True)
            job.artifacts.source, job.artifacts.source_kind = source_url, "url"
            self._jobs[job.id] = job
        self._persist(job)
        return job

    @staticmethod
    def _pipeline_options(job: DemoJob) -> PipelineOptions:
        options = job.options
        source_url = job.material.source_url
        source = source_url or job.artifacts.source or str(job.material.path or "")
        return PipelineOptions(
            source=source, source_kind="url" if source_url else "file", output_dir=job.output_root,
            title=options["title"] or job.material.name, description=options["description"], tags=options["tags"],
            i_have_rights=job.material.authorized,
            require_reuse_allowed=options["require_reuse_allowed"] and bool(source_url),
            cookies_from_browser=options["cookies_from_browser"], cookies_file=options["cookies_file"],
            youtube_po_token_mode=options["youtube_po_token_mode"], youtube_proxy=options["youtube_proxy"],
            download_quality=options["download_quality"], max_seconds=options["max_seconds"] if source_url else None,
            quality_proposal=options.get("quality_proposal"),
            resource_profile=options["resource_profile"],
            subtitle_source=options["subtitle_source"],
            prefer_platform_subtitles=options["prefer_platform_subtitles"] and options["subtitle_source"] in {"auto", "merged"},
            ocr_fallback_to_audio=options["subtitle_source"] == "auto",
            whisper_model_size=options["whisper_model_size"], source_language=options["source_language"],
            beam_size=options["beam_size"], asr_hint=str(options.get("asr_hint") or ""),
            asr_vad_mode=str(options.get("asr_vad_mode") or "standard"),
            ocr_interval=options["ocr_interval"],
            ocr_crop_ratio=options["ocr_crop_ratio"], ocr_min_chars=options["ocr_min_chars"],
            ocr_language=options["ocr_language"],
            audio_channel=options.get("audio_channel", "mix"),
            subtitle_margin_ratio=options["subtitle_margin_ratio"], render_crf=options["render_crf"],
            render_encoder=options["render_encoder"], device=job.device, compute_type=job.compute_type,
            translator=options["translator"], target_lang=options["target_lang"],
            translate_model=options["translate_model"], deepseek_thinking=options.get("deepseek_thinking", "disabled"), smart_translation=options["smart_translation"],
            smart_subtitle_layout=options["smart_subtitle_layout"], font_name=options["font_name"],
            font_size=options["font_size"], subtitle_display_mode=options["subtitle_display_mode"],
            subtitle_preset=options.get("subtitle_preset", "soft"),
            subtitle_color=options["subtitle_color"], subtitle_outline_color=options["subtitle_outline_color"],
            subtitle_outline=options["subtitle_outline"], subtitle_shadow=options["subtitle_shadow"],
            publish_to_bilibili=options["publish_to_bilibili"],
            include_source_link_in_description=options["include_source_link"],
            bilibili_browser=options["bilibili_browser"],
            bilibili_profile_dir=USER_DATA_ROOT / "data" / ("bilibili-edge-profile" if options["bilibili_browser"] == "msedge" else "bilibili-profile"),
            bilibili_wait_for_review=not options["close_after_fill"],
            dub_enabled=bool(options.get("dub_enabled")), dub_mix=options.get("dub_mix", "duck"),
            voxcpm_python=options.get("voxcpm_python", ""), voxcpm_model_dir=options.get("voxcpm_model_dir", ""),
        )

    @staticmethod
    def _config_fingerprint(job: DemoJob) -> str:
        task_options = _public_options(job.options)
        task_options.pop("llm_pricing_mode", None)
        task_options.pop("llm_custom_pricing", None)
        payload = {
            "source": job.artifacts.source, "device": job.device, "compute_type": job.compute_type,
            "options": task_options,
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]

    def _verify_checkpoints(self, job: DemoJob) -> None:
        """Validate completed artifacts and derive the first safe continuation point."""
        if job.work_dir and job.work_dir.is_dir():
            raw_candidate = next(iter(job.work_dir.glob("raw.*")), None)
            if raw_candidate is None:
                raw_candidate = next((
                    path for path in sorted(job.work_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
                    if path.suffix.lower() in MEDIA_EXTENSIONS and not path.name.startswith(("rendered", "trimmed", "kept", "muted", "reordered", "export"))
                ), None)
            conventional = {
                "raw_video": raw_candidate,
                "audio": job.work_dir / "audio.wav",
                "source_segments": job.work_dir / "segments.source.json",
                "source_srt": job.work_dir / "source.srt",
                "translated_segments": job.work_dir / "segments.translated.json",
                "translated_srt": job.work_dir / "zh.srt",
                "publish_metadata": job.work_dir / "publish_metadata.json",
                "rendered_video": job.work_dir / "rendered.mp4",
            }
            for name, path in conventional.items():
                if not getattr(job.artifacts, name) and path and path.is_file():
                    setattr(job.artifacts, name, str(path))
        self._migrate_legacy_artifacts(job)
        translate_row = job.stages.get("translate") or {}
        if (job.work_dir and translate_row.get("status") == "completed" and "glossary_hash" in translate_row
                and translate_row["glossary_hash"] != glossary_hash(load_glossary(job.work_dir))):
            invalidate_downstream(job.stages, "translate")
        first_invalid: str | None = None
        for stage in CORE_STAGES:
            row = job.stages.setdefault(stage, new_stage_states()[stage])
            if row.get("status") == "completed" or (job.workflow_version < WORKFLOW_VERSION and first_invalid is None):
                if validate_stage(stage, job.artifacts):
                    row["status"], row["progress"], row["error"] = "completed", 100, None
                else:
                    row["status"], row["progress"] = "stale", 0
                    first_invalid = first_invalid or stage
            elif first_invalid is None and row.get("status") != "completed":
                first_invalid = stage
        publish_pending = not first_invalid and job.options.get("publish_to_bilibili") and job.stages["publish"].get("status") != "completed"
        dub_pending = not first_invalid and job.options.get("dub_enabled") and job.stages.setdefault("dub", new_stage_states()["dub"]).get("status") != "completed"
        job.next_stage = first_invalid or ("dub" if dub_pending else "publish" if publish_pending else None)
        job.raw_video, job.rendered_video = job.artifacts.path("raw_video"), job.artifacts.path("rendered_video")
        job.edited_segments = job.artifacts.path("translated_segments") if job.artifacts.revision else None
        job.checkpoint_validation = "verified" if not first_invalid else "invalid"
        if not first_invalid and validate_stage("render", job.artifacts):
            if dub_pending:
                if job.status not in {"failed", "cancelled", "interrupted"}:
                    job.status, job.stage, job.progress = "ready", "字幕版已完成，待配音", 100
            elif publish_pending:
                if job.status not in {"failed", "cancelled", "interrupted"}:
                    job.status, job.stage, job.progress = "ready", "成片完成，待投稿辅助", 100
            else:
                job.status, job.stage, job.progress = "completed", "处理完成", 100

    def _migrate_legacy_artifacts(self, job: DemoJob) -> None:
        """Upgrade v1 pointers conservatively; ambiguity never overwrites a deliverable."""
        artifacts = job.artifacts
        if job.workflow_version >= WORKFLOW_VERSION:
            if not artifacts.original_video:
                artifacts.original_video = artifacts.raw_video
            return
        artifacts.original_video = artifacts.original_video or artifacts.raw_video
        work_dir = job.work_dir
        if not work_dir or not work_dir.is_dir():
            job.workflow_version = WORKFLOW_VERSION
            artifacts.edit_state = "clean"
            return
        modified_videos = sorted(
            [path for pattern in ("trimmed-*.mp4", "kept-*.mp4", "muted-*.mp4", "reordered-*.mp4") for path in work_dir.glob(pattern)],
            key=lambda path: path.stat().st_mtime,
        )
        aligned_pairs = [
            (work_dir / "segments.aligned.json", work_dir / "zh.aligned.srt")
        ] if (work_dir / "segments.aligned.json").is_file() or (work_dir / "zh.aligned.srt").is_file() else []
        if not modified_videos and not aligned_pairs:
            artifacts.edit_state = "clean"
            artifacts.revision = 0
            job.workflow_version = WORKFLOW_VERSION
            return
        if len(modified_videos) > 1 or (modified_videos and aligned_pairs):
            artifacts.edit_state = "legacy-ambiguous"
            artifacts.last_edit = "legacy"
            job.workflow_version = WORKFLOW_VERSION
            job.add_log("旧任务包含多组后期产物，已保留现有成片；为防止覆盖，禁止重新渲染。")
            return
        if aligned_pairs:
            segments, subtitle = aligned_pairs[0]
            if validate_segments(segments) and validate_srt(subtitle):
                artifacts.translated_segments, artifacts.translated_srt = str(segments), str(subtitle)
                artifacts.revision, artifacts.last_edit, artifacts.edit_state = 1, "align", "migrated"
            else:
                artifacts.edit_state, artifacts.last_edit = "legacy-ambiguous", "legacy"
            job.workflow_version = WORKFLOW_VERSION
            return
        candidate = modified_videos[0]
        operation = candidate.stem.split("-", 1)[0]
        segment_names = {"trimmed": "segments.trimmed.json", "kept": "segments.kept.json", "reordered": "segments.reordered.json"}
        subtitle_names = {"trimmed": "zh.trimmed.srt", "kept": "zh.kept.srt", "reordered": "zh.reordered.srt"}
        if not validate_media(candidate, "video"):
            artifacts.edit_state, artifacts.last_edit = "legacy-ambiguous", "legacy"
        elif operation == "muted":
            artifacts.raw_video = str(candidate)
            artifacts.revision, artifacts.last_edit, artifacts.edit_state = 1, "mute", "migrated"
        else:
            segments = work_dir / segment_names.get(operation, "")
            subtitle = work_dir / subtitle_names.get(operation, "")
            if validate_segments(segments) and validate_srt(subtitle):
                artifacts.raw_video = str(candidate)
                artifacts.translated_segments, artifacts.translated_srt = str(segments), str(subtitle)
                artifacts.revision, artifacts.last_edit, artifacts.edit_state = 1, operation, "migrated"
            else:
                artifacts.edit_state, artifacts.last_edit = "legacy-ambiguous", "legacy"
        job.workflow_version = WORKFLOW_VERSION

    @staticmethod
    def _prepare_required_stages(job: DemoJob) -> set[str]:
        """Return checks needed for the first incomplete workflow path."""
        required: set[str] = set()
        first_pending: int | None = None
        for index, stage in enumerate(CORE_STAGES):
            if job.stages.get(stage, {}).get("status") != "completed":
                first_pending = index
                break
        if first_pending is not None:
            required.update(CORE_STAGES[first_pending:])
        if job.options.get("dub_enabled") and job.stages.get("dub", {}).get("status") != "completed":
            required.add("dub")
        if job.options.get("publish_to_bilibili") and job.stages.get("publish", {}).get("status") != "completed":
            required.add("publish")
        return required

    def prepare(self, job_id: str) -> DemoJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise RuntimeError("Task not found.")
            if self._running_id == job_id or self._subtitle_model_action_id == job_id or job.status in {"running", "cancelling", "preparing"} or job_id in self._preparing_ids:
                raise RuntimeError("当前阶段或准备检查仍在运行，不能同时执行准备检查。")
            job.cancellation.reset()
            self._preparing_ids.add(job_id)
            job.status, job.stage = "preparing", "正在检查准备"
            self._persist(job)
        try:
            return self._prepare_reserved(job)
        finally:
            with self._lock:
                self._preparing_ids.discard(job_id)
                if job.cancellation.cancelled:
                    job.status, job.stage = "cancelled", "准备检查已取消"
                    job.checks = []
                elif job.status == "preparing":
                    job.status, job.stage = "draft", "准备检查未完成"
                self._persist(job)

    def _prepare_reserved(self, job: DemoJob) -> DemoJob:
        load_dotenv(USER_DATA_ROOT / ".env", override=True)
        self._verify_checkpoints(job)
        required_stages = self._prepare_required_stages(job)
        payload = {
            "source_url": job.material.source_url or "", "material_id": job.material.id,
            "authorized": job.material.authorized, "device": job.device, "compute_type": job.compute_type,
            "source_height": job.material.height or 0, "options": job.options,
        }
        base = preflight(payload, str(job.output_root), required_stages=required_stages)
        checks: list[dict[str, Any]] = []

        def add(check_id: str, label: str, purpose: str, status: str, message: str, action: str | None = None) -> None:
            checks.append({
                "id": check_id, "label": label, "purpose": purpose, "status": status,
                "message": message, "action": action, "needs_recheck": status in {"warning", "blocking"},
            })

        for item in base["blocking"]:
            local_dependency = item["code"] == "ollama-runtime" or item["code"] in OLLAMA_MODEL_TAGS
            add(
                item["code"], "本地翻译模型" if local_dependency else "配置检查",
                "检查 Ollama 与所选模型是否已在本机就绪" if local_dependency else "确保任务可以按当前设置运行",
                "blocking", item["message"], "打开依赖中心" if local_dependency else "修正设置",
            )
        for item in base["warnings"]:
            add(item["code"], "配置提醒", "提前说明可能影响结果的条件", "warning", item["message"], "查看设置")
        if "acquire" in required_stages:
            add("source", "素材与授权", "确认素材来源、授权和处理范围", "passed" if job.material.authorized else "blocking", "素材与授权已确认。" if job.material.authorized else "请先确认处理授权。", "返回素材")
        if "acquire" in required_stages and job.material.source_url:
            scope = "完整视频" if job.options["max_seconds"] is None else f"前 {job.options['max_seconds']} 秒"
            add("download-scope", "下载范围与画质", "控制网络、磁盘和后续处理负载", "passed", f"将获取{scope}，画质上限为 {job.options['download_quality']}。")
            cookies_file = job.options.get("cookies_file")
            cookies_browser = job.options.get("cookies_from_browser")
            if cookies_file:
                valid = _cookies_file_has_login(Path(str(cookies_file)).expanduser())
                add("youtube-cookies", "YouTube 登录凭据", "处理机器人验证或需要登录的视频", "passed" if valid else "blocking", "Cookies 文件包含登录凭据。" if valid else "Cookies 文件缺少 SID/LOGIN_INFO，请登录 YouTube 后重新导出。", "打开设置" if not valid else None)
            elif cookies_browser:
                add("youtube-cookies", "YouTube 登录凭据", "处理机器人验证或需要登录的视频", "warning", f"运行时将读取 {cookies_browser} 的登录 Cookies；请先关闭该浏览器以避免数据库被占用。", "打开设置")
            else:
                add("youtube-cookies", "YouTube 访问", "降低机器人验证导致的下载失败", "warning", "未配置 Cookies；公开视频可能可用，但 YouTube 风控时会阻止下载。", "打开设置")
            try:
                metadata = get_video_metadata(
                    job.material.source_url,
                    job.options.get("cookies_from_browser"), job.options.get("cookies_file"),
                    proxy=job.options.get("youtube_proxy"),
                    po_token_mode=job.options.get("youtube_po_token_mode", "auto"),
                    download_quality=job.options.get("download_quality", "1080p"),
                )
            except Exception as exc:
                add("source-access", "链接访问测试", "在下载媒体前验证链接、登录会话和网络出口", "blocking", str(exc), "打开设置" if re.search(r"cookie|login|sign in|robot|bot", str(exc), re.I) else "返回素材")
            else:
                job.material.name = metadata.title or job.material.name
                job.material.duration_seconds = metadata.duration
                job.material.width, job.material.height = metadata.max_width, metadata.max_height
                proposal = getattr(metadata, "quality_proposal", None)
                if proposal:
                    previous = job.options.get("quality_proposal")
                    proposal = _carry_quality_baseline(proposal, previous if isinstance(previous, dict) else None)
                    proposal["fingerprint"] = self._quality_fingerprint(job, proposal)
                    job.options["quality_proposal"] = proposal
                    if not isinstance(previous, dict) or previous.get("fingerprint") != proposal["fingerprint"]:
                        job.options["quality_consent"] = False
                    if not proposal.get("width") or not proposal.get("height"):
                        job.options.pop("quality_proposal", None)
                        job.options.pop("quality_consent", None)
                        add("source-access", "链接访问测试", "在下载媒体前验证链接、登录会话和网络出口", "blocking", "链接可访问，但无法确定可下载的视频流；请重新读取链接或检查访问设置。", "返回素材")
                    else:
                        add("source-access", "链接访问测试", "在下载媒体前验证链接、登录会话和网络出口", "passed", f"链接可访问：{metadata.title or job.material.name}。")
                elif hasattr(metadata, "quality_proposal"):
                    add("source-access", "链接访问测试", "在下载媒体前验证链接、登录会话和网络出口", "blocking", "无法确定服务器拟下载的视频流；请重新读取链接。", "返回素材")
                else:
                    add("source-access", "链接访问测试", "在下载媒体前验证链接、登录会话和网络出口", "passed", f"链接可访问：{metadata.title or job.material.name}。")
        if required_stages & {"extract", "render"}:
            ffmpeg = resolve_command("ffmpeg")
            ffprobe = resolve_command("ffprobe")
            add("ffmpeg", "FFmpeg", "提取音频并渲染字幕", "passed" if ffmpeg else "blocking", "已检测到 FFmpeg。" if ffmpeg else "缺少 FFmpeg。", "安装 FFmpeg")
            add("ffprobe", "媒体检查器", "验证下载和检查点文件没有损坏", "passed" if ffprobe else "blocking", "已检测到 ffprobe。" if ffprobe else "缺少 ffprobe；通常随 FFmpeg 一起安装。", "安装 FFmpeg")
        mode = job.options["subtitle_source"]
        needs_whisper = mode in {"audio", "auto", "merged"}
        whisper_check_id: str | None = None
        if "extract" in required_stages and needs_whisper:
            size = job.options["whisper_model_size"]
            whisper_check_id = f"whisper-{size}"
            try:
                require_whisper_model(size, USER_DATA_ROOT)
            except RuntimeError as exc:
                action = "安装 Whisper 模型" if size == "small" else "安装所选模型"
                add(f"whisper-{size}", f"Whisper {size}", "把语音转换成带时间轴的源字幕", "blocking", str(exc), action)
            else:
                add(f"whisper-{size}", f"Whisper {size}", "把语音转换成带时间轴的源字幕", "passed", "模型文件完整。")
        if "extract" in required_stages and mode in {"ocr", "merged"}:
            tesseract = resolve_command("tesseract")
            add("tesseract", "Tesseract OCR", "读取画面中已有的字幕文字", "passed" if tesseract else "blocking", "OCR 引擎可用。" if tesseract else "当前字幕来源需要 OCR，但未安装 Tesseract。", "安装 Tesseract")
            if tesseract:
                language = job.options["ocr_language"]
                installed = available_ocr_languages(str(tesseract))
                missing = [item for item in language.split("+") if item not in installed]
                ready, _ = ocr_language_ready(language, str(tesseract))
                add(
                    "ocr-language", "OCR 语言包", "决定 Tesseract 识别哪种画面字幕",
                    "passed" if ready else "blocking",
                    f"已选择 {language}，语言数据完整。" if ready else f"缺少语言包：{', '.join(missing) or language}。当前已安装：{', '.join(sorted(installed)) or '未知'}。",
                    "下载语言包" if not ready else None,
                )
        if "translate" in required_stages:
            provider = job.options["translator"]
            key_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY" if provider == "openai" else ""
            key_ready = not key_name or bool(os.getenv(key_name))
            if provider != "ollama" or not any(item["code"] == "ollama-runtime" or item["code"] in OLLAMA_MODEL_TAGS for item in base["blocking"]):
                connection_ready, connection_message = _probe_translation_service(
                    provider, job.options.get("translate_model"),
                ) if key_ready else (False, f"缺少 {provider} API Key。")
                add("translator", "翻译服务", "生成中文字幕并校正表达", "passed" if connection_ready else "blocking", connection_message, "打开设置" if not connection_ready else None)
        if "extract" in required_stages and needs_whisper and job.device == "cuda":
            cuda = cuda_runtime_status()
            cuda_ready = bool(cuda["available"])
            add(
                "cuda", "CUDA 转写运行库", "使用显卡加速 Whisper 转写",
                "passed" if cuda_ready else "blocking", str(cuda["message"]),
                "打开依赖中心" if not cuda_ready else None,
            )
        if "render" in required_stages:
            encoder = job.options["render_encoder"]
            encoders = render_encoder_status()
            encoder_ready = encoders["nvidia"] if encoder == "nvidia" else encoders["cpu"] or (encoder == "auto" and encoders["nvidia"])
            add("encoder", "渲染编码器", "把中文字幕烧录到最终视频", "passed" if encoder_ready else "blocking", "渲染编码器可用。" if encoder_ready else "当前渲染编码器不可用。", "更改编码器")
        if "publish" in required_stages:
            add("publish", "投稿配置", "确认投稿辅助可以安全打开", "warning", "发布阶段将另行检查登录状态和浏览器配置。", "打开投稿辅助")
        if required_stages & {"acquire", "extract", "render"}:
            try:
                job.output_root.mkdir(parents=True, exist_ok=True)
                probe = job.output_root / ".write-test"
                probe.write_bytes(b"ok"); probe.unlink()
                free = shutil.disk_usage(job.output_root).free
                # URL jobs may be clipped by max_seconds, while local imports must
                # reserve space for their actual duration regardless of that UI
                # option.  Keep a conservative fallback when metadata is absent.
                if job.material.source_url:
                    duration = job.material.duration_seconds or job.options.get("max_seconds") or 600
                    max_seconds = job.options.get("max_seconds")
                    if max_seconds and job.material.duration_seconds:
                        duration = min(duration, max_seconds)
                else:
                    duration = job.material.duration_seconds or 600
                estimate = max(512 * 1024 * 1024, int(duration * 8 * 1024 * 1024))
                disk_status = "passed" if free >= estimate else "blocking"
                add("output", "输出空间", "保存原视频、中间字幕和最终成片", disk_status, f"可用 {format_bytes(free)}，预计至少需要 {format_bytes(estimate)}。", "更改输出目录" if disk_status == "blocking" else None)
            except OSError as exc:
                add("output", "输出目录", "保存任务产物", "blocking", f"输出目录不可写：{exc}", "更改输出目录")
        if not checks:
            add("checkpoints", "已完成阶段", "确认现有检查点仍可继续使用", "passed", "已完成阶段的检查点仍然有效。")
        # Deduplicate the older preflight messages when a richer named check exists.
        rich_ids = {"ffmpeg", "tesseract", "ocr-language", "translator", "cuda", "encoder", "output", "source"}
        if whisper_check_id:
            rich_ids.add(whisper_check_id)
        checks = [row for index, row in enumerate(checks) if row["id"] not in rich_ids or index == max(i for i, item in enumerate(checks) if item["id"] == row["id"])]
        with self._lock:
            if job.cancellation.cancelled or job.id not in self._preparing_ids:
                return job
            job.checks = checks
            blocking = any(row["status"] == "blocking" for row in checks)
            job.error = None
            if blocking:
                job.status, job.stage = "draft", "准备检查未通过"
            elif job.status != "completed":
                job.status, job.stage = "ready", "准备完成"
        self._persist(job)
        return job

    def update_options(self, job_id: str, payload: dict[str, Any]) -> tuple[DemoJob, list[str]]:
        job = self.get(job_id)
        if not job:
            raise RuntimeError("Task not found.")
        if job.status in {"running", "cancelling", "preparing"}:
            raise RuntimeError("请先取消当前阶段，再修改方案。")
        old = {**job.options, "device": job.device, "compute_type": job.compute_type}
        incoming = payload.get("options")
        if incoming is not None and not isinstance(incoming, dict):
            raise ValueError("options must be an object.")
        internal_quality = {key: job.options.get(key) for key in ("quality_proposal", "quality_consent") if key in job.options}
        options = _safe_options({**job.options, **(incoming or {})})
        device, compute = _validate_device_precision(
            payload.get("device", job.device), payload.get("compute_type", job.compute_type),
        )
        new = {**options, "device": device, "compute_type": compute}
        # These server-owned fields are intentionally outside the public
        # options schema; preserve them across unrelated style patches.
        for key, value in internal_quality.items():
            new[key] = value
        options.update({key: value for key, value in internal_quality.items()})
        changed_fields = {name for name in set(old) | set(new) if old.get(name) != new.get(name)}
        stale: list[str] = []
        first = invalidation_stage(changed_fields)
        with self._lock:
            if "output_dir" in changed_fields:
                if any(job.stages[name].get("status") == "completed" for name in WORKFLOW_STAGES[1:]):
                    raise RuntimeError("已有阶段产物时不能更换输出目录；请新建任务，或保留当前目录。")
                job.output_root = _resolve_output_root(options["output_dir"])
                job.work_dir = job.output_root / job.id
                job.work_dir.mkdir(parents=True, exist_ok=True)
            job.options, job.device, job.compute_type = options, device, compute
            if changed_fields & {"download_quality", "max_seconds", "require_reuse_allowed", "cookies_file", "cookies_from_browser", "youtube_proxy", "youtube_po_token_mode"}:
                job.options.pop("quality_proposal", None)
                job.options.pop("quality_consent", None)
            if first:
                stale = invalidate_downstream(job.stages, first)
                job.next_stage = first
                job.error_code = None
                job.suggested_action = None
                # CC reuse is enforced at download time, not during prepare. Keep
                # existing checks so unchecking the box can retry immediately.
                if changed_fields <= {"require_reuse_allowed"} and job.checks:
                    blocking = any(row.get("status") == "blocking" for row in job.checks)
                    job.status = "draft" if blocking else "ready"
                    job.stage = (
                        "已开启仅接受 CC reuse allowed，可重新获取素材"
                        if options["require_reuse_allowed"]
                        else "已关闭 CC 限制，可重新获取素材"
                    )
                elif first in {"render", "publish"} and job.checks:
                    # Subtitle style / publish copy changes must not wipe the
                    # preparation centre the user already passed.
                    blocking = any(row.get("status") == "blocking" for row in job.checks)
                    job.status = "draft" if blocking else "ready"
                    job.stage = (
                        "字幕样式已更新，可重新渲染"
                        if first == "render"
                        else "发布设置已更新"
                    )
                else:
                    job.checks = []
                    job.status, job.stage = "draft", "设置已更新，请重新检查准备"
            elif changed_fields:
                job.stage = "资源模式已更新" if changed_fields == {"resource_profile"} else "设置已更新"
            job.error = None
        self._persist(job)
        return job, stale

    def _prerequisites_complete(self, job: DemoJob, stage: str) -> bool:
        # Optional dub never gates publish; only the core chain must be complete.
        index = WORKFLOW_STAGES.index(stage)
        return all(job.stages[name].get("status") == "completed" for name in CORE_STAGES[:index])

    def _checks_ready(self, job: DemoJob) -> bool:
        return bool(job.checks) and not any(row.get("status") == "blocking" for row in job.checks)

    @staticmethod
    def _quality_fingerprint(job: DemoJob, proposal: dict[str, Any]) -> str:
        payload = {
            "job_id": job.id,
            "source": job.material.source_url or job.artifacts.source,
            "quality": job.options.get("download_quality"),
            "max_seconds": job.options.get("max_seconds"),
            "cookies_from_browser": job.options.get("cookies_from_browser"),
            "cookies_file": bool(job.options.get("cookies_file")),
            "youtube_po_token_mode": job.options.get("youtube_po_token_mode"),
            "youtube_proxy_configured": bool(job.options.get("youtube_proxy")),
            "require_reuse_allowed": bool(job.options.get("require_reuse_allowed")),
            "proposal": {key: proposal.get(key) for key in ("width", "height", "has_audio", "baseline_width", "baseline_height")},
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def _require_quality_consent(self, job: DemoJob, stages: list[str] | None = None) -> None:
        if not job.material.source_url or (stages and "acquire" not in stages and job.stages.get("acquire", {}).get("status") == "completed"):
            return
        proposal = job.options.get("quality_proposal")
        if not isinstance(proposal, dict) or not proposal.get("fingerprint"):
            raise RuntimeError("请先准备素材并确认服务器提供的画质方案。")
        if not proposal.get("width") or not proposal.get("height"):
            raise RuntimeError("服务器无法确定可下载的视频流，请重新准备或检查链接。")
        if proposal.get("fingerprint") != self._quality_fingerprint(job, proposal):
            raise RuntimeError("画质方案已失效，请重新准备检查。")
        if proposal.get("requires_consent") is not True:
            return
        if job.options.get("quality_consent") is not True:
            raise RuntimeError("下载画质方案尚未确认，请先明确同意后再开始。")

    def confirm_quality(self, job_id: str, fingerprint: str) -> DemoJob:
        job = self.get(job_id)
        if not job or not job.material.source_url:
            raise RuntimeError("当前任务不需要 URL 画质确认。")
        proposal = job.options.get("quality_proposal")
        if not isinstance(proposal, dict) or fingerprint != proposal.get("fingerprint") or fingerprint != self._quality_fingerprint(job, proposal):
            raise RuntimeError("画质方案已失效，请重新准备检查后再确认。")
        job.options["quality_consent"] = True
        self._persist(job)
        return job

    def run_stage(self, job_id: str, stage: str) -> DemoJob:
        job = self.get(job_id)
        if not job:
            raise RuntimeError("Task not found.")
        if stage not in WORKFLOW_STAGES:
            raise ValueError("Unknown workflow stage.")
        if not self._checks_ready(job):
            raise RuntimeError("请先完成“开始前准备”检查并解决阻断项。")
        self._require_quality_consent(job, [stage])
        if not self._prerequisites_complete(job, stage):
            raise RuntimeError("上一个阶段尚未完成，不能跳过执行。")
        return self._start_stage_sequence(job, [stage], auto_run=False)

    def run_all(self, job_id: str) -> DemoJob:
        job = self.get(job_id)
        if not job:
            raise RuntimeError("Task not found.")
        if not self._checks_ready(job):
            raise RuntimeError("请先完成“开始前准备”检查并解决阻断项。")
        with self._lock:
            if self._running_id or self._subtitle_model_action_id or job.status in {"running", "cancelling", "preparing"} or job.id in self._preparing_ids:
                raise RuntimeError("已有耗资源阶段在运行，请等待或先取消。")
        # Re-check artifacts at the point of execution.  A file can disappear
        # or become corrupt after prepare (including while the app was closed).
        self._verify_checkpoints(job)
        self._persist(job)
        first_pending = next(
            (index for index, name in enumerate(CORE_STAGES)
             if job.stages[name].get("status") != "completed"),
            None,
        )
        stages = list(CORE_STAGES[first_pending:]) if first_pending is not None else []
        if job.options.get("dub_enabled") and job.stages.setdefault("dub", new_stage_states()["dub"]).get("status") != "completed":
            stages.append("dub")
        if not stages:
            return job
        self._require_quality_consent(job, stages)
        first = stages[0]
        if not self._prerequisites_complete(job, first):
            raise RuntimeError("检查点不连续，请先重新运行最早失效阶段。")
        return self._start_stage_sequence(job, stages, auto_run=True)

    def resume(self, job_id: str) -> DemoJob:
        job = self.get(job_id)
        if not job:
            raise RuntimeError("Task not found.")
        self._verify_checkpoints(job)
        if job.next_stage == "publish":
            raise RuntimeError("投稿辅助可能已上传视频；请先到 B 站创作中心核对草稿，再手动重试投稿辅助。")
        if not job.next_stage:
            self._persist(job)
            return job
        if not self._checks_ready(job):
            raise RuntimeError("恢复前需要重新执行准备检查，确认环境与配置仍然可用。")
        self._require_quality_consent(job, [job.next_stage] if job.next_stage else None)
        return self.run_stage(job_id, job.next_stage)

    @contextmanager
    def _llm_usage_session(self, job: DemoJob, label: str, run_token: str | None = None):
        token = run_token or uuid.uuid4().hex
        with self._lock:
            if run_token is None:
                job.llm_run_token = token
            directory = (job.work_dir or job.output_root / job.id) / f".attempt-llm-{label}-{token[:12]}"
            index_path = directory.parent / "llm_usage_index.json"
            if job.llm_usage_index is None:
                job.llm_usage_index = {"schema_version": 1, "revision": 0, "recorded_since": time.time(), "sessions": {}}
                if job.artifacts.llm_usage_total:
                    # Retain a saved subtotal if its index became unavailable.
                    job.llm_usage_index["baseline"] = copy.deepcopy(job.artifacts.llm_usage_total)
            pricing_options = {key: copy.deepcopy(job.options.get(key)) for key in ("llm_pricing_mode", "llm_custom_pricing")}

        def update(summary: dict[str, Any]) -> None:
            with self._lock:
                if job.id in self._deleted_history_ids:
                    return
                if not update_usage_index(job.llm_usage_index, token, directory / "llm_usage.json", summary):
                    return
                job.artifacts.llm_usage_index_path = str(index_path)
                job.artifacts.llm_usage_total = copy.deepcopy(job.llm_usage_index["total"])
                if job.llm_run_token == token:
                    job.llm_usage = summary
                    job.artifacts.llm_usage_path = str(directory / "llm_usage.json")
                try:
                    atomic_write_manifest(index_path, job.llm_usage_index)
                    # Merge just accounting fields, without rewriting stage/log manifests.
                    job_db.update_usage_artifacts(job.id, {
                        "llm_usage_path": job.artifacts.llm_usage_path,
                        "llm_usage_index_path": job.artifacts.llm_usage_index_path,
                        "llm_usage_total": job.artifacts.llm_usage_total,
                    })
                except Exception:
                    LOGGER.warning("Could not persist cumulative LLM usage for %s", job.id, exc_info=True)

        with self._lock:
            if job.llm_run_token == token:
                job.llm_usage = None
        tracker = LlmUsageTracker(directory, update, scope=label, pricing_options=pricing_options)
        tracker.flush()
        try:
            with usage_session(tracker):
                yield tracker
        finally:
            tracker.flush()
            with self._lock:
                if job.llm_run_token == token:
                    try:
                        self._persist(job)
                    except Exception:
                        LOGGER.warning("Could not persist LLM usage pointer for job %s", job.id, exc_info=True)

    def _start_stage_sequence(self, job: DemoJob, stages: list[str], *, auto_run: bool) -> DemoJob:
        self._require_quality_consent(job, stages)
        previous_state: dict[str, Any] | None = None
        with self._lock:
            if self._batch_job_id and self._batch_job_id != job.id:
                raise RuntimeError("批量队列正在处理其他任务，请等待队列完成后再运行。")
            if self._running_id or self._subtitle_model_action_id or job.status in {"preparing", "cancelling"} or job.id in self._preparing_ids:
                raise RuntimeError("准备检查或耗资源阶段正在运行，请等待或先取消。")
            previous_state = {
                "status": job.status, "stage": job.stage, "error": job.error,
                "current_stage": job.current_stage, "started_at": job.started_at,
                "finished_at": job.finished_at, "next_stage": job.next_stage,
                "auto_run": job.auto_run, "progress": job.progress,
                "stages": copy.deepcopy(job.stages),
                "llm_run_token": job.llm_run_token,
            }
            # Re-running a stage must invalidate it and everything after, otherwise
            # translate/render can remain "completed" and be skipped after extract.
            if stages:
                invalidate_downstream(job.stages, stages[0])
                job.next_stage = stages[0]
            job.cancellation.reset()
            job.auto_run, job.current_stage = auto_run, stages[0]
            job.status, job.stage, job.error = "running", WORKFLOW_STAGE_LABELS[stages[0]], None
            job.started_at, job.finished_at = time.time(), None
            self._running_id = job.id
            job.llm_run_token = uuid.uuid4().hex
            run_token = job.llm_run_token
        try:
            self._persist(job)
        except Exception:
            with self._lock:
                if previous_state is not None:
                    job.status = previous_state["status"]
                    job.stage = previous_state["stage"]
                    job.error = previous_state["error"]
                    job.current_stage = previous_state["current_stage"]
                    job.started_at = previous_state["started_at"]
                    job.finished_at = previous_state["finished_at"]
                    job.next_stage = previous_state["next_stage"]
                    job.auto_run = previous_state["auto_run"]
                    job.progress = previous_state["progress"]
                    job.stages = previous_state["stages"]
                    job.llm_run_token = previous_state["llm_run_token"]
                if self._running_id == job.id:
                    self._running_id = None
            raise
        worker = threading.Thread(target=self._run_stage_sequence, args=(job, stages, run_token), daemon=True, name=f"workflow-{job.id}")
        with self._lock:
            self._worker = worker
        worker.start()
        return job

    def _run_stage_sequence(self, job: DemoJob, stages: list[str], run_token: str) -> None:
        usage_context = self._llm_usage_session(job, "workflow", run_token)
        usage_context.__enter__()
        def log(message: str) -> None:
            with self._lock:
                job.add_log(message)

        def on_event(event: PipelineEvent) -> None:
            with self._lock:
                job.apply_event(event)
                if job.current_stage:
                    job.stages[job.current_stage]["progress"] = max(0, min(99, event.progress))

        def stage_progress(fraction: float) -> None:
            with self._lock:
                if job.current_stage:
                    job.stages[job.current_stage]["progress"] = int(max(0, min(1, fraction)) * 100)

        try:
            load_dotenv(USER_DATA_ROOT / ".env", override=True)
            options = self._pipeline_options(job)
            for stage in stages:
                with self._lock:
                    job.current_stage, job.stage = stage, WORKFLOW_STAGE_LABELS[stage]
                    row = job.stages[stage]
                    row.update({
                        "status": "running", "progress": 0, "error": None,
                        "error_code": None, "recovery": None,
                        "started_at": time.time(), "finished_at": None,
                    })
                    job.error, job.error_code, job.suggested_action = None, None, None
                    job.add_log(f"Stage started: {WORKFLOW_STAGE_LABELS[stage]}")
                self._persist(job)
                # Stage runners update artifacts incrementally. Keep that
                # mutation isolated until the stage returns successfully so a
                # cancellation/failure cannot replace a known-good pointer.
                candidate_artifacts = copy.deepcopy(job.artifacts)
                candidate_artifacts = run_workflow_stage(
                    stage, options, job.work_dir, candidate_artifacts,  # type: ignore[arg-type]
                    PipelineContext(job.cancellation, on_event), log, stage_progress,
                )
                previous_artifacts = job.artifacts
                previous_result = copy.deepcopy(job.result)
                previous_next_stage = job.next_stage
                previous_stage_row = copy.deepcopy(job.stages[stage])
                with self._lock:
                    candidate_artifacts.llm_usage_path = job.artifacts.llm_usage_path
                    candidate_artifacts.llm_usage_index_path = job.artifacts.llm_usage_index_path
                    candidate_artifacts.llm_usage_total = copy.deepcopy(job.artifacts.llm_usage_total)
                    job.artifacts = candidate_artifacts
                with self._lock:
                    row = job.stages[stage]
                    row.update({
                        "status": "completed", "progress": 100, "error": None,
                        "error_code": None, "recovery": None,
                        "finished_at": time.time(), "config_fingerprint": self._config_fingerprint(job),
                    })
                    if stage == "translate":
                        row["glossary_hash"] = glossary_hash(load_glossary(job.work_dir))  # type: ignore[arg-type]
                    job.raw_video, job.rendered_video = job.artifacts.path("raw_video"), job.artifacts.path("rendered_video")
                    next_index = WORKFLOW_STAGES.index(stage) + 1
                    job.next_stage = next((name for name in CORE_STAGES[next_index:] if job.stages[name].get("status") != "completed"), None)
                    if not job.next_stage and stage != "dub" and job.options.get("dub_enabled") and job.stages["dub"].get("status") != "completed":
                        job.next_stage = "dub"
                    if not job.next_stage and job.options.get("publish_to_bilibili") and job.stages["publish"].get("status") != "completed":
                        job.next_stage = "publish"
                    job.result = {
                        "output_dir": public_path(job.work_dir) or "",
                        "source_srt": public_path(job.artifacts.path("source_srt")) or "" if job.artifacts.path("source_srt") else "",
                        "translated_srt": public_path(job.artifacts.path("translated_srt")) or "" if job.artifacts.path("translated_srt") else "",
                        "rendered_video": public_path(job.rendered_video) or "" if job.rendered_video else "",
                        "dub_video": public_path(job.artifacts.path("dub_video")) or "" if job.artifacts.dub_video else "",
                    }
                try:
                    self._persist(job)
                except Exception:
                    # SQLite is authoritative: do not leave an unpersisted
                    # candidate pointer claiming a completed stage.
                    with self._lock:
                        job.artifacts = previous_artifacts
                        job.raw_video = previous_artifacts.path("raw_video")
                        job.rendered_video = previous_artifacts.path("rendered_video")
                        job.result = previous_result
                        job.next_stage = previous_next_stage
                        job.stages[stage] = previous_stage_row
                    raise
            with self._lock:
                job.current_stage, job.auto_run = None, False
                job.finished_at = time.time()
                if job.stages["render"].get("status") == "completed" and job.next_stage == "publish":
                    job.status, job.stage, job.progress = "ready", "成片完成，待投稿辅助", 100
                elif job.stages["render"].get("status") == "completed":
                    job.status, job.stage, job.progress = "completed", "处理完成", 100
                else:
                    job.status, job.stage = "ready", "阶段完成，等待下一步"
                job.checkpoint_validation = "verified" if job.next_stage in {None, "publish"} else "invalid"
                job.add_log("Requested stage sequence completed.")
        except (CancellationRequested, CancellationError):
            with self._lock:
                stage = job.current_stage
                if stage:
                    job.stages[stage].update({"status": "cancelled", "error": None, "error_code": None, "recovery": None, "finished_at": time.time()})
                    job.next_stage = stage
                job.status, job.stage, job.finished_at, job.auto_run = "cancelled", "当前阶段已取消", time.time(), False
                job.error, job.error_code, job.suggested_action = None, None, None
                job.add_log("Current stage was cancelled; completed checkpoints were kept.")
        except Exception as exc:
            log_frontend_error("工作流程失败，已显示在界面")
            with self._lock:
                stage = job.current_stage
                if stage:
                    error_code, suggested_action = _classify_job_error(exc)
                    row = job.stages[stage]
                    row.update({
                        "status": "failed", "error": str(exc), "error_code": error_code,
                        "recovery": _recovery_for_error(exc, stage, job), "finished_at": time.time(),
                    })
                    job.next_stage = stage
                else:
                    error_code, suggested_action = _classify_job_error(exc)
                job.status, job.stage, job.error = "failed", f"{WORKFLOW_STAGE_LABELS.get(stage or '', '阶段')}失败", str(exc)
                job.error_code, job.suggested_action = error_code, suggested_action
                job.finished_at, job.auto_run = time.time(), False
                job.add_log(f"Stage failed: {exc}")
        except BaseException as exc:
            log_frontend_error("工作线程异常终止，已显示在界面")
            stage = job.current_stage or (stages[0] if stages else None)
            detail = str(exc).strip()
            error = f"工作线程异常终止（{type(exc).__name__}）"
            if detail:
                error += f"：{detail}"
            with self._lock:
                if stage:
                    job.stages[stage].update({
                        "status": "failed", "error": error,
                        "error_code": "worker_terminated",
                        "recovery": "已完成的阶段成果保留；请重试当前阶段。",
                        "finished_at": time.time(),
                    })
                    job.next_stage = stage
                job.status = "failed"
                job.stage = f"{WORKFLOW_STAGE_LABELS.get(stage or '', '工作流程')}失败"
                job.error, job.error_code = error, "worker_terminated"
                job.suggested_action = "已完成的阶段成果保留；查看错误详情后重试当前阶段。"
                job.finished_at, job.auto_run = time.time(), False
                job.add_log(f"工作线程异常终止：{type(exc).__name__}" + (f": {detail}" if detail else ""))
            raise
        finally:
            usage_context.__exit__(*sys.exc_info())
            with self._lock:
                if self._running_id == job.id:
                    self._running_id = None
                if self._worker is threading.current_thread():
                    self._worker = None
                job.current_stage = None
            try:
                self._persist(job)
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.stage = "结果保存失败"
                    job.error = f"无法保存任务最终状态：{exc}"
                    job.error_code = "persistence_failed"
                    job.add_log(f"Final persistence failed: {exc}")
                LOGGER.exception("Could not persist final state for job %s", job.id)

    # Batch queue (M2): items run one at a time through create_url -> prepare -> run_all.
    # The batch never runs publish; publish_to_bilibili is forced off for every item.
    @staticmethod
    def _batch_path() -> Path:
        return USER_DATA_ROOT / "batch_queue.json"

    def _save_batch(self) -> None:
        atomic_write_manifest(self._batch_path(), self._batch_items)  # type: ignore[arg-type]

    def _load_batch(self) -> None:
        path = self._batch_path()
        if not path.exists():
            return
        try:
            self._batch_items = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            LOGGER.warning("Corrupt batch queue file, moved to batch_queue.json.bad", exc_info=True)
            path.replace(path.with_name("batch_queue.json.bad"))
            self._batch_items = []
            return
        for item in self._batch_items:
            if item["status"] == "running":
                item["status"] = "pending"
        self._save_batch()
        self._start_batch_worker()

    def batch_snapshot(self) -> dict[str, Any]:
        with self._lock:
            items = [{key: item.get(key) for key in ("id", "url", "title", "job_id", "status", "error")} for item in self._batch_items]
            for row in items:
                job = self._jobs.get(row["job_id"]) if row["job_id"] else None
                row["stage"], row["progress"] = (job.stage, job.progress) if job else (None, None)
            return {"items": items, "active": self._batch_thread is not None}

    def batch_add(self, entries: list[dict[str, Any]], device: str, compute_type: str, options: dict[str, Any] | None, quality_consent: bool) -> dict[str, Any]:
        options = dict(options or {})
        options["publish_to_bilibili"] = False
        with self._lock:
            for entry in entries:
                self._batch_items.append({
                    "id": uuid.uuid4().hex[:10], "url": str(entry["url"]), "title": str(entry.get("title") or ""),
                    "job_id": None, "status": "pending", "error": None,
                    "device": device, "compute_type": compute_type, "options": options,
                    "quality_consent": quality_consent,
                })
            self._save_batch()
        self._start_batch_worker()
        return self.batch_snapshot()

    def batch_retry(self, item_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            for item in self._batch_items:
                if item["status"] in ("failed", "cancelled") and item_id in (None, item["id"]):
                    item["status"], item["error"] = "pending", None
            self._save_batch()
        self._start_batch_worker()
        return self.batch_snapshot()

    def batch_cancel(self) -> dict[str, Any]:
        running_job_id = None
        with self._lock:
            for item in self._batch_items:
                if item["status"] == "pending":
                    item["status"], item["error"] = "cancelled", None
                elif item["status"] == "running":
                    item["status"], item["error"] = "cancelled", "\u5df2\u53d6\u6d88"
                    running_job_id = item["job_id"]
            self._save_batch()
        if running_job_id:
            self.cancel(running_job_id)
        return self.batch_snapshot()

    def _start_batch_worker(self) -> None:
        with self._lock:
            if self._batch_thread is not None or not any(item["status"] == "pending" for item in self._batch_items):
                return
            self._batch_thread = threading.Thread(target=self._batch_loop, daemon=True, name="batch-queue")
            self._batch_thread.start()

    def _batch_loop(self) -> None:
        while True:
            with self._lock:
                item = next((row for row in self._batch_items if row["status"] == "pending"), None)
                if item is None:
                    self._batch_thread = None
                    return
                busy = bool(self._running_id or self._preparing_ids or self._subtitle_model_action_id)
                if not busy:
                    item["status"], item["error"] = "running", None
                    self._batch_job_id = "batch-reserved-" + item["id"]
                    self._save_batch()
            if busy:
                time.sleep(1)
                continue
            try:
                error = self._batch_run_item(item)
            except Exception as exc:
                error = str(exc) or type(exc).__name__
            with self._lock:
                if error == "\u5df2\u53d6\u6d88":
                    item["status"], item["error"] = "cancelled", error
                elif item["status"] != "cancelled":
                    item["status"], item["error"] = ("failed", error) if error else ("done", None)
                self._batch_job_id = None
                self._save_batch()

    def _batch_run_item(self, item: dict[str, Any]) -> str | None:
        job = self.get(item["job_id"]) if item["job_id"] else None
        if job is None:
            job = self.create_url(item["url"], item["device"], item["compute_type"], True, item["options"])
            with self._lock:
                item["job_id"] = job.id
                if not item["title"]:
                    item["title"] = job.material.name
                self._save_batch()
        with self._lock:
            self._batch_job_id = job.id
        if item["status"] == "cancelled":
            return "\u5df2\u53d6\u6d88"
        job.options["publish_to_bilibili"] = False
        self.prepare(job.id)
        if job.status == "cancelled":
            return "已取消"
        if not self._checks_ready(job):
            return "准备检查未通过：" + "；".join(str(row.get("message")) for row in job.checks if row.get("status") == "blocking")
        if item["quality_consent"]:
            job.options["quality_consent"] = True
        self.run_all(job.id)
        with self._lock:
            worker = self._worker if self._running_id == job.id else None
        if worker:
            worker.join()
        if job.stages["render"].get("status") == "completed":
            return None
        if job.status == "cancelled":
            return "已取消"
        return job.error or job.stage or "处理失败"

    def cancel(self, job_id: str) -> DemoJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and (job.status in {"queued", "running", "preparing"} or job.id in self._preparing_ids or self._running_id == job.id):
                job.status, job.stage = "cancelling", "正在取消当前阶段"
                job.add_log("Cancellation requested from workbench.")
                job.cancellation.cancel()
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
            associated_metadata = next(
                (job.artifacts.path("publish_metadata") for job in self._jobs.values()
                 if job.artifacts.path("rendered_video") and job.artifacts.path("rendered_video").resolve() == video.resolve()
                 and job.artifacts.path("publish_metadata") and job.artifacts.path("publish_metadata").is_file()),
                None,
            ) if video else None
            if video and associated_metadata is None:
                # Native file selection can happen before its task is restored
                # into memory after a restart. Consult persisted artifacts, but
                # require an exact rendered-video pointer match.
                for record in job_db.list_jobs(None):
                    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
                    rendered = artifacts.get("rendered_video")
                    metadata = artifacts.get("publish_metadata")
                    if not isinstance(rendered, str) or not isinstance(metadata, str):
                        continue
                    try:
                        if Path(rendered).expanduser().resolve() == video.resolve() and Path(metadata).is_file():
                            associated_metadata = Path(metadata)
                            break
                    except OSError:
                        continue
        if video is None or not video.exists():
            raise ValueError("The selected video is no longer available. Choose it again.")
        copy = _read_publish_metadata(associated_metadata or video.parent / "publish_metadata.json")
        return {
            "token": token,
            "name": video.name,
            "title": copy["title"] or video.stem,
            "tags": copy["tags"],
            "has_metadata": copy["has_metadata"],
        }

    def refresh_publish_metadata(self, job: DemoJob) -> dict[str, Any]:
        work_dir = job.work_dir
        if work_dir is None:
            raise ValueError("当前任务还没有输出目录。")
        source_path = job.artifacts.path("source_segments") or work_dir / "segments.source.json"
        translated_path = job.artifacts.path("translated_segments") or work_dir / "segments.translated.json"
        source_items = load_segments(source_path) if source_path.is_file() else []
        translated_items = load_segments(translated_path) if translated_path.is_file() else []
        if not source_items and not translated_items:
            raise ValueError("还没有字幕，无法生成投稿标题和标签。")
        with self._lock:
            if self._running_id or self._subtitle_model_action_id or job.status in {"queued", "running", "preparing", "cancelling"}:
                raise ValueError("当前有处理或模型请求运行，请完成后再生成标题。")
            self._subtitle_model_action_id = job.id
        try:
            with self._llm_usage_session(job, "metadata"):
                metadata = generate_publish_metadata(
                    source_title=str(job.options.get("title") or job.artifacts.title or job.material.name),
                    source_description=job.artifacts.description,
                    source_segments=source_items,
                    translated_segments=translated_items or source_items,
                    provider=str(job.options.get("translator") or "none"),
                    target_lang=str(job.options.get("target_lang") or "zh-Hans"),
                    model=job.options.get("translate_model"),
                    deepseek_thinking=str(job.options.get("deepseek_thinking") or "disabled"),
                )
        finally:
            self._end_subtitle_model_action(job.id)
        metadata_path = save_publish_metadata(work_dir / "publish_metadata.json", metadata)
        job.artifacts.publish_metadata = str(metadata_path)
        self._persist(job)
        return {"title": metadata.title, "tags": metadata.tags, "has_metadata": True, "generated": True}

    def publish_file_from_token(self, token: str) -> Path | None:
        with self._lock:
            video = self._native_publish_files.get(token)
        return video if video and video.exists() else None

    @staticmethod
    def _cue_layout_limits(job: DemoJob) -> tuple[int, float]:
        line_width, display_limit = 18, 36.5
        video = job.raw_video or job.artifacts.path("raw_video")
        if video and video.is_file():
            try:
                measured = subtitle_line_capacity(
                    video,
                    font_size=int(job.options.get("font_size", 24)),
                    outline=int(job.options.get("subtitle_outline", 3)),
                    shadow=int(job.options.get("subtitle_shadow", 1)),
                )
            except Exception:
                measured = None
            if measured is not None:
                line_width, display_limit = measured, measured + 0.5
        return line_width, display_limit

    def cues(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None:
            raise RuntimeError("Subtitle data is not ready yet.")
        translated_pointer = job.edited_segments or job.artifacts.path("translated_segments")
        source_pointer = job.artifacts.path("source_segments")
        if translated_pointer and not translated_pointer.exists():
            raise RuntimeError("Subtitle data is invalid; a committed subtitle file is missing.")
        if source_pointer and not source_pointer.exists() and not translated_pointer:
            raise RuntimeError("Subtitle data is invalid; a committed subtitle file is missing.")
        if job.work_dir is None and not (translated_pointer or source_pointer):
            return {"cues": [], "ready": False, "translation_ready": False}
        translated_file = translated_pointer or (job.work_dir / "segments.translated.json" if job.work_dir else None)
        translated_ready = bool(translated_file and translated_file.exists())
        try:
            if translated_ready:
                translated = load_segments(translated_file)  # type: ignore[arg-type]
            else:
                source_file = source_pointer or (job.work_dir / "segments.source.json" if job.work_dir else None)
                if source_file is None or not source_file.exists():
                    # A valid draft simply has no subtitle data yet.
                    return {"cues": [], "ready": False, "translation_ready": False}
                translated = load_segments(source_file)  # 翻译中：原文预览
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError("Subtitle data is invalid; please rerun subtitle extraction.") from exc
        # 以当前数据为准：时间轴随编辑/裁剪/保留/重排变化，text 始终是原文
        audio_texts: set[str] = set()
        ocr_texts: set[str] = set()
        attempt_parent = job.artifacts.path("source_segments")
        attempt_parent = attempt_parent.parent if attempt_parent else (
            translated_pointer.parent if translated_pointer else job.work_dir
        )
        for name in ("segments.audio.json", "segments.ocr.json"):
            path = attempt_parent / name
            if not path.exists() and job.work_dir is not None:
                path = job.work_dir / name
            if path.exists():
                try:
                    auxiliary_segments = load_segments(path)
                except (OSError, TypeError, ValueError) as exc:
                    raise RuntimeError("Subtitle data is invalid; please rerun subtitle extraction.") from exc
                for segment in auxiliary_segments:
                    (ocr_texts if name.startswith("segments.ocr") else audio_texts).add(segment.text.strip())
        output: list[dict[str, Any]] = []
        platform_mode = job.artifacts.subtitle_extraction_mode.startswith("platform")
        layout_indexes: set[int] = set()
        dense_indexes: set[int] = set()
        fast_indexes: set[int] = set()
        if translated_ready:
            line_width, display_limit = self._cue_layout_limits(job)
            try:
                layout_indexes = {
                    cue_number - 1
                    for cue_number, _start, _text in translated_layout_issues(
                        translated,
                        smart_layout=bool(job.options.get("smart_subtitle_layout", True)),
                        max_chars_per_line=line_width,
                        max_display_units_per_line=display_limit,
                        display_mode=str(job.options.get("subtitle_display_mode") or "translated"),
                    )
                }
            except Exception:
                layout_indexes = set()
            dense_indexes = set(subtitle_density_overflow_indices(translated))
            fast_indexes = set(subtitle_high_reading_speed_indices(translated))
        duration = job.duration_override or job.material.duration_seconds
        for index, item in enumerate(translated):
            raw_source = item.text
            stripped = raw_source.strip()
            if stripped in ocr_texts and stripped not in audio_texts:
                kind = "ocr"
            elif stripped in audio_texts and stripped not in ocr_texts:
                kind = "audio"
            else:
                kind = "platform" if platform_mode else "merged"
            issues: list[str] = []
            if index in layout_indexes:
                issues.append("layout-overflow")
            if index in dense_indexes:
                issues.append("dense-text")
            if index in fast_indexes:
                issues.append("reading-speed")
            if item.end - item.start < MIN_ESTIMATED_SLOT_DURATION_SEC:
                issues.append("short-duration")
            if (
                not math.isfinite(item.start)
                or not math.isfinite(item.end)
                or item.start < 0
                or item.end <= item.start
                or (duration is not None and item.end > duration + 0.05)
            ):
                issues.append("invalid-time")
            if index + 1 < len(translated) and item.end > translated[index + 1].start + 0.02:
                issues.append("overlap")
            if not stripped:
                issues.append("empty-source")
            if translated_ready and not (item.translated_text or "").strip():
                issues.append("empty-translation")
            output.append({
                "id": index, "start": item.start, "end": item.end,
                "source": raw_source,
                "translated": item.translated_text or item.text,
                "kind": kind,
                "issues": issues,
                "needs_review": bool(item.needs_review),
            })
        fingerprint_path = translated_pointer or source_pointer
        fingerprint = hashlib.sha256(fingerprint_path.read_bytes()).hexdigest() if fingerprint_path and fingerprint_path.is_file() else None
        return {
            "cues": output, "ready": True, "translation_ready": translated_ready,
            "revision": int(job.artifacts.revision),
            "base_fingerprint": fingerprint,
            "base_ids": [item["id"] for item in output],
        }

    def save_cues(self, job_id: str, cues: list[dict[str, Any]], *, base_revision: int | None = None,
                  base_fingerprint: str | None = None, base_ids: list[Any] | None = None,
                  candidate_rows: list[dict[str, Any]] | None = None) -> DemoJob:
        """Atomically save the complete subtitle draft and optional review rows."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.work_dir is None or job.stages.get("translate", {}).get("status") != "completed":
                raise RuntimeError("翻译阶段完成后才能编辑中文字幕。")
            if job.status in {"running", "cancelling"} or self._running_id or self._subtitle_model_action_id:
                raise RuntimeError("当前有处理任务正在运行，请完成后再保存字幕。")
            if base_revision is None or not base_fingerprint or base_ids is None:
                raise RuntimeError("保存字幕必须携带完整版本基准，请重新加载字幕后再保存。")
            source_path = job.edited_segments or job.artifacts.path("translated_segments") or job.work_dir / "segments.translated.json"
            segments = load_segments(source_path)
            if base_revision is not None and int(base_revision) != int(job.artifacts.revision):
                raise RuntimeError("字幕版本已变化，请重新加载后再保存。")
            if base_fingerprint and not hmac.compare_digest(str(base_fingerprint), hashlib.sha256(source_path.read_bytes()).hexdigest()):
                raise RuntimeError("字幕内容已变化，请重新加载后再保存。")
            if base_ids is not None:
                try: received = [int(item) for item in base_ids]
                except (TypeError, ValueError): raise RuntimeError("字幕基础行 ID 无效，请重新加载后再保存。") from None
                if sorted(received) != list(range(len(segments))) or len(received) != len(set(received)):
                    raise RuntimeError("字幕基础行已变化，请重新加载后再保存。")
            if not isinstance(cues, list) or len(cues) != len(segments):
                raise RuntimeError("Subtitle changed while editing; reload the latest cues.")
            kept: list[Segment] = []
            for segment, item in zip(segments, cues):
                if not isinstance(item, dict): raise RuntimeError("Invalid subtitle edit payload.")
                if item.get("deleted", False): continue
                source = str(item.get("source", segment.text)).strip()
                translated = str(item.get("translated", "")).strip()
                if ((not source and str(segment.text or "").strip()) or not translated or len(source) > 2000 or len(translated) > 2000):
                    raise RuntimeError("字幕原文和译文不能为空且不能超过 2000 个字符。")
                try: start, end = float(item.get("start", segment.start)), float(item.get("end", segment.end))
                except (TypeError, ValueError): raise RuntimeError("字幕时间必须是数字。") from None
                if not math.isfinite(start) or not math.isfinite(end) or not (0 <= start < end):
                    raise RuntimeError("Subtitle time is invalid.")
                review = item.get("needs_review", segment.needs_review)
                if type(review) is not bool: raise RuntimeError("needs_review must be true or false.")
                kept.append(Segment(round(start, 2), round(end, 2), source, translated, review))
            candidate_rows = candidate_rows or []
            ids = [str(row.get("candidate_id") or "") for row in candidate_rows if isinstance(row, dict)]
            if len(ids) != len(candidate_rows) or len(ids) != len(set(ids)):
                raise RuntimeError("candidate_rows 的候选 ID 必须唯一。")
            session = self._reconcile_subtitle_commit(job)
            session_candidates = {str(item.get("id")): item for item in (session or {}).get("candidates", [])}
            if candidate_rows and session and not self._review_media_is_current(job, session):
                raise RuntimeError("视频源已修改，请重新准备语音复核后再采用候选。")
            result_candidates = {str(item.get("id")): item for item in (session or {}).get("result_candidates", [])}
            additions: list[Segment] = []
            protected_base_intervals = [(float(item.start), float(item.end)) for item in segments if math.isfinite(float(item.start)) and math.isfinite(float(item.end)) and float(item.end) > float(item.start)]
            for row in candidate_rows:
                candidate_id = str(row.get("candidate_id"))
                result_candidate = result_candidates.get(candidate_id)
                parent_id = str(result_candidate.get("window_id")) if result_candidate else ""
                candidate = session_candidates.get(parent_id)
                if not result_candidate or result_candidate.get("status") == "committed" or not candidate or candidate.get("status") != "done":
                    raise RuntimeError("候选不存在或尚未完成复核。")
                tier = result_candidate.get("tier")
                if tier not in {"strong", "weak"}:
                    raise RuntimeError("诊断候选不能保存。")
                if (tier == "strong" and result_candidate.get("adoptable") is not True) or (tier == "weak" and (result_candidate.get("adoptable") is not False or row.get("confirmed_weak") is not True)):
                    raise RuntimeError("弱复核候选必须先人工确认后才能保存。")
                source = str(row.get("source") or "").strip(); translated = str(row.get("translated") or "").strip()
                if not source or not translated: raise RuntimeError("候选原文和译文不能为空。")
                original_source = "".join(str(word.get("text") or "") for word in (result_candidate.get("words") or [])).strip()
                if original_source and source != original_source and row.get("confirmed_source_edit") is not True:
                    raise RuntimeError("候选原文已修改，请先确认修改后的原文。")
                start, end = float(row.get("start")), float(row.get("end"))
                if not math.isfinite(start) or not math.isfinite(end) or not (0 <= start < end): raise RuntimeError("候选时间无效。")
                if any(start < base_end and end > base_start for base_start, base_end in protected_base_intervals) or any(start < item.end and end > item.start for item in additions):
                    raise RuntimeError("候选与现有字幕或其他候选重叠。")
                additions.append(Segment(round(start, 2), round(end, 2), source, translated, True))
            kept = sorted([*kept, *additions], key=lambda item: (item.start, item.end))
            if not kept: raise RuntimeError("At least one subtitle must remain.")
            revision = job.artifacts.revision + 1
            commit_id = uuid.uuid4().hex
            translated_out = job.work_dir / f"segments.translated.edited.r{revision}.{commit_id}.json"
            translated_srt = job.work_dir / f"zh.edited.r{revision}.{commit_id}.srt"
            source_out = job.work_dir / f"segments.source.edited.r{revision}.{commit_id}.json"
            source_srt = job.work_dir / f"source.edited.r{revision}.{commit_id}.srt"
            base_file_fingerprint = hashlib.sha256(source_path.read_bytes()).hexdigest()
            commit_staging = job.work_dir / f".subtitle-edit.r{revision}.{uuid.uuid4().hex}.staging.commit.json"
            commit_path = subtitle_commit_path(job.work_dir, revision, commit_id)
            committed_receipts: list[dict[str, Any]] = []
            for row in candidate_rows:
                candidate_id = str(row.get("candidate_id"))
                candidate = result_candidates[candidate_id]
                raw_words = []
                for word in candidate.get("words", []) if isinstance(candidate.get("words"), list) else []:
                    raw_words.append({key: word.get(key) for key in ("text", "start", "end", "avg_logprob", "no_speech_prob", "tier") if key in word})
                try:
                    ordinal = int(str(candidate_id).rsplit("-", 1)[-1])
                except ValueError:
                    ordinal = None
                committed_receipts.append({
                    "candidateId": candidate_id,
                    "parentWindowId": candidate.get("window_id"),
                    "ordinal": ordinal,
                    "tier": candidate.get("tier"),
                    "confirmedWeak": row.get("confirmed_weak") is True,
                    "rawText": "".join(str(word.get("text") or "") for word in raw_words),
                    "rawWords": raw_words,
                    "editedSource": str(row.get("source") or "").strip(),
                    "editedStart": float(row.get("start")),
                    "editedEnd": float(row.get("end")),
                })
            old_artifact_commit = job.artifacts.subtitle_commit
            source_only = [Segment(item.start, item.end, item.text, None, item.needs_review) for item in kept]
            try:
                save_segments(translated_out, kept); write_srt(translated_srt, kept, display_mode=job.options["subtitle_display_mode"], smart_layout=job.options["smart_subtitle_layout"])
                save_segments(source_out, source_only); write_srt(source_srt, source_only, display_mode="source")
                new_file_fingerprint = hashlib.sha256(translated_out.read_bytes()).hexdigest()
                receipt = {
                    "version": 1, "commitId": commit_id, "jobId": job.id,
                    "fromRevision": int(job.artifacts.revision), "toRevision": revision,
                    "baseFingerprint": base_file_fingerprint, "newFingerprint": new_file_fingerprint,
                    "sessionFingerprint": (session or {}).get("session_fingerprint"),
                    "committed": committed_receipts, "createdAt": time.time(),
                }
                _atomic_json(commit_staging, receipt)
                commit_staging.replace(commit_path)
            except Exception:
                for path in (translated_out, translated_srt, source_out, source_srt, commit_staging, commit_path): path.unlink(missing_ok=True)
                raise
            old_state = (job.edited_segments, job.artifacts.source_segments, job.artifacts.source_srt, job.artifacts.translated_segments, job.artifacts.translated_srt, job.artifacts.subtitle_commit, job.artifacts.revision, job.artifacts.last_edit, job.artifacts.edit_state, copy.deepcopy(job.stages), job.next_stage, job.status, job.stage, job.progress)
            old_session = copy.deepcopy(session)
            published = False
            try:
                job.edited_segments = translated_out
                job.artifacts.source_segments, job.artifacts.source_srt = str(source_out), str(source_srt)
                job.artifacts.translated_segments, job.artifacts.translated_srt = str(translated_out), str(translated_srt)
                job.artifacts.subtitle_commit = str(commit_path)
                job.artifacts.revision, job.artifacts.last_edit, job.artifacts.edit_state = revision, "subtitle", "modified"
                invalidate_downstream(job.stages, "render")
                job.next_stage, job.status, job.stage, job.progress = "render", "ready", "字幕已保存，等待重新渲染", 0
                self._persist(job)
                published = True
                if session:
                    for item in session.get("result_candidates", []):
                        if str(item.get("id")) in ids: item["status"] = "committed"
                    try:
                        update_session(job.work_dir, candidates=session["candidates"], result_candidates=session.get("result_candidates", []), subtitle_revision=revision, subtitle_fingerprint=new_file_fingerprint)
                    except Exception as session_exc:
                        # The authoritative DB and complete published files are
                        # retained; startup reconciliation can advance this CAS.
                        job.add_log(f"字幕提交回执已发布，但复核会话稍后重同步：{type(session_exc).__name__}")
            except Exception:
                (job.edited_segments, job.artifacts.source_segments, job.artifacts.source_srt, job.artifacts.translated_segments, job.artifacts.translated_srt, job.artifacts.subtitle_commit, job.artifacts.revision, job.artifacts.last_edit, job.artifacts.edit_state, job.stages, job.next_stage, job.status, job.stage, job.progress) = old_state
                if old_session is not None:
                    try: _atomic_json(session_path(job.work_dir), old_session)
                    except OSError: pass
                # Keep a complete published revision and receipt if persistence
                # failed: deleting files could leave an already-written DB
                # pointer dangling.  The next restore can reconcile the receipt.
                commit_staging.unlink(missing_ok=True)
                raise
            job.add_log(f"Subtitle edits saved: {len(kept)} cues, {len(segments) - len(kept)} deleted.")
        return job

    def voice_review(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None:
            raise RuntimeError("Task not found.")
        if not job.work_dir:
            return {"status": "unavailable", "candidates": [], "remaining": 0}
        session = self._reconcile_subtitle_commit(job)
        if session is None:
            return {"status": "unavailable", "candidates": [], "remaining": 0}
        if not self._review_media_is_current(job, session):
            return {"status": "stale_media", "candidates": [], "result_candidates": [], "diagnostics": [], "remaining": 0, "error": "视频源已修改，请重新准备语音复核。"}
        candidates = session.get("candidates") if isinstance(session.get("candidates"), list) else []
        return {
            "status": session.get("status", "pending"),
            "channel": session.get("channel", "mix"),
            "baseline_kind": session.get("baseline_kind", "legacy_partial"),
            "candidates": candidates,
            "windows": session.get("windows", []),
            "result_candidates": session.get("result_candidates", []),
            "diagnostics": session.get("diagnostics", []),
            "remaining": sum(1 for item in candidates if item.get("status") in {"pending", "running"}),
            "recovered_words": session.get("recovered_words", []),
            "review_intervals": session.get("review_intervals", []),
            "actual_decode_seconds": session.get("actual_decode_seconds", 0),
            "resource_profile": normalize_resource_profile(job.options.get("resource_profile", "balanced")),
            "cpu_threads": cpu_thread_limit(job.options.get("resource_profile", "balanced")),
            "resource_info": session.get("resource_info"),
            "error": session.get("error"),
            "updated_at": session.get("updated_at"),
        }

    def prepare_voice_review(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise RuntimeError("Task not found.")
            if self._running_id or self._subtitle_model_action_id or job.status in {"queued", "running", "preparing", "cancelling"} or job_id in self._preparing_ids:
                raise RuntimeError("当前任务正在运行，不能准备语音复核。")
            self._preparing_ids.add(job_id)
        try:
            return self._prepare_voice_review_impl(job_id)
        finally:
            with self._lock:
                self._preparing_ids.discard(job_id)

    def _prepare_voice_review_impl(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None or not job.work_dir:
            raise RuntimeError("Task or output directory is unavailable.")
        if job.status in {"queued", "running", "preparing", "cancelling"} or self._running_id or self._subtitle_model_action_id:
            raise RuntimeError("当前任务正在运行，不能准备语音复核。")
        existing_session = read_session(job.work_dir)
        source = job.artifacts.path("source_segments")
        if not source or not source.is_file():
            raise RuntimeError("请先完成字幕提取，才能准备语音复核。")
        channel = str(job.options.get("audio_channel") or "mix")
        raw = job.artifacts.path("raw_video") or job.raw_video
        if not raw or not raw.is_file():
            raise RuntimeError("当前任务没有可用于语音复核的原视频。")
        if channel in {"left", "right"}:
            channels = audio_stream_channels(raw)
            if channels != 2:
                raise RuntimeError(f"当前音频流为 {channels or '未知'} 声道，左右声道复核只支持明确的双声道音频。")
        current_raw_signature = _path_signature(raw)
        token = hashlib.sha256(str(current_raw_signature or raw).encode("utf-8")).hexdigest()[:12]
        target_audio = job.work_dir / "asr-review" / f"review-audio-{token}-{channel}.wav"
        audio: Path | None = None
        if existing_session:
            prior_audio = Path(str(existing_session.get("audio_path") or ""))
            if (
                self._valid_review_wav(prior_audio)
                and existing_session.get("source_video_signature") == current_raw_signature
                and existing_session.get("channel") == channel
                and existing_session.get("audio_fingerprint") == _file_fingerprint(prior_audio)
            ):
                audio = prior_audio
        if audio is None:
            audio = target_audio
            if not self._valid_review_wav(audio):
                audio.parent.mkdir(parents=True, exist_ok=True)
                temporary_audio = audio.with_name(audio.stem + f".part-{uuid.uuid4().hex[:8]}" + audio.suffix)
                try:
                    extract_audio(raw, temporary_audio, channel=channel)
                    if not self._valid_review_wav(temporary_audio):
                        raise RuntimeError("语音复核音频提取结果不是有效的 16 kHz 单声道 WAV。")
                    temporary_audio.replace(audio)
                except Exception:
                    temporary_audio.unlink(missing_ok=True)
                    raise
        words = source.with_name("words." + source.name[len("segments."):]) if source.name.startswith("segments.") else None
        subtitle_fingerprint_path = job.edited_segments or job.artifacts.path("translated_segments") or source
        baseline_words = words if words and words.is_file() else source
        requested_language = resolve_review_language(
            job.options.get("source_language"), job.artifacts.path("asr_evidence"),
            allow_detected=job.artifacts.subtitle_extraction_mode in {"audio", "audio-fallback", "merged"},
        )
        if requested_language is None and str(job.options.get("source_language") or "").strip().lower() == "auto":
            try:
                provenance = json.loads((job.work_dir / "asr-review" / "language-provenance.json").read_text(encoding="utf-8"))
                if provenance.get("source_video_signature") == current_raw_signature:
                    requested_language = str(provenance.get("language") or "") or None
            except (OSError, ValueError, TypeError):
                pass
        current_audio_fingerprint = _file_fingerprint(audio) if audio.is_file() else None
        current_shift = float(getattr(job.artifacts, "asr_timeline_shift_seconds", 0.0) or 0.0)
        if existing_session and existing_session.get("audio_path") == str(audio) and existing_session.get("audio_fingerprint") == current_audio_fingerprint and existing_session.get("source_video_signature") == current_raw_signature and float(existing_session.get("timeline_shift_seconds") or 0.0) == current_shift and existing_session.get("model") == str(job.options.get("whisper_model_size") or "small") and existing_session.get("language") == requested_language and int(existing_session.get("beam_size") or 5) == int(job.options.get("beam_size") or 5) and existing_session.get("channel") == channel:
            old_baseline = Path(str(existing_session.get("baseline_words_path") or ""))
            if old_baseline.is_file():
                baseline_words = old_baseline
        session = prepare_session(
            work_dir=job.work_dir, job_id=job.id, audio_path=audio,
            baseline_words_path=baseline_words,
            baseline_kind="canonical" if baseline_words == words and words and words.is_file() else "legacy_partial",
            model=str(job.options.get("whisper_model_size") or "small"),
            language=requested_language,
            beam_size=int(job.options.get("beam_size") or 5),
            channel=channel,
            subtitle_revision=int(job.artifacts.revision),
            subtitle_fingerprint=hashlib.sha256(subtitle_fingerprint_path.read_bytes()).hexdigest(),
            timeline_shift_seconds=float(getattr(job.artifacts, "asr_timeline_shift_seconds", 0.0) or 0.0),
            source_video_path=job.artifacts.path("raw_video") or job.raw_video,
        )
        job.artifacts.asr_review = str(session_path(job.work_dir))
        job.add_log(f"语音复核队列已准备：{len(session.get('candidates', []))} 个区间。")
        self._persist(job)
        return self.voice_review(job_id)

    def start_voice_review(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None or not job.work_dir:
            raise RuntimeError("Task or output directory is unavailable.")
        with self._lock:
            if self._running_id or self._subtitle_model_action_id or job.status in {"queued", "running", "preparing", "cancelling"}:
                raise RuntimeError("当前任务正在运行，不能启动语音复核。")
            session = self._reconcile_subtitle_commit(job)
            if session is None:
                raise RuntimeError("请先准备语音复核队列。")
            if not self._review_media_is_current(job, session):
                raise RuntimeError("视频源已修改，请重新准备语音复核。")
            if session.get("status") == "running" or any(item.get("status") == "running" for item in session.get("candidates", [])):
                session = reset_interrupted_round(job.work_dir) or session
            if any(item.get("status") == "pending" for item in session.get("candidates", [])):
                try:
                    resources = review_cuda_memory(job.device, str(session.get("model") or "small"), job.compute_type)
                except ReviewResourceBusy as exc:
                    update_session(job.work_dir, status="paused", error=str(exc), error_code=exc.code, resource_info=exc.resources)
                    job.add_log(str(exc))
                    self._persist(job)
                    return self.voice_review(job_id)
                update_session(job.work_dir, resource_info=resources)
            prepared = start_round(job.work_dir)
            windows = prepared.get("windows", [])
            if not windows:
                return self.voice_review(job_id)
            job.add_log(f"语音复核已启动：本轮 {len(windows)} 个区间，最多 120 秒。")
            job.cancellation.reset()
            self._running_id = job.id
        self._persist(job)
        worker = threading.Thread(target=self._run_voice_review, args=(job,), daemon=True, name=f"asr-review-{job.id}")
        with self._lock:
            self._voice_review_worker = worker
        worker.start()
        return self.voice_review(job_id)

    def _run_voice_review(self, job: DemoJob) -> None:
        control_dir = job.work_dir / "asr-review" / f".worker-{uuid.uuid4().hex[:10]}"
        control_dir.mkdir(parents=True, exist_ok=True)
        request_path = control_dir / "request.json"
        result_path = control_dir / "result.json"
        events_path = control_dir / "events.jsonl"
        request_path.write_text(json.dumps({
            "mode": "asr-review", "work_dir": str(job.work_dir),
            "result_path": str(result_path), "events_path": str(events_path),
            "model_size": str(job.options.get("whisper_model_size") or "small"),
            "device": job.device, "compute_type": job.compute_type,
            "resource_profile": normalize_resource_profile(job.options.get("resource_profile", "balanced")),
            "user_data_root": str(USER_DATA_ROOT),
            "round_id": str((read_session(job.work_dir) or {}).get("round_id") or ""),
            "owner_token": str((read_session(job.work_dir) or {}).get("owner_token") or ""),
        }, ensure_ascii=False), encoding="utf-8")
        process: subprocess.Popen[str] | None = None
        try:
            command = [sys.executable, "--whisper-worker", str(request_path)] if getattr(sys, "frozen", False) else [sys.executable, "-m", "yblocalizer.whisper_worker", str(request_path)]
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            lower_process_priority(process, job.options.get("resource_profile", "balanced"))
            with self._lock:
                self._voice_review_process = process
            while process.poll() is None:
                if job.cancellation.cancelled:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=5)
                    reset_interrupted_round(job.work_dir, status="paused", error="已取消；未完成区间保留待续跑。")
                    return
                time.sleep(0.1)
            if process.returncode:
                detail = ""
                error_code = ""
                if result_path.is_file():
                    try:
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        detail = str(result.get("error") or "")
                        error_code = str(result.get("error_code") or "")
                    except (OSError, ValueError): pass
                if error_code in {"cuda_resource_busy", "cuda_oom"}:
                    job.add_log(f"语音复核已暂停：{detail}")
                    return
                raise RuntimeError(detail or "语音复核子进程失败")
        except Exception as exc:
            log_frontend_error("语音复核失败，已显示在界面")
            job.add_log(f"语音复核失败：{type(exc).__name__}: {exc}")
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)
            shutil.rmtree(control_dir, ignore_errors=True)
            with self._lock:
                if self._voice_review_process is process:
                    self._voice_review_process = None
                if self._running_id == job.id:
                    self._running_id = None
                if self._voice_review_worker is threading.current_thread():
                    self._voice_review_worker = None
            self._persist(job)

    def cancel_voice_review(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None or not job.work_dir:
            raise RuntimeError("Task or output directory is unavailable.")
        session = read_session(job.work_dir)
        if session is None:
            raise RuntimeError("语音复核队列不存在。")
        # Fence and stop the owned process before changing queue state.  This
        # prevents a late worker checkpoint from restoring running windows.
        job.cancellation.cancel()
        with self._lock:
            process = self._voice_review_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        session = reset_interrupted_round(job.work_dir, status="paused", error="已取消；未完成区间保留待续跑。") or session
        job.add_log("语音复核已暂停；已完成区间与候选保留，可继续运行。")
        self._persist(job)
        return self.voice_review(job_id)

    def translate_voice_candidate(self, job_id: str, candidate_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None or not job.work_dir:
            raise RuntimeError("Task or output directory is unavailable.")
        session = self._reconcile_subtitle_commit(job)
        if not session:
            raise RuntimeError("语音复核队列不存在。")
        if not self._review_media_is_current(job, session):
            raise RuntimeError("视频源已修改，请重新准备语音复核。")
        result = next((item for item in session.get("result_candidates", []) if str(item.get("id")) == str(candidate_id)), None)
        candidate = next((item for item in session.get("candidates", []) if str(item.get("id")) == str(candidate_id)), None)
        if candidate is None and isinstance(result, dict):
            candidate = next((item for item in session.get("candidates", []) if str(item.get("id")) == str(result.get("window_id"))), None)
        if candidate is None:
            raise RuntimeError("语音复核候选不存在。")
        words = list(result.get("words") or []) if isinstance(result, dict) else []
        if not words:
            words = [item for item in session.get("recovered_words", []) if item.get("candidate_id") == candidate_id]
        source = "".join(str(item.get("text") or "") for item in words).strip()
        if not source:
            raise RuntimeError("该候选没有可翻译的恢复词，请先完成语音复核。")
        payload = payload or {}
        source = str(payload.get("source_text") or source).strip()
        if not source:
            raise RuntimeError("候选原文不能为空。")
        provider = str(job.options.get("translator") or "none").strip().lower()
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if payload.get("source_hash") and not hmac.compare_digest(str(payload["source_hash"]), source_hash):
            raise RuntimeError("候选原文已变化，请重新加载后再翻译。")
        if provider in {"deepseek", "openai"} and payload.get("confirmed_paid") is not True:
            raise RuntimeError("在线候选翻译可能产生费用，请先确认后再请求。")
        cache_path = job.work_dir / "asr-review" / "translation-cache.json"
        try: cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError): cache = {}
        cache_key = hashlib.sha256(json.dumps({"source": source_hash, "provider": provider, "model": job.options.get("translate_model"), "target": job.options.get("target_lang")}, sort_keys=True).encode()).hexdigest()
        if isinstance(cache, dict) and isinstance(cache.get(cache_key), dict):
            return {**cache[cache_key], "cached": True, "requires_confirmation": provider in {"deepseek", "openai"}}
        with self._lock:
            if self._running_id or self._subtitle_model_action_id:
                raise RuntimeError("当前已有处理或字幕模型请求运行，请完成后再试。")
            self._subtitle_model_action_id = job.id
        context = ""
        canonical = job.artifacts.path("source_segments")
        if canonical and canonical.is_file():
            try:
                nearby = [item.text for item in load_segments(canonical) if abs(item.start - float(result.get("start") or 0)) <= 45]
                context = "\n".join(nearby[:5])
            except (OSError, ValueError, TypeError):
                context = ""
        try:
            with self._llm_usage_session(job, "single"):
                translated = translate_single_cue(
                    source, provider=provider, glossary=load_glossary(job.work_dir),
                    target_lang=str(job.options.get("target_lang") or "zh-Hans"),
                    model=job.options.get("translate_model"),
                    source_lang=str(session.get("language") or "") or None,
                    context=context, smart_translation=bool(job.options.get("smart_translation", True)),
                    deepseek_thinking=str(job.options.get("deepseek_thinking") or "disabled"),
                )
        finally:
            with self._lock:
                if self._subtitle_model_action_id == job.id:
                    self._subtitle_model_action_id = None
        result = {
            "candidate_id": candidate_id, "source": source, "translated": translated,
            "start": result.get("start"), "end": result.get("end"),
            "source_hash": source_hash, "requires_confirmation": provider in {"deepseek", "openai"},
        }
        cache[cache_key] = result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(cache_path)
        return result

    def apply_voice_candidates(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None or not job.work_dir:
            raise RuntimeError("Task or output directory is unavailable.")
        if job.status in {"queued", "running", "preparing", "cancelling"} or self._running_id or self._subtitle_model_action_id:
            raise RuntimeError("当前任务正在运行，不能采用语音复核候选。")
        session = self._reconcile_subtitle_commit(job)
        if not session:
            raise RuntimeError("语音复核队列不存在。")
        if not self._review_media_is_current(job, session):
            raise RuntimeError("视频源已修改，请重新准备语音复核。")
        ids = payload.get("candidate_ids")
        translations = payload.get("translations")
        if not isinstance(ids, list) or len(ids) != len({str(item) for item in ids}):
            raise ValueError("candidate_ids must be a unique list.")
        translations = translations if isinstance(translations, dict) else {}
        confirmed_weak = payload.get("confirmed_weak") if isinstance(payload.get("confirmed_weak"), dict) else {}
        source_path = job.edited_segments or job.artifacts.path("translated_segments")
        if not source_path or not source_path.is_file():
            raise RuntimeError("当前译文字幕不存在。")
        segments = load_segments(source_path)
        candidates = {str(item.get("id")): item for item in session.get("candidates", [])}
        result_candidates = {str(item.get("id")): item for item in session.get("result_candidates", [])}
        additions: list[Segment] = []
        for candidate_id in [str(item) for item in ids]:
            result_candidate = result_candidates.get(candidate_id)
            parent_id = str(result_candidate.get("window_id")) if result_candidate else ""
            candidate = candidates.get(parent_id)
            if not result_candidate or result_candidate.get("status") == "committed" or not candidate or candidate.get("status") != "done":
                raise RuntimeError(f"候选 {candidate_id} 尚未完成复核。")
            text = "".join(str(item.get("text") or "") for item in (result_candidate.get("words") or [])).strip()
            if not text:
                text = "".join(str(item.get("text") or "") for item in session.get("recovered_words", []) if item.get("candidate_id") == candidate_id).strip()
            translated = str(translations.get(candidate_id) or "").strip()
            if not text or not translated:
                raise ValueError(f"候选 {candidate_id} 必须同时提供原文和译文。")
            if result_candidate.get("tier") not in {"strong", "weak"} or (result_candidate.get("tier") == "strong" and result_candidate.get("adoptable") is not True) or (result_candidate.get("tier") == "weak" and result_candidate.get("adoptable") is not False):
                raise RuntimeError("弱复核候选必须先人工确认后才能保存。")
            if result_candidate.get("tier") == "weak" and confirmed_weak.get(candidate_id) is not True:
                raise RuntimeError("弱复核候选必须先确认已重听。")
            start, end = float(result_candidate["start"]), float(result_candidate["end"])
            if any(start < segment.end and end > segment.start for segment in segments) or any(start < item.end and end > item.start for item in additions):
                raise RuntimeError(f"候选 {candidate_id} 与现有字幕时间重叠，不能自动插入。")
            additions.append(Segment(start, end, text, translated, True))
        if not additions:
            return {"preview": True, "candidates": [], "message": "没有选择候选。"}
        # Candidate adoption is preview-only.  The single durable commit path
        # is PUT /cues with base_ids, candidate_rows and CAS metadata.
        return {
            "preview": True,
            "candidates": [
                {"candidate_id": str(item), "start": row.start, "end": row.end, "source": row.text, "translated": row.translated_text, "confirmed_weak": confirmed_weak.get(str(item)) is True}
                for item, row in zip([str(item) for item in ids], additions)
            ],
            "message": "候选仅已加入本地编辑草稿；请在字幕页保存后正式采用。",
        }
    def _begin_subtitle_model_action(self, job_id: str) -> tuple[DemoJob, list[Segment]]:
        with self._lock:
            if self._running_id or self._subtitle_model_action_id:
                raise RuntimeError("当前已有处理或字幕模型请求运行，请完成后再试。")
            job = self._jobs.get(job_id)
            if job is None or job.work_dir is None:
                raise RuntimeError("任务或字幕目录不可用。")
            if job.status in {"queued", "running", "preparing", "cancelling"}:
                raise RuntimeError("任务正在运行，请等待当前阶段结束后再校对。")
            if job.stages.get("translate", {}).get("status") != "completed":
                raise RuntimeError("翻译阶段完成后才能使用字幕模型校对。")
            source_path = job.edited_segments or job.artifacts.path("translated_segments")
            if source_path is None or not source_path.is_file():
                raise RuntimeError("当前任务没有可用的已翻译字幕。")
            try:
                segments = load_segments(source_path)
            except (OSError, TypeError, ValueError) as exc:
                raise RuntimeError("字幕数据无效，请重新加载任务字幕。") from exc
            if not segments:
                raise RuntimeError("当前任务没有可编辑的字幕。")
            self._subtitle_model_action_id = job_id
            return job, segments

    def _end_subtitle_model_action(self, job_id: str) -> None:
        with self._lock:
            if self._subtitle_model_action_id == job_id:
                self._subtitle_model_action_id = None

    @staticmethod
    def _subtitle_model_label(job: DemoJob) -> tuple[str, str]:
        provider = str(job.options.get("translator") or "none").strip().lower()
        provider_label = {
            "deepseek": "DeepSeek",
            "openai": "OpenAI",
            "ollama": "Ollama 本地模型",
        }.get(provider, provider or "未知模型")
        model = str(job.options.get("translate_model") or "").strip()
        if not model:
            model = {
                "deepseek": os.getenv("DEEPSEEK_TRANSLATE_MODEL", "deepseek-v4-flash"),
                "openai": os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4.1-mini"),
                "ollama": "当前 Ollama 模型",
            }.get(provider, provider_label)
        return provider_label, model

    def retranslate_cue(
        self,
        job_id: str,
        cue_id: int,
        source_text: str,
        context_override: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if isinstance(cue_id, bool) or not isinstance(cue_id, int) or cue_id < 0:
            raise ValueError("字幕编号无效。")
        if not isinstance(source_text, str) or len(source_text.strip()) > 2_000:
            raise ValueError("原文必须是 2000 个字符以内的文本。")
        corrected_source = source_text.strip()
        if not corrected_source:
            raise ValueError("原文不能为空。")
        job, segments = self._begin_subtitle_model_action(job_id)
        try:
            if cue_id >= len(segments):
                raise ValueError("字幕编号已变化，请重新加载字幕列表。")
            provider = str(job.options.get("translator") or "none").strip().lower()
            if provider not in {"deepseek", "openai", "ollama"}:
                raise RuntimeError("当前任务未配置可用的翻译模型。")
            context_by_id: dict[int, dict[str, Any]] = {}
            for neighbor_id in (cue_id - 1, cue_id + 1):
                if 0 <= neighbor_id < len(segments):
                    neighbor = segments[neighbor_id]
                    context_by_id[neighbor_id] = {
                        "cue": neighbor_id + 1,
                        "source": neighbor.text,
                        "translation": neighbor.translated_text or "",
                    }
            if context_override is not None:
                if not isinstance(context_override, list) or len(context_override) > 2:
                    raise ValueError("相邻字幕上下文无效。")
                seen_context_ids: set[int] = set()
                for item in context_override:
                    if not isinstance(item, dict):
                        raise ValueError("相邻字幕上下文无效。")
                    neighbor_id = item.get("cue_id")
                    if (
                        isinstance(neighbor_id, bool)
                        or not isinstance(neighbor_id, int)
                        or neighbor_id not in context_by_id
                        or neighbor_id in seen_context_ids
                    ):
                        raise ValueError("相邻字幕上下文编号无效。")
                    source = item.get("source")
                    translation = item.get("translated")
                    if (
                        not isinstance(source, str)
                        or not isinstance(translation, str)
                        or len(source) > 2_000
                        or len(translation) > 2_000
                    ):
                        raise ValueError("相邻字幕上下文长度无效。")
                    context_by_id[neighbor_id] = {
                        "cue": neighbor_id + 1,
                        "source": source,
                        "translation": translation,
                    }
                    seen_context_ids.add(neighbor_id)
            source_language = resolve_review_language(
                job.options.get("source_language"),
                job.artifacts.path("asr_evidence"),
                allow_detected=job.artifacts.subtitle_extraction_mode in {"audio", "audio-fallback"},
            )
            configured_source_language = str(job.options.get("source_language") or "").strip()
            if source_language is None and configured_source_language and configured_source_language.lower() != "auto":
                source_language = configured_source_language
            if source_language is None and job.artifacts.asr_evidence and job.artifacts.subtitle_extraction_mode in {"audio", "audio-fallback"}:
                try:
                    evidence = json.loads(Path(job.artifacts.asr_evidence).read_text(encoding="utf-8"))
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
                    source_language = detected
            with self._llm_usage_session(job, "single"):
                candidate = translate_single_cue(
                    corrected_source, glossary=load_glossary(job.work_dir),
                    provider=provider,
                    target_lang=str(job.options.get("target_lang") or "zh-Hans"),
                    model=str(job.options.get("translate_model") or "") or None,
                    source_lang=source_language,
                    context=json.dumps(
                        [context_by_id[index] for index in sorted(context_by_id)],
                        ensure_ascii=False,
                    ),
                    smart_translation=bool(job.options.get("smart_translation", True)),
                    deepseek_thinking=str(job.options.get("deepseek_thinking") or "disabled"),
                )
            provider_label, model_label = self._subtitle_model_label(job)
            return {
                "cue_id": cue_id,
                "translation": candidate,
                "provider": provider_label,
                "model": model_label,
            }
        finally:
            self._end_subtitle_model_action(job_id)

    def compact_cues(
        self,
        job_id: str,
        cue_ids: list[int],
        cue_overrides: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(cue_ids, list) or not cue_ids or len(cue_ids) > 50:
            raise ValueError("请选择 1–50 条字幕生成精简建议。")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in cue_ids):
            raise ValueError("字幕编号无效。")
        if len(set(cue_ids)) != len(cue_ids):
            raise ValueError("字幕编号不能重复。")
        job, segments = self._begin_subtitle_model_action(job_id)
        try:
            indexes = sorted(cue_ids)
            if indexes[-1] >= len(segments):
                raise ValueError("字幕编号已变化，请重新加载字幕列表。")
            provider = str(job.options.get("translator") or "none").strip().lower()
            if provider not in {"deepseek", "openai", "ollama"}:
                raise RuntimeError("当前任务未配置可用的翻译模型。")
            preview_segments = copy.deepcopy(segments)
            if cue_overrides is not None:
                if not isinstance(cue_overrides, list) or len(cue_overrides) > 50:
                    raise ValueError("字幕草稿数据无效。")
                seen_override_ids: set[int] = set()
                for item in cue_overrides:
                    if not isinstance(item, dict):
                        raise ValueError("字幕草稿数据无效。")
                    override_id = item.get("cue_id")
                    source = item.get("source")
                    translated = item.get("translated")
                    if (
                        isinstance(override_id, bool)
                        or not isinstance(override_id, int)
                        or override_id not in indexes
                        or override_id in seen_override_ids
                        or not isinstance(source, str)
                        or not isinstance(translated, str)
                        or len(source) > 2_000
                        or len(translated) > 2_000
                    ):
                        raise ValueError("字幕草稿数据无效。")
                    preview_segments[override_id].text = source
                    preview_segments[override_id].translated_text = translated
                    seen_override_ids.add(override_id)
            line_width, display_limit = self._cue_layout_limits(job)
            target_units = {
                index: max(1, min(36, int(max(0.1, preview_segments[index].end - preview_segments[index].start) * 12)))
                for index in indexes
            }
            errors: dict[int, str] = {}
            with self._llm_usage_session(job, "formatting"):
                candidates, unresolved = compact_overflowing_subtitles(
                    preview_segments,
                    indexes,
                    provider=provider,
                    target_lang=str(job.options.get("target_lang") or "zh-Hans"),
                    model=str(job.options.get("translate_model") or "") or None,
                    max_chars_per_line=line_width,
                    max_display_units_per_line=display_limit,
                    smart_layout=bool(job.options.get("smart_subtitle_layout", True)),
                    checkpoint_path=None,
                    force_compact=True,
                    target_display_units=target_units,
                    max_attempts=1,
                    errors=errors,
                    display_mode=str(job.options.get("subtitle_display_mode") or "translated"),
                    deepseek_thinking=str(job.options.get("deepseek_thinking") or "disabled"),
                )
            provider_label, model_label = self._subtitle_model_label(job)
            unresolved_indexes = set(unresolved)
            return {
                "provider": provider_label,
                "model": model_label,
                "request_count": sum(1 for index in indexes if preview_segments[index].translated_text),
                "candidates": [
                    {
                        "cue_id": index,
                        "original": preview_segments[index].translated_text or preview_segments[index].text,
                        "candidate": candidates[index].translated_text or preview_segments[index].text,
                        "error": errors.get(index),
                        "still_overflowing": index in unresolved_indexes,
                    }
                    for index in indexes
                ],
            }
        finally:
            self._end_subtitle_model_action(job_id)

    def shutdown(self, timeout: float = 8.0) -> None:
        """Cancel the active stage and preserve a recoverable status before exit."""
        with self._lock:
            job = self._jobs.get(self._running_id or "")
            worker = self._worker
            review_worker = self._voice_review_worker
            review_process = self._voice_review_process
            review_job = self._jobs.get(self._running_id or "") if review_worker else None
            if review_job:
                review_job.cancellation.cancel()
                review_job.add_log("Application is closing; ASR review worker cancellation requested.")
            if review_process and review_process.poll() is None:
                review_process.terminate()
            if job and job.status in {"queued", "running", "cancelling"}:
                job.status, job.stage = "cancelling", "正在安全停止当前阶段"
                job.add_log("Application is closing; cancellation requested.")
                job.cancellation.cancel()
        if not job and not review_job:
            return
        if review_worker and review_worker.is_alive() and review_worker is not threading.current_thread():
            review_worker.join(max(0.0, timeout))
        if review_job:
            self._persist(review_job)
            if review_process and review_process.poll() is None:
                review_process.terminate()
                try: review_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    review_process.kill(); review_process.wait(timeout=5)
            try: reset_interrupted_round(review_job.work_dir, status="pending", error="应用关闭，当前复核窗口可继续。")
            except OSError: pass
            return
        self._persist(job)
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(max(0.0, timeout))
        if not worker or worker.is_alive():
            with self._lock:
                stage = job.current_stage
                if stage:
                    job.stages[stage].update({"status": "interrupted", "finished_at": time.time()})
                    job.next_stage = stage
                job.status, job.stage, job.finished_at, job.auto_run = "interrupted", "应用关闭，当前阶段已中断", time.time(), False
                job.add_log("Shutdown timeout reached; the current stage can be resumed after restart.")
            self._persist(job)

    def align(self, job_id: str) -> DemoJob:
        """Re-align the finished subtitles using OCR anchors (manual fallback)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.work_dir is None or job.raw_video is None or job.status != "completed":
                raise RuntimeError("完成初始任务后才能对齐字幕。")
            if self._running_id or self._subtitle_model_action_id:
                raise RuntimeError("已有任务在运行。")
        work_dir = job.work_dir
        source_parent = job.artifacts.path("source_segments")
        source_parent = source_parent.parent if source_parent else work_dir
        audio_path = source_parent / "segments.audio.json"
        ocr_path = source_parent / "segments.ocr.json"
        if not audio_path.exists() or not ocr_path.exists():
            audio_path, ocr_path = work_dir / "segments.audio.json", work_dir / "segments.ocr.json"
        if not audio_path.exists() or not ocr_path.exists():
            raise RuntimeError("该任务没有同时包含音频与画面字幕，无法自动对齐。")
        with self._lock:
            self._running_id = job.id
            job.cancellation.reset()
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
            current_path = job.artifacts.path("translated_segments") or job.edited_segments or work_dir / "segments.translated.json"
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
            self._finish_modify(job, job.raw_video, f"Aligned subtitles by {offset:+.2f}s.", subtitle=new_srt, segments_file=new_segments, operation="align")
        except CancellationRequested:
            with self._lock:
                job.status, job.stage, job.error = "cancelled", "对齐已取消", None
                job.finished_at = time.time()
                job.add_log("Alignment cancelled; the previous rendered video was kept.")
        except Exception as exc:
            log_frontend_error("对齐失败，已显示在界面")
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
            if self._running_id or self._subtitle_model_action_id:
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
                if any(type(value) not in {int, float} for value in raw_points):
                    raise TypeError
                if any(type(value) is not int for value in raw_order):
                    raise TypeError
                times = [float(p) for p in raw_points]
                indexes = [int(i) for i in raw_order]
            except (TypeError, ValueError):
                raise RuntimeError("重排参数必须是数字。") from None
            if not all(math.isfinite(value) for value in times):
                raise RuntimeError("重排时间点必须是有限数字。")
            boundaries = [0.0] + sorted(t for t in times if 0 < t < duration) + [duration]
            boundaries = [boundaries[0]] + [b for a, b in zip(boundaries, boundaries[1:]) if b > a + 0.1]
            count = len(boundaries) - 1
            if count < 2:
                raise RuntimeError("至少需要两个片段才能重排。")
            if sorted(indexes) != list(range(count)):
                raise RuntimeError("新顺序必须包含每一段且不重复（如 3,1,2）。")
        with self._lock:
            self._running_id = job.id
            job.cancellation.reset()
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
        except CancellationRequested:
            with self._lock:
                job.status, job.stage, job.error = "cancelled", "修改已取消", None
                job.finished_at = time.time()
                job.add_log("Modify cancelled; the previous rendered video was kept.")
        except Exception as exc:
            log_frontend_error("修改失败，已显示在界面")
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
                *ffmpeg_thread_args(job.options["resource_profile"]),
                str(kept_video),
            ],
            cancel_check=lambda: job.cancellation.cancelled,
            resource_profile=job.options["resource_profile"],
        )
        if not kept_video.exists():
            raise RuntimeError("截取保留没有生成输出文件。")
        segments_path = job.artifacts.path("translated_segments") or job.edited_segments or work_dir / "segments.translated.json"
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
            operation="keep",
        )

    def _run_mute(self, job: DemoJob, ranges: list[tuple[float, float]]) -> None:
        work_dir = job.work_dir
        raw = job.raw_video
        muted = work_dir / f"muted-{uuid.uuid4().hex[:8]}.mp4"
        volume_expr = ",".join(f"volume=enable='between(t,{a + 0.05:.3f},{b - 0.05:.3f})':volume=0" for a, b in ranges)
        run_command(
            [
                "ffmpeg", "-y", "-i", str(raw),
                "-af", volume_expr,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-map", "0:v:0?", "-map", "0:a:0?",
                *ffmpeg_thread_args(job.options["resource_profile"]),
                str(muted),
            ],
            cancel_check=lambda: job.cancellation.cancelled,
            resource_profile=job.options["resource_profile"],
        )
        if not muted.exists():
            raise RuntimeError("静音处理没有生成输出文件。")
        subtitle = job.artifacts.path("translated_srt")
        segments_file = job.artifacts.path("translated_segments")
        if not subtitle or not validate_srt(subtitle):
            raise RuntimeError("当前活动字幕无效，静音后无法安全恢复硬字幕成片。")
        self._finish_modify(
            job, muted, f"Muted {len(ranges)} segment(s).",
            subtitle=subtitle, segments_file=segments_file, operation="mute",
        )

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
                        *ffmpeg_thread_args(job.options["resource_profile"]),
                        str(part),
                    ],
                    cancel_check=lambda: job.cancellation.cancelled,
                    resource_profile=job.options["resource_profile"],
                )
                parts.append(part)
            reordered = work_dir / f"reordered-{uuid.uuid4().hex[:8]}.mp4"
            list_file = work_dir / "concat.txt"
            list_file.write_text(
                "\n".join(f"file '{_ffconcat_path(parts[i])}'" for i in indexes) + "\n", encoding="utf-8")
            run_command(
                [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c", "copy", str(reordered),
                ],
                cancel_check=lambda: job.cancellation.cancelled,
                resource_profile=job.options["resource_profile"],
            )
        finally:
            for part in parts:
                part.unlink(missing_ok=True)
            (work_dir / "concat.txt").unlink(missing_ok=True)
        if not reordered.exists():
            raise RuntimeError("重排没有生成输出文件。")

        segments_path = job.artifacts.path("translated_segments") or job.edited_segments or work_dir / "segments.translated.json"
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
            operation="reorder",
        )

    def _finish_modify(
        self,
        job: DemoJob,
        new_raw: Path,
        log_message: str,
        subtitle: Path | None = None,
        segments_file: Path | None = None,
        operation: str = "modify",
    ) -> None:
        work_dir = job.work_dir
        if work_dir is None or not validate_media(new_raw, "video"):
            raise RuntimeError("修改后的视频验证失败，已保留上一版成片。")
        candidate = work_dir / f"rendered.next-{uuid.uuid4().hex[:8]}.mp4"
        if subtitle is not None and validate_srt(subtitle):
            rendered_candidate = burn_subtitles(
                new_raw, subtitle, candidate,
                font_name=job.options["font_name"], font_size=job.options["font_size"],
                primary_color=job.options["subtitle_color"], outline_color=job.options["subtitle_outline_color"],
                outline=job.options["subtitle_outline"], shadow=job.options["subtitle_shadow"],
                raised_margin=job.artifacts.used_ocr_subtitles, crf=job.options["render_crf"],
                margin_ratio=job.options["subtitle_margin_ratio"], encoder=job.options["render_encoder"],
                log=job.add_log, cancel_check=lambda: job.cancellation.cancelled,
                resource_profile=job.options["resource_profile"],
            )
        else:
            shutil.copy2(new_raw, candidate)
            rendered_candidate = candidate
            with self._lock:
                job.add_log("修改后没有剩余有效字幕，成片仅保留视频内容。")
        job.cancellation.check()
        if not validate_media(rendered_candidate, "video"):
            raise RuntimeError("新成片验证失败，已保留上一版成片。")
        rendered = work_dir / "rendered.mp4"
        rendered_candidate.replace(rendered)
        new_duration = _probe_media(new_raw)[0]
        # A video edit changes the source clock and invalidates the extracted
        # audio/word timeline. Preserve the old review session for audit, but
        # never let a later prepare call reuse it for the new raw video.
        review_file = session_path(work_dir)
        prior_session = read_session(work_dir)
        prior_language = (prior_session or {}).get("language")
        prior_raw_signature = _path_signature(new_raw)
        if review_file.is_file():
            stale_review = review_file.with_name(f"session-stale-{uuid.uuid4().hex[:10]}.json")
            review_file.replace(stale_review)
        if prior_language and prior_raw_signature:
            _atomic_json(work_dir / "asr-review" / "language-provenance.json", {"language": prior_language, "source_video_signature": prior_raw_signature, "reason": "video edit preserved source language"})
        current_source_segments = None
        current_source_srt = None
        if segments_file is not None and segments_file.is_file():
            current_source_segments = work_dir / f"segments.source.current.{uuid.uuid4().hex[:8]}.json"
            current_source_srt = work_dir / f"source.current.{uuid.uuid4().hex[:8]}.srt"
            remapped_segments = load_segments(segments_file)
            source_only = [Segment(item.start, item.end, item.text, None, item.needs_review) for item in remapped_segments]
            save_segments(current_source_segments, source_only)
            write_srt(current_source_srt, source_only, display_mode="source")
        with self._lock:
            job.artifacts.original_video = job.artifacts.original_video or job.artifacts.raw_video or str(new_raw)
            job.artifacts.raw_video = str(new_raw)
            job.artifacts.audio = None
            job.artifacts.asr_evidence = None
            job.artifacts.asr_review = None
            job.artifacts.asr_timeline_shift_seconds = 0.0
            job.artifacts.rendered_video = str(rendered)
            job.artifacts.revision += 1
            job.artifacts.last_edit = operation
            job.artifacts.edit_state = "modified"
            if subtitle is not None:
                job.artifacts.translated_srt = str(subtitle)
            if segments_file is not None:
                job.artifacts.translated_segments = str(segments_file)
            if current_source_segments is not None:
                job.artifacts.source_segments = str(current_source_segments)
                job.artifacts.source_srt = str(current_source_srt)
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
            job.stages["render"].update({"status": "completed", "progress": 100, "error": None, "finished_at": job.finished_at})
            publish = job.stages.get("publish")
            if publish and publish.get("status") == "completed":
                publish.update({"status": "stale", "progress": 0, "error": None})
            job.next_stage = None
        self._persist(job)

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
                *ffmpeg_thread_args(job.options["resource_profile"]),
                str(out),
            ],
            cancel_check=lambda: job.cancellation.cancelled,
            resource_profile=job.options["resource_profile"],
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
            if self._running_id or self._subtitle_model_action_id:
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
            job.cancellation.reset()
        try:
            self._run_trim(job, ranges)
        except CancellationRequested:
            with self._lock:
                job.status, job.stage, job.error = "cancelled", "裁剪已取消", None
                job.finished_at = time.time()
                job.add_log("Trim cancelled; the previous rendered video was kept.")
        except Exception as exc:
            log_frontend_error("裁剪失败，已显示在界面")
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
                *ffmpeg_thread_args(job.options["resource_profile"]),
                str(trimmed),
            ],
            cancel_check=lambda: job.cancellation.cancelled,
            resource_profile=job.options["resource_profile"],
        )
        if not trimmed.exists():
            raise RuntimeError("Trimming produced no output video.")

        segments_path = job.artifacts.path("translated_segments") or job.edited_segments or work_dir / "segments.translated.json"
        segments = load_segments(segments_path)
        kept = _remap_segments_after_cut(segments, ranges)
        new_segments = work_dir / "segments.trimmed.json"
        new_srt = work_dir / "zh.trimmed.srt"
        save_segments(new_segments, kept)
        write_srt(new_srt, kept, display_mode=job.options["subtitle_display_mode"], smart_layout=job.options["smart_subtitle_layout"])

        self._finish_modify(
            job, trimmed, f"Trim finished: kept {len(kept)} cues.",
            subtitle=new_srt if kept else None, segments_file=new_segments, operation="trim",
        )

    def rerender(self, job_id: str) -> DemoJob:
        job = self.get(job_id)
        if job and job.checkpoint_validation == "pending":
            self._verify_checkpoints(job)
            self._persist(job)
        if job and job.artifacts.edit_state == "legacy-ambiguous":
            raise RuntimeError("旧任务包含无法确定顺序的后期产物。现有成片已保留，但为防止覆盖，不能重新渲染；请新建任务或手动选择最终版本。")
        if job and job.workflow_version >= WORKFLOW_VERSION and job.stages.get("translate", {}).get("status") == "completed":
            return self.run_stage(job_id, "render")
        with self._lock:
            if self._running_id or self._subtitle_model_action_id:
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
            job.cancellation.reset()
        threading.Thread(target=self._run_rerender, args=(job, subtitle), daemon=True, name=f"rerender-{job.id}").start()
        return job

    def media(self, job_id: str, kind: str) -> Path | None:
        job = self.get(job_id)
        if job is None:
            return None
        if kind == "source":
            return job.raw_video or job.material.path
        if kind == "rendered":
            return job.rendered_video or job.artifacts.path("rendered_video")
        if kind == "dub":
            return job.artifacts.path("dub_video")
        return None

    def _run_rerender(self, job: DemoJob, subtitle: Path) -> None:
        candidate: Path | None = None
        try:
            candidate = job.work_dir / f"rendered.next-{uuid.uuid4().hex[:8]}.mp4"  # type: ignore[operator]
            rendered_candidate = burn_subtitles(
                job.raw_video, subtitle, candidate,  # type: ignore[arg-type]
                font_name=job.options["font_name"], font_size=job.options["font_size"],
                primary_color=job.options["subtitle_color"], outline_color=job.options["subtitle_outline_color"],
                outline=job.options["subtitle_outline"], shadow=job.options["subtitle_shadow"],
                crf=job.options["render_crf"], margin_ratio=job.options["subtitle_margin_ratio"],
                encoder=job.options["render_encoder"], log=job.add_log,
                cancel_check=lambda: job.cancellation.cancelled,
                resource_profile=job.options["resource_profile"],
            )
            job.cancellation.check()
            if not validate_media(rendered_candidate, "video"):
                raise RuntimeError("新成片验证失败，已保留上一版成片。")
            rendered = commit_rendered_candidate(rendered_candidate, job.work_dir)  # type: ignore[arg-type]
        except CancellationRequested:
            with self._lock:
                job.status, job.stage, job.error = "cancelled", "重新渲染已取消", None
                job.finished_at = time.time()
                job.add_log("Re-render cancelled; the previous rendered video was kept.")
        except Exception as exc:
            log_frontend_error("重新渲染失败，已显示在界面")
            with self._lock:
                job.status, job.stage, job.error = "failed", "重新渲染失败", str(exc)
                job.finished_at = time.time()
                job.add_log(f"Re-render failed: {exc}")
        else:
            with self._lock:
                if job.result:
                    job.result["rendered_video"] = public_path(rendered) or rendered.name
                job.rendered_video = rendered
                job.artifacts.rendered_video = str(rendered)
                job.add_log("Edited rendered.mp4 created successfully.")
                job.status, job.stage, job.progress = "completed", "重新渲染完成", 100
                job.finished_at = time.time()
        finally:
            if candidate is not None:
                candidate.unlink(missing_ok=True)
            with self._lock:
                if self._running_id == job.id:
                    self._running_id = None
            self._persist(job)


class WorkbenchHandler(BaseHTTPRequestHandler):
    frontend_dir: Path
    jobs: WorkbenchJobs
    dependencies: DependencyManager
    csrf_token: str
    _external_consent_lock = threading.Lock()
    _external_consents: dict[str, tuple[float, tuple[str, ...]]] = {}
    _external_consent_limit = 512
    server_version = "YBLocalizerWorkbench/2.0"

    @staticmethod
    def _consent_text(value: Any, limit: int = 120) -> str:
        text = re.sub(r"[^\w .:/@+_-]", "", str(value or "").strip())
        return text[:limit]

    def _external_ai_binding(self, path: str, payload: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, str]] | None:
        """Return a stable consent binding for routes that can call online AI."""
        job_id = None
        match = re.match(r"/api/jobs/([^/]+)", path)
        if match:
            job_id = match.group(1)
        elif path == "/api/publish/copy":
            job_id = str(payload.get("job_id") or "") or None
        job = self.jobs.get(job_id) if job_id else None
        options = payload.get("options") if path in {"/api/jobs", "/api/batch/start"} else (job.options if job else None)
        options = options if isinstance(options, dict) else {}
        translator = str(options.get("translator") or ("deepseek" if path in {"/api/jobs", "/api/batch/start"} else "")).lower()
        if translator not in {"deepseek", "openai", "ollama"}:
            return None
        guarded = re.search(r"/(run-all|resume|retranslate|compact|stages/(translate|render)/run|voice-review/candidates/[^/]+/translate)$", path) or (path == "/api/publish/copy" and payload.get("generate") is True) or path == "/api/batch/start"
        if guarded:
            provider, resolved_model_raw, origin, local = ai_provider_request_info(translator, options.get("translate_model"))
            if local:
                return None
            model = self._consent_text(resolved_model_raw)
            operation = "在线 AI 翻译"
            if "/retranslate" in path: operation = "在线 AI 单条重译"
            elif "/compact" in path: operation = "在线 AI 压缩候选"
            elif "voice-review" in path: operation = "在线 AI 语音复核候选"
            elif path == "/api/publish/copy": operation = "在线 AI 生成投稿文案"
            elif path == "/api/jobs": operation = "在线 AI 主任务翻译"
            data_kinds = "OCR/转写字幕、字幕原文、任务标题/简介、相邻上下文"
            if path == "/api/publish/copy": data_kinds = "字幕文本、来源标题和简介、投稿文案上下文"
            elif "voice-review" in path: data_kinds = "语音复核候选文本、相邻字幕上下文"
            material = str(payload.get("material_id") or payload.get("source_url") or job_id or "")
            revision = str(job.artifacts.revision if job else "")
            body_for_binding = {key: value for key, value in payload.items() if key != "external_ai_consent"}
            request_hash = hashlib.sha256(json.dumps(body_for_binding, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            binding = (self.command, path, provider, origin, resolved_model_raw[:256], data_kinds, material[:128], revision[:128], request_hash)
            return binding, {"provider": provider, "endpoint_origin": origin, "model": model, "data_kinds": data_kinds, "operation": operation, "fee_notice": "可能产生服务商 API 费用，请核对服务商账单。"}
        return None

    def _require_external_ai_consent(self, path: str, payload: dict[str, Any]) -> bool:
        result = self._external_ai_binding(path, payload)
        if result is None:
            return True
        binding, disclosure = result
        supplied = payload.get("external_ai_consent")
        now = time.monotonic()
        with self._external_consent_lock:
            for key, (expires, _) in list(self._external_consents.items()):
                if expires <= now:
                    self._external_consents.pop(key, None)
            if isinstance(supplied, dict) and supplied.get("approved") is True:
                token = str(supplied.get("consent_id") or "")
                stored = self._external_consents.get(token)
                if stored and stored[0] > now and hmac.compare_digest("|".join(stored[1]).encode("utf-8"), "|".join(binding).encode("utf-8")):
                    self._external_consents.pop(token, None)
                    payload.pop("external_ai_consent", None)
                    return True
        consent_id = secrets.token_urlsafe(24)
        with self._external_consent_lock:
            while len(self._external_consents) >= self._external_consent_limit:
                oldest = min(self._external_consents, key=lambda key: self._external_consents[key][0])
                self._external_consents.pop(oldest, None)
            self._external_consents[consent_id] = (now + 600, binding)
        self._json({"code": "external_ai_consent_required", "consent_id": consent_id, "disclosure": disclosure}, HTTPStatus.CONFLICT)
        return False

    def do_OPTIONS(self) -> None:  # noqa: N802
        origin = self.headers.get("Origin", "")
        if origin and not self._origin_allowed(origin):
            self._json({"error": "不允许的请求来源。"}, HTTPStatus.FORBIDDEN); return
        self.send_response(HTTPStatus.NO_CONTENT); self._cors_headers(); self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        # Browser fetch percent-encodes Unicode task IDs. Decode once before
        # route matching, otherwise paths for titles containing accented or
        # CJK characters fall through to the frontend HTML handler.
        path = unquote(urlparse(self.path).path)
        if path == "/api/health":
            self._json({"ok": True, "version": __version__}); return
        if path == "/api/bootstrap":
            self._json({
                "version": __version__,
                "csrf_token": self.csrf_token,
                "defaults": default_options(str(OUTPUT_ROOT)),
                "capabilities": capabilities(),
                "demo": {
                    "name": "authorized-demo-10s.mp4", "duration_seconds": 10,
                    "source_media": "/api/demo/source", "rendered_media": "/api/demo/rendered",
                    "cues": "/api/demo/cues",
                },
                "legacy_gui": {"supported": True, "status": "maintenance", "message": "旧 Tk GUI 仅维护兼容性，不再新增功能。"},
            }); return
        if path in {"/api/demo/source", "/api/demo/rendered"}:
            demo_path = DEMO_VIDEO if path.endswith("source") else ASSET_ROOT / "demo" / "artifacts" / "rendered.mp4"
            if demo_path.is_file(): self._serve_media(demo_path)
            else: self._json({"error": "Demo artifact is not available."}, HTTPStatus.NOT_FOUND)
            return
        if path == "/api/demo/cues":
            source_path = ASSET_ROOT / "demo" / "artifacts" / "source.srt"
            translated_path = ASSET_ROOT / "demo" / "artifacts" / "zh.srt"
            self._json({"cues": _demo_cues(source_path, translated_path), "translation_ready": True}); return
        if path == "/api/settings":
            cookies_file = os.getenv("YBLOCALIZER_COOKIES_FILE", "")
            try:
                default_output_dir = str(OUTPUT_ROOT.relative_to(PROJECT_ROOT))
            except ValueError:
                default_output_dir = str(OUTPUT_ROOT)
            dub_status = voxcpm_status()
            self._json({
                "voxcpm_available": dub_status["voxcpm_available"], "voxcpm_reason": dub_status["voxcpm_reason"],
                "voxcpm_help": dub_status["voxcpm_help"],
                "deepseek_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
                "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
                "default_output_dir": default_output_dir,
                "cookies_from_browser": os.getenv("YBLOCALIZER_COOKIES_FROM_BROWSER", ""),
                "cookies_file": cookies_file,
                "cookies_file_valid": _cookies_file_has_login(Path(cookies_file)) if cookies_file else None,
                "youtube_proxy_configured": bool(os.getenv("YBLOCALIZER_YOUTUBE_PROXY")),
                "youtube_po_token": po_token_provider_status(),
            }); return
        if path == "/api/subscriptions":
            self._json(subscription_view()); return
        if path == "/api/subscriptions/weekly":
            self._json(weekly_snapshot()); return
        if path == "/api/tutorial-preferences":
            self._json(_read_tutorial_preferences()); return
        if path == "/api/diagnostics":
            self._json(self._diagnostics()); return
        if path == "/api/dependencies":
            query = parse_qs(urlparse(self.path).query)
            ocr_language = str(query.get("ocr_language", ["eng"])[0])
            if query.get("refresh") == ["1"]:
                clear_dependency_cache()
            self._json(self.dependencies.snapshot(ocr_language)); return
        match = re.fullmatch(r"/api/dependencies/jobs/([a-f0-9]+)", path)
        if match:
            install_job = self.dependencies.get_job(match.group(1))
            self._json(install_job.snapshot() if install_job else {"error": "安装任务不存在。"}, HTTPStatus.OK if install_job else HTTPStatus.NOT_FOUND)
            return
        if path == "/api/templates":
            query = parse_qs(urlparse(self.path).query)
            platform = normalize_publish_platform(str(query.get("platform", ["bilibili"])[0]))
            self._json({
                "platform": platform,
                "templates": [
                    {"name": name, "body": body}
                    for name, body in get_all_templates(platform).items()
                ],
            })
            return
        if path == "/api/outputs":
            self._json(self._outputs_snapshot(OUTPUT_ROOT)); return
        if path == "/api/publish/status":
            browser = "msedge" if str(parse_qs(urlparse(self.path).query).get("browser", ["chromium"])[0]).lower() in {"edge", "msedge"} else "chromium"
            platform = "douyin" if str(parse_qs(urlparse(self.path).query).get("platform", ["bilibili"])[0]).lower() == "douyin" else "bilibili"
            pids = self._publish_profile_process_ids(browser, platform)
            platform_name = "抖音" if platform == "douyin" else "B 站"
            self._json({"profile_busy": bool(pids), "process_count": len(pids), "message": f"{platform_name}自动化浏览器正在使用中。" if pids else f"{platform_name}专用浏览器资料可用。"}); return
        if path == "/api/publish/assist/status":
            self._json(self.jobs.publish_session().snapshot()); return
        if path == "/api/materials":
            self._json({"materials": self.jobs.materials()}); return
        if path == "/api/batch":
            self._json(self.jobs.batch_snapshot()); return
        if path == "/api/dub":
            self._json(self._dub_status()); return
        if path == "/api/glossary":
            glossary_path = global_glossary_path(OUTPUT_ROOT)
            self._json({"entries": read_glossary_file(glossary_path if glossary_path.is_file() else DEFAULT_GLOSSARY_PATH)}); return
        if path == "/api/jobs/restorable":
            try:
                limit = int(parse_qs(urlparse(self.path).query).get("limit", ["20"])[0])
            except ValueError:
                limit = 20
            self._json({"jobs": self.jobs.restorable(limit)}); return
        if path == "/api/history/jobs":
            try:
                limit = int(parse_qs(urlparse(self.path).query).get("limit", ["50"])[0])
            except ValueError:
                limit = 50
            query = parse_qs(urlparse(self.path).query)
            scope = str(query.get("scope", ["current"])[0]).lower()
            output_dir = str(query.get("output_dir", [str(OUTPUT_ROOT)])[0])
            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
            except ValueError:
                self._json({"error": "历史记录偏移量必须是数字。"}, HTTPStatus.BAD_REQUEST); return
            search = str(query.get("q", [""])[0]).strip().casefold()
            category = str(query.get("category", [""])[0]).strip().lower()
            if category not in {"", "active", "attention", "completed"}:
                self._json({"error": "历史记录分类无效。"}, HTTPStatus.BAD_REQUEST); return
            try:
                records = self._history_records(scope, output_dir)
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST); return
            if search:
                records = [record for record in records if search in " ".join(
                    str(record.get(key) or "") for key in ("title", "stage", "status", "error")
                ).casefold()]
            counts = {
                "active": sum(record.get("status") in {"queued", "running", "preparing", "cancelling"} for record in records),
                "attention": sum(record.get("status") not in {"queued", "running", "preparing", "cancelling", "completed"} for record in records),
                "completed": sum(record.get("status") == "completed" for record in records),
            }
            if category == "active":
                records = [record for record in records if record.get("status") in {"queued", "running", "preparing", "cancelling"}]
            elif category == "attention":
                records = [record for record in records if record.get("status") not in {"queued", "running", "preparing", "cancelling", "completed"}]
            elif category == "completed":
                records = [record for record in records if record.get("status") == "completed"]
            page_size = max(1, min(200, limit))
            page = records[offset:offset + page_size]
            self._json({
                "jobs": page, "total": len(records), "offset": offset,
                "has_more": offset + len(page) < len(records), "scope": scope, "counts": counts,
            }); return
        match = re.fullmatch(r"/api/materials/([\w-]+)/media", path)
        if match:
            media = self.jobs.material_path(match.group(1))
            if media is None or not media.exists(): self._json({"error": "Material is not available."}, HTTPStatus.NOT_FOUND)
            else: self._serve_media(media)
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/media/(source|rendered|dub)", path)
        if match:
            media = self.jobs.media(*match.groups())
            if media is None or not media.exists(): self._json({"error": "Requested media is not ready."}, HTTPStatus.NOT_FOUND)
            else: self._serve_media(media)
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/cues", path)
        if match:
            if self.jobs.get(match.group(1)) is None:
                self._json({"error": "Task not found."}, HTTPStatus.NOT_FOUND)
                return
            try: self._json(self.jobs.cues(match.group(1)))
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/voice-review", path)
        if match:
            try: self._json(self.jobs.voice_review(match.group(1)))
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/logs", path)
        if match:
            try:
                limit = int(parse_qs(urlparse(self.path).query).get("limit", ["5000"])[0])
                self._json(self.jobs.full_logs(match.group(1), limit))
            except ValueError:
                self._json({"error": "日志条数必须是数字。"}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})", path)
        if match:
            job = self.jobs.get(match.group(1))
            query = parse_qs(urlparse(self.path).query)
            try:
                log_after = int(query["log_after"][0]) if "log_after" in query else None
                log_limit = int(query.get("log_limit", ["200"])[0])
            except ValueError:
                self._json({"error": "日志游标必须是数字。"}, HTTPStatus.BAD_REQUEST); return
            self._json(job.snapshot(log_after=log_after, log_limit=log_limit) if job else {"error": "Task not found."}, HTTPStatus.OK if job else HTTPStatus.NOT_FOUND)
            return
        self._serve_frontend(path)

    @staticmethod
    def _dub_status() -> dict[str, Any]:
        return {**voxcpm_status(), "dub_mix_modes": list(DUB_MIX_MODES), "default_dub_mix": "duck", "default_dub_enabled": False}

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if not self._authorize_mutation(path):
            return
        if path == "/api/dub/recheck":
            voxcpm_status.cache_clear()
            self._json(self._dub_status()); return
        if path == "/api/materials":
            self._upload_material(); return
        if path == "/api/publish/upload":
            self._upload_publish_video(); return
        try:
            payload = self._payload()
        except PayloadReadError as exc:
            self._json(exc.payload(), exc.status); return
        if path == "/api/diagnostics/export":
            try:
                archive = export_diagnostics_zip(USER_DATA_ROOT, LOG_ROOT)
            except OSError as exc:
                log_frontend_error("诊断包导出失败")
                self._json({"error": "无法写入诊断包。"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                self._json({"ok": True, "path": str(archive)})
            return
        if path == "/api/tutorial-preferences":
            try:
                _write_tutorial_preferences(payload)
            except InputValidationError as exc:
                self._json(exc.payload(), HTTPStatus.BAD_REQUEST); return
            except OSError:
                self._json({"error": "教程偏好无法保存。", "code": "preference_write_failed"}, HTTPStatus.INTERNAL_SERVER_ERROR); return
            self._json({"saved": True, "dismissedVersion": TUTORIAL_VERSION}); return
        if not self._require_external_ai_consent(path, payload):
            return
        if path == "/api/dependencies/ollama/start":
            try:
                result = self.dependencies.start_ollama()
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            else:
                status = HTTPStatus.ACCEPTED if result["state"] == "starting" else HTTPStatus.OK
                self._json(result, status)
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/cues/(\d+)/retranslate", path)
        if match:
            try:
                result = self.jobs.retranslate_cue(
                    match.group(1), int(match.group(2)), payload.get("source"),
                    payload.get("context"),
                )
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST); return
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(result); return
        if path == "/api/dependencies/tesseract/start":
            try:
                result = self.dependencies.start_tesseract_installer()
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            else:
                self._json(result, HTTPStatus.ACCEPTED)
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/cues/compact", path)
        if match:
            try:
                result = self.jobs.compact_cues(
                    match.group(1), payload.get("cue_ids"), payload.get("cues"),
                )
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST); return
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(result); return
        match = re.fullmatch(r"/api/dependencies/([a-z0-9-]+)/install", path)
        if match:
            try:
                install_job = self.dependencies.start(match.group(1), payload.get("confirmed") is True, str(payload.get("ocr_language") or "eng"))
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            else:
                status = HTTPStatus.OK if install_job.status == "completed" else HTTPStatus.ACCEPTED
                self._json({"job": install_job.snapshot()}, status)
            return
        match = re.fullmatch(r"/api/dependencies/jobs/([a-f0-9]+)/cancel", path)
        if match:
            install_job = self.dependencies.cancel(match.group(1))
            self._json(
                {"job": install_job.snapshot()} if install_job else {"error": "安装任务不存在。"},
                HTTPStatus.ACCEPTED if install_job else HTTPStatus.NOT_FOUND,
            )
            return
        if path == "/api/history/clear":
            if payload.get("confirmed") is not True:
                self._json({"error": "Clearing history requires explicit confirmation."}, HTTPStatus.BAD_REQUEST); return
            try:
                scope = str(payload.get("scope", "current")).lower()
                records = self._history_records(scope, str(payload.get("output_dir") or OUTPUT_ROOT), include_private=True)
                result = self.jobs.delete_history([str(record["id"]) for record in records], "record_only", allow_many=True) if records else {"deleted": 0}
                deleted = int(result.get("deleted", 0))
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST); return
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json({
                "deleted": deleted,
                "deleted_ids": [str(record["id"]) for record in records],
                "message": f"已清除 {deleted} 条任务记录；输出视频和字幕文件未删除。",
            })
            return
        if path in {"/api/history/jobs/delete-preview", "/api/history/delete-preview"}:
            try:
                ids = payload.get("ids")
                if not isinstance(ids, list):
                    raise ValueError("ids must be a list of task IDs.")
                plan = self.jobs.history_delete_plan(ids, str(payload.get("mode", "record_only")))
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST); return
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(plan); return
        if path in {"/api/history/jobs/delete", "/api/history/delete"}:
            if payload.get("confirmed") is not True:
                self._json({"error": "Deletion requires explicit confirmation."}, HTTPStatus.BAD_REQUEST); return
            try:
                ids = payload.get("ids")
                if not isinstance(ids, list):
                    raise ValueError("ids must be a list of task IDs.")
                result = self.jobs.delete_history(ids, str(payload.get("mode", "record_only")), str(payload.get("fingerprint") or ""))
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST); return
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(result); return
        if path == "/api/history/test-records/scan":
            candidates = self._test_history_candidates()
            self._json({"candidates": candidates, "count": len(candidates)}); return
        if path == "/api/history/test-records/delete":
            if payload.get("confirmed") is not True:
                self._json({"error": "删除测试记录需要二次确认。"}, HTTPStatus.BAD_REQUEST); return
            requested = {str(value) for value in payload.get("ids", []) if isinstance(value, str)}
            eligible = {str(row["id"]) for row in self._test_history_candidates()}
            rejected = requested - eligible
            if rejected:
                self._json({"error": "部分记录已不符合严格测试记录规则，操作已拒绝。"}, HTTPStatus.CONFLICT); return
            try:
                result = self.jobs.delete_history(sorted(requested), "record_only", allow_many=True)
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST); return
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            deleted = int(result.get("deleted", 0))
            self._json({"deleted": deleted, "message": f"已删除 {deleted} 条测试历史；没有删除任何视频或输出目录。"}); return
        if path == "/api/history/open":
            target = Path(str(payload.get("path", ""))).expanduser()
            if not self._history_path_is_allowed(target):
                self._json({"error": "只能打开任务历史中的输出目录或成品文件。"}, HTTPStatus.BAD_REQUEST); return
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
                url = _validate_public_http_url(payload.get("url", ""), "url")
                cookies_file = str(payload.get("cookies_file", "")).strip() or os.getenv("YBLOCALIZER_COOKIES_FILE", "").strip() or None
                cookies_from_browser = str(payload.get("cookies_from_browser", "")).strip() or None
                if cookies_file:
                    cookies_from_browser = None
                metadata = get_video_metadata(
                    url, cookies_from_browser, cookies_file,
                    proxy=str(payload.get("youtube_proxy", "")).strip() or None,
                    po_token_mode=str(payload.get("youtube_po_token_mode", "auto")),
                    download_quality=str(payload.get("download_quality", "1080p")),
                )
                self._json({"title": metadata.title, "duration": metadata.duration, "license": metadata.license, "view_count": metadata.view_count, "webpage_url": metadata.webpage_url, "thumbnail_url": metadata.thumbnail_url, "max_width": metadata.max_width, "max_height": metadata.max_height, "quality_proposal": metadata.quality_proposal})
            except InputValidationError as exc: self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
            except Exception as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path in {"/api/preflight", "/api/readiness"}:
            try:
                _validate_device_precision(payload.get("device", "cpu"), payload.get("compute_type", "int8"))
                if "authorized" in payload:
                    require_boolean(payload["authorized"], "authorized")
                if "source_url" in payload and payload["source_url"] is not None and payload["source_url"] != "":
                    payload = {**payload, "source_url": _validate_public_http_url(payload["source_url"])}
                result = preflight(
                    payload, str(OUTPUT_ROOT),
                    lambda browser: bool(self._publish_profile_process_ids(browser)),
                )
            except InputValidationError as exc:
                self._json(exc.payload(), HTTPStatus.BAD_REQUEST); return
            if path == "/api/readiness":
                result["issues"] = [item["message"] for item in result["blocking"]]
                result["message"] = "基础配置可以运行。" if result["ready"] else "；".join(result["issues"])
            self._json(result); return
        if path == "/api/publish/native-video":
            try: self._json(self.jobs.publish_file_metadata(str(payload.get("token", ""))))
            except ValueError as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/glossary":
            entries = payload.get("entries")
            if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
                self._json({"error": "entries 必须是对象列表"}, HTTPStatus.BAD_REQUEST); return
            clearing = not any(str(item.get("source") or "").strip() for item in entries)
            if clearing and payload.get("confirmed") is not True:
                self._json({"error": "清空术语表需要确认。"}, HTTPStatus.BAD_REQUEST); return
            self._json({"entries": save_glossary(global_glossary_path(OUTPUT_ROOT), entries)}); return
        if path == "/api/settings":
            try: self._save_settings(payload)
            except InputValidationError as exc: self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
            except ValueError as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            else: self._json({"saved": True})
            return
        if path == "/api/subscriptions":
            try:
                url = require_text(payload.get("url", ""), "url", MAX_URL_LENGTH, allow_empty=False)
                self._json(add_channel(url))
            except InputValidationError as exc:
                self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/subscriptions/remove":
            try:
                channel_id = require_text(payload.get("id", ""), "id", 32, allow_empty=False)
                self._json(remove_channel(channel_id))
            except InputValidationError as exc:
                self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/subscriptions/enabled":
            try:
                self._json(set_subscriptions_enabled(require_boolean(payload.get("enabled"), "enabled")))
            except InputValidationError as exc:
                self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/subscriptions/weekly/refresh":
            try:
                self._json(refresh_weekly_hot(self._weekly_access(payload)))
            except InputValidationError as exc:
                self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/templates":
            try:
                name = require_text(payload.get("name", ""), "name", 50, allow_empty=False)
                body = require_text(payload.get("body", ""), "body", MAX_DESCRIPTION_LENGTH, allow_empty=False)
                platform = normalize_publish_platform(str(payload.get("platform", "bilibili")))
                save_custom_template(name, body, platform=platform)
            except InputValidationError as exc: self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
            except ValueError as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            else: self._json({"saved": True, "platform": platform})
            return
        if path == "/api/publish/description":
            try:
                platform = normalize_publish_platform(str(payload.get("platform", "bilibili")))
                default_template = "中文字幕版" if platform == "douyin" else "授权本地化"
                template_name = require_text(payload.get("template", default_template), "template", 50, allow_empty=False)
                source_link = require_text(payload.get("source_url", ""), "source_url", MAX_URL_LENGTH)
                include_default = platform != "douyin"
                include_link = require_boolean(
                    payload.get("include_source_link", include_default),
                    "include_source_link",
                )
                if platform == "douyin":
                    include_link = False
                custom_text = require_text(payload.get("custom_text", ""), "custom_text", MAX_DESCRIPTION_LENGTH)
                template_body = require_text(payload.get("template_body", ""), "template_body", MAX_DESCRIPTION_LENGTH)
                extra_value = payload.get("extra_lines", [])
                if not isinstance(extra_value, list) or len(extra_value) > 20:
                    raise InputValidationError(
                        "extra_lines 必须是最多 20 项的文本数组。", code="invalid_collection", field="extra_lines",
                        limits={"max_items": 20},
                    )
                extra_lines = [require_text(line, f"extra_lines[{index}]", 500) for index, line in enumerate(extra_value)]
                body = build_publish_description(
                    platform,
                    template_name,
                    source_link,
                    include_link,
                    custom_text,
                    [line for line in extra_lines if line],
                    template_body=template_body or None,
                )
            except InputValidationError as exc:
                self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            else:
                self._json({"description": body, "platform": platform})
            return
        if path == "/api/publish/copy":
            self._publish_copy(payload); return
        if path == "/api/publish/douyin/open":
            try:
                opened = webbrowser.open(DOUYIN_UPLOAD_URL, new=2, autoraise=True)
            except (OSError, webbrowser.Error) as exc:
                self._json({"error": f"打开抖音创作者服务平台失败：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                if opened:
                    self._json({"opened": True, "message": "已在默认浏览器打开抖音创作者服务平台；请手动选择视频、粘贴文案并检查后发布。"}, HTTPStatus.ACCEPTED)
                else:
                    self._json({"error": f"系统未能打开浏览器。请手动访问：{DOUYIN_UPLOAD_URL}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/api/publish/douyin/assist":
            self._start_douyin_publish_assist(payload); return
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
        if path == "/api/outputs/delete-preview":
            try:
                self._json(self._output_delete_plan(payload))
            except ActiveTaskDeletionError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except (ValueError, RuntimeError, OSError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/outputs/delete":
            self._delete_output_paths(payload); return
        if path == "/api/batch/parse":
            options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
            entries: list[dict[str, Any]] = []
            for line in str(payload.get("text") or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.extend(list_playlist_entries(line, options.get("cookies_from_browser"), options.get("cookies_file"), options.get("youtube_proxy")))
                except Exception as exc:
                    entries.append({"url": line, "title": "", "duration": None, "error": str(exc)})
            self._json({"entries": entries}); return
        if path == "/api/batch/start":
            try:
                if payload.get("authorized") is not True:
                    raise ValueError("请先确认拥有处理和转载授权。")
                device, compute_type = _validate_device_precision(payload.get("device", "cuda"), payload.get("compute_type", "float16"))
                entries = [entry for entry in payload.get("entries") or [] if isinstance(entry, dict) and entry.get("url")]
                if not entries:
                    raise ValueError("没有可加入队列的链接。")
                batch_options = payload.get("options")
                if "dub_enabled" in payload and (batch_options is None or isinstance(batch_options, dict)):
                    batch_options = {**(batch_options or {}), "dub_enabled": payload["dub_enabled"] is True}
                result = self.jobs.batch_add(entries, device, compute_type, batch_options, payload.get("quality_consent") is True)
            except (InputValidationError, ValueError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST); return
            self._json(result, HTTPStatus.ACCEPTED); return
        if path == "/api/batch/cancel":
            self._json(self.jobs.batch_cancel()); return
        if path == "/api/batch/retry":
            self._json(self.jobs.batch_retry(payload.get("item_id") or None), HTTPStatus.ACCEPTED); return
        if path == "/api/jobs":
            source_url = payload.get("source_url")
            try:
                if source_url is not None and "material_id" in payload:
                    raise InputValidationError("视频链接与本地素材只能选择一种来源。", code="ambiguous_source", field="source_url")
                device, compute_type = _validate_device_precision(
                    payload.get("device", "cuda"), payload.get("compute_type", "float16"),
                )
                authorized = require_boolean(payload.get("authorized"), "authorized")
                if source_url is not None:
                    source_url = require_text(source_url, "source_url", MAX_URL_LENGTH, allow_empty=False)
                    job = self.jobs.create_url(source_url, device, compute_type, authorized, payload.get("options"))
                else:
                    if not authorized:
                        raise InputValidationError("请先确认拥有处理该视频的授权。", code="authorization_required", field="authorized")
                    material_id = require_text(payload.get("material_id", ""), "material_id", 128, allow_empty=False)
                    job = self.jobs.create(material_id, device, compute_type, payload.get("options"))
            except InputValidationError as exc: self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
            except ValueError as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            else: self._json(job.snapshot(), HTTPStatus.CREATED)
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/(prepare|run-all|resume)", path)
        if match:
            job_id, action = match.groups()
            try:
                job = self.jobs.prepare(job_id) if action == "prepare" else self.jobs.run_all(job_id) if action == "run-all" else self.jobs.resume(job_id)
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST); return
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(job.snapshot(), HTTPStatus.ACCEPTED if action != "prepare" else HTTPStatus.OK)
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/quality/confirm", path)
        if match:
            try:
                fingerprint = require_text(payload.get("fingerprint", ""), "fingerprint", 128, allow_empty=False)
                job = self.jobs.confirm_quality(match.group(1), fingerprint)
            except (ValueError, RuntimeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(job.snapshot()); return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/load", path)
        if match:
            try: job = self.jobs.load(match.group(1))
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(job.snapshot()); return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/stages/(acquire|extract|translate|render|dub|publish)/run", path)
        if match:
            try:
                job = self.jobs.run_stage(match.group(1), match.group(2))
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST); return
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(job.snapshot(), HTTPStatus.ACCEPTED)
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/(cancel|open-output|rerender)", path)
        if match:
            job_id, action = match.groups()
            if action == "cancel": job = self.jobs.cancel(job_id)
            elif action == "rerender":
                if payload.get("dirty") is True:
                    self._json({"error": "字幕存在未保存修改，请先保存后再重新渲染。"}, HTTPStatus.CONFLICT); return
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
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/voice-review/(prepare|start|resume|cancel)", path)
        if match:
            job_id, action = match.groups()
            try:
                result = self.jobs.prepare_voice_review(job_id) if action == "prepare" else self.jobs.cancel_voice_review(job_id) if action == "cancel" else self.jobs.start_voice_review(job_id)
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(result, HTTPStatus.ACCEPTED if action in {"start", "resume"} else HTTPStatus.OK); return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/voice-review/candidates/([^/]+)/translate", path)
        if match:
            try:
                result = self.jobs.translate_voice_candidate(*match.groups(), payload)
            except (RuntimeError, ValueError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(result); return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/voice-review/apply", path)
        if match:
            try:
                result = self.jobs.apply_voice_candidates(match.group(1), payload)
            except (RuntimeError, ValueError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(result); return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/trim", path)
        if match:
            try: job = self.jobs.trim(match.group(1), payload.get("cut_ranges", []))
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(job.snapshot())
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/modify", path)
        if match:
            try: result = self.jobs.modify(match.group(1), str(payload.get("op", "")), payload)
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(result)
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/align", path)
        if match:
            try: job = self.jobs.align(match.group(1))
            except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT); return
            self._json(job.snapshot())
            return
        self._json({"error": "Route not found."}, HTTPStatus.NOT_FOUND)

    def do_PUT(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if not self._authorize_mutation(path):
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/cues", path)
        if not match: self._json({"error": "Route not found."}, HTTPStatus.NOT_FOUND); return
        try:
            payload = self._payload()
        except PayloadReadError as exc:
            self._json(exc.payload(), exc.status); return
        cues = payload.get("cues")
        if not isinstance(cues, list):
            self._json({"error": "cues must be a list of {start, end, translated, deleted}."}, HTTPStatus.BAD_REQUEST); return
        try:
            job = self.jobs.save_cues(
                match.group(1), cues,
                base_revision=payload.get("base_revision"),
                base_fingerprint=payload.get("base_fingerprint"),
                base_ids=payload.get("base_ids"),
                candidate_rows=payload.get("candidate_rows"),
            )
        except RuntimeError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        else: self._json(job.snapshot())

    def do_PATCH(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if not self._authorize_mutation(path):
            return
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/options", path)
        if not match:
            self._json({"error": "Route not found."}, HTTPStatus.NOT_FOUND); return
        try:
            payload = self._payload()
        except PayloadReadError as exc:
            self._json(exc.payload(), exc.status); return
        try:
            job, stale = self.jobs.update_options(match.group(1), payload)
        except InputValidationError as exc:
            self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        else:
            self._json({**job.snapshot(), "stale_stages": stale})

    def do_DELETE(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if not self._authorize_mutation(path):
            return
        match = re.fullmatch(r"/api/templates/(.+)", path)
        if not match: self._json({"error": "Route not found."}, HTTPStatus.NOT_FOUND); return
        try:
            name = require_text(unquote(match.group(1)), "name", 50, allow_empty=False)
            query = parse_qs(urlparse(self.path).query)
            platform = normalize_publish_platform(str(query.get("platform", ["bilibili"])[0]))
        except InputValidationError as exc:
            self._json(exc.payload(), HTTPStatus.BAD_REQUEST); return
        if delete_custom_template(name, platform=platform): self._json({"deleted": True, "platform": platform})
        else: self._json({"error": "Only custom templates can be deleted."}, HTTPStatus.NOT_FOUND)

    def _weekly_access(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Cookies, proxy and PO Token mode, with the same env fallback as metadata."""
        cookies_file = require_text(payload.get("cookies_file", ""), "cookies_file", 1024) or os.getenv("YBLOCALIZER_COOKIES_FILE", "").strip()
        cookies_from_browser = require_text(payload.get("cookies_from_browser", ""), "cookies_from_browser", 32).lower()
        if not cookies_from_browser:
            cookies_from_browser = os.getenv("YBLOCALIZER_COOKIES_FROM_BROWSER", "").strip().lower()
        if cookies_from_browser and cookies_from_browser not in {"chrome", "edge", "firefox", "brave", "chromium"}:
            raise InputValidationError("YouTube Cookies 来源无效。", code="invalid_choice", field="cookies_from_browser")
        if cookies_file:
            cookies_from_browser = ""
        proxy = require_text(payload.get("youtube_proxy", ""), "youtube_proxy", MAX_URL_LENGTH) or os.getenv("YBLOCALIZER_YOUTUBE_PROXY", "").strip()
        po_token_mode = require_text(payload.get("youtube_po_token_mode", "auto"), "youtube_po_token_mode", 16).lower() or "auto"
        if po_token_mode not in {"auto", "off"}:
            raise InputValidationError("YouTube 浏览器验证模式必须是 auto 或 off。", code="invalid_choice", field="youtube_po_token_mode")
        return {
            "cookies_file": cookies_file or None,
            "cookies_from_browser": cookies_from_browser or None,
            "proxy": proxy or None,
            "po_token_mode": po_token_mode,
        }

    def _save_settings(self, payload: dict[str, Any]) -> None:
        translator = require_text(payload.get("translator", "deepseek"), "translator", 32, allow_empty=False).lower()
        key = require_text(payload.get("api_key", ""), "api_key", 1024)
        if translator not in {"deepseek", "openai", "ollama"}:
            raise InputValidationError("设置中的翻译器无效。", code="invalid_choice", field="translator")
        cookies_from_browser = require_text(payload.get("cookies_from_browser", ""), "cookies_from_browser", 32).lower()
        cookies_file = require_text(payload.get("cookies_file", ""), "cookies_file", 1024)
        proxy_supplied = "youtube_proxy" in payload
        youtube_proxy = require_text(payload.get("youtube_proxy", ""), "youtube_proxy", MAX_URL_LENGTH)
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
        if proxy_supplied:
            if youtube_proxy:
                parsed_proxy = urlparse(youtube_proxy)
                if parsed_proxy.scheme not in {"http", "https", "socks5", "socks5h"} or not parsed_proxy.netloc:
                    raise ValueError("代理地址必须是有效的 HTTP(S) 或 SOCKS5 URL。")
            _upsert_env(USER_DATA_ROOT / ".env", {"YBLOCALIZER_YOUTUBE_PROXY": youtube_proxy})
            os.environ["YBLOCALIZER_YOUTUBE_PROXY"] = youtube_proxy
        if key:
            if translator == "ollama":
                raise ValueError("Ollama 本地翻译不使用 API Key。")
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
                "id": str(task.path.relative_to(root)).replace("\\", "/"),
                "name": task.name,
                "title": resolve_task_title(task.path),
                "size": task.size,
                "size_label": format_bytes(task.size),
                "modified_at": task.modified_at,
                "files": [{
                    "id": str(file.path.relative_to(root)).replace("\\", "/"),
                    "name": file.path.name,
                    "kind": file.kind,
                    "size": file.size,
                    "size_label": format_bytes(file.size),
                } for file in task.files],
            } for task in tasks],
        }

    def _diagnostics(self) -> dict[str, Any]:
        """Report first-run prerequisites without exposing local secrets or paths."""
        manager = getattr(self, "dependencies", None)
        rows = manager.snapshot()["dependencies"] if manager else dependency_statuses()
        checks = [
            {key: item[key] for key in ("id", "label", "purpose", "available", "required", "message")}
            for item in rows
        ]
        return {"ready": all(check["available"] for check in checks if check["required"]), "checks": checks}

    @staticmethod
    def _path_is_within(path: Path, root: Path) -> bool:
        try:
            path.expanduser().resolve().relative_to(root.expanduser().resolve())
        except (OSError, ValueError):
            return False
        return True

    def _history_records(
        self, scope: str, output_dir: str, *, include_private: bool = False
    ) -> list[dict[str, Any]]:
        if scope not in {"current", "all"}:
            raise ValueError("History scope must be current or all.")
        root = _resolve_output_root(output_dir)
        records = job_db.list_jobs(None)
        if scope == "current":
            records = [
                record for record in records
                if record.get("output_dir") and self._path_is_within(Path(str(record["output_dir"])), root)
            ]
        if include_private:
            return records
        public: list[dict[str, Any]] = []
        for record in records:
            output_dir_value = str(record.get("output_dir") or "")
            rendered_value = str(record.get("rendered_video") or "")
            dub_value = str((record.get("artifacts") or {}).get("dub_video") or "")
            public.append({
                "id": record["id"],
                "title": record.get("title") or "未命名任务",
                "status": record.get("status") or "unknown",
                "stage": record.get("stage") or "—",
                "progress": record.get("progress") or 0,
                "error": record.get("error"),
                "output_dir": output_dir_value or None,
                "rendered_video": rendered_value or None,
                "output_exists": bool(output_dir_value and Path(output_dir_value).expanduser().exists()),
                "rendered_exists": bool(rendered_value and Path(rendered_value).expanduser().is_file()),
                "dub_video": dub_value or None,
                "dub_exists": bool(dub_value and Path(dub_value).expanduser().is_file()),
                "created_at": record.get("created_at"),
                "finished_at": record.get("finished_at"),
                "llm_usage_total": (record.get("artifacts") or {}).get("llm_usage_total"),
            })
        return public

    def _test_history_candidates(self) -> list[dict[str, Any]]:
        """Identify only records rooted under pytest's own temporary tree."""
        temp_root = Path(tempfile.gettempdir()).resolve()
        candidates: list[dict[str, Any]] = []
        for record in job_db.list_jobs(None):
            value = str(record.get("output_dir") or "")
            if not value:
                continue
            try:
                resolved = Path(value).expanduser().resolve()
                relative = resolved.relative_to(temp_root)
            except (OSError, ValueError):
                continue
            parts = [part.lower() for part in relative.parts]
            if not any(part.startswith("pytest-") or part.startswith("pytest-of-") for part in parts):
                continue
            candidates.append({
                "id": str(record["id"]),
                "title": record.get("title") or "未命名测试任务",
                "status": record.get("status") or "unknown",
                "output_dir": str(resolved),
                "reason": "输出目录位于 pytest 独立临时目录中",
                "created_at": record.get("created_at"),
            })
        return candidates

    def _history_path_is_allowed(self, target: Path) -> bool:
        try:
            resolved = target.expanduser().resolve()
        except OSError:
            return False
        for record in job_db.list_jobs(None):
            for value in (record.get("output_dir"), record.get("rendered_video"), (record.get("artifacts") or {}).get("dub_video")):
                if not value:
                    continue
                try:
                    if resolved == Path(str(value)).expanduser().resolve():
                        return True
                except OSError:
                    continue
        return False

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

    def _output_delete_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        root = _resolve_output_root(str(payload.get("output_dir") or OUTPUT_ROOT))
        values = payload.get("paths")
        if not isinstance(values, list) or not values or len(values) > 200 or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("每次请选择 1～200 个任务或文件。")
        if len(set(values)) != len(values):
            raise ValueError("删除目标不能重复。")
        snapshot = self._outputs_snapshot(root)
        tasks = {task["id"]: task for task in snapshot["tasks"]}
        files = {file["id"]: file for task in snapshot["tasks"] for file in task["files"]}
        if any(value not in tasks and value not in files for value in values):
            raise ValueError("只能删除当前输出扫描中列出的任务或文件。请刷新后重试。")
        if any(file_id.startswith(f"{task_id}/") for task_id in values if task_id in tasks for file_id in values if file_id in files):
            raise ValueError("已选整个任务时不能同时选择其内部文件。")
        paths = [(root / value).resolve() for value in values]
        if self.jobs.output_paths_active(paths):
            raise ActiveTaskDeletionError("运行中的任务输出不可删除，请先取消并等待当前阶段结束。")
        records = {str(row["id"]): row for row in job_db.list_jobs(None)}
        targets: list[dict[str, Any]] = []
        for value, target in zip(values, paths):
            if _path_contains_reparse(root / value):
                raise RuntimeError("输出目录包含符号链接或 reparse point，已拒绝删除。")
            is_task = value in tasks
            registered = is_task and value in records
            if registered:
                history_preview = self.jobs.history_delete_plan([value], "record_and_outputs")
                if history_preview["tasks"][0].get("disposition") != "verified_files":
                    raise RuntimeError(
                        "该任务输出无法通过安全校验；请到任务库使用仅删除记录或混合删除预览。"
                    )
            target_files = sorted(target.rglob("*")) if is_task else [target]
            signatures = [{"path": file.relative_to(target if is_task else root).as_posix(), "size": file.stat().st_size, "mtime_ns": file.stat().st_mtime_ns} for file in target_files if file.is_file()]
            targets.append({
                "id": value, "name": tasks[value]["name"] if is_task else files[value]["name"],
                "kind": "task" if is_task else "file", "record_will_remove": bool(registered),
                "file_count": len(signatures), "bytes": sum(item["size"] for item in signatures),
                "files": signatures,
            })
        plan = {"root": str(root), "paths": values, "targets": targets, "target_count": len(targets),
                "file_count": sum(item["file_count"] for item in targets), "bytes": sum(item["bytes"] for item in targets)}
        plan["fingerprint"] = WorkbenchJobs._history_fingerprint(plan)
        return plan

    def _delete_output_paths(self, payload: dict[str, Any]) -> None:
        if payload.get("confirmed") is not True:
            self._json({"error": "Deletion requires explicit confirmation."}, HTTPStatus.BAD_REQUEST); return
        try:
            plan = self._output_delete_plan(payload)
            fingerprint = payload.get("fingerprint")
            if not fingerprint or not hmac.compare_digest(str(fingerprint), plan["fingerprint"]):
                raise RuntimeError("删除预览已过期，请刷新后重新确认。")
            root = Path(plan["root"])
            results: list[dict[str, Any]] = []
            for item in plan["targets"]:
                value = item["id"]
                if item["record_will_remove"]:
                    history_plan = self.jobs.history_delete_plan([value], "record_and_outputs")
                    history_result = self.jobs.delete_history([value], "record_and_outputs", history_plan["fingerprint"])
                    detail = history_result["results"][0]
                    results.append({"id": value, "name": item["name"], "status": detail["status"],
                                    "record_removed": detail["record_removed"], "bytes": detail["bytes"], "error": detail["error"]})
                    continue
                try:
                    delete_paths([root / value], root)
                except Exception as exc:
                    LOGGER.exception("Failed to delete output item %s", value)
                    results.append({"id": value, "name": item["name"], "status": "partial", "record_removed": False, "bytes": 0, "error": str(exc)})
                else:
                    results.append({"id": value, "name": item["name"], "status": "deleted", "record_removed": False, "bytes": item["bytes"], "error": None})
            deleted = sum(item["bytes"] for item in results if item["status"] == "deleted")
            removed_history = sum(1 for item in results if item["record_removed"])
        except ActiveTaskDeletionError as exc: self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except (ValueError, RuntimeError, OSError) as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        else: self._json({"deleted_bytes": deleted, "deleted_size": format_bytes(deleted), "removed_history": removed_history, "results": results})

    def _publish_profile_dir(self, browser: str, platform: str = "bilibili") -> Path:
        prefix = "douyin" if platform == "douyin" else "bilibili"
        suffix = "edge-profile" if browser == "msedge" else "profile"
        return USER_DATA_ROOT / "data" / f"{prefix}-{suffix}"

    def _publish_profile_process_ids(self, browser: str, platform: str = "bilibili") -> list[str]:
        profile = str(self._publish_profile_dir(browser, platform).resolve()).replace("'", "''")
        command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -notin @('powershell.exe','pwsh.exe') -and $_.CommandLine -and $_.CommandLine -like '*" + profile + "*' } | Select-Object -ExpandProperty ProcessId"
        try:
            completed = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=5)
            return [line.strip() for line in completed.stdout.splitlines() if line.strip().isdigit()]
        except (OSError, subprocess.SubprocessError):
            return []

    def _readiness(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = preflight(payload, str(OUTPUT_ROOT), lambda browser: bool(self._publish_profile_process_ids(browser)))
        issues = [item["message"] for item in result["blocking"]]
        return {**result, "issues": issues, "message": "基础配置可以运行。" if result["ready"] else "；".join(issues)}

    def _start_publish_check(self, payload: dict[str, Any]) -> None:
        if self.jobs.publish_session().status in {"running", "waiting_review"}:
            self._json({"error": "已有投稿辅助浏览器正在运行，请先关闭后再打开登录检查。"}, HTTPStatus.CONFLICT); return
        browser = "msedge" if str(payload.get("browser", "chromium")).lower() in {"edge", "msedge"} else "chromium"
        session = self.jobs.begin_publish_session()
        session.status, session.message = "running", "正在打开 B 站登录检查页…"
        session.logs = [session.message]

        def runner() -> None:
            try:
                open_upload_page(profile_dir=self._publish_profile_dir(browser), browser=browser, log=self.jobs._publish_log)
            except Exception as exc:
                log_frontend_error("B 站登录检查失败，已显示在界面")
                session.status, session.error, session.message = "failed", str(exc), f"打开登录检查页失败：{exc}"
            else:
                session.status, session.message = "finished", "B 站登录检查页已关闭。"
        threading.Thread(target=runner, daemon=True, name="bilibili-check").start()
        self._json({"started": True, "message": "B 站登录检查页已打开；不会上传或投稿。"}, HTTPStatus.ACCEPTED)

    def _resolve_publish_video(self, payload: dict[str, Any]) -> tuple[Path, DemoJob | None]:
        job_id = str(payload.get("job_id", "") or "").strip()
        job = self.jobs.get(job_id) if job_id else None
        video = self.jobs.media(job.id, "rendered") if job else None
        if video is None:
            native_token = str(payload.get("native_video_token", "") or "").strip()
            video = self.jobs.publish_file_from_token(native_token) if native_token else None
        video_id = str(payload.get("video_id", "") or "").strip()
        if video is None and video_id and video_id != "__native__":
            root = _resolve_output_root(str(payload.get("output_dir") or OUTPUT_ROOT))
            candidate = (root / video_id).resolve()
            candidate.relative_to(root)
            video = candidate
            if job is None:
                job = self.jobs.get(candidate.parent.name)
        if video is None or not video.is_file() or video.suffix.lower() not in MEDIA_EXTENSIONS:
            raise ValueError("请先选择一份已渲染的成片。")
        if job is None:
            job = self.jobs.get(video.parent.name)
        return video, job

    def _compose_publish_fields(self, payload: dict[str, Any], video: Path, job: DemoJob | None) -> tuple[str, str, list[str]]:
        metadata_path = job.artifacts.path("publish_metadata") if job else None
        copy = _read_publish_metadata(metadata_path if metadata_path and metadata_path.is_file() else video.parent / "publish_metadata.json")
        title = require_text(payload.get("title", ""), "title", MAX_TITLE_LENGTH) or copy["title"]
        if job and not title:
            title = str(job.options.get("title") or job.artifacts.title or job.material.name)
        title = title or video.stem
        description = require_text(payload.get("description", ""), "description", MAX_DESCRIPTION_LENGTH)
        if not description:
            source_url = ""
            if job:
                source_url = job.material.source_url or job.artifacts.webpage_url or ""
            platform = normalize_publish_platform(str(payload.get("platform", "bilibili")))
            include_link = platform != "douyin"
            if "include_source_link" in payload:
                include_link = require_boolean(payload.get("include_source_link"), "include_source_link")
            if platform == "douyin":
                include_link = False
            default_template = "中文字幕版" if platform == "douyin" else "授权本地化"
            template_name = require_text(payload.get("template", default_template), "template", 50) or default_template
            template_body = require_text(payload.get("template_body", ""), "template_body", MAX_DESCRIPTION_LENGTH) or None
            extra_value = payload.get("extra_lines", [])
            extra_lines = []
            if isinstance(extra_value, list):
                extra_lines = [require_text(line, f"extra_lines[{index}]", 500) for index, line in enumerate(extra_value) if str(line).strip()]
            description = build_publish_description(
                platform,
                template_name,
                source_url,
                include_link,
                "",
                extra_lines,
                template_body=template_body,
            )
        tags = _validate_tags(payload.get("tags", [])) or list(copy["tags"])
        if job and not tags:
            tags = [str(tag).strip() for tag in (job.options.get("tags") or []) if str(tag).strip()]
        return title, description, tags

    def _publish_copy(self, payload: dict[str, Any]) -> None:
        try:
            generate = require_boolean(payload.get("generate", False), "generate")
            video, job = self._resolve_publish_video(payload)
            if generate:
                if job is None:
                    raise ValueError("找不到对应任务，无法根据字幕重新生成标题和标签。")
                copy = self.jobs.refresh_publish_metadata(job)
            else:
                metadata_path = job.artifacts.path("publish_metadata") if job else None
                copy = _read_publish_metadata(metadata_path if metadata_path and metadata_path.is_file() else video.parent / "publish_metadata.json")
                if not copy["title"]:
                    copy["title"] = (str(job.options.get("title") or job.artifacts.title or job.material.name) if job else video.stem)
                copy["generated"] = False
            source_url = ""
            if job:
                source_url = job.material.source_url or job.artifacts.webpage_url or ""
        except InputValidationError as exc:
            self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
        except (ValueError, OSError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        else:
            self._json({
                "title": copy["title"],
                "tags": copy["tags"],
                "has_metadata": copy["has_metadata"],
                "generated": bool(copy.get("generated")),
                "source_url": source_url,
                "video_name": video.name,
            })

    def _start_publish_assist(self, payload: dict[str, Any]) -> None:
        if payload.get("confirmed") is not True:
            self._json({"error": "请先确认拥有授权，并在投稿前人工检查。"}, HTTPStatus.BAD_REQUEST); return
        try:
            browser = "msedge" if str(payload.get("browser", "chromium")).lower() in {"edge", "msedge"} else "chromium"
            if self.jobs.publish_session().status in {"running", "waiting_review"}:
                raise ValueError("已有投稿辅助浏览器正在运行，请先关闭后再开始新的辅助。")
            close_after_fill = require_boolean(payload.get("close_after_fill", False), "close_after_fill")
            if self._publish_profile_process_ids(browser):
                raise ValueError("B 站自动化浏览器正在使用中。请先关闭后台浏览器，再开始投稿辅助。")
            video, job = self._resolve_publish_video(payload)
            title, description, tags = self._compose_publish_fields(payload, video, job)
            cover = next((video.parent / name for name in ("cover.jpg", "cover.png", "thumbnail.jpg", "thumbnail.png") if (video.parent / name).exists()), None)
            session = self.jobs.begin_publish_session()

            def assist_runner() -> None:
                try:
                    assist_publish(
                        video_path=video, title=title, description=description, tags=tags, cover_path=cover,
                        profile_dir=self._publish_profile_dir(browser), browser=browser,
                        screenshot_path=video.parent / "bilibili-upload-page.png",
                        wait_for_review=not close_after_fill,
                        log=self.jobs._publish_log,
                    )
                except Exception as exc:
                    log_frontend_error("B 站投稿辅助失败，已显示在界面")
                    session.status, session.error = "failed", str(exc)
                    session.message = f"B 站投稿辅助失败：{exc}"
                else:
                    if session.status != "failed":
                        session.status, session.message = "finished", "B 站投稿辅助已结束（浏览器已关闭）。"
            threading.Thread(target=assist_runner, daemon=True, name="bilibili-assist").start()
        except InputValidationError as exc: self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
        except (ValueError, OSError) as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        else: self._json({"started": True, "message": "正在打开 B 站投稿辅助。填写完成后请人工检查并手动投稿。"}, HTTPStatus.ACCEPTED)

    def _start_douyin_publish_assist(self, payload: dict[str, Any]) -> None:
        if payload.get("confirmed") is not True:
            self._json({"error": "请确认要将所选成片上传到抖音，并在发布前人工检查。"}, HTTPStatus.BAD_REQUEST); return
        try:
            browser = "msedge" if str(payload.get("browser", "chromium")).lower() in {"edge", "msedge"} else "chromium"
            if self.jobs.publish_session().status in {"running", "waiting_review"}:
                raise ValueError("已有投稿辅助浏览器正在运行，请先关闭后再开始新的辅助。")
            if self._publish_profile_process_ids(browser, "douyin"):
                raise ValueError("抖音专用自动化浏览器正在使用中。请先关闭该窗口，再开始新的投稿辅助。")
            video, job = self._resolve_publish_video(payload)
            title, description, tags = self._compose_publish_fields(payload, video, job)
            session = self.jobs.begin_publish_session()
            session.message = "正在打开抖音创作者中心投稿页…"
            session.logs = [session.message]

            def assist_runner() -> None:
                try:
                    assist_douyin_publish(
                        video_path=video, title=title, description=description, tags=tags,
                        profile_dir=self._publish_profile_dir(browser, "douyin"), browser=browser,
                        wait_for_review=True, log=self.jobs._publish_log,
                    )
                except Exception as exc:
                    log_frontend_error("抖音投稿辅助失败，已显示在界面")
                    session.status, session.error = "failed", str(exc)
                    session.message = f"抖音投稿辅助失败：{exc}"
                else:
                    if session.status != "failed":
                        session.status, session.message = "finished", "抖音投稿辅助已结束（浏览器已关闭）。"
            threading.Thread(target=assist_runner, daemon=True, name="douyin-assist").start()
        except InputValidationError as exc: self._json(exc.payload(), HTTPStatus.BAD_REQUEST)
        except (ValueError, OSError) as exc: self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        else: self._json({"started": True, "message": "抖音投稿辅助已启动。成片上传并填写后会停在发布前，等待你检查。"}, HTTPStatus.ACCEPTED)

    def _close_publish_browser(self, payload: dict[str, Any]) -> None:
        if payload.get("confirmed") is not True:
            self._json({"error": "Closing a browser requires explicit confirmation."}, HTTPStatus.BAD_REQUEST); return
        platform = "douyin" if str(payload.get("platform", "bilibili")).lower() == "douyin" else "bilibili"
        profiles = [str(self._publish_profile_dir(browser, platform).resolve()) for browser in ("chromium", "msedge")]
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
            cleanup = temporary or getattr(self, "_active_upload_temp", None)
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)
            self._active_upload_temp = None

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
        finally:
            # Native publish registration retains only the validated path token;
            # browser fallback files are cleaned when registration fails.
            if temporary is None:
                cleanup = getattr(self, "_active_upload_temp", None)
                if cleanup is not None:
                    cleanup.unlink(missing_ok=True)
            self._active_upload_temp = None

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
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(UPLOAD_ROOT).free
        required_bytes = remaining * 2 + UPLOAD_RESERVE_BYTES
        if free_bytes < required_bytes:
            raise ValueError(
                f"磁盘空间不足：上传和验证该视频至少需要约 {format_bytes(required_bytes)} 可用空间。"
            )
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
                self._active_upload_temp = temporary
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
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise PayloadReadError(
                    "请求缺少 Content-Length。", status=HTTPStatus.LENGTH_REQUIRED,
                    code="content_length_required",
                )
            length = int(raw_length)
        except ValueError:
            raise PayloadReadError(
                "Content-Length 必须是有效整数。", status=HTTPStatus.BAD_REQUEST,
                code="invalid_content_length",
            ) from None
        if length <= 0:
            raise PayloadReadError(
                "请求体不能为空。", status=HTTPStatus.BAD_REQUEST, code="empty_body",
            )
        if length > MAX_JSON_BODY:
            raise PayloadReadError(
                "请求体不能超过 1 MiB。", status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                code="body_too_large",
            )
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise PayloadReadError(
                "请求体在传输完成前中断。", status=HTTPStatus.BAD_REQUEST,
                code="incomplete_body",
            )

        def reject_constant(value: str) -> None:
            raise ValueError(f"Non-finite JSON number: {value}")

        try:
            value = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
        except UnicodeDecodeError:
            raise PayloadReadError(
                "请求体必须使用 UTF-8 编码。", status=HTTPStatus.BAD_REQUEST,
                code="invalid_utf8",
            ) from None
        except json.JSONDecodeError:
            raise PayloadReadError(
                "请求体不是有效 JSON。", status=HTTPStatus.BAD_REQUEST,
                code="invalid_json",
            ) from None
        except ValueError as exc:
            raise PayloadReadError(
                "JSON 不能包含 NaN 或 Infinity。", status=HTTPStatus.BAD_REQUEST,
                code="non_finite_number",
            ) from exc
        if not isinstance(value, dict):
            raise PayloadReadError(
                "JSON 请求体必须是对象。", status=HTTPStatus.BAD_REQUEST,
                code="object_required",
            )
        return value

    def _authorize_mutation(self, path: str) -> bool:
        """Reject drive-by browser requests before reading a request body."""
        host = self.headers.get("Host", "")
        expected_port = int(self.server.server_address[1])
        allowed_hosts = {
            f"127.0.0.1:{expected_port}", f"localhost:{expected_port}",
            "127.0.0.1:5173", "localhost:5173",
        }
        if host.lower() not in allowed_hosts:
            self._json({"error": "不允许的主机地址。"}, HTTPStatus.FORBIDDEN); return False
        origin = self.headers.get("Origin", "")
        if origin and not self._origin_allowed(origin):
            self._json({"error": "不允许的请求来源。"}, HTTPStatus.FORBIDDEN); return False
        supplied = self.headers.get("X-YBL-CSRF", "")
        if not supplied or not hmac.compare_digest(supplied, self.csrf_token):
            self._json({"error": "请求令牌无效，请刷新应用后重试。"}, HTTPStatus.FORBIDDEN); return False
        media_type = self.headers.get("Content-Type", "").partition(";")[0].strip().lower()
        multipart_paths = {"/api/materials", "/api/publish/upload"}
        expected = "multipart/form-data" if path in multipart_paths else "application/json"
        if media_type != expected:
            self._json({"error": f"该接口只接受 {expected}。"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE); return False
        return True

    def _origin_allowed(self, origin: str) -> bool:
        parsed = urlparse(origin)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            return False
        port = parsed.port or 80
        return port in {int(self.server.server_address[1]), 5173}

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self._cors_headers(); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store, no-cache, must-revalidate"); self.send_header("Pragma", "no-cache"); self.send_header("Expires", "0"); self.end_headers(); self.wfile.write(body)

    def _cors_headers(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin and self._origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-YBL-CSRF")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, OPTIONS")

    def _serve_media(self, path: Path) -> None:
        size = path.stat().st_size; start, end = 0, size - 1; status = HTTPStatus.OK
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match:
            start = int(match.group(1) or 0); end = min(int(match.group(2) or end), end); status = HTTPStatus.PARTIAL_CONTENT
        if start > end or start >= size: self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE); return
        length = end - start + 1
        self.send_response(status); self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4"); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
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
    recovered = job_db.recover_interrupted_jobs(time.time())
    if recovered:
        LOGGER.warning("Recovered %d interrupted job(s) from the previous process", recovered)
    Handler.frontend_dir, Handler.jobs = frontend_dir.resolve(), WorkbenchJobs(restore=True)
    Handler.csrf_token = secrets.token_urlsafe(32)
    Handler.dependencies = DependencyManager(USER_DATA_ROOT)

    class WorkbenchHTTPServer(ThreadingHTTPServer):
        request_queue_size = MAX_HTTP_CONNECTIONS

        def __init__(self, *server_args: Any, **server_kwargs: Any) -> None:
            self._connection_slots = threading.BoundedSemaphore(MAX_HTTP_CONNECTIONS)
            super().__init__(*server_args, **server_kwargs)

        def process_request(self, request: Any, client_address: Any) -> None:
            if not self._connection_slots.acquire(blocking=False):
                body = json.dumps({
                    "error": "本地服务请求过多，请稍后重试。",
                    "code": "too_many_connections",
                }, ensure_ascii=False).encode("utf-8")
                response = (
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Type: application/json; charset=utf-8\r\n"
                    + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
                    + body
                )
                try:
                    # Drain the already-sent request headers before closing.
                    # Closing a Windows socket with unread inbound bytes can
                    # turn this graceful 503 into WSAECONNABORTED on the
                    # client, hiding the useful overload message.
                    request.settimeout(0.1)
                    received = bytearray()
                    try:
                        while len(received) < 64 * 1024 and b"\r\n\r\n" not in received:
                            chunk = request.recv(4096)
                            if not chunk:
                                break
                            received.extend(chunk)
                    except (OSError, TimeoutError):
                        pass
                    request.sendall(response)
                    try:
                        request.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                finally:
                    self.close_request(request)
                return
            try:
                super().process_request(request, client_address)
            except Exception:
                self._connection_slots.release()
                raise

        def process_request_thread(self, request: Any, client_address: Any) -> None:
            try:
                super().process_request_thread(request, client_address)
            finally:
                self._connection_slots.release()

        def handle_error(self, request: Any, client_address: Any) -> None:
            log_frontend_error("未处理的本地接口异常")
            super().handle_error(request, client_address)

        def server_close(self) -> None:
            Handler.jobs.shutdown()
            Handler.dependencies.shutdown()
            super().server_close()

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
