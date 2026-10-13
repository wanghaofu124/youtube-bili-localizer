from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import os
import shutil
import subprocess
import time
import uuid
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from .cancellation import check_cancelled as legacy_check_cancelled
from .dependencies import resolve_command
from .models import VideoJob
from .runtime import CancellationRequested
from .performance import download_rate_limit, lower_process_priority, normalize_resource_profile
from .util import require_command, slugify


class YouTubeAccessError(RuntimeError):
    """A stable, user-facing YouTube failure with a machine-readable code."""

    def __init__(self, code: str, message: str, suggested_action: str) -> None:
        super().__init__(message)
        self.code = code
        self.suggested_action = suggested_action


class DownloadResolutionMismatch(RuntimeError):
    """A downloaded file does not match the resolution yt-dlp selected."""

    code = "download_resolution_mismatch"

    def __init__(self, selected: tuple[int, int], actual: tuple[int, int] | None) -> None:
        expected_text = f"{selected[0]}×{selected[1]}"
        actual_text = f"{actual[0]}×{actual[1]}" if actual else "无法读取"
        super().__init__(
            f"下载画质校验未通过：下载器选择了 {expected_text} 视频流，但成品文件实测为 {actual_text}。"
            "已停止后续处理，没有把较低画质当作原画继续使用。请重试获取素材；如需降低画质，请在任务失败后明确确认。"
        )


def _extended_io_path(path: Path) -> Path:
    """Use Win32 extended-length syntax only for filesystem I/O.

    Containment and symlink checks intentionally continue to use ordinary
    resolved paths; this conversion is only for the actual open/stat/rename
    operations in the resume-protection area.
    """
    if os.name != "nt":
        return path
    raw = str(path.resolve())
    if raw.startswith("\\\\?\\"):
        return Path(raw)
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw.lstrip("\\"))
    return Path("\\\\?\\" + raw)


class DownloadProposalMismatch(RuntimeError):
    code = "download_proposal_mismatch"

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or "下载前媒体流方案已变化，需要重新准备并确认当前画质方案。")


@dataclass(frozen=True)
class _PartialSnapshot:
    path: Path
    backup: Path
    size: int
    sha256: str
    directory: Path


def _sha256_prefix(path: Path, length: int) -> str | None:
    digest = hashlib.sha256()
    remaining = length
    try:
        with _extended_io_path(path).open("rb") as handle:
            while remaining:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    return None
                digest.update(block)
                remaining -= len(block)
        return digest.hexdigest()
    except OSError:
        return None


def _snapshot_existing_partials(work_dir: Path, log: Callable[[str], None] | None) -> list[_PartialSnapshot]:
    partials: list[tuple[Path, os.stat_result]] = []
    for path in sorted(work_dir.glob("*.part")):
        if path.is_symlink():
            raise RuntimeError("发现符号链接形式的未完成下载；为避免覆盖其他文件，已停止获取素材。")
        try:
            details = path.stat()
        except OSError as exc:
            raise RuntimeError(f"无法安全检查未完成下载文件：{path.name}") from exc
        if not path.is_file():
            raise RuntimeError("发现非普通文件形式的未完成下载；为避免覆盖其他文件，已停止获取素材。")
        if details.st_size > 0:
            partials.append((path, details))
    if not partials:
        return []

    if len(partials) > 16:
        raise RuntimeError("未完成下载文件数量异常；为避免误改现有文件，已停止获取素材。")
    total_size = sum(details.st_size for _, details in partials)
    if shutil.disk_usage(work_dir).free < total_size + 8 * 1024 * 1024:
        raise RuntimeError(
            "磁盘剩余空间不足，无法在安全续传前保护现有未完成文件；原文件未改动，本次下载未开始。"
        )

    protection_root = work_dir / ".resume-protection"
    if protection_root.exists() and (
        protection_root.is_symlink() or protection_root.resolve().parent != work_dir.resolve()
    ):
        raise RuntimeError("续传保护目录路径不安全；原未完成文件未改动，本次下载未开始。")
    _extended_io_path(protection_root).mkdir(exist_ok=True)
    snapshot_dir = protection_root / uuid.uuid4().hex
    _extended_io_path(snapshot_dir).mkdir()
    snapshots: list[_PartialSnapshot] = []
    try:
        for path, before in partials:
            backup = snapshot_dir / path.name
            digest = hashlib.sha256()
            copied = 0
            try:
                with _extended_io_path(path).open("rb") as source, _extended_io_path(backup).open("xb") as target:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        target.write(block)
                        digest.update(block)
                        copied += len(block)
                    target.flush()
                    os.fsync(target.fileno())
            except FileNotFoundError as exc:
                if os.name == "nt" and len(str(_extended_io_path(backup))) >= 260:
                    raise RuntimeError(
                        "续传保护副本创建失败：保护路径超过 Windows 文件路径限制；"
                        "原未完成文件未改动，请缩短任务输出目录后重试。"
                    ) from exc
                raise RuntimeError("续传保护副本创建失败，原未完成文件未改动。") from exc
            shutil.copystat(_extended_io_path(path), _extended_io_path(backup))
            after = _extended_io_path(path).stat()
            if (
                copied != before.st_size
                or after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
                or _extended_io_path(backup).stat().st_size != before.st_size
            ):
                raise RuntimeError(f"未完成下载在保护期间发生变化：{path.name}；原文件保持不动。")
            snapshots.append(_PartialSnapshot(path, backup, before.st_size, digest.hexdigest(), snapshot_dir))
    except BaseException:
        shutil.rmtree(_extended_io_path(snapshot_dir), ignore_errors=True)
        try:
            _extended_io_path(protection_root).rmdir()
        except OSError:
            pass
        raise

    if log:
        log(f"续传保护已就绪：保留了 {len(snapshots)} 个已有部分文件，共 {total_size} 字节。")
    return snapshots


def _clean_empty_snapshot_dirs(snapshot_dir: Path) -> None:
    try:
        _extended_io_path(snapshot_dir).rmdir()
    except OSError:
        return
    try:
        _extended_io_path(snapshot_dir.parent).rmdir()
    except OSError:
        pass


def _restore_changed_partials(snapshots: list[_PartialSnapshot], log: Callable[[str], None] | None) -> None:
    for snapshot in snapshots:
        current_size = 0
        current_matches = False
        try:
            if snapshot.path.is_file() and not snapshot.path.is_symlink():
                current_size = _extended_io_path(snapshot.path).stat().st_size
                current_matches = (
                    current_size >= snapshot.size
                    and _sha256_prefix(snapshot.path, snapshot.size) == snapshot.sha256
                )
        except OSError:
            current_matches = False

        if current_matches:
            try:
                _extended_io_path(snapshot.backup).unlink(missing_ok=True)
            except OSError:
                if log:
                    log("续传校验通过，但无法清理临时保护副本；保护副本仍保留在续传保护目录。")
                continue
            if log:
                log(f"续传检查通过：已有数据 {snapshot.size} 字节，当前部分文件 {current_size} 字节，未覆盖原数据。")
            _clean_empty_snapshot_dirs(snapshot.directory)
            continue

        restarted_candidate_exists = snapshot.path.exists() or snapshot.path.is_symlink()
        try:
            if restarted_candidate_exists:
                rejected_dir = snapshot.directory / "restarted"
                _extended_io_path(rejected_dir).mkdir(exist_ok=True)
                rejected = rejected_dir / f"{snapshot.path.name}.{uuid.uuid4().hex[:8]}.restarted"
                _extended_io_path(snapshot.path).replace(_extended_io_path(rejected))
            _extended_io_path(snapshot.backup).replace(_extended_io_path(snapshot.path))
        except OSError as exc:
            if log:
                log(
                    f"续传保护无法恢复原路径：{type(exc).__name__}；原数据副本仍保存在保护目录中，"
                    "未完成下载文件均已保留。"
                )
            continue
        if log:
            attempted_state = (
                f"失败时文件大小 {current_size} 字节；已恢复原部分，并将改写版本隔离保留。"
                if restarted_candidate_exists
                else "失败时原路径文件已消失；已从保护副本恢复原部分。"
            )
            log(f"检测到下载器重启或改写已有部分文件（原大小 {snapshot.size} 字节，{attempted_state}）")
        _clean_empty_snapshot_dirs(snapshot.directory)


def _discard_partial_snapshots(snapshots: list[_PartialSnapshot]) -> None:
    for directory in {snapshot.directory for snapshot in snapshots}:
        shutil.rmtree(_extended_io_path(directory), ignore_errors=True)
        _clean_empty_snapshot_dirs(directory)


def _safe_diagnostic_text(value: str) -> str:
    text = " ".join(str(value).split())

    def redact_url(match: re.Match[str]) -> str:
        raw = match.group(0).rstrip(".,;:)]}")
        trailing = match.group(0)[len(raw):]
        parsed = urlparse(raw)
        host = parsed.hostname or "redacted-host"
        return f"{parsed.scheme}://{host}/[redacted]" + ("?[redacted]" if parsed.query else "") + trailing

    text = re.sub(r"(?i)\b(?:https?|socks5h?|socks4a?)://[^\s\"'<>]+", redact_url, text)
    text = re.sub(
        r"(?i)\b(cookie|set-cookie|authorization|proxy-authorization|token|signature|sig|api[_-]?key|password)\s*[:=]\s*[^,\s;]+",
        r"\1=[redacted]",
        text,
    )
    text = re.sub(r"(?i)\b[A-Z]:\\[^\s\"'<>]+", "[local path]", text)
    return text[:500]


def _safe_exception_chain(exc: BaseException, limit: int = 6) -> str:
    entries: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(entries) < limit and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        message = _safe_diagnostic_text(str(current))
        entries.append(f"{name}: {message}" if message else name)
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return " <- ".join(entries)


class _YtDlpTaskLogger:
    """Persist useful yt-dlp diagnostics without echoing routine progress or secrets."""

    def __init__(self, log: Callable[[str], None]) -> None:
        self._log = log

    def debug(self, message: str) -> None:
        message = str(message)
        if (
            "Retrying" in message
            or message.startswith("[download] Got error:")
            or message.startswith("[download] Unable to resume")
            or message.startswith("[download] Resuming download")
        ):
            self._log(f"yt-dlp: {_safe_diagnostic_text(message)}")

    def warning(self, message: str) -> None:
        self._log(f"yt-dlp warning: {_safe_diagnostic_text(message)}")

    def error(self, message: str) -> None:
        self._log(f"yt-dlp error: {_safe_diagnostic_text(message)}")


def po_token_provider_status() -> dict[str, str | bool | None]:
    """Report whether the optional yt-dlp PO Token provider can be used."""
    try:
        available = importlib.util.find_spec("yt_dlp_plugins.extractor.getpot_wpc") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    return {
        "available": available,
        "browser_path": _find_chromium_browser(),
    }


def _find_chromium_browser() -> str | None:
    """Find a browser suitable for the WPC PO Token provider without opening it."""
    explicit = os.getenv("YBLOCALIZER_PO_BROWSER", "").strip()
    candidates = [Path(explicit)] if explicit else []
    for command in ("chrome", "chrome.exe", "msedge", "msedge.exe", "chromium", "chromium.exe"):
        resolved = resolve_command(command)
        if resolved:
            candidates.append(Path(resolved))
    for root_env, relative in (
        ("ProgramFiles", "Google/Chrome/Application/chrome.exe"),
        ("ProgramFiles(x86)", "Google/Chrome/Application/chrome.exe"),
        ("LOCALAPPDATA", "Google/Chrome/Application/chrome.exe"),
        ("ProgramFiles", "Microsoft/Edge/Application/msedge.exe"),
        ("ProgramFiles(x86)", "Microsoft/Edge/Application/msedge.exe"),
        ("LOCALAPPDATA", "Microsoft/Edge/Application/msedge.exe"),
    ):
        root = os.environ.get(root_env, "").strip()
        if root:
            candidates.append(Path(root) / Path(relative))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _validate_proxy(proxy: str | None) -> str | None:
    value = (proxy or "").strip()
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https", "socks4", "socks4a", "socks5", "socks5h"} or not parsed.hostname:
        raise ValueError("代理地址格式无效，请使用 http(s)://主机:端口 或 socks5(h)://主机:端口。")
    return value


def _youtube_access_opts(proxy: str | None, po_token_mode: str) -> dict:
    mode = str(po_token_mode or "auto").strip().lower()
    if mode not in {"auto", "off"}:
        raise ValueError("YouTube 浏览器验证模式必须是 auto 或 off。")
    opts: dict = {}
    valid_proxy = _validate_proxy(proxy)
    if valid_proxy:
        opts["proxy"] = valid_proxy
    if mode == "off":
        return opts

    status = po_token_provider_status()
    if not status["available"]:
        # Source checkouts remain usable without the optional package. The
        # packaged release includes it and preflight exposes any packaging bug.
        return opts
    extractor_args: dict[str, dict[str, list[str]]] = {
        "youtube": {
            "player_client": ["mweb"],
            "fetch_pot": ["auto"],
        }
    }
    browser_path = status.get("browser_path")
    if browser_path:
        extractor_args["youtubepot-wpc"] = {"browser_path": [str(browser_path)]}
    opts["extractor_args"] = extractor_args
    return opts


def _js_runtime_opts() -> dict:
    """Best-effort options for YouTube's n-challenge solver.

    Modern yt-dlp needs a JavaScript runtime (node/deno/bun) plus the remote
    ``ejs`` challenge solver script to obtain playable stream URLs.  When no
    runtime is installed the previous behaviour is kept (warning only).
    """
    runtimes = {
        name: {"path": None}
        for name in ("deno", "node", "bun", "quickjs")
        if resolve_command(name)
    }
    if not runtimes:
        # PyInstaller 打包的 EXE 有时 PATH 不完整，探测不到 node；
        # 兜底检查常见安装位置，避免 yt-dlp 缺少 JS 运行时解 n-sig 导致
        # YouTube 以 “Sign in to confirm you're not a bot” 拦截下载。
        node_candidates = [
            Path(os.environ.get("ProgramFiles", "")) / "nodejs" / "node.exe",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "nodejs" / "node.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "nodejs" / "node.exe",
        ]
        for candidate in node_candidates:
            if candidate.is_file():
                runtimes["node"] = {"path": str(candidate)}
                break
    if not runtimes:
        return {}
    return {"js_runtimes": runtimes, "remote_components": ["ejs:github"]}


@dataclass(slots=True)
class VideoMetadata:
    title: str | None
    duration: float | None
    license: str | None
    view_count: int | None
    webpage_url: str | None
    thumbnail_url: str | None
    max_width: int | None
    max_height: int | None
    quality_proposal: dict[str, Any] | None = None


def _quality_proposal(info: dict, quality: str) -> dict[str, Any]:
    """Describe yt-dlp's actual selection; never rank its format table ourselves."""
    requested = info.get("requested_formats")
    selected = requested if isinstance(requested, list) and requested else [info]
    video = next((item for item in selected if isinstance(item, dict) and str(item.get("vcodec") or "none") != "none"), {})
    audio = next((item for item in selected if isinstance(item, dict) and str(item.get("acodec") or "none") != "none" and item is not video), {})
    width, height = _selected_video_dimensions(info) or (0, 0)
    cap = {"720p": 720, "1080p": 1080}.get(str(quality).lower())
    muxed = str(video.get("acodec") or "none") != "none"
    return {
        "version": 3,
        "quality": str(quality),
        "video_format_id": str(video.get("format_id") or ""),
        "audio_format_id": "" if muxed else str(audio.get("format_id") or ""),
        "width": width or None,
        "height": height or None,
        "has_audio": muxed or bool(audio),
        "requires_consent": bool(cap and (height <= 0 or height < cap)),
    }


def _format_for_quality(
    quality: str,
    minimum_resolution: tuple[int, int] | None = None,
) -> str:
    normalized = str(quality or "1080p").strip().lower()
    heights = {"720p": 720, "1080p": 1080}
    if normalized != "original" and normalized not in heights:
        raise ValueError("下载画质必须是 720p、1080p 或 original。")
    width_floor = height_floor = 0
    if minimum_resolution is not None:
        try:
            width_floor, height_floor = (int(minimum_resolution[0]), int(minimum_resolution[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise DownloadProposalMismatch("画质确认记录不完整，无法安全开始下载。") from exc
        if width_floor <= 0 or height_floor <= 0:
            raise DownloadProposalMismatch("画质确认记录不完整，无法安全开始下载。")

    cap = heights.get(normalized)
    filters = "" if minimum_resolution is None else f"[width>={width_floor}][height>={height_floor}]"
    if cap is not None:
        filters += f"[height<={cap}]"
    # Each alternative carries the same confirmed-resolution floor. yt-dlp may
    # use a muxed stream when that is the best available match, but can never
    # fall through to an unbounded low-resolution "best" format.
    return f"bestvideo{filters}+bestaudio/best{filters}"


def _quarantine_below_resolution(
    work_dir: Path,
    minimum_resolution: tuple[int, int],
) -> list[Path]:
    """Move stale low-resolution finals aside so yt-dlp cannot reuse them."""
    minimum_width, minimum_height = minimum_resolution
    quarantine_dir = work_dir / ".quarantined-low-resolution"
    moved: list[Path] = []
    media_suffixes = {".mp4", ".mkv", ".webm", ".mov"}
    for candidate in work_dir.iterdir():
        if not candidate.is_file() or candidate.suffix.lower() not in media_suffixes:
            continue
        dimensions = _probe_video_dimensions(candidate)
        if dimensions and dimensions[0] >= minimum_width and dimensions[1] >= minimum_height:
            continue
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        size = f"{dimensions[0]}x{dimensions[1]}" if dimensions else "unverified"
        preserved = quarantine_dir / f"{candidate.stem}-{size}-{uuid.uuid4().hex[:8]}{candidate.suffix}"
        candidate.replace(preserved)
        moved.append(preserved)
    return moved


def _format_meets_proposal(current: dict[str, Any], promised: dict[str, Any]) -> None:
    """Format IDs may change; the promised resolution and audio may not."""
    try:
        width, height = int(current.get("width") or 0), int(current.get("height") or 0)
        minimum_width = int(promised.get("width") or 0)
        minimum_height = int(promised.get("height") or 0)
    except (TypeError, ValueError) as exc:
        raise DownloadProposalMismatch("原有画质方案不完整，请重新执行准备检查。") from exc
    if minimum_width <= 0 or minimum_height <= 0:
        raise DownloadProposalMismatch("当前画质方案不完整，请重新执行准备检查。")
    if width < minimum_width or height < minimum_height:
        raise DownloadProposalMismatch(
            f"先前确认的画质为 {minimum_width}×{minimum_height}，本次实际画质为 {width}×{height}。"
            "应用不会把较低画质文件提交为有效素材；请重新准备并明确确认新画质，或稍后重试。"
        )
    if promised.get("has_audio") is True and not current.get("has_audio"):
        raise DownloadProposalMismatch("先前确认的素材含音频，但当前可用视频流没有音频。请重新准备并确认。")


def _selected_video_dimensions(info: dict) -> tuple[int, int] | None:
    """Return the selected video stream size reported by yt-dlp, not max available."""
    selected = info.get("requested_formats")
    if isinstance(selected, list):
        candidates: list[tuple[int, int]] = []
        for item in selected:
            if not isinstance(item, dict) or str(item.get("vcodec") or "none").lower() == "none":
                continue
            try:
                width, height = int(item.get("width") or 0), int(item.get("height") or 0)
            except (TypeError, ValueError):
                continue
            if width > 0 and height > 0:
                candidates.append((width, height))
        if candidates:
            return max(candidates, key=lambda size: (size[1], size[0]))

    try:
        width, height = int(info.get("width") or 0), int(info.get("height") or 0)
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def _probe_video_dimensions(path: Path) -> tuple[int, int] | None:
    ffprobe = resolve_command("ffprobe")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [
                str(ffprobe), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0:
            return None
        streams = json.loads(completed.stdout).get("streams") or []
        if not streams:
            return None
        width, height = int(streams[0].get("width") or 0), int(streams[0].get("height") or 0)
        return (width, height) if width > 0 and height > 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")


def _platform_subtitle_languages(source_language: str | None) -> list[str]:
    language = (source_language or "en").strip().lower().replace("_", "-")
    base = language.split("-", 1)[0] or "en"
    return list(dict.fromkeys((language, base, f"{base}-orig")))


def _downloaded_subtitle_path(info: dict, work_dir: Path, video_id: str) -> Path | None:
    requested = info.get("requested_subtitles") or {}
    if isinstance(requested, dict):
        for item in requested.values():
            if not isinstance(item, dict):
                continue
            filepath = str(item.get("filepath") or "").strip()
            if filepath:
                candidate = Path(filepath)
                if not candidate.is_absolute():
                    candidate = work_dir / candidate
                if candidate.is_file() and candidate.suffix.lower() == ".srt":
                    return candidate
    candidates = sorted(
        work_dir.glob(f"{video_id}.*.srt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _try_download_platform_subtitles(
    yt_dlp_module,
    url: str,
    work_dir: Path,
    outtmpl: str,
    video_id: str,
    source_language: str | None,
    cookies_from_browser: str | None,
    cookies_file: str | None,
    proxy: str | None,
    po_token_mode: str,
    check: Callable[[], None],
    log: Callable[[str], None] | None,
) -> Path | None:
    """Download optional captions without making video acquisition depend on them."""
    options = {
        "outtmpl": outtmpl,
        "noplaylist": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": _platform_subtitle_languages(source_language),
        "subtitlesformat": "srt/best",
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 10,
        "retries": 0,
        "extractor_retries": 0,
    }
    options.update(_js_runtime_opts())
    options.update(_youtube_access_opts(proxy, po_token_mode))
    if cookies_file:
        options["cookiefile"] = cookies_file
    elif cookies_from_browser:
        options["cookiesfrombrowser"] = (cookies_from_browser,)
    try:
        check()
        with yt_dlp_module.YoutubeDL(options) as ydl:
            subtitle_info = ydl.extract_info(url, download=True)
        check()
    except Exception as exc:
        # Cancellation must remain authoritative. Any other subtitle-only
        # network error is a warning because Whisper is the real fallback.
        check()
        if log:
            log(f"平台字幕获取失败，已保留原视频并将自动回退 Whisper：{exc}")
        return None
    return _downloaded_subtitle_path(subtitle_info or {}, work_dir, video_id)


def _max_video_size(info: dict) -> tuple[int | None, int | None]:
    candidates: list[tuple[int, int]] = []
    for item in info.get("formats") or []:
        if not isinstance(item, dict) or item.get("vcodec") == "none":
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width > 0 and height > 0:
            candidates.append((width, height))
    if candidates:
        return max(candidates, key=lambda size: (size[1], size[0]))
    width, height = int(info.get("width") or 0), int(info.get("height") or 0)
    return (width or None, height or None)


@contextmanager
def _cancellable_external_ffmpeg(
    cancel_check: Callable[[], None],
    resource_profile: str = "balanced",
):
    """Make yt-dlp's blocking external FFmpeg wait observe the job token."""
    try:
        from yt_dlp.downloader import external
    except ImportError:
        yield
        return
    original = external.Popen
    profile = normalize_resource_profile(resource_profile)

    class CancellablePopen(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            lower_process_priority(self, profile)

        def wait(self, timeout=None):  # type: ignore[override]
            if timeout is not None:
                return super().wait(timeout=timeout)
            while True:
                try:
                    cancel_check()
                except BaseException:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(self.pid), "/T", "/F"],
                            check=False,
                            capture_output=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                    else:
                        super().terminate()
                    try:
                        super().wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        super().kill()
                        super().wait(timeout=3)
                    raise
                try:
                    return super().wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    continue

    external.Popen = CancellablePopen
    try:
        yield
    finally:
        external.Popen = original


def get_video_metadata(
    url: str,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
    proxy: str | None = None,
    po_token_mode: str = "auto",
    download_quality: str = "1080p",
) -> VideoMetadata:
    _validate_video_url(url)
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. Run: pip install -r requirements.txt") from exc

    opts = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": 15,
        "retries": 1,
        "extractor_retries": 1,
    }
    opts.update(_js_runtime_opts())
    opts.update(_youtube_access_opts(proxy, po_token_mode))
    opts["format"] = _format_for_quality(download_quality)
    if cookies_file:
        _validate_cookies_file(cookies_file)
        opts["cookiefile"] = cookies_file
    elif cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        _raise_with_youtube_hint(exc, cookies_from_browser, po_token_mode)
    max_width, max_height = _max_video_size(info)
    return VideoMetadata(
        title=info.get("title"),
        duration=info.get("duration"),
        license=info.get("license"),
        view_count=info.get("view_count"),
        webpage_url=info.get("webpage_url") or url,
        thumbnail_url=_best_thumbnail_url(info),
        max_width=max_width,
        max_height=max_height,
        quality_proposal=_quality_proposal(info, download_quality),
    )


def download_with_ytdlp(
    url: str,
    work_dir: Path,
    title: str | None = None,
    max_seconds: int | None = None,
    require_reuse_allowed: bool = False,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
    proxy: str | None = None,
    po_token_mode: str = "auto",
    download_quality: str = "1080p",
    progress: Callable[[float], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    resource_profile: str = "balanced",
    source_language: str | None = None,
    prefer_platform_subtitles: bool = True,
    log: Callable[[str], None] | None = None,
    quality_proposal: dict[str, Any] | None = None,
) -> VideoJob:
    check = cancel_check or legacy_check_cancelled
    profile = normalize_resource_profile(resource_profile)
    _validate_video_url(url)
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. Run: pip install -r requirements.txt") from exc

    work_dir.mkdir(parents=True, exist_ok=True)
    # yt-dlp owns stream selection and its .part continuation; do not force
    # overwrite, disable resume, or clear partial downloads here.
    outtmpl = str(work_dir / "%(id)s.%(ext)s")
    minimum_resolution: tuple[int, int] | None = None
    if quality_proposal is not None:
        try:
            minimum_resolution = (
                int(quality_proposal.get("width") or 0),
                int(quality_proposal.get("height") or 0),
            )
        except (TypeError, ValueError) as exc:
            raise DownloadProposalMismatch("已确认的画质记录不完整，请重新准备素材。") from exc
        if minimum_resolution[0] <= 0 or minimum_resolution[1] <= 0:
            raise DownloadProposalMismatch("已确认的画质记录不完整，请重新准备素材。")

    def download_progress(status: dict) -> None:
        check()
        if progress is None:
            return
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            downloaded = status.get("downloaded_bytes") or 0
            if total:
                progress(max(0.0, min(1.0, downloaded / total)))
        elif status.get("status") == "finished":
            progress(1.0)

    opts = {
        "format": _format_for_quality(download_quality, minimum_resolution),
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "restrictfilenames": False,
        "writeinfojson": True,
        "quiet": False,
        "no_warnings": False,
        "progress_hooks": [download_progress],
        "postprocessor_hooks": [download_progress],
        "socket_timeout": 15,
        "continuedl": True,
        "nopart": False,
        "retries": 10,
    }

    def sleep_before_http_retry(n: int | float) -> float:
        try:
            retry_number = int(n)
        except (TypeError, ValueError):
            retry_number = 0
        remaining = float(min(2 ** max(0, retry_number), 8))
        if log:
            log(f"yt-dlp HTTP 重试 {retry_number + 1}/10，退避 {int(remaining)} 秒。")
        # yt-dlp normally sleeps the returned delay; wait here so cancellation
        # remains responsive during backoff, then return zero to avoid sleeping twice.
        while remaining > 0:
            check()
            pause = min(0.25, remaining)
            time.sleep(pause)
            remaining -= pause
        return 0.0

    opts["retry_sleep_functions"] = {"http": sleep_before_http_retry}
    if log:
        opts["logger"] = _YtDlpTaskLogger(log)
    rate_limit = download_rate_limit(profile)
    if rate_limit is not None:
        opts["ratelimit"] = rate_limit
    opts.update(_js_runtime_opts())
    opts.update(_youtube_access_opts(proxy, po_token_mode))
    if max_seconds:
        require_command("ffmpeg")
        opts["download_ranges"] = yt_dlp.utils.download_range_func(None, [(0, max_seconds)])
        # Our only supported range starts at zero.  Forcing a keyframe makes
        # yt-dlp re-encode the whole slice (often 4K AV1 -> H.264 on the CPU),
        # which was the main cause of the desktop freezing during downloads.
        # Stream-copying the leading range is much faster and stays cancellable.
        opts["force_keyframes_at_cuts"] = False
    if cookies_file:
        _validate_cookies_file(cookies_file)
        opts["cookiefile"] = cookies_file
    elif cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    partial_snapshots: list[_PartialSnapshot] = []
    try:
        check()
        partial_snapshots = _snapshot_existing_partials(work_dir, log)
        with _cancellable_external_ffmpeg(check, profile):
            with yt_dlp.YoutubeDL(opts) as ydl:
                # The common path performs a single extraction.  A metadata-only
                # preflight is retained solely when the user explicitly requires
                # YouTube's CC/reuse flag before any media is downloaded.
                if require_reuse_allowed:
                    info = ydl.extract_info(url, download=False)
                    license_text = info.get("license") or ""
                    if "reuse allowed" not in license_text.lower():
                        raise RuntimeError(
                            "该视频未被 YouTube 标记为“可转载”（reuse allowed，yt-dlp 元数据中许可证为空或未知）。\n"
                            "如果你确认已获得处理或转载授权，请在素材页取消勾选「仅接受 CC reuse allowed」后重新开始任务。"
                        )
                if minimum_resolution is not None:
                    quarantined = _quarantine_below_resolution(work_dir, minimum_resolution)
                    if quarantined and log:
                        log(
                            f"发现 {len(quarantined)} 个不符合已确认分辨率的旧缓存，已保留到隔离目录；"
                            "本次只会选择达到已确认画质的流。"
                        )
                info = ydl.extract_info(url, download=True)
        check()
    except Exception as exc:
        _restore_changed_partials(partial_snapshots, log)
        if isinstance(exc, CancellationRequested):
            raise
        if log:
            log(f"素材获取异常详情（URL 与凭据已脱敏）：{_safe_exception_chain(exc)}")
        if quality_proposal is not None and _is_requested_format_unavailable_error(exc):
            width, height = minimum_resolution or (0, 0)
            raise DownloadProposalMismatch(
                f"当前源站没有达到已确认的 {width}×{height} 视频流，本次未下载较低画质。"
                "请重新读取素材信息，并在确认新的画质方案后重试。"
            ) from exc
        _raise_with_youtube_hint(exc, cookies_from_browser, po_token_mode)

    try:
        video_id = info.get("id") or slugify(info.get("title") or "video")
        candidates = sorted(work_dir.glob(f"{video_id}.*"), key=lambda path: path.stat().st_mtime, reverse=True)
        video_files = [path for path in candidates if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
        if not video_files:
            video_files = sorted(
                [path for path in work_dir.iterdir() if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        if not video_files:
            raise RuntimeError(f"Download finished but no video file was found in {work_dir}")

        selected_dimensions = _selected_video_dimensions(info or {})
        actual_dimensions = _probe_video_dimensions(video_files[0])
        if not actual_dimensions:
            raise RuntimeError("下载后无法通过 ffprobe 读取视频分辨率。")
        if selected_dimensions and (
            actual_dimensions[0] < selected_dimensions[0]
            or actual_dimensions[1] < selected_dimensions[1]
        ):
            raise DownloadResolutionMismatch(selected_dimensions, actual_dimensions)
        if quality_proposal:
            _format_meets_proposal(
                {"width": actual_dimensions[0], "height": actual_dimensions[1], "has_audio": _quality_proposal(info, download_quality)["has_audio"]},
                quality_proposal,
            )
    except Exception as exc:
        _restore_changed_partials(partial_snapshots, log)
        if log:
            log(f"下载后素材校验诊断（URL 与凭据已脱敏）：{_safe_exception_chain(exc)}")
        raise

    if any(work_dir.glob("*.part")):
        _restore_changed_partials(partial_snapshots, log)
    else:
        _discard_partial_snapshots(partial_snapshots)
    if actual_dimensions and log:
        log(
            f"原片分辨率校验通过：{actual_dimensions[0]}×{actual_dimensions[1]}；"
            f"画质选择为 {download_quality}，未自动降低画质。"
        )

    thumbnail_url = _best_thumbnail_url(info)
    thumbnail_path = _download_thumbnail(thumbnail_url, work_dir, video_id, proxy) if thumbnail_url else None
    source_subtitles = None
    if prefer_platform_subtitles and _is_youtube_url(url):
        source_subtitles = _try_download_platform_subtitles(
            yt_dlp, url, work_dir, outtmpl, video_id, source_language,
            cookies_from_browser, cookies_file, proxy, po_token_mode, check, log,
        )

    job = VideoJob(
        job_id=work_dir.name,
        source=url,
        source_kind="url",
        work_dir=work_dir,
        title=title or info.get("title"),
        description=info.get("description"),
        raw_video=video_files[0],
        source_subtitles=source_subtitles,
        license=info.get("license"),
        view_count=info.get("view_count"),
        webpage_url=info.get("webpage_url") or url,
        thumbnail_url=thumbnail_url,
        thumbnail_path=thumbnail_path,
    )
    return job


def import_local_video(path: Path, work_dir: Path, title: str | None = None) -> VideoJob:
    if not path.exists():
        raise FileNotFoundError(path)
    work_dir.mkdir(parents=True, exist_ok=True)
    destination = work_dir / f"raw{path.suffix.lower()}"
    if path.resolve() != destination.resolve():
        shutil.copy2(path, destination)
    return VideoJob(
        job_id=work_dir.name,
        source=str(path),
        source_kind="file",
        work_dir=work_dir,
        title=title or path.stem,
        raw_video=destination,
    )


def _raise_with_youtube_hint(
    exc: Exception,
    cookies_from_browser: str | None,
    po_token_mode: str = "auto",
) -> None:
    if isinstance(exc, YouTubeAccessError):
        raise exc
    message = str(exc)
    lowered = message.lower()
    if cookies_from_browser and (
        "Could not copy Chrome cookie database" in message
        or "failed to load cookies" in message
        or "Permission denied" in message
    ):
        raise YouTubeAccessError(
            "youtube_cookies_unreadable",
            f"无法读取 {cookies_from_browser} 浏览器 Cookies。请关闭对应浏览器后重试，"
            "或在设置中改用导出的 cookies.txt。",
            "关闭对应浏览器后重试，或改用 cookies.txt。",
        ) from exc

    if cookies_from_browser and (
        "dpapi" in lowered
        or "failed to decrypt" in lowered
        or "could not decrypt" in lowered
    ):
        raise YouTubeAccessError(
            "youtube_cookies_decrypt_failed",
            f"无法解密 {cookies_from_browser} 浏览器的 Cookies（DPAPI 加密）。"
            "请先在「设置」里选择 Cookies 来源浏览器，然后完全退出该浏览器再重新开始任务。"
            "如果仍然失败，请改用浏览器导出的 cookies.txt。",
            "完全退出浏览器后重试，或改用 cookies.txt。",
        ) from exc

    if "too many requests" in lowered or "http error 429" in lowered:
        raise YouTubeAccessError(
            "youtube_rate_limited",
            "YouTube 暂时限制了当前网络的请求频率（HTTP 429）。这不是字幕或翻译配置问题。",
            "暂停一段时间后重试；如使用代理，请更换为稳定且未被滥用的出口。",
        ) from exc

    provider_markers = (
        "po token", "pot provider", "proof of origin", "gvs token",
        "missing required po", "webpoclient", "youtubepot-wpc",
    )
    if any(marker in lowered for marker in provider_markers):
        status = po_token_provider_status()
        if not status["available"]:
            raise YouTubeAccessError(
                "youtube_po_provider_missing",
                "当前安装包缺少 YouTube 浏览器验证组件，无法自动获取本次请求所需的 PO Token。",
                "更新或修复安装 YouTube Bili Localizer 后重试。",
            ) from exc
        raise YouTubeAccessError(
            "youtube_po_token_failed",
            "YouTube 要求浏览器验证，但本次 PO Token 获取或使用失败。浏览器可能短暂出现或在后台启动。",
            "确认 Chrome 或 Edge 可以正常打开 YouTube，然后重试；仍失败时再检查代理与 Cookies。",
        ) from exc

    browser_markers = (
        "could not find chrome", "browser executable", "failed to start browser",
        "cannot find chrome", "chrome not found", "browser path",
    )
    if any(marker in lowered for marker in browser_markers):
        raise YouTubeAccessError(
            "youtube_browser_missing",
            "自动浏览器验证无法启动：未找到可用的 Chrome 或 Edge。",
            "安装或修复 Chrome/Edge，或在设置中关闭“自动浏览器验证”。",
        ) from exc

    bot_challenge_markers = (
        "confirm you're not a bot",
        "confirm you’re not a bot",
        "not a bot",
    )
    if any(marker in lowered for marker in bot_challenge_markers):
        status = po_token_provider_status()
        if str(po_token_mode).lower() == "off":
            raise YouTubeAccessError(
                "youtube_browser_verification_required",
                "YouTube 要求浏览器验证；当前任务关闭了自动 PO Token 获取。",
                "在设置中开启“自动浏览器验证（推荐）”后重试。",
            ) from exc
        if not status["available"]:
            raise YouTubeAccessError(
                "youtube_po_provider_missing",
                "YouTube 要求浏览器验证，但当前安装包缺少自动获取 PO Token 的组件。",
                "更新或修复安装 YouTube Bili Localizer 后重试。",
            ) from exc
        raise YouTubeAccessError(
            "youtube_po_token_failed",
            "YouTube 要求浏览器验证，但自动获取的 PO Token 未被本次请求接受。",
            "确认 Chrome/Edge 可以正常打开 YouTube 后重试；仍失败时再检查代理出口。",
        ) from exc

    signin_markers = (
        "please sign in",
        "sign in to",
        "you must sign in",
        "login required",
        "log in to",
        "accounts.google.com",
    )
    if any(marker in lowered for marker in signin_markers):
        raise YouTubeAccessError(
            "youtube_auth_required",
            "YouTube 要求登录或重新验证当前会话。自动浏览器验证只能补齐媒体请求令牌，不能代替账号登录。",
            "需要登录才能观看的视频，请重新导出有效 cookies.txt；公开内容可先不使用 Cookies 重试。",
        ) from exc

    if "http error 403" in lowered or "forbidden" in lowered or "request was blocked" in lowered:
        if str(po_token_mode).lower() == "off":
            action = "在设置中开启“自动浏览器验证（推荐）”后重试。"
        else:
            action = "先确认 Chrome/Edge 可访问 YouTube；随后检查代理出口，必要时更新 yt-dlp 与验证组件。"
        raise YouTubeAccessError(
            "youtube_forbidden",
            "YouTube 拒绝了媒体请求（HTTP 403）。可能是本次媒体令牌失效、网络出口受限，或视频需要登录；不能仅凭 403 判定为 Cookies 问题。",
            action,
        ) from exc
    if _is_stale_partial_download_error(exc):
        raise YouTubeAccessError(
            "youtube_stale_partial",
            "媒体流续传失败（HTTP 416 或文件被占用无法重命名）；已下载的部分仍保留。",
            "请先关闭占用文件的播放器或杀毒扫描后重试；若源站媒体流已变更，请重新执行准备检查。",
        ) from exc
    if _is_network_timeout_error(exc):
        raise YouTubeAccessError(
            "youtube_network_timeout",
            "下载 YouTube 媒体流时连接中断或超时；未完成文件已保留，不会自动降低画质。",
            "检查代理/VPN 连接后重试「获取素材」；yt-dlp 会尝试续传同一画质的已下载部分。",
        ) from exc
    raise exc


def _is_stale_partial_download_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "http error 416" in message
        or "requested range not satisfiable" in message
        or "unable to rename file" in message
        or "winerror 32" in message
        or ("另一个程序正在使用此文件" in str(exc))
    )


def _is_network_timeout_error(exc: BaseException) -> bool:
    """Identify interrupted network transfers without suggesting lower quality."""
    message = str(exc).lower()
    return any(marker in message for marker in (
        "winerror 10060",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "unexpected_eof_while_reading",
        "eof occurred in violation of protocol",
    ))


def _is_requested_format_unavailable_error(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return "requested format is not available" in message or "requested format not available" in message


# Compatibility name for external callers that imported the old private helper.
_raise_with_cookie_hint = _raise_with_youtube_hint


def _validate_video_url(url: str) -> None:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("请输入完整的 http(s) 视频链接，例如 https://www.youtube.com/watch?v=...")


def _validate_cookies_file(path: str | None) -> None:
    if not path:
        return
    value = Path(path).expanduser()
    if not value.is_file():
        raise RuntimeError(
            f"Cookies 文件不存在：{value}。请先在「设置」里选择导出的 cookies.txt 文件。"
        )


def _best_thumbnail_url(info: dict) -> str | None:
    thumbnails = info.get("thumbnails") or []
    candidates: list[tuple[int, str]] = []
    for item in thumbnails:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        preference = int(item.get("preference") or 0)
        candidates.append((width * height + preference, url))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    url = str(info.get("thumbnail") or "").strip()
    return url if url.startswith(("http://", "https://")) else None


def _download_thumbnail(
    url: str,
    work_dir: Path,
    video_id: str,
    proxy: str | None = None,
) -> Path | None:
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        valid_proxy = _validate_proxy(proxy)
        response_context = (
            build_opener(ProxyHandler({"http": valid_proxy, "https": valid_proxy})).open(request, timeout=30)
            if valid_proxy
            else urlopen(request, timeout=30)
        )
        with response_context as response:
            content = response.read()
    except Exception:
        return None
    if not content:
        return None
    original = work_dir / f"{video_id}.thumbnail"
    original.write_bytes(content)
    destination = work_dir / "cover.jpg"
    try:
        from PIL import Image

        with Image.open(original) as image:
            image = image.convert("RGB")
            image.save(destination, format="JPEG", quality=94)
        original.unlink(missing_ok=True)
        return destination
    except Exception:
        original.unlink(missing_ok=True)
        return None
