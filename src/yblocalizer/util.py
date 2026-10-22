from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from .cancellation import CancellationRequested
from .dependencies import resolve_command
from .performance import lower_process_priority, normalize_resource_profile


def require_command(name: str) -> None:
    if resolve_command(name) is None:
        raise RuntimeError(f"Required command not found on PATH: {name}")


def _discover_windows_command(name: str) -> Path | None:
    if not sys.platform.startswith("win"):
        return None
    executable = f"{name}.exe" if not name.lower().endswith(".exe") else name
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None
    winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    if not winget_packages.exists():
        return None
    matches = sorted(winget_packages.rglob(executable), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_json(path: Path, default: Any) -> Any:
    """Read UTF-8 JSON. Return default when the file is missing, unreadable, or invalid."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _safe_console_print(message: str) -> None:
    """Write diagnostics without letting the local console encoding stop work."""
    stream = sys.stdout
    if stream is None:
        return
    try:
        print(message, file=stream, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        escaped = message.encode(encoding, errors="backslashreplace").decode(encoding)
        try:
            print(escaped, file=stream, flush=True)
        except (OSError, UnicodeError, ValueError):
            pass
    except (OSError, ValueError):
        # GUI/frozen launchers may have no attached console at all.
        pass


def run(
    command: list[str],
    cwd: Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    resource_profile: str = "balanced",
    progress_callback: Callable[[float], None] | None = None,
    progress_duration: float | None = None,
) -> None:
    printable = " ".join(command)
    _safe_console_print(f"$ {printable}")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        command, cwd=cwd, text=True, errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    lower_process_priority(process, normalize_resource_profile(resource_profile))
    output: list[str] = []
    cancelled = False
    last_progress = 0.0

    def _report_progress(line: str) -> None:
        nonlocal last_progress
        if not progress_callback:
            return
        if line.startswith(("out_time_us=", "out_time_ms=")) and progress_duration and progress_duration > 0:
            try:
                elapsed = int(line.partition("=")[2]) / 1_000_000
            except ValueError:
                return
            fraction = max(0.0, min(0.99, elapsed / progress_duration))
            if fraction > last_progress:
                last_progress = fraction
                progress_callback(fraction)
        elif line.strip() == "progress=end" and last_progress < 1.0:
            last_progress = 1.0
            progress_callback(1.0)

    def _watch_cancel() -> None:
        nonlocal cancelled
        while process.poll() is None:
            if cancel_check and cancel_check():
                cancelled = True
                _terminate_process_tree(process)
                return
            time.sleep(0.1)

    watcher = threading.Thread(target=_watch_cancel, daemon=True) if cancel_check else None
    if watcher:
        watcher.start()
    try:
        while True:
            try:
                line = process.stdout.readline() if process.stdout else ""
            except (OSError, ValueError):
                line = ""
            if not line and process.poll() is not None:
                break
            if line:
                value = line.rstrip()
                output.append(value)
                _report_progress(value)
        if cancelled:
            raise CancellationRequested("任务已被用户中断")
        if process.returncode != 0:
            tail = "\n".join(output[-40:])
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {printable}\n{tail}")
    finally:
        if process.stdout:
            process.stdout.close()
        if watcher:
            watcher.join(timeout=4)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate only the known child command after the user presses cancel."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def slugify(value: str, fallback: str = "video") -> str:
    value = re.sub(r"[^\w\-.]+", "-", value, flags=re.UNICODE).strip("-._")
    value = re.sub(r"-{2,}", "-", value)
    return value[:80] or fallback


_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{6,})",
    re.IGNORECASE,
)
_MEDIA_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".mp3", ".wav", ".m4a")


def extract_youtube_video_id(url: str | None) -> str | None:
    text = (url or "").strip()
    if not text:
        return None
    match = _YOUTUBE_ID_RE.search(text)
    return match.group(1) if match else None


def _folder_safe_slug(value: str, *, max_len: int = 48, fallback: str = "") -> str:
    """Slug for job folder / job id segments (no dots — keep URL routes working)."""
    text = (value or "").strip()
    lowered = text.lower()
    for suffix in _MEDIA_SUFFIXES:
        if lowered.endswith(suffix):
            text = text[: -len(suffix)]
            break
    # Prefer pathlib stem when the value looks like a file path.
    if "/" in text or "\\" in text:
        text = Path(text).stem
    slug = slugify(text, fallback=fallback or "task")
    slug = slug.replace(".", "-").strip("-._")
    return (slug[:max_len] or (fallback or "task"))


def job_folder_label(
    *,
    title: str | None = None,
    source_url: str | None = None,
    source_name: str | None = None,
    max_len: int = 48,
) -> str:
    """Human-readable folder label from title / YouTube id / local file name."""
    video_id = extract_youtube_video_id(source_url)
    title_slug = _folder_safe_slug(title or "", max_len=max_len, fallback="")
    if title_slug and title_slug != "task" and video_id:
        # Keep both: easy to spot the video and still unique across re-runs.
        id_part = video_id[:12]
        remain = max(8, max_len - len(id_part) - 1)
        return f"{id_part}_{title_slug[:remain]}"
    if title_slug and title_slug != "task":
        return title_slug
    if video_id:
        return video_id
    file_slug = _folder_safe_slug(source_name or "", max_len=max_len, fallback="")
    if file_slug and file_slug != "task":
        return file_slug
    return "task"


def allocate_job_folder_name(
    output_root: Path,
    *,
    title: str | None = None,
    source_url: str | None = None,
    source_name: str | None = None,
    now: datetime | None = None,
) -> str:
    """Allocate a unique work-dir name: YYYYMMDD-HHMMSS_label[-n]."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    label = job_folder_label(title=title, source_url=source_url, source_name=source_name)
    base = f"{stamp}_{label}"
    # Windows path safety: no trailing dots/spaces.
    base = base.rstrip(". ")
    candidate = base
    suffix = 2
    root = output_root
    while (root / candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
        if suffix > 99:
            from uuid import uuid4

            candidate = f"{base}-{uuid4().hex[:6]}"
            break
    return candidate


def timestamp_id(prefix: str = "job", *, title: str | None = None) -> str:
    """Legacy CLI helper; prefers readable stamp_label when a title is given."""
    if title and title.strip():
        # Ignore output_root collision here; callers that need uniqueness should
        # use allocate_job_folder_name with a real output root.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label = job_folder_label(title=title)
        return f"{stamp}_{label}"
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def project_root() -> Path:
    return Path.cwd()


def ensure_rights_confirmed(confirmed: bool) -> None:
    if confirmed:
        return
    raise SystemExit(
        "Refusing to continue without --i-have-rights. Use this tool only for videos you own, "
        "videos you have explicit permission to process, or content whose license allows this use."
    )


def warn_optional_dependency(package: str, install_hint: str) -> None:
    print(f"Missing optional dependency: {package}", file=sys.stderr)
    print(f"Install hint: {install_hint}", file=sys.stderr)
