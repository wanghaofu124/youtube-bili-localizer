"""Crash log and one-click diagnostics export.

Logs go to the existing user-data ``logs`` directory. The diagnostics zip is a
redacted copy of those logs plus version and config; it never contains API
keys, tokens, passwords, cookies, Bilibili login state, or browser profiles.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
import zipfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

from . import __version__

LOG_NAME = "workbench.log"
_HANDLER_FLAG = "yblocalizer_log"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_MAX_BYTES = 1_048_576
_BACKUP_COUNT = 3
_RECENT_LOG_FILES = 8
_RECENT_LOG_BYTES = 256 * 1024
_SECRET_VALUE_MIN = 8

# Names whose values are credentials or login material.
_SENSITIVE_NAME = re.compile(
    r"(api[_-]?key|token|password|passwd|secret|cookie|sessdata|bili_jct|dedeuserid|login_info|authorization|proxy)",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    r"(?i)([\w.-]*(?:api[_-]?key|token|password|passwd|secret|cookie|sessdata|bili_jct|dedeuserid|login_info|authorization|proxy)[\w.-]*)"
    r"(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|\S+)"
)
_URL_USERINFO = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@")
_APP_SECRET_KEYS = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "YBLOCALIZER_COOKIES_FILE",
    "YBLOCALIZER_COOKIES_FROM_BROWSER",
    "YBLOCALIZER_YOUTUBE_PROXY",
)
_PACKAGES = ("yt-dlp", "faster-whisper", "playwright", "pywebview")
_PROFILE_MARKERS = (
    "bilibili-profile",
    "bilibili-edge-profile",
    "douyin-profile",
    "douyin-edge-profile",
    "ms-playwright",
)

_hooks_installed = False


def configure_logging(log_dir: Path) -> None:
    """Attach one rotating file handler for the workbench log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = str((log_dir / LOG_NAME).resolve())
    root = logging.getLogger()
    for handler in list(root.handlers):
        if not getattr(handler, _HANDLER_FLAG, False):
            continue
        if getattr(handler, "baseFilename", None) == path:
            return
        root.removeHandler(handler)
        handler.close()
    others = [item for item in root.handlers if not getattr(item, _HANDLER_FLAG, False)]
    handler = RotatingFileHandler(
        path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    setattr(handler, _HANDLER_FLAG, True)
    root.addHandler(handler)
    if not others and (root.level == logging.NOTSET or root.level > logging.INFO):
        root.setLevel(logging.INFO)


def log_unhandled(where: str, exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
    logging.getLogger("yblocalizer.workbench").error(
        "未处理异常（%s）", where, exc_info=(exc_type, exc, tb),
    )


def log_frontend_error(summary: str) -> None:
    """Record an exception whose message is shown in the workbench."""
    logging.getLogger("yblocalizer.workbench").error("%s", summary, exc_info=True)


def install_excepthooks() -> None:
    """Log unhandled exceptions from the main thread and worker threads."""
    global _hooks_installed
    if _hooks_installed and getattr(sys.excepthook, "_yblocalizer_crash", False):
        return
    previous = sys.excepthook

    def hook(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        log_unhandled("主线程", exc_type, exc, tb)
        if previous is not hook and previous is not sys.__excepthook__:
            previous(exc_type, exc, tb)

    hook._yblocalizer_crash = True  # type: ignore[attr-defined]
    sys.excepthook = hook

    previous_thread = threading.excepthook

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        name = args.thread.name if args.thread is not None else ""
        log_unhandled(f"线程 {name}", args.exc_type, args.exc_value, args.exc_traceback)
        if previous_thread is not thread_hook and previous_thread is not threading.__excepthook__:
            previous_thread(args)

    thread_hook._yblocalizer_crash = True  # type: ignore[attr-defined]
    threading.excepthook = thread_hook
    _hooks_installed = True


def export_diagnostics_zip(data_root: Path, log_dir: Path, dest: Path | None = None) -> Path:
    """Write a diagnostics zip next to the logs. Secrets are redacted."""
    log_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest or (log_dir / f"diagnostics-{time.strftime('%Y%m%d-%H%M%S')}.zip")
    secrets = _secret_values(data_root / ".env")
    system = {
        "app_version": __version__,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": sys.version,
        "dependencies": _dependency_versions(),
    }
    files = [(path, _read_recent(path)) for path in _recent_logs(log_dir, archive_path)]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("system.json", redact_text(json.dumps(system, ensure_ascii=False, indent=2), secrets))
        archive.writestr("config.env", redact_text(_sanitized_config(data_root / ".env"), secrets))
        for path, text in files:
            archive.writestr(f"logs/{path.name}", redact_text(text, secrets))
    return archive_path


def redact_text(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    text = _ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", text)
    return _URL_USERINFO.sub(r"\1[redacted]@", text)


def _secret_values(env_path: Path) -> list[str]:
    values: list[str] = []
    keys = set(_APP_SECRET_KEYS)
    if env_path.is_file():
        for key, value in _parse_env(env_path):
            if _SENSITIVE_NAME.search(key):
                keys.add(key)
                values.append(value)
    for key in keys:
        value = os.environ.get(key, "")
        if value:
            values.append(value)
    unique = {value for value in values if len(value) >= _SECRET_VALUE_MIN}
    return sorted(unique, key=len, reverse=True)


def _sanitized_config(env_path: Path) -> str:
    if not env_path.is_file():
        return "# no local .env\n"
    lines: list[str] = []
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            lines.append(line)
            continue
        key, _value = line.split("=", 1)
        if _SENSITIVE_NAME.search(key):
            lines.append(f"{key.strip()}=[redacted]")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _parse_env(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        pairs.append((key.strip(), value.strip().strip("\"'")))
    return pairs


def _recent_logs(log_dir: Path, archive_path: Path) -> list[Path]:
    if not log_dir.is_dir():
        return []
    skip = archive_path.resolve()
    files: list[Path] = []
    for path in log_dir.iterdir():
        if not path.is_file():
            continue
        if path.resolve() == skip:
            continue
        name = path.name
        if any(marker in name.lower() for marker in _PROFILE_MARKERS):
            continue
        if name == LOG_NAME or name.startswith(f"{LOG_NAME}.") or path.suffix == ".log":
            files.append(path)
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:_RECENT_LOG_FILES]


def _read_recent(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - _RECENT_LOG_BYTES))
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _dependency_versions() -> dict[str, str]:
    versions = {name: _package_version(name) for name in _PACKAGES}
    versions["ffmpeg"] = _ffmpeg_version()
    return versions


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _ffmpeg_version() -> str:
    from .dependencies import resolve_command

    exe = resolve_command("ffmpeg")
    if exe is None:
        return "not found"
    try:
        completed = subprocess.run(
            [str(exe), "-version"], capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "not found"
    lines = (completed.stdout or completed.stderr).splitlines()
    return lines[0].strip() if lines else "not found"
