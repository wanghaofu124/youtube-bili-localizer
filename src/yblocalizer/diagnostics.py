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
import traceback
import zipfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import unquote

from . import __version__

LOG_NAME = "workbench.log"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_MAX_BYTES = 1_048_576
_BACKUP_COUNT = 3
_RECENT_LOG_BYTES = 256 * 1024
_SECRET_VALUE_MIN = 4

# Names whose values are credentials or login material.
_SENSITIVE_NAME = re.compile(
    r"(access_key|api[_-]?key|authorization|bili_jct|cookie|csrf|dedeuserid|"
    r"login_info|passwd|password|proxy|secret|sessdata|sign|token|key)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)\S+")
_ASSIGNMENT = re.compile(
    r"(?i)([\"']?)([\w.-]*(?:api[_-]?key|authorization|bili_jct|cookie|dedeuserid|"
    r"login_info|passwd|password|proxy|secret|sessdata|token)[\w.-]*)([\"']?)"
    r"(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^&\s]+)"
)
_QUERY_PARAM = re.compile(
    r"(?i)([\"']?)(?<![\w])(access_key|csrf|sign|key)(?![\w])([\"']?)"
    r"(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^&\s]+)"
)
_TOKEN_WORD = re.compile(r"(?i)((?<![\w])token\s+)(?!=)(\S+)")
_URL_USERINFO = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@")
_PACKAGES = ("yt-dlp", "faster-whisper", "playwright", "pywebview")


class _RedactingFilter(logging.Filter):
    """Strip secrets before a record is written to workbench.log."""

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = _secret_values(_env_path())
        record.msg = redact_text(record.getMessage(), secrets)
        record.args = None
        if record.exc_info and record.exc_info[0] is not None:
            record.exc_text = redact_text("".join(traceback.format_exception(*record.exc_info)), secrets)
        elif isinstance(record.exc_text, str):
            record.exc_text = redact_text(record.exc_text, secrets)
        return True


def configure_logging(log_dir: Path) -> None:
    """Attach one rotating file handler for the workbench log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = str((log_dir / LOG_NAME).resolve())
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, RotatingFileHandler) and handler.baseFilename == path:
            return
    handler = RotatingFileHandler(
        path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(_RedactingFilter())
    if not root.handlers:
        root.setLevel(logging.INFO)
    root.addHandler(handler)


def log_unhandled(where: str, exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
    logging.getLogger("yblocalizer.workbench").error(
        "未处理异常（%s）", where, exc_info=(exc_type, exc, tb),
    )


def log_frontend_error(summary: str) -> None:
    """Record an exception whose message is shown in the workbench."""
    logging.getLogger("yblocalizer.workbench").error("%s", summary, exc_info=True)


def install_excepthooks() -> None:
    """Log unhandled exceptions, then continue with the previous hook."""
    previous = sys.excepthook

    def hook(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        log_unhandled("主线程", exc_type, exc, tb)
        previous(exc_type, exc, tb)

    sys.excepthook = hook

    previous_thread = threading.excepthook

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        name = args.thread.name if args.thread is not None else ""
        log_unhandled(f"线程 {name}", args.exc_type, args.exc_value, args.exc_traceback)
        previous_thread(args)

    threading.excepthook = thread_hook


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
    files = [(path, _read_recent(path)) for path in _recent_logs(log_dir)]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("system.json", redact_text(json.dumps(system, ensure_ascii=False, indent=2), secrets))
        archive.writestr("config.env", redact_text(_sanitized_config(data_root / ".env"), secrets))
        for path, text in files:
            archive.writestr(f"logs/{path.name}", _hide_home(redact_text(text, secrets)))
    return archive_path


def redact_text(text: str, secrets: list[str]) -> str:
    text = unquote(text)
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    text = _BEARER.sub(r"\1[redacted]", text)
    text = _ASSIGNMENT.sub(_redact_assignment, text)
    text = _QUERY_PARAM.sub(_redact_assignment, text)
    text = _TOKEN_WORD.sub(r"\1[redacted]", text)
    return _URL_USERINFO.sub(r"\1[redacted]@", text)


def _redact_assignment(match: re.Match[str]) -> str:
    return f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}[redacted]"


def _env_path() -> Path:
    root = os.environ.get("YBLOCALIZER_DATA_DIR") or str(Path(os.environ.get("APPDATA") or Path.home()) / "YouTubeBiliLocalizer")
    return Path(root) / ".env"


def _secret_values(env_path: Path) -> list[str]:
    values: list[str] = []
    if env_path.is_file():
        for key, value in _parse_env(env_path):
            if value and _SENSITIVE_NAME.search(key):
                values.append(value)
    for key, value in os.environ.items():
        if value and _SENSITIVE_NAME.search(key):
            values.append(value)
    unique = {value for value in values if len(value) >= _SECRET_VALUE_MIN}
    return sorted(unique, key=len, reverse=True)


def _hide_home(text: str) -> str:
    home = Path.home()
    for variant in sorted({str(home), home.as_posix()}, key=len, reverse=True):
        if variant:
            text = text.replace(variant, "~")
    return text


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


def _recent_logs(log_dir: Path) -> list[Path]:
    """Only the rotating workbench log, not per-job logs."""
    if not log_dir.is_dir():
        return []
    files = [
        path for path in log_dir.iterdir()
        if path.is_file() and (path.name == LOG_NAME or path.name.startswith(f"{LOG_NAME}."))
    ]
    files.sort(key=lambda path: path.name)
    return files


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
