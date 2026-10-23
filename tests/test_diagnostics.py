from http import HTTPStatus
import logging
from logging.handlers import RotatingFileHandler
import sys
import threading
import zipfile
from pathlib import Path

import pytest

from yblocalizer import __version__
from yblocalizer.diagnostics import (
    configure_logging,
    export_diagnostics_zip,
    install_excepthooks,
    log_frontend_error,
)
from yblocalizer.workbench_api import WorkbenchHandler


_API_KEY = "sk-secret-value"
_OPENAI_KEY = "sk-openai-secret"
_PROXY = "http://user:password@127.0.0.1:7890"
_LOGIN_STATE = "sessdata-login-state-do-not-leak"
_COOKIE_FILE = "cookie-file-only-marker"
_PROFILE_FILE = "profile-only-secret-marker"


def _detach_log_handler() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "yblocalizer_log", False):
            root.removeHandler(handler)
            handler.close()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_deliberate_exception_is_written_to_the_log(tmp_path: Path) -> None:
    configure_logging(tmp_path)
    install_excepthooks()
    try:
        try:
            raise RuntimeError("deliberate-crash-marker")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())

        def boom() -> None:
            raise RuntimeError("worker-crash-marker")

        worker = threading.Thread(target=boom, name="diagnostics-worker")
        worker.start()
        worker.join()

        try:
            raise RuntimeError("frontend-crash-marker")
        except RuntimeError:
            log_frontend_error("界面可见的后端错误")

        attached = [handler for handler in logging.getLogger().handlers if getattr(handler, "yblocalizer_log", False)]
        assert len(attached) == 1
        assert isinstance(attached[0], RotatingFileHandler)
        assert attached[0].backupCount == 3
        text = (tmp_path / "workbench.log").read_text(encoding="utf-8")
    finally:
        _detach_log_handler()

    assert "deliberate-crash-marker" in text
    assert "worker-crash-marker" in text
    assert "frontend-crash-marker" in text
    assert "Traceback (most recent call last)" in text
    assert "未处理异常（主线程）" in text
    assert "未处理异常（线程 diagnostics-worker）" in text


def test_diagnostics_zip_redacts_secrets_and_lists_versions(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    log_dir = data_root / "logs"
    log_dir.mkdir(parents=True)
    (data_root / ".env").write_text(
        "\n".join([
            f"DEEPSEEK_API_KEY={_API_KEY}",
            f"OPENAI_API_KEY={_OPENAI_KEY}",
            "DEEPSEEK_TRANSLATE_MODEL=deepseek-v4-flash",
            "YBLOCALIZER_COOKIES_FILE=C:/Users/me/cookies.txt",
            f"YBLOCALIZER_YOUTUBE_PROXY={_PROXY}",
            "OPENAI_BASE_URL=https://api.openai.com/v1",
        ]) + "\n",
        encoding="utf-8",
    )
    (log_dir / "workbench.log").write_text(
        "\n".join([
            f"crash DEEPSEEK_API_KEY={_API_KEY}",
            f"embedded {_API_KEY} in the message",
            f"SESSDATA={_LOGIN_STATE}",
            f"proxy {_PROXY}",
            "Cookie: LOGIN_INFO=youtube-login-cookie",
        ]) + "\n",
        encoding="utf-8",
    )
    profile = data_root / "data" / "bilibili-profile"
    profile.mkdir(parents=True)
    (profile / "Cookies").write_text(f"SESSDATA={_PROFILE_FILE}\n", encoding="utf-8")
    (data_root / "cookies.txt").write_text(f"SID\t{_COOKIE_FILE}\n", encoding="utf-8")

    archive_path = export_diagnostics_zip(data_root, log_dir, tmp_path / "out.zip")

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        blob = "\n".join(archive.read(name).decode("utf-8") for name in names)
        system = archive.read("system.json").decode("utf-8")
        config = archive.read("config.env").decode("utf-8")
        log = archive.read("logs/workbench.log").decode("utf-8")

    assert "system.json" in names
    assert "config.env" in names
    assert "logs/workbench.log" in names
    assert not any("bilibili-profile" in name or name.endswith("Cookies") or name.endswith("cookies.txt") for name in names)
    assert __version__ in system
    assert "app_version" in system
    for key in ("yt-dlp", "faster-whisper", "playwright", "pywebview", "ffmpeg"):
        assert key in system
    assert sys.version.split()[0] in system
    assert "deepseek-v4-flash" in config
    assert "https://api.openai.com/v1" in config
    assert "crash DEEPSEEK_API_KEY=" in log
    for secret in (_API_KEY, _OPENAI_KEY, _PROXY, _LOGIN_STATE, _COOKIE_FILE, _PROFILE_FILE, "youtube-login-cookie", "user:password"):
        assert secret not in blob
    assert "DEEPSEEK_API_KEY=[redacted]" in config
    assert "YBLOCALIZER_COOKIES_FILE=[redacted]" in config
    assert "YBLOCALIZER_YOUTUBE_PROXY=[redacted]" in config


def test_diagnostics_export_route_writes_a_redacted_zip(isolated_user_data: Path) -> None:
    log_dir = isolated_user_data / "logs"
    (isolated_user_data / ".env").write_text(f"DEEPSEEK_API_KEY={_API_KEY}\n", encoding="utf-8")
    (log_dir / "workbench.log").write_text(f"shown {_API_KEY}\n", encoding="utf-8")

    handler = object.__new__(WorkbenchHandler)
    handler.path = "/api/diagnostics/export"
    handler._authorize_mutation = lambda _path: True  # type: ignore[method-assign]
    handler._payload = lambda: {}  # type: ignore[method-assign]
    result: dict[str, object] = {}
    handler._json = lambda payload, status=HTTPStatus.OK: result.update(payload=payload, status=status)  # type: ignore[method-assign]

    handler.do_POST()

    assert result["status"] == HTTPStatus.OK
    payload = result["payload"]
    assert isinstance(payload, dict)
    archive_path = Path(str(payload["path"]))
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        blob = "\n".join(archive.read(name).decode("utf-8") for name in archive.namelist())
    assert _API_KEY not in blob
    assert "logs/workbench.log" in zipfile.ZipFile(archive_path).namelist()
