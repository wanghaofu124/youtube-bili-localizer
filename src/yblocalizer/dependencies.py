"""First-run dependency discovery and installation.

The workbench API deliberately delegates host changes to this module.  It is
the only place allowed to start package-manager/model-browser installation
processes, which keeps the HTTP handlers and the media pipeline declarative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable


WEBVIEW2_DOWNLOAD_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"
WINGET_HELP_URL = "https://learn.microsoft.com/windows/package-manager/winget/"
_SCAN_CACHE_TTL_SECONDS = 10.0
_scan_cache: dict[tuple[str, str], tuple[float, list[Path]]] = {}
_scan_cache_lock = threading.Lock()


class DependencyInstallCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DependencySpec:
    id: str
    label: str
    purpose: str
    required: bool
    install_kind: str | None
    size_hint: str
    action_url: str | None = None


SPECS: tuple[DependencySpec, ...] = (
    DependencySpec("ffmpeg", "FFmpeg", "提取音频与硬字幕渲染", True, "winget", "约 180 MB", WINGET_HELP_URL),
    DependencySpec("whisper-tiny", "Whisper tiny 模型", "轻量音频转写模型", False, "whisper", "约 75 MB"),
    DependencySpec("whisper-base", "Whisper base 模型", "基础音频转写模型", False, "whisper", "约 150 MB"),
    DependencySpec("whisper-small", "Whisper small 模型", "默认音频转写模型，可提前下载", False, "whisper", "约 500 MB"),
    DependencySpec("whisper-medium", "Whisper medium 模型", "更高精度音频转写模型", False, "whisper", "约 1.5 GB"),
    DependencySpec("whisper-large-v3", "Whisper large-v3 模型", "最高质量音频转写模型", False, "whisper", "约 3 GB"),
    DependencySpec("node", "Node.js", "提高 YouTube 新版视频链接解析兼容性", False, "winget", "约 80 MB", WINGET_HELP_URL),
    DependencySpec("youtube-po-token", "YouTube 浏览器验证组件", "自动获取 YouTube 媒体请求所需的 PO Token", False, None, "已包含在安装包中"),
    DependencySpec("tesseract", "Tesseract OCR", "OCR 与音频+OCR 字幕模式", False, "winget", "约 100 MB", WINGET_HELP_URL),
    DependencySpec("playwright-chromium", "投稿辅助浏览器", "B 站登录检查、上传与填表", False, "playwright", "约 300 MB"),
    DependencySpec("webview2", "Microsoft WebView2", "显示桌面应用窗口", True, None, "通常随 Windows/Edge 安装", WEBVIEW2_DOWNLOAD_URL),
)

_COMMAND_CANDIDATES: dict[str, tuple[Path, ...]] = {
    "node": (
        Path(os.environ.get("ProgramFiles", "")) / "nodejs" / "node.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "nodejs" / "node.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "nodejs" / "node.exe",
    ),
    "tesseract": (
        Path(os.environ.get("ProgramFiles", "")) / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Tesseract-OCR" / "tesseract.exe",
    ),
}


def default_user_data_root() -> Path:
    return Path(os.environ.get("APPDATA") or Path.home()) / "YouTubeBiliLocalizer"


def clear_dependency_cache() -> None:
    with _scan_cache_lock:
        _scan_cache.clear()


def _cached_package_matches(packages_root: Path, executable: str) -> list[Path]:
    key = (str(packages_root).lower(), executable.lower())
    now = time.monotonic()
    with _scan_cache_lock:
        cached = _scan_cache.get(key)
        if cached and now - cached[0] < _SCAN_CACHE_TTL_SECONDS:
            return list(cached[1])
    try:
        matches = sorted(
            packages_root.glob(f"**/{executable}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:4]
    except OSError:
        matches = []
    with _scan_cache_lock:
        _scan_cache[key] = (now, matches)
    return matches


def resolve_command(name: str, tools_root: Path | None = None) -> Path | None:
    """Find a command in PATH, the app-local tool folder, or WinGet links.

    The result is also made visible to child processes in the current app
    session, so a freshly installed FFmpeg works without restarting the EXE.
    """
    found = shutil.which(name)
    if found:
        return Path(found)
    executable = name if name.lower().endswith(".exe") else f"{name}.exe"
    candidates: list[Path] = []
    root = tools_root or Path(os.environ.get("YBLOCALIZER_TOOLS_DIR", "") or default_user_data_root() / "tools")
    if sys.platform.startswith("win"):
        candidates.extend([
            root / name / executable,
            root / name / "bin" / executable,
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / executable,
        ])
        candidates.extend(_COMMAND_CANDIDATES.get(name.removesuffix(".exe"), ()))
        winget_packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        if winget_packages.is_dir():
            candidates.extend(_cached_package_matches(winget_packages, executable))
    for candidate in candidates:
        try:
            if candidate.is_file():
                parent = str(candidate.parent)
                current = os.environ.get("PATH", "")
                if parent.lower() not in {part.lower() for part in current.split(os.pathsep) if part}:
                    os.environ["PATH"] = f"{parent}{os.pathsep}{current}"
                return candidate
        except OSError:
            continue
    return None


def webview2_available() -> bool:
    if not sys.platform.startswith("win"):
        return True
    try:
        import winreg
    except ImportError:
        return False
    client = r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C72EC9A3C}"
    locations = (
        (winreg.HKEY_CURRENT_USER, client),
        (winreg.HKEY_LOCAL_MACHINE, client),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{{F3017226-FE2A-4295-8BDF-00C72EC9A3C}}"),
    )
    for hive, key_path in locations:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                version = str(winreg.QueryValueEx(key, "pv")[0]).strip()
                if version and version != "0.0.0.0":
                    return True
        except OSError:
            continue
    # Edge itself normally carries a compatible runtime even if the Evergreen
    # registration is unavailable to a restricted user account.
    edge = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft" / "EdgeWebView" / "Application"
    return edge.is_dir() and any(edge.glob("*/msedgewebview2.exe"))


def _model_root(user_data_root: Path, size: str = "small") -> Path:
    return user_data_root / "models" / f"faster-whisper-{size}"


def resolve_whisper_model(model_size: str, user_data_root: Path | None = None) -> str:
    root = _model_root(user_data_root or default_user_data_root(), model_size)
    required = ("model.bin", "config.json", "tokenizer.json")
    return str(root) if all((root / name).is_file() for name in required) else model_size


def require_whisper_model(model_size: str, user_data_root: Path | None = None) -> Path:
    """Return a complete managed model, never allowing an implicit download.

    ``faster-whisper`` accepts a model name and downloads it on construction.
    That behaviour is useful for scripts but surprising in a desktop workflow:
    it can start a multi-hundred-megabyte transfer after the source video has
    already been downloaded.  The workbench installs models explicitly in its
    preparation centre, so transcription must only receive a local directory.
    """
    size = str(model_size or "small").strip().lower()
    root = _model_root(user_data_root or default_user_data_root(), size)
    required = ("model.bin", "config.json", "tokenizer.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Whisper {size} 模型尚未完整安装。请返回“开始前准备”安装模型并重新检查；"
            "转写阶段不会自动下载模型。"
        )
    return root


def _playwright_root() -> Path:
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if configured:
        return Path(configured)
    managed = default_user_data_root() / "browsers" / "ms-playwright"
    if managed.is_dir():
        return managed
    return Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"


def _playwright_available() -> bool:
    root = _playwright_root()
    return root.is_dir() and bool(list(root.glob("chromium-*")) or list(root.glob("chromium_headless_shell-*")))


def dependency_statuses(user_data_root: Path | None = None) -> list[dict[str, Any]]:
    root = user_data_root or default_user_data_root()
    winget_available = resolve_command("winget", root / "tools") is not None
    command_available = {
        "ffmpeg": resolve_command("ffmpeg", root / "tools") is not None,
        "node": resolve_command("node", root / "tools") is not None,
        "tesseract": resolve_command("tesseract", root / "tools") is not None,
        "youtube-po-token": _module_available("yt_dlp_plugins.extractor.getpot_wpc"),
    }
    model_available = {
        f"whisper-{size}": resolve_whisper_model(size, root) != size
        for size in ("tiny", "base", "small", "medium", "large-v3")
    }
    available = {
        **command_available,
        **model_available,
        "playwright-chromium": _playwright_available(),
        "webview2": webview2_available(),
    }
    rows = []
    for spec in SPECS:
        can_install = bool(spec.install_kind)
        if spec.install_kind == "winget":
            can_install = winget_available and sys.platform.startswith("win")
        if spec.install_kind == "whisper":
            can_install = _module_available("faster_whisper")
        if spec.install_kind == "playwright":
            can_install = _module_available("playwright")
        present = bool(available[spec.id])
        rows.append({
            "id": spec.id,
            "label": spec.label,
            "purpose": spec.purpose,
            "required": spec.required,
            "available": present,
            "installable": can_install and not present,
            "size_hint": spec.size_hint,
            "action_url": spec.action_url,
            "message": "已安装，可以使用。" if present else (
                "点击安装，完成后无需重启应用。" if can_install else
                "当前环境缺少 WinGet；点击说明安装 App Installer 后重试。" if spec.install_kind == "winget" else
                "当前环境无法自动安装，请使用说明链接。" if spec.action_url else
                "当前环境无法自动安装。"
            ),
        })
    return rows


def dependency_capabilities(user_data_root: Path | None = None) -> dict[str, dict[str, Any]]:
    return {
        row["id"]: {key: row[key] for key in ("label", "required", "purpose", "available")}
        for row in dependency_statuses(user_data_root)
        if row["id"] in {"ffmpeg", "node", "tesseract"}
    }


def _module_available(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass(slots=True)
class InstallJob:
    id: str
    dependency_id: str
    status: str = "queued"
    progress: int = 0
    progress_kind: str = "indeterminate"
    message: str = "正在准备安装…"
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    _process: subprocess.Popen[str] | None = field(default=None, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dependency_id": self.dependency_id,
            "status": self.status,
            "progress": self.progress,
            "progress_kind": self.progress_kind,
            "message": self.message,
            "logs": self.logs[-30:],
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancel_requested": self._cancel_event.is_set(),
        }


class DependencyManager:
    """Owns a single serialized host-install queue for the local desktop app."""

    def __init__(self, user_data_root: Path | None = None) -> None:
        self.user_data_root = (user_data_root or default_user_data_root()).resolve()
        self.tools_root = self.user_data_root / "tools"
        self.models_root = self.user_data_root / "models"
        os.environ.setdefault("YBLOCALIZER_TOOLS_DIR", str(self.tools_root))
        self._jobs: dict[str, InstallJob] = {}
        self._active_id: str | None = None
        self._lock = threading.RLock()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = self._jobs.get(self._active_id or "")
            rows = dependency_statuses(self.user_data_root)
            return {
                "ready": all(row["available"] for row in rows if row["required"]),
                "dependencies": rows,
                "active_job": active.snapshot() if active else None,
            }

    def get_job(self, job_id: str) -> InstallJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, dependency_id: str, confirmed: bool) -> InstallJob:
        if not confirmed:
            raise ValueError("安装依赖前需要明确确认。")
        spec = next((item for item in SPECS if item.id == dependency_id), None)
        if spec is None:
            raise ValueError("未知依赖组件。")
        status = next(item for item in dependency_statuses(self.user_data_root) if item["id"] == dependency_id)
        if status["available"]:
            job = InstallJob(
                uuid.uuid4().hex, dependency_id, status="completed", progress=100,
                progress_kind="determinate", message="组件已经安装，无需重复操作。", finished_at=time.time(),
            )
            with self._lock:
                self._jobs[job.id] = job
            return job
        if not status["installable"]:
            raise RuntimeError(status["message"])
        with self._lock:
            if self._active_id and (active := self._jobs.get(self._active_id)) and active.status in {"queued", "running", "cancelling"}:
                raise RuntimeError(f"正在安装 {active.dependency_id}，请等待当前任务完成。")
            job = InstallJob(uuid.uuid4().hex, dependency_id)
            self._jobs[job.id] = job
            self._active_id = job.id
        job._thread = threading.Thread(target=self._run, args=(job, spec), daemon=True, name=f"dependency-{dependency_id}")
        job._thread.start()
        return job

    def cancel(self, job_id: str) -> InstallJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.status not in {"queued", "running", "cancelling"}:
                return job
            job._cancel_event.set()
            job.status = "cancelling"
            job.message = "正在取消安装；网络请求会在当前步骤结束后停止…"
            process = job._process
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        return job

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            active = self._jobs.get(self._active_id or "")
        if not active:
            return
        self.cancel(active.id)
        thread = active._thread
        if thread and thread.is_alive():
            thread.join(timeout=max(0.0, timeout))

    @staticmethod
    def _check_cancelled(job: InstallJob) -> None:
        if job._cancel_event.is_set():
            raise DependencyInstallCancelled("安装已由用户取消。")

    def _run(self, job: InstallJob, spec: DependencySpec) -> None:
        job.status, job.progress = "running", 5
        try:
            self._check_cancelled(job)
            if spec.install_kind == "winget":
                self._install_winget(job, spec.id)
            elif spec.install_kind == "whisper":
                self._install_whisper(job, spec.id.removeprefix("whisper-"))
            elif spec.install_kind == "playwright":
                self._install_playwright(job)
            else:
                raise RuntimeError("该组件不能从应用内安装。")
            self._check_cancelled(job)
            clear_dependency_cache()
            refreshed = next(item for item in dependency_statuses(self.user_data_root) if item["id"] == spec.id)
            if not refreshed["available"]:
                raise RuntimeError("安装程序已结束，但仍未检测到组件。请查看安装日志后重试。")
            job.status, job.progress, job.progress_kind, job.message = "completed", 100, "determinate", "安装完成，可以继续使用。"
        except DependencyInstallCancelled as exc:
            job.status, job.error, job.message = "cancelled", None, str(exc)
            job.logs.append(str(exc))
        except Exception as exc:
            if job._cancel_event.is_set():
                job.status, job.error, job.message = "cancelled", None, "安装已由用户取消。"
                job.logs.append(job.message)
            else:
                job.status, job.error, job.message = "failed", str(exc), "安装失败；修正网络或权限问题后可以重试。"
                job.logs.append(str(exc))
        finally:
            job.finished_at = time.time()
            with self._lock:
                if self._active_id == job.id:
                    self._active_id = None

    def _install_winget(self, job: InstallJob, dependency_id: str) -> None:
        self._check_cancelled(job)
        package_ids = {"ffmpeg": "Gyan.FFmpeg", "node": "OpenJS.NodeJS.LTS", "tesseract": "UB-Mannheim.TesseractOCR"}
        winget = resolve_command("winget", self.tools_root)
        if winget is None:
            raise RuntimeError("未检测到 WinGet，无法自动安装此组件。")
        command = [
            str(winget), "install", "--id", package_ids[dependency_id], "--exact", "--source", "winget",
            "--accept-source-agreements", "--accept-package-agreements", "--disable-interactivity", "--silent",
        ]
        if dependency_id == "ffmpeg":
            target = self.tools_root / "ffmpeg"
            target.mkdir(parents=True, exist_ok=True)
            command.extend(["--scope", "user", "--location", str(target)])
        job.message, job.progress = f"正在通过 WinGet 安装 {dependency_id}…", 15
        self._stream_command(job, command)
        job.progress, job.message = 90, "安装程序已完成，正在重新检测组件…"

    def _install_whisper(self, job: InstallJob, model_size: str) -> None:
        self._check_cancelled(job)
        job.progress, job.message = 10, f"正在下载 Whisper {model_size} 模型；首次下载可能需要几分钟…"
        from faster_whisper.utils import download_model
        target = _model_root(self.user_data_root, model_size)
        target.mkdir(parents=True, exist_ok=True)
        download_model(model_size, output_dir=str(target))
        self._check_cancelled(job)
        job.progress, job.message = 92, "模型下载完成，正在校验文件…"

    def _install_playwright(self, job: InstallJob) -> None:
        self._check_cancelled(job)
        from playwright._impl._driver import compute_driver_executable, get_driver_env
        node, cli = compute_driver_executable()
        root = self.user_data_root / "browsers" / "ms-playwright"
        root.mkdir(parents=True, exist_ok=True)
        env = get_driver_env()
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(root)
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(root)
        job.progress, job.message = 12, "正在下载投稿辅助浏览器…"
        self._stream_command(job, [node, cli, "install", "chromium"], env=env)
        job.progress, job.message = 92, "浏览器下载完成，正在校验…"

    def _stream_command(self, job: InstallJob, command: list[str], env: dict[str, str] | None = None) -> None:
        self._check_cancelled(job)
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=env,
            creationflags=creationflags,
        )
        with self._lock:
            job._process = process
            cancel_now = job._cancel_event.is_set()
        if cancel_now:
            _terminate_process_tree(process)
        try:
            for line in iter(process.stdout.readline, "") if process.stdout else ():
                text = line.strip()
                if text:
                    job.logs.append(text)
                    job.logs[:] = job.logs[-100:]
            return_code = process.wait()
        finally:
            if process.stdout:
                process.stdout.close()
            with self._lock:
                job._process = None
        if job._cancel_event.is_set():
            raise DependencyInstallCancelled("安装已由用户取消。")
        if return_code != 0:
            tail = "\n".join(job.logs[-12:])
            raise RuntimeError(f"安装程序退出码 {return_code}。\n{tail}".strip())


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
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
