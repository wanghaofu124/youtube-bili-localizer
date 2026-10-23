"""First-run dependency discovery and installation.

The workbench API deliberately delegates host changes to this module.  It is
the only place allowed to start package-manager/model-browser installation
processes, which keeps the HTTP handlers and the media pipeline declarative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
import uuid
import hashlib
import tempfile
import zipfile
from typing import Any, Callable

from .diagnostics import log_frontend_error


WEBVIEW2_DOWNLOAD_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"
WINGET_HELP_URL = "https://learn.microsoft.com/windows/package-manager/winget/"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download/windows"
OLLAMA_HOST = "127.0.0.1:11434"
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}"
FFMPEG_PORTABLE_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-08-13-05/ffmpeg-N-127252-ga25ba44c0c-win64-gpl.zip"
FFMPEG_PORTABLE_SHA256 = "2fac0659fa45a3782e760a92a664d0c3244733146adea463ea153a29335601ee"
FFMPEG_PORTABLE_SIZE = 200236273
FFMPEG_PORTABLE_UNPACKED_SIZE = 520515244
NODE_PORTABLE_URL = "https://nodejs.org/dist/v24.21.0/node-v24.21.0-win-x64.zip"
NODE_PORTABLE_SHA256 = "158f7685b44de51f6c0df1d153526cbcd3e1bc739a8dfc607721cef75de9e541"
NODE_PORTABLE_SIZE = 37618919
NODE_PORTABLE_UNPACKED_SIZE = 106986507
TESSERACT_INSTALLER_URL = "https://github.com/tesseract-ocr/tesseract/releases/download/5.5.3/tesseract-ocr-w64-setup-5.5.3.20260724.exe"
TESSERACT_INSTALLER_SHA256 = "bee9e3434bd94fd65387d9be28cd467a41f61b1275383b55b0f59a1331270ae4"
TESSERACT_INSTALLER_SIZE = 26573224
TESSDATA_FAST_BASE_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd"
TESSDATA_FAST_FILES: dict[str, tuple[int, str]] = {
    "eng": (4113088, "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"),
    "chi_sim": (2469156, "a5fcb6f0db1e1d6d8522f39db4e848f05984669172e584e8d76b6b3141e1f730"),
    "chi_tra": (2366642, "529c5b5797d64b126065cd55f2bb4c7fd7b15790798091b1ff259941a829330b"),
    "deu": (1525436, "19d219bbb6672c869d20a9636c6816a81eb9a71796cb93ebe0cb1530e2cdb22d"),
    "fra": (1130365, "ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca"),
    "spa": (2294433, "6f2e04d02774a18f01bed44b1111f2cd7f3ba7ac9dc4373cd3f898a40ea6b464"),
    "rus": (3861738, "e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225"),
    "jpn": (2471260, "1f5de9236d2e85f5fdf4b3c500f2d4926f8d9449f28f5394472d9e8d83b91b4d"),
}
CUDA_GUIDE_URL = "https://developer.nvidia.com/cuda-12-8-1-download-archive"
CUDNN_GUIDE_URL = "https://developer.nvidia.com/cudnn-downloads"
OLLAMA_MODEL_TAGS: dict[str, str] = {
    "ollama-qwen35-4b": "qwen3.5:4b",
    "ollama-qwen35-9b": "qwen3.5:9b",
    "ollama-translategemma-12b": "translategemma:12b",
}
OLLAMA_SPECIALIZED_TRANSLATION_MODELS = frozenset({"translategemma:12b"})
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
    model_tag: str | None = None
    download_url: str | None = None
    download_sha256: str | None = None
    download_size: int | None = None


SPECS: tuple[DependencySpec, ...] = (
    DependencySpec("ffmpeg", "FFmpeg", "提取音频与硬字幕渲染", True, "winget", "约 180 MB", "https://ffmpeg.org/download.html", download_url=FFMPEG_PORTABLE_URL, download_sha256=FFMPEG_PORTABLE_SHA256, download_size=FFMPEG_PORTABLE_SIZE),
    DependencySpec("whisper-tiny", "Whisper tiny 模型", "轻量音频转写模型", False, "whisper", "约 75 MB"),
    DependencySpec("whisper-base", "Whisper base 模型", "基础音频转写模型", False, "whisper", "约 150 MB"),
    DependencySpec("whisper-small", "Whisper small 模型", "默认音频转写模型，可提前下载", False, "whisper", "约 500 MB"),
    DependencySpec("whisper-medium", "Whisper medium 模型", "更高精度音频转写模型", False, "whisper", "约 1.5 GB"),
    DependencySpec("whisper-large-v3", "Whisper large-v3 模型", "最高质量音频转写模型", False, "whisper", "约 3 GB"),
    DependencySpec("node", "Node.js", "提高 YouTube 新版视频链接解析兼容性", False, "winget", "约 80 MB", "https://nodejs.org/en/download", download_url=NODE_PORTABLE_URL, download_sha256=NODE_PORTABLE_SHA256, download_size=NODE_PORTABLE_SIZE),
    DependencySpec("youtube-po-token", "YouTube 浏览器验证组件", "自动获取 YouTube 媒体请求所需的 PO Token", False, None, "已包含在安装包中"),
    DependencySpec("tesseract", "Tesseract OCR", "OCR 与音频+OCR 字幕模式", False, "winget", "约 100 MB", "https://github.com/UB-Mannheim/tesseract/wiki", download_url=TESSERACT_INSTALLER_URL, download_sha256=TESSERACT_INSTALLER_SHA256, download_size=TESSERACT_INSTALLER_SIZE),
    DependencySpec("playwright-chromium", "投稿辅助浏览器", "B 站登录检查、上传与填表", False, "playwright", "约 300 MB"),
    DependencySpec("webview2", "Microsoft WebView2", "显示桌面应用窗口", True, None, "通常随 Windows/Edge 安装", WEBVIEW2_DOWNLOAD_URL),
    DependencySpec("ollama-runtime", "Ollama 本地运行环境", "在本机运行免费翻译模型；不会随应用打包", False, None, "单独安装", OLLAMA_DOWNLOAD_URL),
    DependencySpec("ollama-qwen35-4b", "Qwen 3.5 4B 本地翻译模型", "本地翻译，适合内存较有限的电脑；首次下载约 3.4 GB", False, "ollama-model", "约 3.4 GB", model_tag="qwen3.5:4b"),
    DependencySpec("ollama-qwen35-9b", "Qwen 3.5 9B 本地翻译模型", "本地翻译；首次下载约 6.6 GB", False, "ollama-model", "约 6.6 GB", model_tag="qwen3.5:9b"),
    DependencySpec("ollama-translategemma-12b", "TranslateGemma 12B 本地翻译模型", "专用多语言翻译模型；模型文件约 8.1 GB，字幕术语仍需人工复核", False, "ollama-model", "约 8.1 GB", model_tag="translategemma:12b"),
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
    "ollama": (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe",
    ),
}


def default_user_data_root() -> Path:
    return Path(os.environ.get("APPDATA") or Path.home()) / "YouTubeBiliLocalizer"


def clear_dependency_cache() -> None:
    with _scan_cache_lock:
        _scan_cache.clear()


def _download_verified(url: str, destination: Path, *, expected_sha256: str, expected_size: int, max_size: int, cancel: Callable[[], None] | None = None) -> None:
    """Download one pinned official artifact into a caller-owned staging path."""
    if expected_size <= 0 or expected_size > max_size:
        raise RuntimeError("下载制品大小限制无效。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "YouTubeBiliLocalizer dependency installer"})
    digest = hashlib.sha256()
    total = 0
    with build_opener(ProxyHandler({})).open(request, timeout=30) as response, destination.open("wb") as output:
        while True:
            if cancel:
                cancel()
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_size:
                raise RuntimeError("下载制品超过安全大小限制。")
            digest.update(chunk)
            output.write(chunk)
    if total != expected_size or digest.hexdigest().lower() != expected_sha256.lower():
        destination.unlink(missing_ok=True)
        raise RuntimeError("下载制品大小或 SHA-256 校验失败。")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _portable_target_valid(target: Path, expected: set[str]) -> bool:
    try:
        for name in expected:
            matches = list(target.rglob(name))
            if not matches:
                return False
            for path in matches:
                with path.open("rb") as stream:
                    header = stream.read(2)
                if path.stat().st_size < 1024 or header != b"MZ":
                    return False
        return True
    except OSError:
        return False


def _safe_extract_zip(archive: Path, target: Path, *, cancel: Callable[[], None] | None = None, max_unpacked_size: int = 500_000_000, expected_executables: set[str] | None = None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    total_size = 0
    required = {item.lower() for item in (expected_executables or set())}
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            if cancel:
                cancel()
            name = member.filename.replace("\\", "/")
            path = Path(name)
            if not name or path.is_absolute() or path.drive or ".." in path.parts or member.is_dir():
                if member.is_dir():
                    continue
                raise RuntimeError("下载压缩包包含不安全路径。")
            total_size += int(member.file_size)
            if total_size > max_unpacked_size:
                raise RuntimeError("下载压缩包解压后超过安全大小限制。")
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise RuntimeError("下载压缩包包含不安全链接。")
            output = (target / path).resolve()
            if target.resolve() not in output.parents:
                raise RuntimeError("下载压缩包越过目标目录。")
            output.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, output.open("wb") as destination:
                while True:
                    if cancel:
                        cancel()
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
            if output.name.lower() in required:
                required.discard(output.name.lower())
    if required:
        raise RuntimeError("便携包中没有找到预期的可执行文件：" + "、".join(sorted(required)))


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
        managed_root = root / name
        if managed_root.is_dir():
            try:
                candidates.extend(sorted(managed_root.glob(f"**/{executable}"), key=lambda path: path.stat().st_mtime, reverse=True)[:4])
            except OSError:
                pass
        if name.lower() == "ffprobe":
            ffmpeg_root = root / "ffmpeg"
            if ffmpeg_root.is_dir():
                try:
                    candidates.extend(sorted(ffmpeg_root.glob("**/ffprobe.exe"), key=lambda path: path.stat().st_mtime, reverse=True)[:4])
                except OSError:
                    pass
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


def _ollama_json(path: str, *, timeout: float = 2.0) -> dict[str, Any]:
    """Read only Ollama's fixed loopback API; bypass system proxies explicitly."""
    request = Request(f"{OLLAMA_BASE_URL}{path}", headers={"Accept": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Ollama returned an invalid response.")
    return value


def ollama_local_status() -> dict[str, Any]:
    """Return runtime and allowlisted-model status without loading a model."""
    executable = resolve_command("ollama")
    try:
        _ollama_json("/api/version")
        try:
            raw_tags = _ollama_json("/api/tags").get("models", [])
            installed = {
                str(item.get("name") or item.get("model") or "").strip()
                for item in raw_tags if isinstance(item, dict)
            } if isinstance(raw_tags, list) else set()
        except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
            return {
                "runtime_state": "unavailable", "runtime_available": False,
                "runtime_message": "Ollama 已响应，但无法读取已安装模型。请检查 Ollama 后重试。",
                "cli_available": bool(executable),
                "models": {dependency_id: False for dependency_id in OLLAMA_MODEL_TAGS},
            }
        return {
            "runtime_state": "ready", "runtime_available": True,
            "runtime_message": "Ollama 正在运行。",
            "cli_available": bool(executable),
            "models": {dependency_id: tag in installed for dependency_id, tag in OLLAMA_MODEL_TAGS.items()},
        }
    except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
        if executable:
            state, message = "stopped", "Ollama 已安装但当前未运行。点击“启动 Ollama”后，应用会自动重新检测；无需重启本应用。"
        else:
            state, message = "missing", "尚未安装 Ollama。请先从官方页面下载安装并启动；运行环境不会打进本应用。"
        return {
            "runtime_state": state, "runtime_available": False,
            "runtime_message": message,
            "cli_available": bool(executable),
            "models": {dependency_id: False for dependency_id in OLLAMA_MODEL_TAGS},
        }


def managed_tessdata_root(user_data_root: Path | None = None) -> Path:
    return (user_data_root or default_user_data_root()) / "tessdata"


def tesseract_installer_path(user_data_root: Path | None = None) -> Path:
    return (user_data_root or default_user_data_root()) / "tools" / "tesseract" / "5.5.3" / "tesseract-5.5.3-setup.exe"


_OCR_TOKEN_RE = re.compile(r"[a-z0-9_]{2,32}")


def parse_ocr_languages(value: str | None, *, default: str = "eng") -> tuple[str, ...]:
    """Parse the shared OCR language contract without dropping user input."""
    raw = default if value is None else str(value).strip().lower()
    if not raw:
        raise ValueError("OCR 语言不能为空；请使用 eng、chi_sim 或 eng+chi_sim。")
    pieces = raw.split("+")
    if len(pieces) > 4:
        raise ValueError("OCR 最多支持 4 个语言代码，请减少选择。")
    if any(not piece or _OCR_TOKEN_RE.fullmatch(piece) is None for piece in pieces):
        raise ValueError("OCR 语言代码无效；请使用 eng、chi_sim 或 eng+chi_sim。")
    return tuple(dict.fromkeys(pieces))


def _ocr_tokens(value: str | None) -> list[str]:
    return list(parse_ocr_languages(value))


def _system_ocr_languages(command: Path | None) -> set[str]:
    if not command:
        return set()
    try:
        completed = subprocess.run([str(command), "--list-langs"], capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.strip().lower() for line in (completed.stdout + "\n" + completed.stderr).splitlines() if re.fullmatch(r"[A-Za-z0-9_+-]+", line.strip()) and not line.lower().startswith("list of available")}


def nvidia_cuda_dll_directories() -> list[Path]:
    roots: list[Path] = []
    explicit = os.environ.get("YBLOCALIZER_CUDA_DLL_DIR", "").strip()
    if explicit:
        roots.append(Path(explicit))
    try:
        import nvidia
        roots.extend(Path(item) for item in getattr(nvidia, "__path__", ()))
        if getattr(nvidia, "__file__", None):
            roots.append(Path(nvidia.__file__).parent)
    except (ImportError, OSError):
        pass
    if os.name == "nt":
        try:
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for access in (winreg.KEY_READ, winreg.KEY_READ | winreg.KEY_WOW64_32KEY, winreg.KEY_READ | winreg.KEY_WOW64_64KEY):
                    try:
                        with winreg.OpenKey(hive, r"Software\Python\PythonCore", 0, access) as key:
                            index = 0
                            while True:
                                try:
                                    version = winreg.EnumKey(key, index)
                                except OSError:
                                    break
                                index += 1
                                try:
                                    with winreg.OpenKey(key, version + r"\InstallPath", 0, access) as install:
                                        roots.append(Path(str(winreg.QueryValue(install, None))) / "Lib" / "site-packages" / "nvidia")
                                except OSError:
                                    continue
                    except OSError:
                        continue
        except ImportError:
            pass
        toolkit = Path(os.environ.get("ProgramFiles", "")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if toolkit.is_dir():
            roots.extend(path for path in toolkit.glob("v*/bin") if path.is_dir())
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for root in roots:
        if root.name.lower() == "bin":
            candidates.append(root)
        else:
            candidates.extend((root / "cuda_runtime" / "bin", root / "cublas" / "bin", root / "cudnn" / "bin"))
    return list(dict.fromkeys(path for path in candidates if path.is_dir()))


def prepend_nvidia_cuda_dll_paths(paths: list[Path] | None = None) -> list[Path]:
    candidates = paths if paths is not None else nvidia_cuda_dll_directories()
    existing = os.environ.get("PATH", "")
    current_lower = {item.lower() for item in existing.split(os.pathsep) if item}
    missing = [str(item) for item in candidates if str(item).lower() not in current_lower]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + [existing])
    return candidates


def cuda_runtime_status() -> dict[str, object]:
    if resolve_command("nvidia-smi") is None:
        return {"available": False, "missing": ["NVIDIA driver"], "missing_groups": {"cublas": ["NVIDIA driver"], "cudnn": ["NVIDIA driver"]}, "message": "未检测到 NVIDIA 显卡驱动（nvidia-smi）。"}
    paths = prepend_nvidia_cuda_dll_paths()
    groups = {
        "cublas": ("cublas64_12.dll", "cublasLt64_12.dll") if os.name == "nt" else ("libcublas.so.12",),
        "cudnn": ("cudnn64_9.dll", "cudnn_ops64_9.dll") if os.name == "nt" else ("libcudnn.so.9",),
    }
    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    missing_groups: dict[str, list[str]] = {}
    for group, names in groups.items():
        missing: list[str] = []
        for name in names:
            resolved = shutil.which(name) or next((str(path / name) for path in paths if (path / name).is_file()), None)
            try:
                loader(resolved or name)
            except (OSError, TypeError):
                missing.append(name)
        if missing:
            missing_groups[group] = missing
    missing = [name for values in missing_groups.values() for name in values]
    return {"available": not missing, "missing": missing, "missing_groups": missing_groups, "message": "NVIDIA 驱动、cuBLAS 和 cuDNN 均可加载。" if not missing else "显卡驱动可用，但缺少或无法加载 CUDA 转写运行库：" + "、".join(missing) + "。"}


def dependency_statuses(user_data_root: Path | None = None, ocr_language: str = "eng") -> list[dict[str, Any]]:
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
    ollama = ollama_local_status()
    available = {
        **command_available,
        **model_available,
        "playwright-chromium": _playwright_available(),
        "webview2": webview2_available(),
        "ollama-runtime": bool(ollama["runtime_available"]),
        **ollama["models"],
    }
    rows = []
    for spec in SPECS:
        can_install = bool(spec.install_kind)
        if spec.install_kind == "winget":
            can_install = sys.platform.startswith("win") and (winget_available or bool(spec.download_url))
        if spec.install_kind == "whisper":
            can_install = _module_available("faster_whisper")
        if spec.install_kind == "playwright":
            can_install = _module_available("playwright")
        if spec.install_kind == "ollama-model":
            can_install = bool(ollama["runtime_available"] and ollama.get("cli_available"))
        present = bool(available[spec.id])
        state = "ready" if present else "missing"
        if spec.id == "ollama-runtime":
            state = str(ollama["runtime_state"])
        elif spec.install_kind == "ollama-model" and not ollama["runtime_available"]:
            state = "blocked"
        elif spec.install_kind == "ollama-model" and not present and not ollama.get("cli_available"):
            state = "blocked"
        if spec.id == "ollama-runtime":
            message = str(ollama["runtime_message"])
        elif spec.install_kind == "ollama-model" and not ollama["runtime_available"]:
            message = str(ollama["runtime_message"])
        elif spec.install_kind == "ollama-model" and not present and not ollama.get("cli_available"):
            message = "Ollama 正在运行，但没有找到 ollama 命令。请重新安装官方运行环境，或使用 Ollama 自行下载模型后重新检测。"
        elif present:
            message = "已安装，可以使用。"
        elif spec.install_kind == "ollama-model":
            message = "尚未下载。点击下载后模型会保存在 Ollama 管理的本机模型目录；下载完成可立即使用。"
        else:
            message = "点击安装，完成后无需重启应用。" if can_install else (
                "当前环境缺少 WinGet；点击说明安装 App Installer 后重试。" if spec.install_kind == "winget" else
                "当前环境无法自动安装，请使用说明链接。" if spec.action_url else
                "当前环境无法自动安装。"
            )
        action_url = spec.action_url
        action_label = None
        if spec.install_kind == "winget" and not winget_available and not present:
            action_label = "下载便携版" if spec.id in {"ffmpeg", "node"} else "下载安装器"
        rows.append({
            "id": spec.id,
            "label": spec.label,
            "purpose": spec.purpose,
            "required": spec.required,
            "available": present,
            "installable": can_install and not present,
            "size_hint": spec.size_hint,
            "action_url": action_url if spec.id != "ollama-runtime" or state == "missing" else None,
            "message": message,
            "state": state,
            "action_label": "下载模型" if spec.install_kind == "ollama-model" else ("下载 Ollama" if spec.id == "ollama-runtime" and state == "missing" else action_label),
            "download_url": spec.download_url,
            "depends_on": "ollama-runtime" if spec.install_kind == "ollama-model" else None,
        })
    tesseract_command = resolve_command("tesseract", root / "tools")
    try:
        selected_ocr = list(parse_ocr_languages(ocr_language))
        ocr_error = None
    except ValueError as exc:
        selected_ocr = []
        ocr_error = str(exc)
    system_ocr = _system_ocr_languages(tesseract_command)
    managed_root = managed_tessdata_root(root)
    managed_ocr = {token for token in selected_ocr if (managed_root / f"{token}.traineddata").is_file()}
    supported = bool(selected_ocr) and all(token in TESSDATA_FAST_FILES for token in selected_ocr)
    ocr_ready = bool(selected_ocr) and (set(selected_ocr) <= system_ocr or set(selected_ocr) <= managed_ocr)
    rows.append({
        "id": "ocr-language", "label": "OCR 语言包", "purpose": "为当前 OCR 语言下载官方 Tesseract 识别数据",
        "required": False, "available": ocr_ready, "installable": bool(tesseract_command and supported and not ocr_ready),
        "size_hint": "按所选语言下载", "action_url": "https://github.com/tesseract-ocr/tessdata_fast",
        "message": ocr_error or ("当前所选语言包可用。" if ocr_ready else ("点击下载所选语言包。" if supported else f"暂不支持自动下载：{', '.join(token for token in selected_ocr if token not in TESSDATA_FAST_FILES)}。")),
        "state": "blocked" if ocr_error else ("ready" if ocr_ready else "missing"), "action_label": "修改 OCR 语言" if ocr_error else ("下载语言包" if supported else "查看语言包"),
        "download_url": None, "depends_on": "tesseract",
    })
    installer_available = tesseract_installer_path(root).is_file()
    for row in rows:
        if row["id"] == "tesseract":
            row["installer_available"] = installer_available
            if installer_available and not row["available"]:
                row["message"] = "安装程序已下载，点击启动安装器；完成后重新检测。"
                row["action_label"] = "启动安装器"
                row["installable"] = False
    cuda_status = cuda_runtime_status()
    driver_ready = "NVIDIA driver" not in cuda_status.get("missing", [])
    missing_groups = cuda_status.get("missing_groups", {})
    for gpu_id, label, names, available in (
        ("nvidia-driver", "NVIDIA 驱动", (), driver_ready),
        ("cuda-cublas", "CUDA cuBLAS 12", ("cublas64_12.dll", "cublasLt64_12.dll"), not missing_groups.get("cublas")),
        ("cuda-cudnn", "CUDA cuDNN 9", ("cudnn64_9.dll", "cudnn_ops64_9.dll"), not missing_groups.get("cudnn")),
    ):
        rows.append({
            "id": gpu_id, "label": label, "purpose": "faster-whisper 1.2.1 / CTranslate2 4.8.1 的 GPU 运行库",
            "required": False, "available": available, "installable": False,
            "size_hint": "官方引导", "action_url": (CUDA_GUIDE_URL if gpu_id == "cuda-cublas" else CUDNN_GUIDE_URL if gpu_id == "cuda-cudnn" else "https://www.nvidia.com/drivers"),
            "message": "已检测到。" if available else ("未检测到；请按官方版本说明安装。" if driver_ready else "未检测到 NVIDIA 驱动。"),
            "state": "ready" if available else "missing", "action_label": "打开官方安装指引",
            "download_url": None, "depends_on": None,
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
    ocr_language: str = "eng"
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
            "ocr_language": self.ocr_language,
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
        self._ollama_process: subprocess.Popen[bytes] | None = None
        self._ollama_mode: str | None = None
        self._ollama_started_at = 0.0
        self._ollama_log_path = self.user_data_root / "logs" / "ollama-start.log"

    def snapshot(self, ocr_language: str = "eng") -> dict[str, Any]:
        with self._lock:
            active = self._jobs.get(self._active_id or "")
            rows = dependency_statuses(self.user_data_root, ocr_language)
            process = self._ollama_process
            if self._ollama_mode == "serve" and process is not None and process.poll() is not None:
                runtime = next((row for row in rows if row["id"] == "ollama-runtime"), None)
                if runtime is not None and not runtime["available"]:
                    runtime["message"] = self._ollama_exit_message(process)
            return {
                "ready": all(row["available"] for row in rows if row["required"]),
                "dependencies": rows,
                "active_job": active.snapshot() if active else None,
            }

    def get_job(self, job_id: str) -> InstallJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _ollama_exit_message(self, process: subprocess.Popen[bytes]) -> str:
        try:
            with self._ollama_log_path.open("rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 1500))
                detail = log.read().decode("utf-8", errors="replace").strip()
        except OSError:
            detail = ""
        suffix = f"：{detail}" if detail else "。请检查 Ollama 安装或端口占用。"
        return f"Ollama 启动失败（退出码 {process.returncode}）{suffix}"

    def start_ollama(self) -> dict[str, str]:
        """Launch the installed Windows app, or an owned CLI server if needed."""
        with self._lock:
            status = ollama_local_status()
            if status["runtime_available"]:
                return {"state": "ready", "message": "Ollama 已在运行，可以继续。"}
            if status["runtime_state"] != "stopped":
                raise RuntimeError(str(status["runtime_message"]))
            if self._ollama_process is not None and self._ollama_process.poll() is None:
                return {"state": "starting", "message": "Ollama 正在启动，应用会自动重新检测。"}
            if self._ollama_mode == "app" and time.monotonic() - self._ollama_started_at < 15:
                return {"state": "starting", "message": "Ollama 正在启动，应用会自动重新检测。"}

            executable = resolve_command("ollama", self.tools_root)
            if executable is None:
                raise RuntimeError("没有找到 Ollama 程序。请先安装 Ollama，再重新检查。")

            desktop_app = executable.with_name("ollama app.exe")
            use_desktop_app = os.name == "nt" and desktop_app.is_file()
            command = [str(desktop_app)] if use_desktop_app else [str(executable), "serve"]
            launch_env = os.environ.copy()
            launch_env["OLLAMA_HOST"] = OLLAMA_HOST

            launch_options: dict[str, Any] = {}
            if os.name == "nt":
                launch_options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            else:
                launch_options["start_new_session"] = True
            try:
                if use_desktop_app:
                    process = subprocess.Popen(
                        command, cwd=str(executable.parent), env=launch_env,
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, close_fds=True, **launch_options,
                    )
                else:
                    self._ollama_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self._ollama_log_path.open("wb") as log:
                        process = subprocess.Popen(
                            command, cwd=str(executable.parent), env=launch_env,
                            stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, close_fds=True, **launch_options,
                        )
            except OSError as exc:
                raise RuntimeError(f"启动 Ollama 失败：{exc}") from exc
            self._ollama_process = process
            self._ollama_mode = "app" if use_desktop_app else "serve"
            self._ollama_started_at = time.monotonic()
            if not use_desktop_app:
                time.sleep(0.3)
                if process.poll() is not None:
                    raise RuntimeError(self._ollama_exit_message(process))
            return {"state": "starting", "message": "已启动 Ollama，正在等待本地服务就绪…"}

    def start_tesseract_installer(self) -> dict[str, str]:
        installer = tesseract_installer_path(self.user_data_root)
        if not installer.is_file():
            raise RuntimeError("Tesseract 安装器尚未下载，请先点击下载安装器。")
        spec = next(item for item in SPECS if item.id == "tesseract")
        try:
            if installer.stat().st_size != spec.download_size or _sha256_file(installer) != spec.download_sha256:
                raise ValueError
        except (OSError, ValueError):
            raise RuntimeError("Tesseract 安装器校验失败，请删除后重新下载。")
        try:
            options: dict[str, Any] = {}
            if os.name == "nt":
                options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen([str(installer)], cwd=str(installer.parent), close_fds=True, **options)
        except OSError as exc:
            raise RuntimeError(f"无法启动 Tesseract 安装器：{exc}") from exc
        return {"state": "starting", "message": "已启动 Tesseract 安装器；完成后点击重新检测。"}

    def start(self, dependency_id: str, confirmed: bool, ocr_language: str = "eng") -> InstallJob:
        if not confirmed:
            raise ValueError("安装依赖前需要明确确认。")
        parse_ocr_languages(ocr_language)
        spec = next((item for item in SPECS if item.id == dependency_id), None)
        if spec is None and dependency_id == "ocr-language":
            spec = DependencySpec("ocr-language", "OCR 语言包", "为当前 OCR 语言下载官方 Tesseract 识别数据", False, "ocr-language", "按所选语言下载", "https://github.com/tesseract-ocr/tessdata_fast")
        if spec is None:
            raise ValueError("未知依赖组件。")
        status = next(item for item in dependency_statuses(self.user_data_root, ocr_language) if item["id"] == dependency_id)
        if status["available"]:
            job = InstallJob(
                uuid.uuid4().hex, dependency_id, ocr_language, status="completed", progress=100,
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
            job = InstallJob(uuid.uuid4().hex, dependency_id, ocr_language=ocr_language)
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
            process = self._ollama_process if self._ollama_mode == "serve" else None
            self._ollama_process = None
        if active:
            self.cancel(active.id)
            thread = active._thread
            if thread and thread.is_alive():
                thread.join(timeout=max(0.0, timeout))
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)

    @staticmethod
    def _check_cancelled(job: InstallJob) -> None:
        if job._cancel_event.is_set():
            raise DependencyInstallCancelled("安装已由用户取消。")

    def _run(self, job: InstallJob, spec: DependencySpec) -> None:
        job.status, job.progress = "running", 5
        try:
            self._check_cancelled(job)
            if spec.install_kind == "winget":
                if resolve_command("winget", self.tools_root) is not None:
                    self._install_winget(job, spec.id)
                elif spec.id == "tesseract":
                    self._download_tesseract_installer(job, spec)
                else:
                    self._install_portable_zip(job, spec)
            elif spec.install_kind == "whisper":
                self._install_whisper(job, spec.id.removeprefix("whisper-"))
            elif spec.install_kind == "playwright":
                self._install_playwright(job)
            elif spec.install_kind == "ollama-model":
                if not spec.model_tag or OLLAMA_MODEL_TAGS.get(spec.id) != spec.model_tag:
                    raise RuntimeError("未知的本地翻译模型。")
                self._install_ollama_model(job, spec.model_tag)
            elif spec.id == "ocr-language":
                self._install_ocr_language(job)
            else:
                raise RuntimeError("该组件不能从应用内安装。")
            self._check_cancelled(job)
            clear_dependency_cache()
            refreshed = next(item for item in dependency_statuses(self.user_data_root, job.ocr_language) if item["id"] == spec.id)
            if spec.id != "tesseract" and not refreshed["available"]:
                raise RuntimeError("安装程序已结束，但仍未检测到组件。请查看安装日志后重试。")
            job.status, job.progress, job.progress_kind, job.message = "completed", 100, "determinate", (
                "安装器已下载，请运行安装器并在完成后点击重新检测。" if spec.id == "tesseract" else "安装完成，可以继续使用。"
            )
        except DependencyInstallCancelled as exc:
            job.status, job.error, job.message = "cancelled", None, str(exc)
            job.logs.append(str(exc))
        except Exception as exc:
            if job._cancel_event.is_set():
                job.status, job.error, job.message = "cancelled", None, "安装已由用户取消。"
                job.logs.append(job.message)
            else:
                log_frontend_error("依赖安装失败，已显示在界面")
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

    def _install_portable_zip(self, job: InstallJob, spec: DependencySpec) -> None:
        if not spec.download_url or not spec.download_sha256 or not spec.download_size:
            raise RuntimeError("该组件没有可校验的官方便携制品。")
        staging = self.tools_root / ".staging" / f"{spec.id}-{uuid.uuid4().hex}"
        archive = staging.with_suffix(".part")
        unpacked = staging.with_name(staging.name + "-unpacked")
        try:
            job.progress, job.message = 8, f"正在下载官方 {spec.label} 便携版…"
            _download_verified(spec.download_url, archive, expected_sha256=spec.download_sha256, expected_size=spec.download_size, max_size=260_000_000, cancel=lambda: self._check_cancelled(job))
            target = self.tools_root / spec.id / (spec.download_sha256[:12])
            expected = {"ffmpeg.exe", "ffprobe.exe"} if spec.id == "ffmpeg" else {"node.exe"}
            if _portable_target_valid(target, expected):
                job.progress, job.message = 92, "便携版已安装，正在重新检测组件…"
                return
            unpacked_limit = (FFMPEG_PORTABLE_UNPACKED_SIZE + 16_000_000) if spec.id == "ffmpeg" else (NODE_PORTABLE_UNPACKED_SIZE + 8_000_000)
            _safe_extract_zip(archive, unpacked, cancel=lambda: self._check_cancelled(job), max_unpacked_size=unpacked_limit, expected_executables=expected)
            target.parent.mkdir(parents=True, exist_ok=True)
            target_resolved = target.resolve()
            expected_parent = (self.tools_root / spec.id).resolve()
            if expected_parent not in target_resolved.parents or target_resolved.parent != expected_parent:
                raise RuntimeError("便携版目标路径不安全。")
            if target.exists():
                shutil.rmtree(target)
            os.replace(unpacked, target)
            job.progress, job.message = 92, "便携版已安装，正在重新检测组件…"
        finally:
            archive.unlink(missing_ok=True)
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if unpacked.exists():
                shutil.rmtree(unpacked, ignore_errors=True)

    def _download_tesseract_installer(self, job: InstallJob, spec: DependencySpec) -> None:
        if not spec.download_url or not spec.download_sha256 or not spec.download_size:
            raise RuntimeError("Tesseract 官方安装器信息不完整。")
        installer = self.tools_root / "tesseract" / "5.5.3" / "tesseract-5.5.3-setup.exe"
        job.progress, job.message = 8, "正在下载 UB Mannheim Tesseract 官方安装器…"
        _download_verified(spec.download_url, installer.with_suffix(".part"), expected_sha256=spec.download_sha256, expected_size=spec.download_size, max_size=80_000_000, cancel=lambda: self._check_cancelled(job))
        os.replace(installer.with_suffix(".part"), installer)
        job.progress, job.message = 95, f"安装器已下载：{installer}。请运行后重新检测。"

    def _install_ocr_language(self, job: InstallJob) -> None:
        tokens = _ocr_tokens(job.ocr_language)
        unsupported = [token for token in tokens if token not in TESSDATA_FAST_FILES]
        if unsupported:
            raise RuntimeError(f"官方语言包暂不支持：{', '.join(unsupported)}。")
        target = managed_tessdata_root(self.user_data_root)
        staging = target / f".staging-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            for index, token in enumerate(tokens):
                size, digest = TESSDATA_FAST_FILES[token]
                part = staging / f"{token}.traineddata.part"
                job.progress = 8 + int(index * 80 / max(1, len(tokens)))
                job.message = f"正在下载 OCR 语言包 {token}…"
                _download_verified(f"{TESSDATA_FAST_BASE_URL}/{token}.traineddata", part, expected_sha256=digest, expected_size=size, max_size=20_000_000, cancel=lambda: self._check_cancelled(job))
                os.replace(part, staging / f"{token}.traineddata")
            target.mkdir(parents=True, exist_ok=True)
            for token in tokens:
                os.replace(staging / f"{token}.traineddata", target / f"{token}.traineddata")
        finally:
            shutil.rmtree(staging, ignore_errors=True)

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

    def _install_ollama_model(self, job: InstallJob, model_tag: str) -> None:
        if model_tag not in OLLAMA_MODEL_TAGS.values():
            raise RuntimeError("不允许下载未登记的 Ollama 模型。")
        if not ollama_local_status()["runtime_available"]:
            raise RuntimeError("Ollama 尚未运行。请先启动 Ollama，再重新下载模型。")
        ollama = resolve_command("ollama", self.tools_root)
        if ollama is None:
            raise RuntimeError("未找到 Ollama 命令。请重新安装 Ollama 并启动后重试。")
        job.progress = 5
        job.progress_kind = "indeterminate"
        job.message = f"正在从 Ollama 官方模型库下载 {model_tag}；下载可安全取消，已下载的数据由 Ollama 保留。"
        pull_env = os.environ.copy()
        pull_env["OLLAMA_HOST"] = OLLAMA_HOST
        self._stream_command(job, [str(ollama), "pull", model_tag], env=pull_env)
        job.progress, job.progress_kind = 92, "determinate"
        job.message = "模型下载完成，正在验证可用状态…"

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
                    if job.dependency_id in OLLAMA_MODEL_TAGS:
                        percent_match = re.search(r"\b(\d{1,3}(?:\.\d+)?)%", text)
                        if percent_match:
                            percent = max(0.0, min(100.0, float(percent_match.group(1))))
                            job.progress = min(90, 5 + int(percent * 0.85))
                            job.progress_kind = "determinate"
                            job.message = f"正在下载本地翻译模型… {int(percent)}%"
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
