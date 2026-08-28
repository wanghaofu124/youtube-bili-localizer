from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import os
import shutil
import subprocess
from typing import Callable
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from .cancellation import check_cancelled as legacy_check_cancelled
from .dependencies import resolve_command
from .models import VideoJob
from .performance import download_rate_limit, lower_process_priority, normalize_resource_profile
from .util import require_command, slugify


class YouTubeAccessError(RuntimeError):
    """A stable, user-facing YouTube failure with a machine-readable code."""

    def __init__(self, code: str, message: str, suggested_action: str) -> None:
        super().__init__(message)
        self.code = code
        self.suggested_action = suggested_action


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


def _format_for_quality(quality: str) -> str:
    normalized = str(quality or "1080p").strip().lower()
    heights = {"720p": 720, "1080p": 1080}
    if normalized == "original":
        return "bestvideo+bestaudio/best"
    if normalized not in heights:
        raise ValueError("下载画质必须是 720p、1080p 或 original。")
    height = heights[normalized]
    return (
        f"bestvideo[height<={height}]+bestaudio/"
        f"best[height<={height}]/worst"
    )


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
) -> VideoJob:
    check = cancel_check or legacy_check_cancelled
    profile = normalize_resource_profile(resource_profile)
    _validate_video_url(url)
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. Run: pip install -r requirements.txt") from exc

    work_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(work_dir / "%(id)s.%(ext)s")

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
        "format": _format_for_quality(download_quality),
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
    }
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

    try:
        check()
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
                info = ydl.extract_info(url, download=True)
        check()
    except Exception as exc:
        _raise_with_youtube_hint(exc, cookies_from_browser, po_token_mode)

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

    thumbnail_url = _best_thumbnail_url(info)
    thumbnail_path = _download_thumbnail(thumbnail_url, work_dir, video_id, proxy) if thumbnail_url else None

    job = VideoJob(
        job_id=work_dir.name,
        source=url,
        source_kind="url",
        work_dir=work_dir,
        title=title or info.get("title"),
        description=info.get("description"),
        raw_video=video_files[0],
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
    raise exc


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
