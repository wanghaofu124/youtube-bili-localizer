from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .cancellation import check_cancelled
from .models import VideoJob
from .util import require_command, slugify


def _js_runtime_opts() -> dict:
    """Best-effort options for YouTube's n-challenge solver.

    Modern yt-dlp needs a JavaScript runtime (node/deno/bun) plus the remote
    ``ejs`` challenge solver script to obtain playable stream URLs.  When no
    runtime is installed the previous behaviour is kept (warning only).
    """
    runtimes = {
        name: {"path": None}
        for name in ("deno", "node", "bun", "quickjs")
        if shutil.which(name)
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


def get_video_metadata(
    url: str,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
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
    if cookies_file:
        _validate_cookies_file(cookies_file)
        opts["cookiefile"] = cookies_file
    elif cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        _raise_with_cookie_hint(exc, cookies_from_browser)
    return VideoMetadata(
        title=info.get("title"),
        duration=info.get("duration"),
        license=info.get("license"),
        view_count=info.get("view_count"),
        webpage_url=info.get("webpage_url") or url,
        thumbnail_url=_best_thumbnail_url(info),
    )


def download_with_ytdlp(
    url: str,
    work_dir: Path,
    title: str | None = None,
    max_seconds: int | None = None,
    require_reuse_allowed: bool = False,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
    progress: Callable[[float], None] | None = None,
) -> VideoJob:
    _validate_video_url(url)
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. Run: pip install -r requirements.txt") from exc

    work_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(work_dir / "%(id)s.%(ext)s")

    def download_progress(status: dict) -> None:
        check_cancelled()
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
        "format": "bestvideo+bestaudio/best",
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
    opts.update(_js_runtime_opts())
    if max_seconds:
        require_command("ffmpeg")
        opts["download_ranges"] = yt_dlp.utils.download_range_func(None, [(0, max_seconds)])
        opts["force_keyframes_at_cuts"] = True
    if cookies_file:
        _validate_cookies_file(cookies_file)
        opts["cookiefile"] = cookies_file
    elif cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    try:
        check_cancelled()
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            license_text = info.get("license") or ""
            if require_reuse_allowed and "reuse allowed" not in license_text.lower():
                raise RuntimeError(
                    "该视频未被 YouTube 标记为“可转载”（reuse allowed，yt-dlp 元数据中许可证为空或未知）。\n"
                    "如果你确认已获得处理或转载授权，请在素材页取消勾选「仅接受 CC reuse allowed」后重新开始任务。"
                )
            info = ydl.extract_info(url, download=True)
        check_cancelled()
    except Exception as exc:
        _raise_with_cookie_hint(exc, cookies_from_browser)

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
    thumbnail_path = _download_thumbnail(thumbnail_url, work_dir, video_id) if thumbnail_url else None

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


def _raise_with_cookie_hint(exc: Exception, cookies_from_browser: str | None) -> None:
    message = str(exc)
    lowered = message.lower()
    if cookies_from_browser and (
        "Could not copy Chrome cookie database" in message
        or "failed to load cookies" in message
        or "Permission denied" in message
    ):
        raise RuntimeError(
            f"无法读取 {cookies_from_browser} 浏览器 Cookies。请关闭对应浏览器后重试，"
            "或在 GUI 的 YouTube Cookies 里选择空值后再运行。"
        ) from exc

    if cookies_from_browser and (
        "dpapi" in lowered
        or "failed to decrypt" in lowered
        or "could not decrypt" in lowered
    ):
        raise RuntimeError(
            f"无法解密 {cookies_from_browser} 浏览器的 Cookies（DPAPI 加密）。"
            "请先在「设置」里选择 Cookies 来源浏览器，然后完全退出该浏览器再重新开始任务。"
            "如果仍然失败，请改选 Edge 或 Firefox，或改用浏览器导出的 cookies.txt。"
        ) from exc

    bot_markers = (
        "not a bot",
        "confirm you're not a bot",
        "sign in to confirm",
        "request was blocked",
        "http error 403",
        "nsig",
        "player requests",
        "this account has been terminated",
        "forbidden",
    )
    signin_markers = (
        "please sign in",
        "sign in to",
        "you must sign in",
        "login required",
        "log in to",
        "accounts.google.com",
    )
    if any(marker in lowered for marker in bot_markers) or any(
        marker in lowered for marker in signin_markers
    ):
        raise RuntimeError(
            "YouTube 拦截了自动请求（机器人验证或登录要求）。如果已经配置了 cookies.txt 仍出现此提示，"
            "说明当前网络出口 IP 或登录会话已被 YouTube 风控标记，不是本地配置问题。\n"
            "建议按顺序尝试：\n"
            "1) 更换代理节点或网络出口（换 IP 最有效）；\n"
            "2) 在 Chrome 中退出并重新登录 YouTube，重新导出 cookies.txt 覆盖旧文件；\n"
            "3) 等待几小时风控解除后再试。"
        ) from exc
    raise exc


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


def _download_thumbnail(url: str, work_dir: Path, video_id: str) -> Path | None:
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
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
