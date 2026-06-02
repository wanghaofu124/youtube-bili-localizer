from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

from .models import VideoJob
from .util import require_command, slugify


@dataclass(slots=True)
class VideoMetadata:
    title: str | None
    duration: float | None
    license: str | None
    view_count: int | None
    webpage_url: str | None
    thumbnail_url: str | None


def get_video_metadata(url: str, cookies_from_browser: str | None = None) -> VideoMetadata:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. Run: pip install -r requirements.txt") from exc

    opts = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    if cookies_from_browser:
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
) -> VideoJob:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. Run: pip install -r requirements.txt") from exc

    work_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(work_dir / "%(id)s.%(ext)s")
    opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "restrictfilenames": False,
        "writeinfojson": True,
        "quiet": False,
        "no_warnings": False,
    }
    if max_seconds:
        require_command("ffmpeg")
        opts["download_ranges"] = yt_dlp.utils.download_range_func(None, [(0, max_seconds)])
        opts["force_keyframes_at_cuts"] = True
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            license_text = info.get("license") or ""
            if require_reuse_allowed and "reuse allowed" not in license_text.lower():
                raise RuntimeError(
                    "This YouTube video is not marked as reuse allowed by yt-dlp metadata. "
                    f"license={license_text or 'unknown'}"
                )
            info = ydl.extract_info(url, download=True)
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
    if cookies_from_browser and (
        "Could not copy Chrome cookie database" in message
        or "failed to load cookies" in message
        or "Permission denied" in message
    ):
        raise RuntimeError(
            f"无法读取 {cookies_from_browser} 浏览器 Cookies。请关闭对应浏览器后重试，"
            "或在 GUI 的 YouTube Cookies 里选择空值后再运行。"
        ) from exc
    raise exc


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
