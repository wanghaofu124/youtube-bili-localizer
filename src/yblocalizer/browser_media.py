"""Read TikTok and Rumble pages with the local Chrome install.

yt-dlp's own clients currently receive a maintenance page, a WAF shell, or a
Cloudflare interstitial from these two sites. A normal Chrome window still
gets the public video metadata, and the media file can then be saved from
that page.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

from .dependencies import resolve_command
from .runtime import CancellationRequested
from .source_sites import site_for_url, video_id_for_url

_CHALLENGE_MARKERS = ("Just a moment", "请稍候", "安全验证", "cf-browser-verification")
_RUMBLE_EMBED = re.compile(r"embed/([0-9a-z]{4,})", re.IGNORECASE)
_RUMBLE_VIDEO_KEY = re.compile(r"""["']video["']\s*:\s*["']([0-9a-z]+)["']""", re.IGNORECASE)


def browser_video_facts(url: str, proxy: str | None = None) -> dict[str, Any]:
    """Title, duration, and a direct media URL. Does not save the file."""
    site = site_for_url(url)
    if site == "tiktok":
        return _with_chrome(url, proxy, headless=True, read=_read_tiktok)
    if site == "rumble":
        return _with_chrome(url, proxy, headless=False, read=lambda page: _read_rumble(page, url))
    raise RuntimeError("这个站点没有浏览器读取路径。")


def download_browser_media(
    url: str,
    destination: Path,
    proxy: str | None = None,
    max_seconds: int | None = None,
    log: Callable[[str], None] | None = None,
    cancel_check: Callable[[], None] = lambda: None,
    download_quality: str = "1080p",
) -> dict[str, Any]:
    cancel_check()
    site = site_for_url(url)
    max_height = {"720p": 720, "1080p": 1080}.get(download_quality)
    headless = site != "rumble"
    if log:
        log("yt-dlp 没有拿到视频，改用本机 Chrome 打开页面。")

    def read(page) -> dict[str, Any]:
        facts = _read_tiktok(page) if site == "tiktok" else _read_rumble(page, url, max_height)
        referer = "https://www.tiktok.com/" if site == "tiktok" else "https://rumble.com/"
        _save_media(
            page, facts["play_url"], destination, referer,
            proxy=proxy, max_seconds=max_seconds, cancel_check=cancel_check, max_height=max_height,
        )
        return facts

    facts = _with_chrome(url, proxy, headless=headless, read=read, cancel_check=cancel_check)
    if max_seconds and max_seconds > 0:
        _trim_copy(destination, max_seconds, cancel_check)
        facts["duration"] = max_seconds
    return facts


def _with_chrome(url: str, proxy: str | None, *, headless: bool, read, cancel_check: Callable[[], None] = lambda: None) -> dict[str, Any]:
    cancel_check()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("需要安装 Playwright 才能用本机 Chrome 打开这个站点。") from exc

    launch: dict[str, Any] = {"channel": "chrome", "headless": headless}
    if not headless:
        launch["args"] = ["--disable-blink-features=AutomationControlled", "--window-size=800,600"]
        launch["ignore_default_args"] = ["--enable-automation"]
    if proxy:
        launch["proxy"] = {"server": proxy}
    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(**launch)
            except Exception:
                launch.pop("channel", None)
                browser = playwright.chromium.launch(**launch)
            try:
                page = browser.new_page()
                page.goto(_rumble_api_url(url) or url, wait_until="domcontentloaded", timeout=45000)
                _wait_until_clear(page, cancel_check)
                cancel_check()
                return read(page)
            finally:
                browser.close()
    except CancellationRequested:
        raise
    except Exception as exc:
        if "需要" in str(exc) or "没有返回" in str(exc):
            raise
        raise RuntimeError(f"本机 Chrome 没有打开这个页面。{exc}") from exc


def _page_is_challenge(page) -> bool:
    sample = f"{page.title()}\n{page.content()[:12000]}"
    return any(marker in sample for marker in _CHALLENGE_MARKERS)


def _wait_until_clear(page, cancel_check: Callable[[], None] = lambda: None) -> None:
    for _ in range(20):
        cancel_check()
        if not _page_is_challenge(page):
            return
        page.wait_for_timeout(1000)
    raise RuntimeError("站点的浏览器验证没有在时限内完成。")


def _read_tiktok(page) -> dict[str, Any]:
    try:
        page.wait_for_selector(
            "#__UNIVERSAL_DATA_FOR_REHYDRATION__",
            state="attached",
            timeout=20000,
        )
    except Exception as exc:
        raise RuntimeError("Chrome 打开了 TikTok，但页面里没有视频数据。") from exc
    payload = page.evaluate(
        """() => JSON.parse(document.querySelector('#__UNIVERSAL_DATA_FOR_REHYDRATION__').textContent)"""
    )
    detail = (
        payload.get("__DEFAULT_SCOPE__", {})
        .get("webapp.video-detail", {})
    )
    status = detail.get("statusCode")
    if status not in (None, 0):
        raise RuntimeError(f"这个 TikTok 视频当前打不开（状态 {status}）。")
    item = (detail.get("itemInfo") or {}).get("itemStruct") or {}
    video = item.get("video") or {}
    play_url = _tiktok_play_url(video)
    if not play_url:
        raise RuntimeError("TikTok 页面没有给出可下载的视频地址。")
    return {
        "title": item.get("desc") or None,
        "duration": video.get("duration"),
        "width": video.get("width"),
        "height": video.get("height"),
        "play_url": play_url,
        "thumbnail_url": video.get("cover") or video.get("originCover"),
        "webpage_url": page.url,
    }


def _read_rumble(page, source_url: str, max_height: int | None = None) -> dict[str, Any]:
    payload = None if _page_is_challenge(page) else _json_from_page(page)
    if not isinstance(payload, dict) or not _rumble_media_url(payload):
        token = video_id_for_url(source_url) or _rumble_embed_id(page.content())
        if not token:
            raise RuntimeError("Rumble 页面没有给出视频编号。")
        page.goto(
            f"https://rumble.com/embedJS/u3/?request=video&ver=2&v={token}",
            wait_until="domcontentloaded",
            timeout=45000,
        )
        _wait_until_clear(page)
        payload = _json_from_page(page)
    if not isinstance(payload, dict):
        raise RuntimeError("Rumble 没有返回视频信息。")
    play_url = _rumble_media_url(payload, max_height)
    if not play_url:
        raise RuntimeError("Rumble 没有返回可下载的视频地址。")
    return {
        "title": payload.get("title") or payload.get("name"),
        "duration": payload.get("duration"),
        "width": payload.get("w") or payload.get("width"),
        "height": payload.get("h") or payload.get("height"),
        "play_url": play_url,
        "thumbnail_url": payload.get("i") if isinstance(payload.get("i"), str) else None,
        "webpage_url": source_url,
    }


def _rumble_api_url(url: str) -> str | None:
    """Public slug token. The watch page itself sits behind a browser check."""
    if site_for_url(url) != "rumble":
        return None
    token = video_id_for_url(url)
    if not token:
        return None
    return f"https://rumble.com/embedJS/u3/?request=video&ver=2&v={token}"


def _rumble_embed_id(html: str) -> str | None:
    match = _RUMBLE_EMBED.search(html) or _RUMBLE_VIDEO_KEY.search(html)
    return match.group(1) if match else None


def _rumble_media_url(payload: dict, max_height: int | None = None) -> str | None:
    """Playable file. Timeline previews also end in .mp4 and are not the video."""
    for key in ("u", "ua"):
        media = payload.get(key)
        if isinstance(media, dict):
            found = _best_named_url(media.get("mp4"), max_height)
            if found:
                return found
    for key in ("ua", "u"):
        media = payload.get(key)
        if isinstance(media, dict):
            found = _best_named_url(media.get("hls"), max_height)
            if found:
                return found
    return None


def _best_named_url(node: Any, max_height: int | None = None) -> str | None:
    if isinstance(node, str) and node.startswith("http"):
        return node
    if not isinstance(node, dict):
        return None
    if isinstance(node.get("url"), str) and node["url"].startswith("http"):
        return node["url"]
    variants: list[tuple[int, str]] = []
    for key, value in node.items():
        if not isinstance(value, dict) or not isinstance(value.get("url"), str):
            continue
        meta = value.get("meta") or {}
        try:
            height = int(meta.get("h") or key)
        except (TypeError, ValueError):
            height = 0
        variants.append((height, value["url"]))
    if not variants:
        return None
    eligible = [item for item in variants if max_height is None or item[0] <= max_height]
    return (max(eligible) if eligible else min(variants))[1]


def _rumble_mp4(payload: dict) -> str | None:
    for key in ("u", "ua"):
        media = payload.get(key)
        if isinstance(media, dict):
            found = _best_named_url(media.get("mp4"))
            if found:
                return found
    return None


def _json_from_page(page) -> dict | None:
    try:
        text = page.locator("body").inner_text()
    except Exception:
        text = ""
    return _json_object(text) or _json_object(page.content())


def _json_object(text: str) -> dict | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _tiktok_play_url(video: dict) -> str | None:
    for key in ("playAddr", "downloadAddr"):
        value = video.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
        if isinstance(value, dict):
            src = value.get("src") or value.get("url")
            if isinstance(src, str) and src.startswith("http"):
                return src
    ranked: list[tuple[int, str]] = []
    for row in video.get("bitrateInfo") or []:
        if not isinstance(row, dict):
            continue
        gear = str(row.get("GearName") or row.get("gear_name") or "")
        if "bytevc2" in gear.lower():
            continue
        address = row.get("PlayAddr") or {}
        urls = address.get("UrlList") or []
        try:
            weight = int(address.get("DataSize") or 0)
        except (TypeError, ValueError):
            weight = 0
        for item in urls:
            if isinstance(item, str) and item.startswith("http"):
                ranked.append((weight, item))
    if not ranked:
        return None
    ranked.sort()
    return ranked[-1][1]


def _browser_auth(page) -> tuple[str, str]:
    cookies = page.context.cookies()
    cookie = "; ".join(
        f"{item['name']}={item['value']}" for item in cookies if item.get("name") and item.get("value")
    )
    try:
        agent = str(page.evaluate("() => navigator.userAgent"))
    except Exception:
        agent = "Mozilla/5.0"
    return cookie, agent


def _save_media(
    page,
    media_url: str,
    destination: Path,
    referer: str,
    proxy: str | None = None,
    max_seconds: int | None = None,
    cancel_check: Callable[[], None] = lambda: None,
    max_height: int | None = None,
) -> None:
    cancel_check()
    destination.parent.mkdir(parents=True, exist_ok=True)
    cookie, agent = _browser_auth(page)
    if ".m3u8" in media_url.lower():
        if _hls_prefix(page, media_url, destination, referer, max_seconds, cancel_check, max_height):
            return
        raise RuntimeError("站点的视频流未能完整保存。")
    if max_seconds and max_seconds > 0 and _ffmpeg_prefix(
        media_url, destination, referer, cookie, agent, proxy, max_seconds, cancel_check,
    ):
        return
    if max_seconds and max_seconds > 0 and _hls_prefix(page, media_url, destination, referer, max_seconds, cancel_check, max_height):
        return
    _stream_media(media_url, destination, referer, cookie, agent, proxy, cancel_check)


def _run_media_command(command: list[str], cancel_check: Callable[[], None] = lambda: None) -> subprocess.CompletedProcess:
    cancel_check()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        while True:
            cancel_check()
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                cancel_check()
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise


def _ffmpeg_prefix(
    media_url: str,
    destination: Path,
    referer: str,
    cookie: str,
    user_agent: str,
    proxy: str | None,
    seconds: int,
    cancel_check: Callable[[], None] = lambda: None,
) -> bool:
    ffmpeg = resolve_command("ffmpeg")
    if ffmpeg is None:
        return False
    header = f"Referer: {referer}\r\nUser-Agent: {user_agent}\r\n"
    if cookie:
        header += f"Cookie: {cookie}\r\n"
    command = [str(ffmpeg), "-y"]
    if proxy:
        command.extend(["-http_proxy", proxy])
    command.extend([
        "-headers", header,
        "-i", media_url,
        "-t", str(seconds),
        "-c", "copy",
        str(destination),
    ])
    completed = _run_media_command(command, cancel_check)
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size < 1024:
        destination.unlink(missing_ok=True)
        command = [item for item in command if item != "-c" and item != "copy"]
        insert_at = command.index(str(destination))
        command[insert_at:insert_at] = ["-c:v", "libx264", "-c:a", "aac"]
        completed = _run_media_command(command, cancel_check)
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size < 1024:
        destination.unlink(missing_ok=True)
        return False
    return True


def _hls_prefix(page, media_url: str, destination: Path, referer: str, seconds: int | None, cancel_check: Callable[[], None] = lambda: None, max_height: int | None = None) -> bool:
    cancel_check()
    try:
        playlist = page.context.request.get(
            media_url, headers={"Referer": referer}, timeout=30000,
        ).text()
    except Exception:
        return False
    if "#EXTM3U" not in playlist:
        return False
    if "#EXT-X-STREAM-INF" in playlist:
        variant = _hls_variant_url(playlist, media_url, max_height)
        if not variant:
            return False
        try:
            playlist = page.context.request.get(
                variant, headers={"Referer": referer}, timeout=30000,
            ).text()
        except Exception:
            return False
        media_url = variant
    segments = _hls_segment_urls(playlist, media_url, seconds or 10**7)
    if not segments:
        return False
    temporary = destination.with_suffix(".ts")
    try:
        with temporary.open("wb") as handle:
            for segment in segments:
                cancel_check()
                response = page.context.request.get(
                    segment, headers={"Referer": referer}, timeout=60000,
                )
                if response.status not in {200, 206}:
                    return False
                handle.write(response.body())
        with temporary.open("rb") as handle:
            head = handle.read(1)
        if temporary.stat().st_size < 1024 or head != b"\x47":
            return False
        ffmpeg = resolve_command("ffmpeg")
        if ffmpeg is None:
            return False
        command = [str(ffmpeg), "-y", "-i", str(temporary)]
        if seconds:
            command.extend(["-t", str(seconds)])
        command.extend(["-c:v", "libx264", "-c:a", "aac", str(destination)])
        completed = _run_media_command(command, cancel_check)
        return completed.returncode == 0 and destination.is_file() and destination.stat().st_size >= 1024
    except CancellationRequested:
        raise
    except Exception:
        destination.unlink(missing_ok=True)
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _hls_variant_url(playlist: str, base_url: str, max_height: int | None = 720) -> str | None:
    """Pick a variant. The default cap stays 720 so existing callers keep their stream."""
    pending: int | None = None
    variants: list[tuple[int, str]] = []
    for raw in playlist.splitlines():
        line = raw.strip()
        if line.startswith("#EXT-X-STREAM-INF"):
            match = re.search(r"RESOLUTION=\d+x(\d+)", line)
            pending = int(match.group(1)) if match else 0
            continue
        if not line or line.startswith("#") or pending is None:
            continue
        variants.append((pending, urljoin(base_url, line)))
        pending = None
    if not variants:
        return None
    eligible = [item for item in variants if max_height is None or item[0] <= max_height]
    return (max(eligible) if eligible else min(variants))[1]


def _hls_segment_urls(playlist: str, base_url: str, seconds: int) -> list[str]:
    chosen: list[str] = []
    pending: float | None = None
    total = 0.0
    for raw in playlist.splitlines():
        line = raw.strip()
        if line.startswith("#EXTINF:"):
            try:
                pending = float(line.split(":", 1)[1].split(",", 1)[0])
            except ValueError:
                pending = 0.0
            continue
        if not line or line.startswith("#") or pending is None:
            continue
        chosen.append(urljoin(base_url, line))
        total += pending
        pending = None
        if total >= seconds:
            break
    return chosen


def _stream_media(
    media_url: str,
    destination: Path,
    referer: str,
    cookie: str,
    user_agent: str,
    proxy: str | None,
    cancel_check: Callable[[], None] = lambda: None,
) -> None:
    import urllib.request

    request = urllib.request.Request(
        media_url,
        headers={"Referer": referer, "User-Agent": user_agent, "Cookie": cookie},
    )
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    cancel_check()
    partial = destination.with_name(destination.name + ".part")
    try:
        with opener.open(request, timeout=60) as response, partial.open("wb") as handle:
            while True:
                cancel_check()
                chunk = getattr(response, "read1", response.read)(256 * 1024)
                cancel_check()
                if not chunk:
                    break
                handle.write(chunk)
    except CancellationRequested:
        raise
    except Exception as exc:
        raise RuntimeError(f"视频文件下载失败。{exc.__class__.__name__}") from exc
    size = partial.stat().st_size
    with partial.open("rb") as handle:
        head = handle.read(16)
    if size < 1024 or head.startswith(b"<"):
        partial.unlink(missing_ok=True)
        raise RuntimeError("视频地址没有返回可播放的文件。")
    cancel_check()
    partial.replace(destination)


def _trim_copy(path: Path, seconds: int, cancel_check: Callable[[], None] = lambda: None) -> None:
    ffmpeg = resolve_command("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("截取前几秒需要 FFmpeg。")
    temporary = path.with_name(f"{path.stem}.trim.mp4")
    try:
        completed = _run_media_command(
            [str(ffmpeg), "-y", "-i", str(path), "-t", str(seconds), "-c", "copy", str(temporary)],
            cancel_check,
        )
        if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size < 1024:
            temporary.unlink(missing_ok=True)
            completed = _run_media_command(
                [str(ffmpeg), "-y", "-i", str(path), "-t", str(seconds), "-c:v", "libx264", "-c:a", "aac", str(temporary)],
                cancel_check,
            )
        if completed.returncode != 0 or not temporary.is_file():
            detail = (completed.stderr or "").strip().splitlines()
            tail = detail[-1] if detail else "ffmpeg failed"
            raise RuntimeError(f"截取前 {seconds} 秒失败。{tail}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
