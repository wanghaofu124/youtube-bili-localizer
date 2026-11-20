"""Read a public post without an account.

X publishes an embed JSON. Bluesky publishes an unauthenticated API.
Reddit's anonymous JSON is often blocked, while the public embed page in
a normal Chrome window still names the video, and that video's CDN file
is then readable on its own.
"""

from __future__ import annotations

import html
import json
import math
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import ProxyHandler, Request, build_opener

from .browser_media import _ffmpeg_prefix, _hls_segment_urls, _hls_variant_url, _run_media_command, _stream_media, _trim_copy
from .dependencies import resolve_command
from .runtime import CancellationRequested
from .source_sites import site_for_url, video_id_for_url

_X_720_BITRATE = 1_000_000

PUBLIC_POST_SITES = frozenset({"x", "reddit", "bluesky"})

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_REDDIT_MEDIA_ID = re.compile(r"v\.redd\.it/([A-Za-z0-9]+)")
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 600.0


def syndication_token(tweet_id: str) -> str:
    """Public embed token. Same formula the X embed page uses."""
    from yt_dlp.jsinterp import js_number_to_string

    raw = js_number_to_string((int(tweet_id) / 1e15) * math.pi, 36)
    return raw.translate(str.maketrans({ord("0"): None, ord("."): None}))


def reddit_media_id(page_html: str) -> str | None:
    match = _REDDIT_MEDIA_ID.search(page_html or "")
    return match.group(1) if match else None


def hls_audio_playlist(playlist: str, base_url: str) -> str | None:
    for line in (playlist or "").splitlines():
        if "TYPE=AUDIO" not in line:
            continue
        match = re.search(r'URI="([^"]+)"', line)
        if match:
            return urljoin(base_url, match.group(1))
    return None


def bluesky_handle_and_rkey(url: str) -> tuple[str, str] | None:
    segments = [part for part in urlparse(str(url).strip()).path.split("/") if part]
    if "profile" not in segments or "post" not in segments:
        return None
    profile = segments.index("profile")
    post = segments.index("post")
    if profile + 1 >= len(segments) or post + 1 >= len(segments):
        return None
    return segments[profile + 1], segments[post + 1]


def public_post_entry(url: str, proxy: str | None = None) -> dict[str, Any] | None:
    """One batch row for a public post. None when the post has no video."""
    facts = resolve_public_post(url, proxy)
    if not facts or not facts.get("media_url"):
        return None
    duration = facts.get("duration")
    try:
        duration = int(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None
    return {"url": url, "title": str(facts.get("title") or ""), "duration": duration}


def resolve_public_post(url: str, proxy: str | None = None) -> dict[str, Any] | None:
    cached = _cached(url)
    if cached is not None:
        return cached
    site = site_for_url(url)
    if site == "x":
        facts = _x_post(url, proxy)
    elif site == "reddit":
        facts = _reddit_post(url, proxy)
    elif site == "bluesky":
        facts = _bluesky_post(url, proxy)
    else:
        return None
    if facts and facts.get("media_url"):
        _store(url, facts)
    return facts


def quality_height_cap(quality: str | None) -> int | None:
    """720 and 1080 are ceilings. Original has no ceiling."""
    normalized = str(quality or "1080p").strip().lower()
    if normalized == "720p":
        return 720
    if normalized == "1080p":
        return 1080
    return None


def select_x_variant(variants: list[dict[str, Any]], quality: str | None) -> str | None:
    """X only publishes bitrates. 720p stays at or under about 1 Mbps."""
    parsed: list[tuple[int, str]] = []
    for item in variants:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        try:
            rate = int(item.get("bitrate") or 0)
        except (TypeError, ValueError):
            rate = 0
        parsed.append((rate, str(item["url"])))
    if not parsed:
        return None
    if str(quality or "1080p").strip().lower() == "720p":
        under = [item for item in parsed if 0 < item[0] <= _X_720_BITRATE]
        if under:
            return max(under)[1]
        positive = [item for item in parsed if item[0] > 0] or parsed
        return min(positive)[1]
    return max(parsed)[1]


def download_public_post(
    url: str,
    destination: Path,
    proxy: str | None = None,
    max_seconds: int | None = None,
    log: Callable[[str], None] | None = None,
    download_quality: str = "1080p",
    cancel_check: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    cancel_check()
    resolved = resolve_public_post(url, proxy)
    cancel_check()
    facts = dict(resolved) if resolved else None
    if facts and facts.get("variants"):
        chosen = select_x_variant(facts["variants"], download_quality)
        if chosen:
            facts["media_url"] = chosen
            facts["kind"] = "file"
    media_url = facts.get("media_url") if facts else None
    if not facts or not media_url:
        raise RuntimeError("这条公开帖子里没有可下载的视频。")
    if log:
        log("公开页面可以读到视频，开始保存。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    referer = str(facts.get("referer") or "https://x.com/")
    kind = str(facts.get("kind") or "file")
    seconds = int(max_seconds) if max_seconds and int(max_seconds) > 0 else None
    saved = False
    if kind == "hls":
        saved = _save_hls(
            media_url, destination, referer, proxy, seconds,
            max_height=quality_height_cap(download_quality),
            cancel_check=cancel_check,
        )
    if not saved and seconds is not None:
        saved = _ffmpeg_prefix(media_url, destination, referer, "", _UA, proxy, seconds, cancel_check)
    if not saved and kind == "file":
        _stream_media(media_url, destination, referer, "", _UA, proxy, cancel_check)
        if seconds is not None:
            _trim_copy(destination, seconds, cancel_check)
        saved = True
    if not saved or not destination.is_file():
        destination.unlink(missing_ok=True)
        raise RuntimeError("公开视频没有保存成可播放的文件。")
    facts["has_audio"] = file_has_audio(destination)
    cancel_check()
    return facts


def file_has_audio(path: Path) -> bool:
    ffprobe = resolve_command("ffprobe")
    if ffprobe is None or not path.is_file():
        return False
    completed = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return completed.returncode == 0 and "audio" in completed.stdout


def _x_post(url: str, proxy: str | None) -> dict[str, Any] | None:
    tweet_id = video_id_for_url(url)
    if not tweet_id:
        return None
    payload = _http_json(
        f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&token={syndication_token(tweet_id)}",
        proxy,
        {"User-Agent": "Googlebot"},
    )
    variants, hls_url, duration = _x_media(payload)
    media_url = select_x_variant(variants, "1080p")
    kind = "file"
    if not media_url and hls_url:
        media_url, kind = hls_url, "hls"
    if not media_url:
        return None
    return {
        "title": _one_line(payload.get("text")),
        "media_url": media_url,
        "kind": kind,
        "duration": duration,
        "variants": variants,
        "referer": "https://x.com/",
        "webpage_url": url,
    }


def _x_media(payload: dict) -> tuple[list[dict[str, Any]], str | None, float | None]:
    variants: list[dict[str, Any]] = []
    hls_url = None
    duration = None
    for item in payload.get("mediaDetails") or []:
        if not isinstance(item, dict):
            continue
        info = item.get("video_info") if isinstance(item.get("video_info"), dict) else {}
        if duration is None and info.get("duration_millis"):
            try:
                duration = float(info["duration_millis"]) / 1000.0
            except (TypeError, ValueError):
                duration = None
        for variant in info.get("variants") or []:
            if not isinstance(variant, dict) or not variant.get("url"):
                continue
            content_type = str(variant.get("content_type") or "")
            if content_type == "video/mp4":
                try:
                    rate = int(variant.get("bitrate") or 0)
                except (TypeError, ValueError):
                    rate = 0
                variants.append({"bitrate": rate, "url": str(variant["url"])})
            elif "mpegURL" in content_type and hls_url is None:
                hls_url = str(variant["url"])
    return variants, hls_url, duration


def _reddit_post(url: str, proxy: str | None) -> dict[str, Any] | None:
    if not video_id_for_url(url):
        return None
    page_html = _reddit_embed_html(_reddit_embed_url(url), proxy)
    media_id = reddit_media_id(page_html)
    if not media_id:
        raise RuntimeError("公开嵌入页没有给出视频。")
    return {
        "title": _html_title(page_html),
        "media_url": f"https://v.redd.it/{media_id}/HLSPlaylist.m3u8",
        "kind": "hls",
        "referer": "https://www.reddit.com/",
        "webpage_url": url,
    }


def _reddit_embed_url(url: str) -> str:
    path = urlparse(str(url).strip()).path or "/"
    if not path.endswith("/"):
        path += "/"
    return f"https://www.redditmedia.com{path}?embed=true"


def _reddit_embed_html(url: str, proxy: str | None) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("需要安装 Playwright 才能读取 Reddit 的公开嵌入页。") from exc

    launch: dict[str, Any] = {
        "channel": "chrome",
        "headless": False,
        "args": ["--disable-blink-features=AutomationControlled", "--window-size=800,600"],
        "ignore_default_args": ["--enable-automation"],
    }
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
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                html_text = ""
                for _ in range(8):
                    html_text = page.content()
                    if "v.redd.it" in html_text:
                        break
                    page.wait_for_timeout(1000)
                return html_text
            finally:
                browser.close()
    except Exception as exc:
        if "需要" in str(exc):
            raise
        raise RuntimeError("本机 Chrome 没有打开这条公开帖子。")


def _bluesky_post(url: str, proxy: str | None) -> dict[str, Any] | None:
    parsed = bluesky_handle_and_rkey(url)
    if parsed is None:
        return None
    handle, rkey = parsed
    identity = _http_json(
        "https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle?handle=" + quote(handle),
        proxy,
        {"Accept": "application/json", "User-Agent": _UA},
    )
    did = str(identity.get("did") or "")
    if not did.startswith("did:"):
        raise RuntimeError("公开接口没有返回这条帖子。")
    uri = quote(f"at://{did}/app.bsky.feed.post/{rkey}", safe="")
    payload = _http_json(
        "https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread?uri="
        + uri
        + "&depth=0&parentHeight=0",
        proxy,
        {"Accept": "application/json", "User-Agent": _UA},
    )
    post = ((payload.get("thread") or {}).get("post") or {})
    if not isinstance(post, dict):
        raise RuntimeError("公开接口没有返回这条帖子。")
    embed = post.get("embed") if isinstance(post.get("embed"), dict) else {}
    playlist = _bluesky_playlist(embed)
    if not playlist:
        return None
    record = post.get("record") if isinstance(post.get("record"), dict) else {}
    aspect = embed.get("aspectRatio") if isinstance(embed.get("aspectRatio"), dict) else {}
    return {
        "title": _one_line(record.get("text")),
        "media_url": playlist,
        "kind": "hls",
        "referer": "https://bsky.app/",
        "webpage_url": url,
        "width": aspect.get("width"),
        "height": aspect.get("height"),
    }


def _bluesky_playlist(embed: dict) -> str | None:
    playlist = embed.get("playlist")
    if isinstance(playlist, str) and playlist.startswith("http"):
        return playlist
    media = embed.get("media")
    if isinstance(media, dict):
        return _bluesky_playlist(media)
    return None


def _save_hls(
    media_url: str,
    destination: Path,
    referer: str,
    proxy: str | None,
    seconds: int | None,
    max_height: int | None = 720,
    cancel_check: Callable[[], None] = lambda: None,
) -> bool:
    cancel_check()
    try:
        playlist = _http_text(media_url, proxy, referer)
    except RuntimeError:
        return False
    if "#EXTM3U" not in playlist:
        return False
    audio_url = hls_audio_playlist(playlist, media_url)
    if "#EXT-X-STREAM-INF" in playlist:
        variant = _hls_variant_url(playlist, media_url, max_height)
        if not variant:
            return False
        try:
            playlist = _http_text(variant, proxy, referer)
        except RuntimeError:
            return False
        media_url = variant
    limit = seconds if seconds is not None else 10**7
    video_parts = _hls_segment_urls(playlist, media_url, limit)
    if not video_parts:
        return False
    audio_parts: list[str] = []
    if audio_url:
        try:
            audio_parts = _hls_segment_urls(_http_text(audio_url, proxy, referer), audio_url, limit)
        except RuntimeError:
            audio_parts = []
    video_ts = destination.with_suffix(".video.ts")
    audio_ts = destination.with_suffix(".audio.ts")
    try:
        if not _write_segments(video_parts, video_ts, proxy, referer, cancel_check):
            return False
        have_audio = bool(audio_parts) and _write_segments(audio_parts, audio_ts, proxy, referer, cancel_check)
        ffmpeg = resolve_command("ffmpeg")
        if ffmpeg is None:
            return False
        command = [str(ffmpeg), "-y", "-i", str(video_ts)]
        if have_audio:
            command.extend(["-i", str(audio_ts)])
        if seconds is not None:
            command.extend(["-t", str(seconds)])
        if have_audio:
            command.extend(["-c:v", "libx264", "-c:a", "aac", str(destination)])
        else:
            command.extend(["-c:v", "libx264", "-an", str(destination)])
        completed = _run_media_command(command, cancel_check)
        return completed.returncode == 0 and destination.is_file() and destination.stat().st_size >= 1024
    except CancellationRequested:
        raise
    except Exception:
        destination.unlink(missing_ok=True)
        return False
    finally:
        video_ts.unlink(missing_ok=True)
        audio_ts.unlink(missing_ok=True)


def _write_segments(urls: list[str], path: Path, proxy: str | None, referer: str, cancel_check: Callable[[], None] = lambda: None) -> bool:
    try:
        with path.open("wb") as handle:
            for segment_url in urls:
                cancel_check()
                handle.write(_http_bytes(segment_url, proxy, referer))
                cancel_check()
    except CancellationRequested:
        path.unlink(missing_ok=True)
        raise
    except Exception:
        path.unlink(missing_ok=True)
        return False
    with path.open("rb") as handle:
        head = handle.read(1)
    if path.stat().st_size < 1024 or head != b"\x47":
        path.unlink(missing_ok=True)
        return False
    return True


def _http_json(url: str, proxy: str | None, headers: dict[str, str]) -> dict[str, Any]:
    try:
        payload = json.loads(_http_bytes(url, proxy, "", headers).decode("utf-8"))
    except json.JSONDecodeError:
        raise RuntimeError("公开接口没有返回这条帖子。")
    if not isinstance(payload, dict):
        raise RuntimeError("公开接口没有返回这条帖子。")
    return payload


def _http_text(url: str, proxy: str | None, referer: str) -> str:
    return _http_bytes(url, proxy, referer).decode("utf-8", "replace")


def _http_bytes(url: str, proxy: str | None, referer: str, headers: dict[str, str] | None = None) -> bytes:
    request_headers = {"User-Agent": _UA}
    if referer:
        request_headers["Referer"] = referer
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers)
    opener = build_opener(*([ProxyHandler({"http": proxy, "https": proxy})] if proxy else []))
    try:
        with opener.open(request, timeout=30) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError):
        raise RuntimeError("公开接口没有返回这条帖子。")


def _html_title(page_html: str) -> str:
    heading = re.search(r"<h1[^>]*>(.*?)</h1>", page_html or "", re.I | re.S)
    if heading:
        text = _one_line(html.unescape(re.sub(r"<[^>]+>", "", heading.group(1))))
        if text:
            return text
    match = re.search(r'property="og:title"[^>]*content="([^"]*)"', page_html or "", re.I)
    if not match:
        match = re.search(r"<title>([^<]+)</title>", page_html, re.I)
    if not match:
        return ""
    return _one_line(html.unescape(match.group(1)))


def _one_line(value: Any) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text[:80]


def _cached(url: str) -> dict[str, Any] | None:
    now = time.time()
    with _CACHE_LOCK:
        found = _CACHE.get(url)
        if not found or now - found[0] > _CACHE_TTL:
            return None
        return dict(found[1])


def _store(url: str, facts: dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _CACHE[url] = (time.time(), dict(facts))
