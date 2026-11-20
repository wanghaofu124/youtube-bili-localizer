"""Overseas video sites the link downloader promises to accept.

Domestic hosts stay out of this list. Bilibili and Douyin remain publish
targets, not download sources.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse
import re

SUPPORTED_SITE_LABELS: tuple[str, ...] = (
    "YouTube",
    "Vimeo",
    "Dailymotion",
    "TikTok",
    "Niconico",
    "Rumble",
    "X",
    "Instagram",
    "Facebook",
    "Reddit",
    "Bluesky",
    "Tumblr",
)

# Host suffixes after a leading www. is removed. A host matches when it equals
# the suffix or is a subdomain of it.
_SITE_HOSTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("youtube", frozenset({"youtube.com", "youtu.be", "youtube-nocookie.com"})),
    ("vimeo", frozenset({"vimeo.com"})),
    ("dailymotion", frozenset({"dailymotion.com", "dai.ly"})),
    ("tiktok", frozenset({"tiktok.com"})),
    ("niconico", frozenset({"nicovideo.jp", "nico.ms"})),
    ("rumble", frozenset({"rumble.com"})),
    ("x", frozenset({"x.com", "twitter.com"})),
    ("instagram", frozenset({"instagram.com"})),
    ("facebook", frozenset({"facebook.com"})),
    ("reddit", frozenset({"reddit.com"})),
    ("bluesky", frozenset({"bsky.app"})),
    ("tumblr", frozenset({"tumblr.com"})),
)

_NICONICO_ID = re.compile(r"(?:sm|so|nm)\d+", re.IGNORECASE)
_RUMBLE_ID = re.compile(r"(v[0-9a-z]{4,})", re.IGNORECASE)


class UnsupportedVideoSite(ValueError):
    """A public URL whose host is outside the promised site list."""

    def __init__(self, host: str) -> None:
        self.host = host
        label = host or "该站点"
        names = "、".join(SUPPORTED_SITE_LABELS)
        super().__init__(f"当前不支持从 {label} 下载。已支持 {names}。")


def host_of(url: str) -> str:
    host = (urlparse(str(url).strip()).hostname or "").rstrip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def site_for_url(url: str) -> str | None:
    host = host_of(url)
    if not host:
        return None
    for site, suffixes in _SITE_HOSTS:
        if any(host == suffix or host.endswith("." + suffix) for suffix in suffixes):
            return site
    return None


def is_youtube_url(url: str) -> bool:
    return site_for_url(url) == "youtube"


def require_supported_video_url(url: str) -> str:
    """Return the URL when its host is on the promised site list."""
    text = str(url).strip()
    site = site_for_url(text)
    if site is None:
        raise UnsupportedVideoSite(host_of(text))
    return text


def youtube_video_id(url: str) -> str | None:
    if not is_youtube_url(url):
        return None
    parsed = urlparse(str(url).strip())
    host = (parsed.hostname or "").lower()
    segments = [segment for segment in parsed.path.split("/") if segment]
    if host == "youtu.be" or host.endswith(".youtu.be"):
        return segments[0] if segments else None
    value = (parse_qs(parsed.query).get("v") or [None])[0]
    if value:
        return value
    if len(segments) >= 2 and segments[0] in {"embed", "live", "shorts", "v"}:
        return segments[1]
    return None


def _path_segments(url: str) -> list[str]:
    return [segment for segment in urlparse(str(url).strip()).path.split("/") if segment]


def video_id_for_url(url: str) -> str | None:
    """Stable id parsed from the URL. Short links without an id return None."""
    site = site_for_url(url)
    if site == "youtube":
        return youtube_video_id(url)
    segments = _path_segments(url)
    if site == "vimeo":
        for segment in reversed(segments):
            if segment.isdigit():
                return segment
        return None
    if site == "dailymotion":
        host = host_of(url)
        if host == "dai.ly" and segments:
            return segments[0]
        if "video" in segments:
            index = segments.index("video")
            if index + 1 < len(segments):
                return segments[index + 1]
        return None
    if site == "tiktok":
        if "video" in segments:
            index = segments.index("video")
            if index + 1 < len(segments) and segments[index + 1].isdigit():
                return segments[index + 1]
        if host_of(url) in {"vm.tiktok.com", "vt.tiktok.com"} and segments:
            return segments[0]
        return None
    if site == "niconico":
        match = _NICONICO_ID.search(urlparse(str(url).strip()).path)
        return match.group(0).lower() if match else None
    if site == "rumble":
        match = _RUMBLE_ID.search("/" + "/".join(segments))
        return match.group(1).lower() if match else None
    if site == "x":
        return _segment_after(segments, ("status", "statuses"), digits_only=True)
    if site == "instagram":
        code = _segment_after(segments, ("p", "tv", "reel", "reels"))
        if code:
            return code
        if "stories" in segments:
            index = segments.index("stories")
            if index + 2 < len(segments) and segments[index + 2].isdigit():
                return segments[index + 2]
        return None
    if site == "facebook":
        query = parse_qs(urlparse(str(url).strip()).query)
        for key in ("v", "video_id", "story_fbid"):
            value = (query.get(key) or [None])[0]
            if value:
                return value
        for marker in ("reel", "videos", "posts"):
            token = _segment_after(segments, (marker,))
            if token and (token.isdigit() or token.startswith("pfbid")):
                return token
        return None
    if site == "reddit":
        return _segment_after(segments, ("comments",))
    if site == "bluesky":
        return _segment_after(segments, ("post",))
    if site == "tumblr":
        return _segment_after(segments, ("post", "video"), digits_only=True)
    return None


def _segment_after(segments: list[str], markers: tuple[str, ...], digits_only: bool = False) -> str | None:
    for marker in markers:
        if marker not in segments:
            continue
        index = segments.index(marker)
        if index + 1 >= len(segments):
            continue
        token = segments[index + 1]
        if digits_only and not token.isdigit():
            continue
        return token
    return None


def vimeo_player_url(url: str) -> str | None:
    """Public player page. Vimeo's app token currently answers 401."""
    text = str(url).strip()
    if site_for_url(text) != "vimeo":
        return None
    video_id = video_id_for_url(text)
    if not video_id:
        return None
    player = f"https://player.vimeo.com/video/{video_id}"
    segments = _path_segments(text)
    if segments and segments[-1] != video_id and re.fullmatch(r"[A-Za-z0-9]+", segments[-1]):
        player += "?h=" + segments[-1]
    return player


def source_identity(url: str) -> str:
    """Cache key. YouTube keeps the historical ``youtube:<id>`` form."""
    text = str(url).strip()
    site = site_for_url(text)
    if site is None:
        return text
    video_id = video_id_for_url(text)
    if video_id:
        return f"{site}:{video_id}"
    parsed = urlparse(text)
    path = parsed.path or "/"
    return f"{site}:{parsed.scheme}://{host_of(text)}{path}"
