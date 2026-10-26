"""Local YouTube channel subscriptions and the 7-day upload list.

Subscriptions live next to the other workbench preferences. The weekly list is
a JSON cache of flat yt-dlp tab results, not a second download pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid
from urllib.parse import unquote, urlparse

from .download import list_channel_uploads


SUBSCRIPTIONS_NAME = "channel-subscriptions.json"
CACHE_NAME = "weekly-hot.json"
MAX_CHANNELS = 30
CACHE_TTL_SECONDS = 12 * 60 * 60
CHANNEL_WORKERS = 3
REQUEST_GAP_SECONDS = 0.75
PLAYLIST_ITEMS = "1-20"

_io_lock = threading.Lock()
_refresh_lock = threading.Lock()

_CHANNEL_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com"}


def _root() -> Path:
    from .workbench_api import USER_DATA_ROOT
    return USER_DATA_ROOT


def _now(now: float | None) -> float:
    return time.time() if now is None else float(now)


def _iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _subscriptions_path() -> Path:
    return _root() / SUBSCRIPTIONS_NAME


def _cache_path() -> Path:
    return _root() / CACHE_NAME


def normalize_channel_videos_url(value: str) -> str:
    """Accept a channel root and store the uploads tab."""
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in _CHANNEL_HOSTS or parsed.username or parsed.password:
        raise ValueError("请输入 YouTube 频道链接，例如 https://www.youtube.com/@handle")
    parts = [part for part in unquote(parsed.path).split("/") if part]
    if not parts:
        raise ValueError("请输入频道主页链接（/@handle 或 /channel/UC…），不是单个视频。")
    head = parts[0]
    if head.startswith("@") and len(head) > 1:
        key = head
    elif head in {"channel", "c", "user"} and len(parts) >= 2 and parts[1] not in {"videos", "featured", "streams", "shorts", "playlists", "about"}:
        if head == "channel" and not parts[1].startswith("UC"):
            raise ValueError("频道 ID 应以 UC 开头，例如 https://www.youtube.com/channel/UC…")
        key = f"{head}/{parts[1]}"
    else:
        raise ValueError("请输入频道主页链接（/@handle 或 /channel/UC…），不是单个视频。")
    if not re.fullmatch(r"(@[^/]+|channel/UC[^/]+|c/[^/]+|user/[^/]+)", key):
        raise ValueError("频道链接格式无效。")
    return f"https://www.youtube.com/{key}/videos"


def _empty_subscriptions() -> dict:
    return {"enabled": False, "channels": []}


def _load_subscriptions() -> dict:
    path = _subscriptions_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _empty_subscriptions()
    if not isinstance(data, dict):
        return _empty_subscriptions()
    channels = []
    for item in data.get("channels") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        channel_id = str(item.get("id") or "").strip()
        if not url or not channel_id:
            continue
        channels.append({
            "id": channel_id,
            "url": url,
            "title": str(item.get("title") or ""),
            "added_at": str(item.get("added_at") or ""),
        })
    return {"enabled": data.get("enabled") is True, "channels": channels}


def _save_subscriptions(data: dict) -> None:
    _write_json(_subscriptions_path(), {"enabled": data["enabled"] is True, "channels": data["channels"]})


def _load_cache() -> dict:
    path = _cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"refreshed_at": None, "videos": [], "channel_results": []}
    if not isinstance(data, dict):
        return {"refreshed_at": None, "videos": [], "channel_results": []}
    videos = [item for item in data.get("videos") or [] if isinstance(item, dict) and item.get("url")]
    results = [item for item in data.get("channel_results") or [] if isinstance(item, dict)]
    refreshed = data.get("refreshed_at")
    return {
        "refreshed_at": refreshed if isinstance(refreshed, str) else None,
        "videos": videos,
        "channel_results": results,
    }


def _save_cache(refreshed_at: str, videos: list[dict], channel_results: list[dict]) -> None:
    _write_json(_cache_path(), {
        "refreshed_at": refreshed_at,
        "videos": videos,
        "channel_results": channel_results,
    })


def _in_window(timestamp: int | None, now: float) -> bool:
    """Keep uploads from the last 7 calendar days, UTC.

    approximate_date snaps to midnight, so a strict 168-hour cutoff drops the
    oldest day of the week once the clock passes midnight. One future day is
    allowed because that same rounding can land slightly ahead of now.
    """
    if timestamp is None:
        return False
    upload_day = datetime.fromtimestamp(timestamp, timezone.utc).date()
    today = datetime.fromtimestamp(now, timezone.utc).date()
    delta = (today - upload_day).days
    return -1 <= delta <= 6


def _parse_stamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def get_subscriptions() -> dict:
    with _io_lock:
        return _load_subscriptions()


def subscription_view() -> dict:
    data = get_subscriptions()
    return {"enabled": data["enabled"], "channels": data["channels"]}


def weekly_snapshot(now: float | None = None) -> dict:
    stamp = _now(now)
    with _io_lock:
        subs = _load_subscriptions()
        cache = _load_cache()
    ids = {item["id"] for item in subs["channels"]}
    enabled = subs["enabled"] is True
    videos = []
    results = []
    if enabled:
        videos = [item for item in cache["videos"] if item.get("channel_id") in ids and _in_window(item.get("timestamp"), stamp)]
        videos.sort(key=lambda item: item.get("timestamp") or 0, reverse=True)
        results = [item for item in cache["channel_results"] if item.get("id") in ids]
    refreshed = _parse_stamp(cache.get("refreshed_at"))
    stale = bool(enabled and subs["channels"] and (refreshed is None or stamp - refreshed >= CACHE_TTL_SECONDS))
    return {
        "enabled": enabled,
        "channels": subs["channels"],
        "videos": videos,
        "channel_results": results,
        "refreshed_at": cache.get("refreshed_at"),
        "stale": stale,
        "refreshing": False,
        "window_days": 7,
    }


def add_channel(url: str) -> dict:
    normalized = normalize_channel_videos_url(url)
    with _io_lock:
        data = _load_subscriptions()
        if any(item["url"] == normalized for item in data["channels"]):
            raise ValueError("这个频道已经在关注列表里。")
        if len(data["channels"]) >= MAX_CHANNELS:
            raise ValueError(f"最多关注 {MAX_CHANNELS} 个频道。")
        data["channels"].append({
            "id": uuid.uuid4().hex[:12],
            "url": normalized,
            "title": "",
            "added_at": _iso(time.time()),
        })
        _save_subscriptions(data)
    return weekly_snapshot()


def remove_channel(channel_id: str) -> dict:
    with _io_lock:
        data = _load_subscriptions()
        remaining = [item for item in data["channels"] if item["id"] != channel_id]
        if len(remaining) == len(data["channels"]):
            raise ValueError("没有这个频道。")
        data["channels"] = remaining
        _save_subscriptions(data)
    return weekly_snapshot()


def set_subscriptions_enabled(enabled: bool) -> dict:
    with _io_lock:
        data = _load_subscriptions()
        data["enabled"] = enabled
        _save_subscriptions(data)
    return weekly_snapshot()


def _video_row(channel: dict, entry: dict, channel_title: str) -> dict | None:
    url = str(entry.get("url") or "").strip()
    timestamp = entry.get("timestamp")
    if not url or not isinstance(timestamp, int):
        return None
    duration = entry.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        duration = None
    return {
        "url": url,
        "title": str(entry.get("title") or url),
        "duration": duration,
        "upload_date": entry.get("upload_date"),
        "timestamp": timestamp,
        "channel_title": channel_title,
        "channel_url": channel["url"],
        "channel_id": channel["id"],
        "thumbnail_url": entry.get("thumbnail_url") or None,
    }


def _fetch_channel(channel: dict, access: dict) -> dict:
    try:
        listed = list_channel_uploads(
            channel["url"],
            access.get("cookies_from_browser"),
            access.get("cookies_file"),
            access.get("proxy"),
            po_token_mode=str(access.get("po_token_mode") or "auto"),
            playlist_items=PLAYLIST_ITEMS,
            sleep_requests=REQUEST_GAP_SECONDS,
        )
    except Exception as exc:
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else "刷新失败"
        message = message.removeprefix("ERROR: ").strip()[:300] or "刷新失败"
        return {"ok": False, "title": channel.get("title") or "", "entries": [], "error": message}
    return {
        "ok": True,
        "title": listed.get("title") or channel.get("title") or "",
        "entries": listed.get("entries") or [],
        "error": None,
    }


def refresh_weekly_hot(access: dict, now: float | None = None) -> dict:
    """Refresh recent uploads. One channel failing does not cancel the others."""
    if not _refresh_lock.acquire(blocking=False):
        view = weekly_snapshot(now)
        view["refreshing"] = True
        return view
    try:
        stamp = _now(now)
        with _io_lock:
            subs = _load_subscriptions()
        if not subs["enabled"] or not subs["channels"]:
            return weekly_snapshot(stamp)
        channels = list(subs["channels"])
        results: list[dict | None] = [None] * len(channels)
        gap = threading.Lock()
        last = {"t": 0.0}

        def run(index: int, channel: dict) -> None:
            with gap:
                wait = REQUEST_GAP_SECONDS - (time.monotonic() - last["t"])
                if last["t"] and wait > 0:
                    time.sleep(wait)
                last["t"] = time.monotonic()
            results[index] = _fetch_channel(channel, access)

        workers = min(CHANNEL_WORKERS, len(channels))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run, index, channel) for index, channel in enumerate(channels)]
            for future in futures:
                future.result()

        fetched = {channel["id"]: result for channel, result in zip(channels, results) if result}
        videos: list[dict] = []
        channel_results: list[dict] = []
        seen: set[str] = set()
        with _io_lock:
            current = _load_subscriptions()
            for channel in current["channels"]:
                result = fetched.get(channel["id"])
                if result is None:
                    continue
                if result["ok"] and result["title"]:
                    channel["title"] = result["title"]
                kept = 0
                if result["ok"]:
                    for entry in result["entries"]:
                        if not _in_window(entry.get("timestamp") if isinstance(entry.get("timestamp"), int) else None, stamp):
                            continue
                        row = _video_row(channel, entry, result["title"] or channel.get("title") or "")
                        if row is None or row["url"] in seen:
                            continue
                        seen.add(row["url"])
                        videos.append(row)
                        kept += 1
                channel_results.append({
                    "id": channel["id"],
                    "url": channel["url"],
                    "title": result["title"] or channel.get("title") or channel["url"],
                    "ok": bool(result["ok"]),
                    "error": result["error"],
                    "count": kept,
                })
            videos.sort(key=lambda item: item["timestamp"], reverse=True)
            _save_subscriptions(current)
            _save_cache(_iso(stamp), videos, channel_results)
        return weekly_snapshot(stamp)
    finally:
        _refresh_lock.release()
