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
CHANNEL_WORKERS = 3
REQUEST_GAP_SECONDS = 0.75

_PREF_BOUNDS = {
    "window_days": (1, 30, 7),
    "top_k": (1, 50, 20),
    "fetch_count": (1, 50, 20),
    "cache_ttl_hours": (1, 168, 12),
}

_io_lock = threading.Lock()
_refresh_lock = threading.Lock()
_progress_lock = threading.Lock()
_progress = {"on": False, "done": 0, "total": 0}

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


def _bound(value, low: int, high: int, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return min(high, max(low, value))


def _preferences(data: dict) -> dict:
    prefs = {
        key: _bound(data.get(key), low, high, default)
        for key, (low, high, default) in _PREF_BOUNDS.items()
    }
    prefs["show_thumbnails"] = data.get("show_thumbnails") is not False
    return prefs


def _empty_subscriptions() -> dict:
    return {"enabled": False, "channels": [], **_preferences({})}


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
    loaded = data if isinstance(data, dict) else {}
    return {"enabled": loaded.get("enabled") is True, "channels": channels, **_preferences(loaded)}


def _save_subscriptions(data: dict) -> None:
    _write_json(_subscriptions_path(), {
        "enabled": data["enabled"] is True,
        "channels": data["channels"],
        **_preferences(data),
    })


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


def _order_videos(videos: list[dict]) -> str:
    """Return ``views`` or ``recent`` and sort ``videos`` in place.

    Flat tab rows often include ``view_count``. The list is ranked by plays
    only when every row has one, so the label can say 热. A missing count
    (including an older cache row) keeps upload order and the 新片 label.
    Nothing re-opens each video to fill the count.
    """
    if videos and all(isinstance(item.get("view_count"), int) for item in videos):
        videos.sort(key=lambda item: (item["view_count"], item.get("timestamp") or 0), reverse=True)
        return "views"
    videos.sort(key=lambda item: item.get("timestamp") or 0, reverse=True)
    return "recent"


def _in_window(timestamp: int | None, now: float, window_days: int) -> bool:
    """Keep uploads from the last `window_days` UTC calendar days.

    approximate_date snaps to midnight, so a strict hour cutoff drops the
    oldest day once the clock passes midnight. One future day is allowed
    because that same rounding can land slightly ahead of now.
    """
    if timestamp is None:
        return False
    upload_day = datetime.fromtimestamp(timestamp, timezone.utc).date()
    today = datetime.fromtimestamp(now, timezone.utc).date()
    delta = (today - upload_day).days
    return -1 <= delta <= window_days - 1


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


def _progress_view() -> dict | None:
    with _progress_lock:
        if not _progress["on"]:
            return None
        return {"done": _progress["done"], "total": _progress["total"]}


def _progress_start(total: int) -> None:
    with _progress_lock:
        _progress.update(on=True, done=0, total=total)


def _progress_step() -> None:
    with _progress_lock:
        _progress["done"] += 1


def _progress_stop() -> None:
    with _progress_lock:
        _progress["on"] = False


def subscription_view() -> dict:
    data = get_subscriptions()
    return {
        "enabled": data["enabled"],
        "channels": data["channels"],
        **_preferences(data),
    }


def weekly_snapshot(now: float | None = None) -> dict:
    stamp = _now(now)
    with _io_lock:
        subs = _load_subscriptions()
        cache = _load_cache()
    ids = {item["id"] for item in subs["channels"]}
    enabled = subs["enabled"] is True
    videos = []
    results = []
    window_days = subs["window_days"]
    if enabled:
        videos = [item for item in cache["videos"] if item.get("channel_id") in ids and _in_window(item.get("timestamp"), stamp, window_days)]
        results = [item for item in cache["channel_results"] if item.get("id") in ids]
    sort_mode = _order_videos(videos)
    if enabled:
        videos = videos[:subs["top_k"]]
    refreshed = _parse_stamp(cache.get("refreshed_at"))
    ttl = subs["cache_ttl_hours"] * 60 * 60
    stale = bool(enabled and subs["channels"] and (refreshed is None or stamp - refreshed >= ttl))
    progress = _progress_view()
    return {
        "enabled": enabled,
        "channels": subs["channels"],
        "videos": videos,
        "channel_results": results,
        "refreshed_at": cache.get("refreshed_at"),
        "stale": stale,
        "refreshing": progress is not None,
        "window_days": window_days,
        "top_k": subs["top_k"],
        "fetch_count": subs["fetch_count"],
        "show_thumbnails": subs["show_thumbnails"],
        "cache_ttl_hours": subs["cache_ttl_hours"],
        "channel_progress": progress,
        "sort_mode": sort_mode,
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


def update_subscription_settings(patch: dict) -> dict:
    with _io_lock:
        data = _load_subscriptions()
        for key, (low, high, _default) in _PREF_BOUNDS.items():
            if key not in patch:
                continue
            value = patch[key]
            if isinstance(value, bool):
                raise ValueError("请填写整数。")
            try:
                number = int(value)
            except (TypeError, ValueError):
                raise ValueError("请填写整数。")
            data[key] = min(high, max(low, number))
        if "show_thumbnails" in patch:
            data["show_thumbnails"] = patch["show_thumbnails"] is True
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
        "view_count": entry.get("view_count") if isinstance(entry.get("view_count"), int) else None,
    }


def _take_cached(videos: list[dict], channel_id: str, seen: set[str], stamp: float, window_days: int) -> list[dict]:
    kept = []
    for item in videos:
        if item.get("channel_id") != channel_id or item.get("url") in seen:
            continue
        timestamp = item.get("timestamp") if isinstance(item.get("timestamp"), int) else None
        if not _in_window(timestamp, stamp, window_days):
            continue
        seen.add(item["url"])
        kept.append(item)
    return kept


def _fetch_channel(channel: dict, access: dict, playlist_items: str) -> dict:
    try:
        listed = list_channel_uploads(
            channel["url"],
            access.get("cookies_from_browser"),
            access.get("cookies_file"),
            access.get("proxy"),
            po_token_mode=str(access.get("po_token_mode") or "auto"),
            playlist_items=playlist_items,
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


def refresh_weekly_hot(access: dict, now: float | None = None, channel_id: str | None = None) -> dict:
    """Refresh recent uploads. One channel failing does not cancel the others.

    ``channel_id`` refreshes that channel only. Channels that are not fetched,
    and a fetched channel that fails, keep their previous in-window rows.
    """
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
        if channel_id:
            channels = [item for item in channels if item["id"] == channel_id]
            if not channels:
                raise ValueError("没有这个频道。")
        playlist_items = f"1-{subs['fetch_count']}"
        _progress_start(len(channels))
        results: list[dict | None] = [None] * len(channels)
        gap = threading.Lock()
        last = {"t": 0.0}

        def run(index: int, channel: dict) -> None:
            with gap:
                wait = REQUEST_GAP_SECONDS - (time.monotonic() - last["t"])
                if last["t"] and wait > 0:
                    time.sleep(wait)
                last["t"] = time.monotonic()
            results[index] = _fetch_channel(channel, access, playlist_items)
            _progress_step()

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
            window_days = current["window_days"]
            old = _load_cache()
            old_results = {item.get("id"): item for item in old["channel_results"]}
            for channel in current["channels"]:
                result = fetched.get(channel["id"])
                if result is None:
                    videos.extend(_take_cached(old["videos"], channel["id"], seen, stamp, window_days))
                    previous = old_results.get(channel["id"])
                    if previous:
                        channel_results.append(previous)
                    continue
                if result["ok"] and result["title"]:
                    channel["title"] = result["title"]
                kept = 0
                if result["ok"]:
                    for entry in result["entries"]:
                        if not _in_window(entry.get("timestamp") if isinstance(entry.get("timestamp"), int) else None, stamp, window_days):
                            continue
                        row = _video_row(channel, entry, result["title"] or channel.get("title") or "")
                        if row is None or row["url"] in seen:
                            continue
                        seen.add(row["url"])
                        videos.append(row)
                        kept += 1
                else:
                    restored = _take_cached(old["videos"], channel["id"], seen, stamp, window_days)
                    videos.extend(restored)
                    kept = len(restored)
                channel_results.append({
                    "id": channel["id"],
                    "url": channel["url"],
                    "title": result["title"] or channel.get("title") or channel["url"],
                    "ok": bool(result["ok"]),
                    "error": result["error"],
                    "count": kept,
                })
            _order_videos(videos)
            _save_subscriptions(current)
            _save_cache(_iso(stamp), videos, channel_results)
        _progress_stop()
        return weekly_snapshot(stamp)
    finally:
        _progress_stop()
        _refresh_lock.release()
