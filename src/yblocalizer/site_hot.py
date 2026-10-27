"""Site-wide weekly hot list from YouTube search, cached next to channel weekly.

YouTube's trending feed is gone. Each category is a handful of search pages
sorted by views and limited to videos uploaded this week (``sp``). Flat
yt-dlp rows are merged, Shorts and livestreams dropped, then ranked locally.
A failed refresh keeps the previous cache and does not touch channel weekly.
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
from urllib.parse import parse_qs, quote_plus, urlparse

from .download import list_playlist_entries


CACHE_NAME = "site-hot.json"
# Sort by view count + uploaded this week + videos only.
# CAM%3D is all-time. EgIIAw%3D%3D is this week by relevance.
SEARCH_SP = "CAMSBAgDEAE%3D"
FETCH_ITEMS = "1:100"
SHORTS_MAX_SECONDS = 60
LONG_VIDEO_SECONDS = 45 * 60
SITE_TOP = 40
WORKERS = 4
REQUEST_GAP_SECONDS = 0.75
CACHE_TTL_HOURS = 12
DEFAULT_REGION = "US"
LIVE_STATUSES = {"is_live", "is_upcoming", "post_live"}

# 综合 is the translation-friendly set. Music, film, vlogs and news stay on their own chips,
# otherwise their view counts fill the default list.
_MUSIC_RE = re.compile(r"official music video|\(song\)|\blyrics\b", re.IGNORECASE)
_LIVE_VOD_RE = re.compile(r"is\s+live|livestream|live\s+stream", re.IGNORECASE)
CATEGORIES: list[tuple[str, str, list[str]]] = [
    ("mix", "综合", ["ai", "tech", "gaming", "science", "explained", "programming", "documentary"]),
    ("tech", "科技/AI", ["ai", "tech"]),
    ("gaming", "游戏", ["gaming"]),
    ("science", "科普", ["science", "explained", "documentary"]),
    ("film", "影视", ["trailer", "movie review"]),
    ("programming", "编程", ["programming"]),
    ("life", "生活", ["cooking", "vlog"]),
    ("news", "资讯", ["news", "sports"]),
    ("music", "音乐", ["music"]),
]
REGIONS: list[tuple[str, str]] = [("US", "美国"), ("GB", "英国"), ("", "不限")]

_CATEGORY_QUERIES = {key: queries for key, _label, queries in CATEGORIES}
_REGION_IDS = {key for key, _label in REGIONS}

_io_lock = threading.Lock()
_refresh_lock = threading.Lock()
_progress_lock = threading.Lock()
_progress = {"on": False, "done": 0, "total": 0, "category": ""}


def _root() -> Path:
    from .workbench_api import USER_DATA_ROOT
    return USER_DATA_ROOT


def _now(now: float | None) -> float:
    return time.time() if now is None else float(now)


def _iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _empty_category() -> dict:
    return {"refreshed_at": None, "videos": [], "error": None, "query_errors": [], "region": None}


def _empty_store() -> dict:
    return {"region": DEFAULT_REGION, "categories": {}}


def _load() -> dict:
    path = _root() / CACHE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _empty_store()
    if not isinstance(data, dict):
        return _empty_store()
    region = data.get("region")
    categories = {}
    for key, item in (data.get("categories") or {}).items():
        if key not in _CATEGORY_QUERIES or not isinstance(item, dict):
            continue
        videos = [row for row in item.get("videos") or [] if isinstance(row, dict) and row.get("url")]
        refreshed = item.get("refreshed_at")
        errors = [str(line) for line in item.get("query_errors") or [] if str(line).strip()]
        categories[key] = {
            "refreshed_at": refreshed if isinstance(refreshed, str) else None,
            "videos": videos,
            "error": str(item["error"]) if item.get("error") else None,
            "query_errors": errors,
            "region": item.get("region") if isinstance(item.get("region"), str) else None,
        }
    return {"region": region if region in _REGION_IDS else DEFAULT_REGION, "categories": categories}


def _save(data: dict) -> None:
    _write_json(_root() / CACHE_NAME, data)


def _search_url(query: str, region: str) -> str:
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp={SEARCH_SP}&hl=en"
    if region:
        url += f"&gl={region}"
    return url


def _video_url(url: str) -> str | None:
    video_id = parse_qs(urlparse(url).query).get("v", [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return None
    return f"https://www.youtube.com/watch?v={video_id}"


def _musicish(row: dict) -> bool:
    channel = row["channel_title"].lower()
    if "music" in channel or "vevo" in channel:
        return True
    return _MUSIC_RE.search(row["title"]) is not None


def _live_vod(row: dict) -> bool:
    return _LIVE_VOD_RE.search(f"{row.get('title') or ''} {row.get('channel_title') or ''}") is not None


def _too_long(row: dict) -> bool:
    duration = row.get("duration")
    return isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > LONG_VIDEO_SECONDS


def _mix_tier(row: dict) -> int:
    """0 normal, 1 live replay, 2 longer than 45 minutes."""
    if _live_vod(row):
        return 1
    if _too_long(row):
        return 2
    return 0


def _channel_key(row: dict) -> str:
    channel_id = str(row.get("channel_id") or "").strip()
    if channel_id:
        return f"id:{channel_id}"
    return f"name:{str(row.get('channel_title') or '').strip().lower()}"


def _site_row(entry: dict) -> dict | None:
    url = _video_url(str(entry.get("url") or ""))
    if url is None or entry.get("live_status") in LIVE_STATUSES:
        return None
    duration = entry.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= SHORTS_MAX_SECONDS:
        return None
    view_count = entry.get("view_count")
    if isinstance(view_count, bool) or not isinstance(view_count, int):
        view_count = None
    timestamp = entry.get("timestamp") if isinstance(entry.get("timestamp"), int) else None
    return {
        "url": url,
        "title": str(entry.get("title") or url),
        "duration": duration,
        "upload_date": entry.get("upload_date") if isinstance(entry.get("upload_date"), str) else None,
        "timestamp": timestamp,
        "channel_title": str(entry.get("channel") or ""),
        "channel_url": str(entry.get("channel_url") or ""),
        "channel_id": str(entry.get("channel_id") or ""),
        "thumbnail_url": entry.get("thumbnail_url") or None,
        "view_count": view_count,
    }


def merge_site_entries(groups: list[list[dict]], *, demote_music: bool = False) -> list[dict]:
    """Dedupe by video, drop Shorts and livestreams, rank by view count.

    The mix list also demotes music, live replays, and videos longer than 45
    minutes, then keeps at most two videos from each channel.
    """
    best: dict[str, dict] = {}
    for group in groups:
        for entry in group:
            row = _site_row(entry)
            if row is None:
                continue
            previous = best.get(row["url"])
            if previous is None or (row["view_count"] or -1) > (previous["view_count"] or -1):
                best[row["url"]] = row
    videos = list(best.values())
    if demote_music:
        videos.sort(
            key=lambda item: (
                2 - _mix_tier(item),
                not _musicish(item),
                item["view_count"] is not None,
                item["view_count"] or 0,
                item.get("timestamp") or 0,
            ),
            reverse=True,
        )
        kept: list[dict] = []
        counts: dict[str, int] = {}
        for item in videos:
            key = _channel_key(item)
            if counts.get(key, 0) >= 2:
                continue
            counts[key] = counts.get(key, 0) + 1
            kept.append(item)
            if len(kept) >= SITE_TOP:
                break
        return kept
    videos.sort(
        key=lambda item: (
            item["view_count"] is not None,
            item["view_count"] or 0,
            item.get("timestamp") or 0,
        ),
        reverse=True,
    )
    return videos


def _progress_view() -> dict | None:
    with _progress_lock:
        if not _progress["on"]:
            return None
        return {"done": _progress["done"], "total": _progress["total"], "category": _progress["category"]}


def _progress_start(total: int, category: str) -> None:
    with _progress_lock:
        _progress.update(on=True, done=0, total=total, category=category)


def _progress_step() -> None:
    with _progress_lock:
        _progress["done"] += 1


def _progress_stop() -> None:
    with _progress_lock:
        _progress["on"] = False


def site_snapshot(category: str = "mix", now: float | None = None) -> dict:
    if category not in _CATEGORY_QUERIES:
        raise ValueError("没有这个分类。")
    stamp = _now(now)
    with _io_lock:
        data = _load()
    cached = data["categories"].get(category) or _empty_category()
    refreshed = _parse_stamp(cached.get("refreshed_at"))
    region_changed = bool(cached.get("refreshed_at")) and cached.get("region") != data["region"]
    stale = refreshed is None or stamp - refreshed >= CACHE_TTL_HOURS * 3600 or region_changed
    progress = _progress_view()
    return {
        "category": category,
        "categories": [{"id": key, "label": label} for key, label, _queries in CATEGORIES],
        "region": data["region"],
        "regions": [{"id": key, "label": label} for key, label in REGIONS],
        "cached_region": cached.get("region"),
        "videos": list(cached["videos"]),
        "refreshed_at": cached.get("refreshed_at"),
        "stale": stale,
        "refreshing": progress is not None,
        "progress": progress,
        "error": cached.get("error"),
        "query_errors": list(cached.get("query_errors") or []),
        "sort_mode": "views",
    }


def set_site_region(region: str, category: str = "mix") -> dict:
    if region not in _REGION_IDS:
        raise ValueError("没有这个地区。")
    if category not in _CATEGORY_QUERIES:
        raise ValueError("没有这个分类。")
    with _io_lock:
        data = _load()
        data["region"] = region
        _save(data)
    return site_snapshot(category)


def _fetch_query(query: str, access: dict, region: str) -> dict:
    try:
        entries = list_playlist_entries(
            _search_url(query, region),
            access.get("cookies_from_browser"),
            access.get("cookies_file"),
            access.get("proxy"),
            FETCH_ITEMS,
            po_token_mode=str(access.get("po_token_mode") or "off"),
            approximate_date=True,
            sleep_requests=REQUEST_GAP_SECONDS,
        )
    except Exception as exc:
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else "刷新失败"
        message = message.removeprefix("ERROR: ").strip()[:300] or "刷新失败"
        return {"ok": False, "query": query, "entries": [], "error": message}
    return {"ok": True, "query": query, "entries": entries, "error": None}


def refresh_site_hot(access: dict, category: str = "mix", now: float | None = None, region: str | None = None) -> dict:
    """Refresh one category. An empty pull keeps the previous rows."""
    if category not in _CATEGORY_QUERIES:
        raise ValueError("没有这个分类。")
    if region is not None and region not in _REGION_IDS:
        raise ValueError("没有这个地区。")
    if not _refresh_lock.acquire(blocking=False):
        view = site_snapshot(category, now)
        view["refreshing"] = True
        return view
    try:
        stamp = _now(now)
        with _io_lock:
            data = _load()
            if region is not None:
                data["region"] = region
                _save(data)
            chosen = data["region"]
        queries = _CATEGORY_QUERIES[category]
        _progress_start(len(queries), category)
        results: list[dict | None] = [None] * len(queries)
        gap = threading.Lock()
        last = {"t": 0.0}

        def run(index: int, query: str) -> None:
            with gap:
                wait = REQUEST_GAP_SECONDS - (time.monotonic() - last["t"])
                if last["t"] and wait > 0:
                    time.sleep(wait)
                last["t"] = time.monotonic()
            results[index] = _fetch_query(query, access, chosen)
            _progress_step()

        with ThreadPoolExecutor(max_workers=min(WORKERS, len(queries))) as pool:
            futures = [pool.submit(run, index, query) for index, query in enumerate(queries)]
            for future in futures:
                future.result()

        groups = []
        errors = []
        for result in results:
            if result is None:
                continue
            if result["ok"]:
                groups.append(result["entries"])
            else:
                errors.append(f"{result['query']}: {result['error']}")
        videos = merge_site_entries(groups, demote_music=category == "mix")[:SITE_TOP]
        with _io_lock:
            data = _load()
            current = data["categories"].get(category) or _empty_category()
            if videos:
                current = {
                    "refreshed_at": _iso(stamp),
                    "videos": videos,
                    "error": None,
                    "query_errors": errors,
                    "region": chosen,
                }
            else:
                current["error"] = "全站热门没有拉到视频。YouTube 搜索排序可能已经改了，上次的列表还在。"
                current["query_errors"] = errors
            data["categories"][category] = current
            _save(data)
        _progress_stop()
        return site_snapshot(category, stamp)
    finally:
        _progress_stop()
        _refresh_lock.release()
