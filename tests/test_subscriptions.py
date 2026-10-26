import sys
from datetime import datetime, timezone
from http import HTTPStatus
from types import SimpleNamespace

from yblocalizer import download
from yblocalizer.subscriptions import (
    _progress_view,
    add_channel,
    normalize_channel_videos_url,
    refresh_weekly_hot,
    remove_channel,
    set_subscriptions_enabled,
    update_subscription_settings,
    weekly_snapshot,
)
from yblocalizer.workbench_api import WorkbenchHandler


NOW = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc).timestamp()


def test_channel_url_normalizes_to_videos_tab():
    assert normalize_channel_videos_url("https://www.youtube.com/@Demo") == "https://www.youtube.com/@Demo/videos"
    assert normalize_channel_videos_url("https://m.youtube.com/channel/UCabcdefghij1234567890ab/featured") == "https://www.youtube.com/channel/UCabcdefghij1234567890ab/videos"
    assert normalize_channel_videos_url("https://www.youtube.com/@Demo/videos?view=0") == "https://www.youtube.com/@Demo/videos"


def test_channel_url_rejects_a_single_video():
    try:
        normalize_channel_videos_url("https://www.youtube.com/watch?v=abcdefghijk")
    except ValueError as exc:
        assert "频道" in str(exc)
    else:
        raise AssertionError("video URL should be rejected")


def test_duplicate_channel_is_rejected():
    add_channel("https://www.youtube.com/@Demo")
    try:
        add_channel("https://www.youtube.com/@Demo/videos")
    except ValueError as exc:
        assert "已经" in str(exc)
    else:
        raise AssertionError("duplicate channel should be rejected")


def test_weekly_refresh_ranks_by_upload_time_and_isolates_failures(monkeypatch):
    monkeypatch.setattr("yblocalizer.subscriptions.REQUEST_GAP_SECONDS", 0)
    add_channel("https://www.youtube.com/@good")
    add_channel("https://www.youtube.com/@bad")
    set_subscriptions_enabled(True)

    def fake_list(url, *_args, **_kwargs):
        if url.endswith("/@bad/videos"):
            raise RuntimeError("channel down")
        edge = int(datetime(2026, 10, 20, tzinfo=timezone.utc).timestamp())
        too_old = int(datetime(2026, 10, 19, tzinfo=timezone.utc).timestamp())
        return {"title": "Good", "channel_id": "UC1", "entries": [
            {"url": "https://www.youtube.com/watch?v=older", "title": "Older", "duration": 8, "timestamp": int(NOW - 2 * 3600)},
            {"url": "https://www.youtube.com/watch?v=newer", "title": "Newer", "duration": 12, "timestamp": int(NOW - 3600)},
            {"url": "https://www.youtube.com/watch?v=edge", "title": "Edge", "duration": 9, "timestamp": edge},
            {"url": "https://www.youtube.com/watch?v=tooold", "title": "Too old", "duration": 9, "timestamp": too_old},
            {"url": "https://www.youtube.com/watch?v=ancient", "title": "Ancient", "duration": 3, "timestamp": int(NOW - 40 * 86400)},
            {"url": "https://www.youtube.com/watch?v=nodate", "title": "No date", "duration": 4, "timestamp": None},
        ]}

    monkeypatch.setattr("yblocalizer.subscriptions.list_channel_uploads", fake_list)
    view = refresh_weekly_hot({"cookies_from_browser": None, "cookies_file": None, "proxy": None, "po_token_mode": "off"}, now=NOW)
    assert [item["title"] for item in view["videos"]] == ["Newer", "Older", "Edge"]
    assert view["sort_mode"] == "recent"
    assert view["stale"] is False
    failed = [item for item in view["channel_results"] if not item["ok"]]
    assert len(failed) == 1 and failed[0]["error"] == "channel down"
    assert weekly_snapshot(now=NOW + 13 * 3600)["stale"] is True
    removed = remove_channel(view["channels"][0]["id"])
    assert removed["videos"] == []


def test_failed_channel_keeps_previous_in_window_videos(monkeypatch):
    monkeypatch.setattr("yblocalizer.subscriptions.REQUEST_GAP_SECONDS", 0)
    add_channel("https://www.youtube.com/@good")
    add_channel("https://www.youtube.com/@bad")
    set_subscriptions_enabled(True)
    recent = int(NOW - 3600)
    too_old = int(datetime(2026, 10, 19, tzinfo=timezone.utc).timestamp())
    state = {"bad_down": False}

    def fake_list(url, *_args, **_kwargs):
        if url.endswith("/@bad/videos"):
            if state["bad_down"]:
                raise RuntimeError("channel down")
            return {"title": "Bad", "entries": [
                {"url": "https://www.youtube.com/watch?v=keep", "title": "Keep", "duration": 20, "timestamp": recent, "upload_date": "20261026", "thumbnail_url": "https://i.ytimg.com/vi/keep/hq.jpg"},
                {"url": "https://www.youtube.com/watch?v=drop", "title": "Drop", "duration": 5, "timestamp": too_old, "thumbnail_url": "https://i.ytimg.com/vi/drop/hq.jpg"},
            ]}
        return {"title": "Good", "entries": [
            {"url": "https://www.youtube.com/watch?v=gone", "title": "Gone", "duration": 8, "timestamp": recent},
            {"url": "https://www.youtube.com/watch?v=stay", "title": "Stay", "duration": 9, "timestamp": recent + 10},
        ] if not state["bad_down"] else [
            {"url": "https://www.youtube.com/watch?v=stay", "title": "Stay", "duration": 9, "timestamp": recent + 10},
        ]}

    monkeypatch.setattr("yblocalizer.subscriptions.list_channel_uploads", fake_list)
    access = {"cookies_from_browser": None, "cookies_file": None, "proxy": None, "po_token_mode": "off"}
    first = refresh_weekly_hot(access, now=NOW)
    assert [item["title"] for item in first["videos"]] == ["Stay", "Gone", "Keep"]
    state["bad_down"] = True
    second = refresh_weekly_hot(access, now=NOW)
    assert [item["title"] for item in second["videos"]] == ["Stay", "Keep"]
    kept = next(item for item in second["videos"] if item["title"] == "Keep")
    assert kept["channel_title"] == "Bad"
    assert kept["duration"] == 20
    assert kept["upload_date"] == "20261026"
    assert kept["thumbnail_url"] == "https://i.ytimg.com/vi/keep/hq.jpg"
    failed = next(item for item in second["channel_results"] if not item["ok"])
    assert failed["error"] == "channel down"
    assert failed["count"] == 1


def test_view_counts_rank_the_week_until_one_count_is_missing(monkeypatch):
    monkeypatch.setattr("yblocalizer.subscriptions.REQUEST_GAP_SECONDS", 0)
    add_channel("https://www.youtube.com/@good")
    set_subscriptions_enabled(True)
    newer = int(NOW - 3600)
    older = int(NOW - 2 * 3600)
    access = {"cookies_from_browser": None, "cookies_file": None, "proxy": None, "po_token_mode": "off"}
    rows = {"missing": False}

    def fake_list(_url, *_args, **_kwargs):
        older_row = {"url": "https://www.youtube.com/watch?v=older", "title": "Older", "duration": 12, "timestamp": older, "view_count": 90}
        newer_row = {"url": "https://www.youtube.com/watch?v=newer", "title": "Newer", "duration": 8, "timestamp": newer, "view_count": 10}
        if rows["missing"]:
            newer_row.pop("view_count")
        return {"title": "Good", "entries": [newer_row, older_row]}

    monkeypatch.setattr("yblocalizer.subscriptions.list_channel_uploads", fake_list)
    ranked = refresh_weekly_hot(access, now=NOW)
    assert [item["title"] for item in ranked["videos"]] == ["Older", "Newer"]
    assert [item["view_count"] for item in ranked["videos"]] == [90, 10]
    assert ranked["sort_mode"] == "views"
    rows["missing"] = True
    fallback = refresh_weekly_hot(access, now=NOW)
    assert [item["title"] for item in fallback["videos"]] == ["Newer", "Older"]
    assert fallback["videos"][0]["view_count"] is None
    assert fallback["sort_mode"] == "recent"


def test_one_channel_retry_keeps_the_other_and_reports_progress(monkeypatch):
    monkeypatch.setattr("yblocalizer.subscriptions.REQUEST_GAP_SECONDS", 0)
    monkeypatch.setattr("yblocalizer.subscriptions.CHANNEL_WORKERS", 1)
    add_channel("https://www.youtube.com/@good")
    add_channel("https://www.youtube.com/@bad")
    set_subscriptions_enabled(True)
    update_subscription_settings({"fetch_count": 5, "show_thumbnails": False})
    seen = []

    def fake_list(url, *_args, **kwargs):
        seen.append((url, kwargs.get("playlist_items"), _progress_view()["done"]))
        title = "Bad" if url.endswith("/@bad/videos") else "Good"
        slug = "bad" if title == "Bad" else "good"
        return {"title": title, "entries": [{
            "url": f"https://www.youtube.com/watch?v={slug}",
            "title": f"{title} clip",
            "duration": 4,
            "timestamp": int(NOW - (3600 if title == "Bad" else 1000)),
            "thumbnail_url": f"https://i.ytimg.com/vi/{slug}/hq.jpg",
        }]}

    monkeypatch.setattr("yblocalizer.subscriptions.list_channel_uploads", fake_list)
    access = {"cookies_from_browser": None, "cookies_file": None, "proxy": None, "po_token_mode": "off"}
    first = refresh_weekly_hot(access, now=NOW)
    assert seen == [
        ("https://www.youtube.com/@good/videos", "1-5", 0),
        ("https://www.youtube.com/@bad/videos", "1-5", 1),
    ]
    assert first["show_thumbnails"] is False
    assert first["fetch_count"] == 5
    assert first["channel_progress"] is None
    assert [item["title"] for item in first["videos"]] == ["Good clip", "Bad clip"]

    def fail_bad(url, *_args, **_kwargs):
        if url.endswith("/@bad/videos"):
            raise RuntimeError("channel down")
        return {"title": "Good", "entries": [{
            "url": "https://www.youtube.com/watch?v=replacement",
            "title": "Replacement",
            "duration": 6,
            "timestamp": int(NOW - 10),
        }]}

    monkeypatch.setattr("yblocalizer.subscriptions.list_channel_uploads", fail_bad)
    bad_id = next(item["id"] for item in first["channels"] if item["url"].endswith("/@bad/videos"))
    second = refresh_weekly_hot(access, now=NOW, channel_id=bad_id)
    assert [item["title"] for item in second["videos"]] == ["Good clip", "Bad clip"]
    failed = next(item for item in second["channel_results"] if item["id"] == bad_id)
    assert failed["ok"] is False and failed["count"] == 1
    narrow = update_subscription_settings({"top_k": 1, "window_days": 0})
    assert narrow["top_k"] == 1
    assert narrow["window_days"] == 1
    assert [item["title"] for item in narrow["videos"]] == ["Good clip"]


def test_disabled_refresh_does_not_call_ytdlp(monkeypatch):
    called = []
    monkeypatch.setattr("yblocalizer.subscriptions.list_channel_uploads", lambda *_args, **_kwargs: called.append(True))
    add_channel("https://www.youtube.com/@demo")
    view = refresh_weekly_hot({}, now=NOW)
    assert called == []
    assert view["enabled"] is False
    assert view["videos"] == []


def test_weekly_route_returns_empty_snapshot():
    handler = object.__new__(WorkbenchHandler)
    handler.path = "/api/subscriptions/weekly"
    result: dict = {}
    handler._json = lambda payload, status=HTTPStatus.OK: result.update(payload=payload, status=status)  # type: ignore[method-assign]
    handler.do_GET()
    assert result["payload"]["enabled"] is False
    assert result["payload"]["videos"] == []
    assert result["payload"]["window_days"] == 7
    assert result["payload"]["top_k"] == 20
    assert result["payload"]["fetch_count"] == 20
    assert result["payload"]["show_thumbnails"] is True
    assert result["payload"]["cache_ttl_hours"] == 12
    assert result["payload"]["channel_progress"] is None
    assert result["payload"]["sort_mode"] == "recent"


def test_channel_listing_reuses_flat_extract_and_metadata_access(monkeypatch):
    seen = {}

    class FakeYDL:
        def __init__(self, opts):
            seen.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url, download=False):
            return {
                "channel": "Demo",
                "channel_id": "UCdemo",
                "entries": [{
                    "id": "abcdefghijk",
                    "ie_key": "Youtube",
                    "url": "abcdefghijk",
                    "title": "Clip",
                    "duration": 12,
                    "timestamp": 1_700_000_000,
                    "view_count": 1200,
                    "thumbnails": [
                        {"url": "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg", "width": 480, "height": 360},
                        {"url": "https://i.ytimg.com/vi/abcdefghijk/sddefault.jpg", "width": 640, "height": 480},
                    ],
                }],
            }

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    called = {}

    def access(proxy, mode, client="mweb"):
        called["proxy"] = proxy
        called["mode"] = mode
        called["client"] = client
        return {"proxy": proxy, "extractor_args": {"youtube": {"player_client": [client], "fetch_pot": ["always"]}}}

    monkeypatch.setattr(download, "_youtube_access_opts", access)
    monkeypatch.setattr(download, "_js_runtime_opts", lambda: {})
    listed = download.list_channel_uploads(
        "https://www.youtube.com/@demo/videos",
        cookies_from_browser="firefox",
        proxy="http://127.0.0.1:7890",
        po_token_mode="auto",
    )
    assert called == {"proxy": "http://127.0.0.1:7890", "mode": "auto", "client": "mweb"}
    assert seen["extract_flat"] == "in_playlist"
    assert seen["playlist_items"] == "1-20"
    assert seen["sleep_interval_requests"] == 0.75
    assert seen["cookiesfrombrowser"] == ("firefox",)
    assert seen["extractor_args"]["youtubetab"]["approximate_date"] == [""]
    assert seen["extractor_args"]["youtube"]["player_client"] == ["mweb"]
    assert seen["extractor_args"]["youtube"]["lang"] == ["en"]
    assert listed["title"] == "Demo"
    assert listed["entries"][0]["url"] == "https://www.youtube.com/watch?v=abcdefghijk"
    assert listed["entries"][0]["upload_date"] == "20231114"
    assert listed["entries"][0]["timestamp"] == 1_700_000_000
    assert listed["entries"][0]["thumbnail_url"] == "https://i.ytimg.com/vi/abcdefghijk/sddefault.jpg"
    assert listed["entries"][0]["view_count"] == 1200


def test_plain_playlist_listing_stays_flat_without_access_opts(monkeypatch):
    seen = {}

    class FakeYDL:
        def __init__(self, opts):
            seen.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url, download=False):
            return {"entries": [{"url": "abc", "ie_key": "Youtube", "title": "One", "duration": 10}]}

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    entries = download.list_playlist_entries("https://www.youtube.com/playlist?list=PL1")
    assert seen["playlist_items"] == "1-200"
    assert "extractor_args" not in seen
    assert "sleep_interval_requests" not in seen
    assert entries == [{"url": "https://www.youtube.com/watch?v=abc", "title": "One", "duration": 10}]
