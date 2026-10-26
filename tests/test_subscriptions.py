import sys
from datetime import datetime, timezone
from http import HTTPStatus
from types import SimpleNamespace

from yblocalizer import download
from yblocalizer.subscriptions import (
    add_channel,
    normalize_channel_videos_url,
    refresh_weekly_hot,
    remove_channel,
    set_subscriptions_enabled,
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
    assert view["stale"] is False
    failed = [item for item in view["channel_results"] if not item["ok"]]
    assert len(failed) == 1 and failed[0]["error"] == "channel down"
    assert weekly_snapshot(now=NOW + 13 * 3600)["stale"] is True
    removed = remove_channel(view["channels"][0]["id"])
    assert removed["videos"] == []


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
