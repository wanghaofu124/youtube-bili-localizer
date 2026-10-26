from urllib.parse import parse_qs, urlparse

from yblocalizer.site_hot import merge_site_entries, refresh_site_hot, set_site_region, site_snapshot


def test_merge_drops_shorts_and_live_then_ranks_by_views():
    groups = [
        [
            {"url": "https://www.youtube.com/watch?v=short000000", "title": "Short", "duration": 60, "view_count": 9_999_999, "timestamp": 1, "channel": "A"},
            {"url": "https://www.youtube.com/watch?v=live0000000", "title": "Live", "duration": 600, "live_status": "is_live", "view_count": 9_999_999, "channel": "B"},
            {"url": "https://www.youtube.com/watch?v=low00000000", "title": "Low", "duration": 120, "view_count": 10, "timestamp": 2, "channel": "C", "upload_date": "20261020", "thumbnail_url": "https://i.ytimg.com/x.jpg"},
            {"url": "https://www.youtube.com/watch?v=high0000000", "title": "High old", "duration": 200, "view_count": 40, "timestamp": 1, "channel": "D"},
        ],
        [
            {"url": "https://www.youtube.com/watch?v=high0000000", "title": "High", "duration": 200, "view_count": 50, "timestamp": 1, "channel": "D", "channel_url": "https://www.youtube.com/@d"},
            {"url": "https://www.youtube.com/watch?v=mid00000000", "title": "Mid", "duration": 61, "view_count": 30, "timestamp": 3, "channel": "E"},
            {"url": "https://www.youtube.com/watch?v=nocount0000", "title": "No count", "duration": 80, "view_count": None, "timestamp": 9, "channel": "F"},
            {"url": "https://www.youtube.com/watch?v=song0000000", "title": "Hit (Song)", "duration": 200, "view_count": 9_000_000, "timestamp": 4, "channel": "Zee Music"},
            {"url": "https://www.youtube.com/playlist?list=PL1", "title": "Not a video", "duration": 80, "view_count": 1, "channel": "G"},
        ],
    ]
    ranked = merge_site_entries(groups)
    assert [item["title"] for item in ranked] == ["Hit (Song)", "High", "Mid", "Low", "No count"]
    rows = merge_site_entries(groups, demote_music=True)
    assert [item["title"] for item in rows] == ["High", "Mid", "Low", "No count", "Hit (Song)"]
    assert rows[0]["view_count"] == 50
    assert rows[0]["channel_title"] == "D"
    assert rows[0]["channel_url"] == "https://www.youtube.com/@d"
    assert rows[2]["thumbnail_url"] == "https://i.ytimg.com/x.jpg"
    assert rows[2]["upload_date"] == "20261020"


def test_failed_refresh_keeps_cache_and_mix_skips_music(monkeypatch):
    monkeypatch.setattr("yblocalizer.site_hot.REQUEST_GAP_SECONDS", 0)
    empty = site_snapshot()
    assert empty["category"] == "mix"
    assert empty["videos"] == []
    assert empty["region"] == "US"
    assert any(item["id"] == "music" for item in empty["categories"])

    calls = []

    def fake(url, _cookies_from_browser=None, _cookies_file=None, _proxy=None, playlist_items=None, **kwargs):
        calls.append((url, playlist_items, kwargs.get("approximate_date")))
        query = parse_qs(urlparse(url).query)["search_query"][0]
        if query == "ai":
            raise RuntimeError("search broke")
        return [{
            "url": "https://www.youtube.com/watch?v=clip0000000",
            "title": "Clip",
            "duration": 180,
            "view_count": 100,
            "timestamp": 10,
            "upload_date": "20261025",
            "channel": "Chan",
            "channel_url": "https://www.youtube.com/@chan",
            "thumbnail_url": "https://i.ytimg.com/vi/clip0000000/hq.jpg",
        }]

    monkeypatch.setattr("yblocalizer.site_hot.list_playlist_entries", fake)
    access = {"cookies_from_browser": None, "cookies_file": None, "proxy": None, "po_token_mode": "off"}
    view = refresh_site_hot(access, category="mix")
    queries = [parse_qs(urlparse(url).query)["search_query"][0] for url, _items, _dated in calls]
    assert "music" not in queries
    assert "cooking" not in queries and "trailer" not in queries and "vlog" not in queries
    assert "ai" in queries and "gaming" in queries and "programming" in queries
    assert calls[0][1] == "1:100"
    assert calls[0][2] is True
    assert all("sp=CAMSBAgDEAE%3D" in url and "hl=en" in url and "gl=US" in url for url, _items, _dated in calls)
    assert view["videos"][0]["title"] == "Clip"
    assert view["videos"][0]["channel_title"] == "Chan"
    assert view["error"] is None
    assert any(item.startswith("ai:") for item in view["query_errors"])
    assert view["sort_mode"] == "views"

    def boom(*_args, **_kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr("yblocalizer.site_hot.list_playlist_entries", boom)
    again = refresh_site_hot(access, category="mix")
    assert [item["title"] for item in again["videos"]] == ["Clip"]
    assert again["refreshed_at"] == view["refreshed_at"]
    assert "上次的列表还在" in again["error"]

    moved = set_site_region("GB", "mix")
    assert moved["region"] == "GB"
    assert moved["stale"] is True
    assert moved["videos"][0]["title"] == "Clip"
    try:
        refresh_site_hot(access, category="nope")
    except ValueError as exc:
        assert "分类" in str(exc)
    else:
        raise AssertionError("unknown category should be rejected")
