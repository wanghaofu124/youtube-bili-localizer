"""Site allowlist, cache identity, and YouTube-only download options. No network."""

from types import SimpleNamespace

import pytest

from yblocalizer.download import _media_access_opts, download_with_ytdlp
from yblocalizer.browser_media import _rumble_mp4
from yblocalizer.source_sites import (
    UnsupportedVideoSite,
    require_supported_video_url,
    source_identity,
    vimeo_player_url,
    youtube_video_id,
)
from yblocalizer.workbench_api import WorkbenchJobs, _openable_video_url
from yblocalizer.workbench_config import InputValidationError


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abcdefghijk",
        "https://youtu.be/abcdefghijk",
        "https://vimeo.com/123456789",
        "https://dai.ly/xabcde",
        "https://www.tiktok.com/@name/video/1234567890123456789",
        "https://www.nicovideo.jp/watch/sm12345",
        "https://rumble.com/vdemo1-title.html",
        "https://x.com/name/status/1234567890123456789",
        "https://www.instagram.com/reel/AbCdEf123/",
        "https://www.facebook.com/reel/1234567890",
        "https://www.reddit.com/r/videos/comments/6rrwyj/title/",
        "https://bsky.app/profile/name.bsky.social/post/3l4omssdl632g",
        "https://example.tumblr.com/post/54196191430/title",
    ],
)
def test_first_batch_hosts_are_accepted(url: str) -> None:
    assert require_supported_video_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "https://www.bilibili.com/video/BV1xx411c7mD",
        "https://www.douyin.com/video/123",
        "https://example.com/video",
        "https://www.twitch.tv/videos/1",
    ],
)
def test_hosts_outside_the_list_are_rejected(url: str) -> None:
    with pytest.raises(UnsupportedVideoSite, match="当前不支持"):
        require_supported_video_url(url)


def test_youtube_identity_keeps_the_historical_video_id() -> None:
    assert youtube_video_id("https://www.youtube.com/watch?v=abcdefghijk&t=12") == "abcdefghijk"
    assert youtube_video_id("https://youtu.be/abcdefghijk") == "abcdefghijk"
    assert youtube_video_id("https://www.youtube.com/shorts/abcdefghijk") == "abcdefghijk"
    assert source_identity("https://www.youtube.com/watch?v=abcdefghijk&list=PL1") == "youtube:abcdefghijk"
    assert source_identity("https://youtu.be/abcdefghijk?t=3") == "youtube:abcdefghijk"


@pytest.mark.parametrize(
    ("url", "identity"),
    [
        ("https://vimeo.com/123456789", "vimeo:123456789"),
        ("https://www.dailymotion.com/video/xabcde", "dailymotion:xabcde"),
        ("https://dai.ly/xabcde", "dailymotion:xabcde"),
        ("https://www.tiktok.com/@name/video/1234567890123456789", "tiktok:1234567890123456789"),
        ("https://www.nicovideo.jp/watch/sm12345", "niconico:sm12345"),
        ("https://rumble.com/vdemo1-title.html", "rumble:vdemo1"),
        ("https://x.com/name/status/1234567890123456789", "x:1234567890123456789"),
        ("https://twitter.com/name/status/1234567890123456789", "x:1234567890123456789"),
        ("https://www.instagram.com/reel/AbCdEf123/", "instagram:AbCdEf123"),
        ("https://www.facebook.com/page/videos/3676516585958356", "facebook:3676516585958356"),
        ("https://www.reddit.com/r/videos/comments/6rrwyj/title/", "reddit:6rrwyj"),
        ("https://bsky.app/profile/name.bsky.social/post/3l4omssdl632g", "bluesky:3l4omssdl632g"),
        ("https://example.tumblr.com/post/54196191430/title", "tumblr:54196191430"),
    ],
)
def test_other_sites_use_site_and_video_id(url: str, identity: str) -> None:
    assert source_identity(url) == identity
    assert source_identity(url + "?utm=1") == identity


def test_vimeo_format_prefers_the_direct_file() -> None:
    from yblocalizer.download import _format_selector
    selected = _format_selector("https://vimeo.com/27246366", "720p")
    assert selected.startswith("best[protocol^=http][ext=mp4][height<=720]/")
    youtube = _format_selector("https://www.youtube.com/watch?v=abcdefghijk", "720p")
    assert "protocol^=http" not in youtube


def test_hls_segment_urls_keep_only_the_opening() -> None:
    from yblocalizer.browser_media import _hls_segment_urls
    playlist = (
        "#EXTM3U\n#EXTINF:14.4,\n?r_type=video/mp2t&r_range=512-100\n"
        "#EXTINF:9.6,\n?r_type=video/mp2t&r_range=200-300\n"
    )
    assert _hls_segment_urls(playlist, "https://cdn.example/a.m3u8?x=1", 5) == [
        "https://cdn.example/a.m3u8?r_type=video/mp2t&r_range=512-100"
    ]


def test_tiktok_play_url_skips_unplayable_codec() -> None:
    from yblocalizer.browser_media import _tiktok_play_url
    video = {
        "bitrateInfo": [
            {"GearName": "bytevc2_720", "PlayAddr": {"UrlList": ["https://cdn.example/bytevc2.mp4"], "DataSize": 9}},
            {"GearName": "h264_540", "PlayAddr": {"UrlList": ["https://cdn.example/h264.mp4"], "DataSize": 3}},
        ]
    }
    assert _tiktok_play_url(video) == "https://cdn.example/h264.mp4"
    assert _tiktok_play_url({"playAddr": "https://cdn.example/direct.mp4"}) == "https://cdn.example/direct.mp4"


def test_vimeo_watch_urls_use_the_public_player_page() -> None:
    assert vimeo_player_url("https://vimeo.com/76979871") == "https://player.vimeo.com/video/76979871"
    assert vimeo_player_url("https://player.vimeo.com/video/76979871") == "https://player.vimeo.com/video/76979871"
    assert (
        vimeo_player_url("https://vimeo.com/123456789/abcdef")
        == "https://player.vimeo.com/video/123456789?h=abcdef"
    )
    assert vimeo_player_url("https://www.youtube.com/watch?v=abcdefghijk") is None


def test_rumble_watch_url_uses_the_public_slug_token() -> None:
    from yblocalizer.browser_media import _rumble_api_url
    assert _rumble_api_url(
        "https://rumble.com/vdmum1-moose-the-dog-helps-girls-dig-a-snow-fort.html"
    ) == "https://rumble.com/embedJS/u3/?request=video&ver=2&v=vdmum1"


def test_rumble_ignores_the_timeline_preview() -> None:
    from yblocalizer.browser_media import _rumble_media_url
    payload = {
        "u": {"timeline": {"url": "https://cdn.example/timeline.mp4", "meta": {"h": 180, "w": 320}}},
        "ua": {"hls": {"auto": {"url": "https://cdn.example/master.m3u8"}}, "mp4": {"720": {"url": "https://cdn.example/720.mp4", "meta": {"h": 720}}}},
    }
    assert _rumble_media_url(payload) == "https://cdn.example/720.mp4"


def test_rumble_embed_json_uses_the_short_mp4_key() -> None:
    payload = {"w": 1280, "h": 720, "u": {"mp4": {"url": "https://cdn.example/video.mp4"}}}
    assert _rumble_mp4(payload) == "https://cdn.example/video.mp4"


def test_media_access_keeps_youtube_extractor_args_and_skips_them_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": False, "browser_path": None},
    )
    proxy = "http://127.0.0.1:7890"
    youtube = _media_access_opts("https://www.youtube.com/watch?v=abcdefghijk", proxy, "auto")
    vimeo = _media_access_opts("https://vimeo.com/123456789", proxy, "auto")
    x_post = _media_access_opts("https://x.com/name/status/1234567890123456789", proxy, "auto")
    assert youtube["proxy"] == proxy
    assert youtube["extractor_args"]["youtube"]["player_client"] == ["mweb"]
    assert vimeo == {"proxy": proxy}
    assert x_post == {"proxy": proxy}
    assert "extractor_args" not in vimeo


def test_vimeo_failure_does_not_use_the_youtube_hint(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    seen: list[dict] = []

    class FakeYoutubeDL:
        def __init__(self, options):
            seen.append(options)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url: str, download: bool):
            raise RuntimeError("HTTP Error 403: Forbidden")

    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": False, "browser_path": None},
    )
    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    with pytest.raises(RuntimeError, match="这个站点没有返回可下载的视频") as captured:
        download_with_ytdlp("https://vimeo.com/123456789", tmp_path, po_token_mode="auto")

    assert "PO Token" not in str(captured.value)
    assert seen and "extractor_args" not in seen[0]


def test_vimeo_requests_platform_subtitles(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: list[dict] = []
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda _path: (1280, 720))

    class FakeYoutubeDL:
        def __init__(self, options):
            captured.append(options)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, url: str, download: bool):
            if captured[-1].get("writesubtitles"):
                return {"id": "123456789", "title": "Demo", "webpage_url": url}
            (tmp_path / "123456789.mp4").write_bytes(b"video")
            return {"id": "123456789", "title": "Demo", "webpage_url": url}

    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": False, "browser_path": None},
    )
    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    job = download_with_ytdlp("https://vimeo.com/123456789", tmp_path, prefer_platform_subtitles=True)
    assert job.source == "https://vimeo.com/123456789"
    assert job.raw_video == tmp_path / "123456789.mp4"
    assert any(options.get("writesubtitles") is True for options in captured)
    assert all("extractor_args" not in options for options in captured)


def test_download_rejects_unlisted_hosts_before_ytdlp(tmp_path) -> None:
    with pytest.raises(UnsupportedVideoSite, match="bilibili.com"):
        download_with_ytdlp("https://www.bilibili.com/video/BV1xx411c7mD", tmp_path)


def test_create_url_rejects_unlisted_hosts(tmp_path) -> None:
    with pytest.raises(InputValidationError, match="当前不支持") as captured:
        WorkbenchJobs().create_url(
            "https://example.com/video", "cpu", "int8", True, {"output_dir": str(tmp_path)},
        )
    assert captured.value.code == "unsupported_site"


def test_openable_links_cover_the_shortlist_and_bilibili() -> None:
    assert _openable_video_url("https://vimeo.com/123456789").endswith("123456789")
    assert _openable_video_url("https://x.com/name/status/1234567890123456789").endswith("1234567890123456789")
    assert _openable_video_url("https://www.bilibili.com/video/BV1xx411c7mD")
    with pytest.raises(InputValidationError):
        _openable_video_url("https://example.com/watch?v=abcdefghijk")
