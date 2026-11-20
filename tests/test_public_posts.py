from __future__ import annotations

import pytest

from yblocalizer.browser_media import _hls_variant_url
from yblocalizer.download import _finish_public_download, list_playlist_entries
from yblocalizer.public_posts import (
    _html_title,
    bluesky_handle_and_rkey,
    download_public_post,
    hls_audio_playlist,
    reddit_media_id,
    select_x_variant,
    syndication_token,
)


def test_syndication_token_matches_the_public_embed_formula():
    token = syndication_token("910031516746514432")
    assert token == syndication_token("910031516746514432")
    assert token and "0" not in token and "." not in token


def test_reddit_embed_html_names_the_public_video():
    page = '<script>https://v.redd.it/zv89llsvexdz/DASHPlaylist.mpd</script>'
    assert reddit_media_id(page) == "zv89llsvexdz"
    assert reddit_media_id("<html>no video</html>") is None


def test_hls_audio_playlist_stays_beside_the_master():
    playlist = '#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="audio",URI="HLS_AUDIO_160.m3u8"\n'
    assert hls_audio_playlist(playlist, "https://v.redd.it/abc/HLSPlaylist.m3u8") == (
        "https://v.redd.it/abc/HLS_AUDIO_160.m3u8"
    )


def test_bluesky_post_url_keeps_the_handle_and_record_key():
    assert bluesky_handle_and_rkey(
        "https://bsky.app/profile/name.bsky.social/post/3l7rdfxhyds2f"
    ) == ("name.bsky.social", "3l7rdfxhyds2f")


def test_reddit_heading_is_the_post_title():
    page = "<title>Reddit - 全网主阵地</title><h1>那次小心脏病发作。</h1>"
    assert _html_title(page) == "那次小心脏病发作。"


def test_batch_parse_uses_the_public_post(monkeypatch):
    url = "https://www.reddit.com/r/videos/comments/6rrwyj/title/"

    def fake_entry(page_url, proxy=None):
        assert page_url == url
        assert proxy == "http://proxy.example"
        return {"url": page_url, "title": "公开帖子", "duration": 12}

    monkeypatch.setattr("yblocalizer.download.public_post_entry", fake_entry)
    assert list_playlist_entries(url, proxy="http://proxy.example") == [
        {"url": url, "title": "公开帖子", "duration": 12}
    ]


def test_batch_parse_reports_a_post_without_video(monkeypatch):
    monkeypatch.setattr("yblocalizer.download.public_post_entry", lambda url, proxy=None: None)
    with pytest.raises(RuntimeError, match="没有可下载的视频"):
        list_playlist_entries("https://x.com/name/status/910031516746514432")


_PLAYLIST = "\n".join([
    "#EXTM3U",
    "#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=1280x720",
    "720.m3u8",
    "#EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=1920x1080",
    "1080.m3u8",
    "#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=2560x1440",
    "1440.m3u8",
])
_BASE = "https://v.redd.it/abc/HLSPlaylist.m3u8"
_X_VARIANTS = [
    {"bitrate": 256000, "url": "https://cdn.example/low.mp4"},
    {"bitrate": 832000, "url": "https://cdn.example/mid.mp4"},
    {"bitrate": 2176000, "url": "https://cdn.example/high.mp4"},
]


def test_hls_height_follows_the_quality_ceiling():
    assert _hls_variant_url(_PLAYLIST, _BASE).endswith("720.m3u8")
    assert _hls_variant_url(_PLAYLIST, _BASE, 720).endswith("720.m3u8")
    assert _hls_variant_url(_PLAYLIST, _BASE, 1080).endswith("1080.m3u8")
    assert _hls_variant_url(_PLAYLIST, _BASE, None).endswith("1440.m3u8")


def test_x_bitrate_follows_the_quality_choice():
    assert select_x_variant(_X_VARIANTS, "720p").endswith("mid.mp4")
    assert select_x_variant(_X_VARIANTS, "1080p").endswith("high.mp4")
    assert select_x_variant(_X_VARIANTS, "original").endswith("high.mp4")
    assert select_x_variant([{"bitrate": 2176000, "url": "https://cdn.example/high.mp4"}], "720p").endswith("high.mp4")


def test_public_download_reselects_x_and_caps_hls(tmp_path, monkeypatch):
    seen = {}

    def fake_resolve(url, proxy=None):
        if "x.com" in url:
            return {
                "media_url": "https://cdn.example/high.mp4",
                "kind": "file",
                "variants": _X_VARIANTS,
                "referer": "https://x.com/",
                "title": "帖子",
            }
        return {
            "media_url": "https://v.redd.it/abc/HLSPlaylist.m3u8",
            "kind": "hls",
            "referer": "https://www.reddit.com/",
            "title": "帖子",
        }

    def fake_stream(media_url, destination, *args):
        seen["mp4"] = media_url
        destination.write_bytes(b"x" * 2048)

    def fake_save(media_url, destination, referer, proxy, seconds, max_height=720, cancel_check=None):
        seen["height"] = max_height
        destination.write_bytes(b"x" * 2048)
        return True

    monkeypatch.setattr("yblocalizer.public_posts.resolve_public_post", fake_resolve)
    monkeypatch.setattr("yblocalizer.public_posts._stream_media", fake_stream)
    monkeypatch.setattr("yblocalizer.public_posts._save_hls", fake_save)
    monkeypatch.setattr("yblocalizer.public_posts.file_has_audio", lambda path: True)
    download_public_post("https://x.com/name/status/1", tmp_path / "x.mp4", download_quality="720p")
    download_public_post(
        "https://www.reddit.com/r/videos/comments/6rrwyj/title/",
        tmp_path / "reddit.mp4",
        download_quality="original",
    )
    assert seen["mp4"].endswith("mid.mp4")
    assert seen["height"] is None


def test_public_video_stays_when_platform_subtitles_fail(tmp_path, monkeypatch):
    calls = {"count": 0}

    def fake_download(url, destination, **kwargs):
        destination.write_bytes(b"video")
        return {"title": "公开", "has_audio": True, "webpage_url": url}

    def fake_subs(*args, **kwargs):
        calls["count"] += 1
        return None

    monkeypatch.setattr("yblocalizer.download.download_public_post", fake_download)
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda path: (1280, 720))
    monkeypatch.setattr("yblocalizer.download._try_download_platform_subtitles", fake_subs)
    job = _finish_public_download(
        "https://x.com/name/status/910031516746514432",
        tmp_path,
        None,
        None,
        None,
        None,
        download_quality="720p",
        prefer_platform_subtitles=True,
        yt_dlp_module=object(),
    )
    assert job.raw_video is not None and job.raw_video.is_file()
    assert job.source_subtitles is None
    assert calls["count"] == 1
    _finish_public_download(
        "https://x.com/name/status/910031516746514432",
        tmp_path,
        None,
        None,
        None,
        None,
        prefer_platform_subtitles=False,
        yt_dlp_module=object(),
    )
    assert calls["count"] == 1


def test_confirmed_quality_still_rejects_a_smaller_public_file(tmp_path, monkeypatch):
    from yblocalizer.download import DownloadProposalMismatch

    def fake_download(url, destination, **kwargs):
        destination.write_bytes(b"video")
        return {"title": "公开", "has_audio": True}

    monkeypatch.setattr("yblocalizer.download.download_public_post", fake_download)
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda path: (640, 360))
    with pytest.raises(DownloadProposalMismatch, match="640×360"):
        _finish_public_download(
            "https://x.com/name/status/1",
            tmp_path,
            None,
            None,
            {"width": 1280, "height": 720, "has_audio": True},
            None,
            prefer_platform_subtitles=True,
            yt_dlp_module=object(),
        )
    assert (tmp_path / "public-post.mp4").is_file()
