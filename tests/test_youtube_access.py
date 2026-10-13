from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import yt_dlp as real_yt_dlp

from yblocalizer.download import (
    DownloadProposalMismatch,
    DownloadResolutionMismatch,
    YouTubeAccessError,
    _cancellable_external_ffmpeg,
    _format_for_quality,
    _quarantine_below_resolution,
    _quality_proposal,
    _raise_with_youtube_hint,
    _youtube_access_opts,
    download_with_ytdlp,
)
from yblocalizer.cancellation import CancellationRequested
from yblocalizer.workbench_api import _classify_job_error, _public_options
from yblocalizer.workbench_config import normalize_options


def test_po_token_auto_uses_mweb_provider_and_same_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": True, "browser_path": "C:/Browser/chrome.exe"},
    )

    options = _youtube_access_opts("http://127.0.0.1:7890", "auto")

    assert options["proxy"] == "http://127.0.0.1:7890"
    assert options["extractor_args"]["youtube"] == {
        "player_client": ["mweb"],
        "fetch_pot": ["auto"],
    }
    assert options["extractor_args"]["youtubepot-wpc"]["browser_path"] == [
        "C:/Browser/chrome.exe"
    ]


def test_po_token_off_does_not_probe_or_open_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: (_ for _ in ()).throw(AssertionError("provider must not be probed")),
    )
    assert _youtube_access_opts(None, "off") == {}


@pytest.mark.parametrize("proxy", ["127.0.0.1:7890", "file:///tmp/proxy", "socks://localhost:1"])
def test_proxy_requires_an_explicit_supported_scheme(proxy: str) -> None:
    with pytest.raises(ValueError, match="代理地址格式无效"):
        _youtube_access_opts(proxy, "off")


def test_normal_download_extracts_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[bool] = []
    captured: dict = {}
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda _path: (1280, 720))

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options
            captured.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url: str, download: bool):
            calls.append(download)
            if download:
                (tmp_path / "demo.mp4").write_bytes(b"video")
            return {"id": "demo", "title": "Demo", "license": "", "webpage_url": _url}

    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": False, "browser_path": None},
    )
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    job = download_with_ytdlp("https://example.test/video", tmp_path)

    assert calls == [True]
    assert job.raw_video == tmp_path / "demo.mp4"
    assert "height<=1080" in captured["format"]
    assert "bestvideo+bestaudio/best" not in captured["format"]
    assert captured["continuedl"] is True
    assert "overwrites" not in captured
    assert captured["nopart"] is False


def test_original_proposal_reports_ytdlp_selection_not_our_bitrate_rank() -> None:
    formats = [
        {"format_id": "315", "width": 3840, "height": 2160, "fps": 50, "vbr": 26325,
         "vcodec": "vp9", "acodec": "none", "ext": "webm", "url": "https://example.test/315"},
        {"format_id": "401", "width": 3840, "height": 2160, "fps": 50, "vbr": 13212,
         "vcodec": "av01.0.13M.08", "acodec": "none", "ext": "mp4", "url": "https://example.test/401"},
        {"format_id": "251", "vcodec": "none", "acodec": "opus", "abr": 160,
         "ext": "webm", "url": "https://example.test/251"},
    ]
    ydl = real_yt_dlp.YoutubeDL({"format": _format_for_quality("original"), "quiet": True})
    selected = ydl.process_video_result({"id": "demo", "title": "demo", "formats": formats}, download=False)
    proposal = _quality_proposal(selected, "original")
    assert (proposal["video_format_id"], proposal["audio_format_id"], proposal["height"]) == ("401", "251", 2160)


def test_confirmed_resolution_is_a_floor_for_every_ytdlp_fallback() -> None:
    selector = _format_for_quality("original", (3840, 2160))
    assert selector == "bestvideo[width>=3840][height>=2160]+bestaudio/best[width>=3840][height>=2160]"

    formats = [
        {"format_id": "360", "width": 640, "height": 360, "vcodec": "avc1", "acodec": "mp4a", "ext": "mp4", "url": "https://example.test/360"},
        {"format_id": "4k-video", "width": 3840, "height": 2160, "vcodec": "av01", "acodec": "none", "ext": "webm", "url": "https://example.test/video"},
        {"format_id": "audio", "vcodec": "none", "acodec": "opus", "ext": "webm", "url": "https://example.test/audio"},
    ]
    ydl = real_yt_dlp.YoutubeDL({"format": selector, "quiet": True})
    selected = ydl.process_video_result({"id": "demo", "title": "demo", "formats": formats}, download=False)
    assert [item["format_id"] for item in selected["requested_formats"]] == ["4k-video", "audio"]

    low_only = {"formats": [formats[0]], "has_merged_format": True, "incomplete_formats": False}
    try:
        available = list(ydl.build_format_selector(selector)(low_only))
    except real_yt_dlp.utils.ExtractorError:
        available = []
    assert available == []


def test_stale_low_resolution_cache_is_preserved_out_of_ytdlp_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    low = tmp_path / "video.mp4"
    high = tmp_path / "other.mp4"
    low.write_bytes(b"old-low")
    high.write_bytes(b"good-4k")
    dimensions = {low: (640, 360), high: (3840, 2160)}
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda path: dimensions.get(path))

    preserved = _quarantine_below_resolution(tmp_path, (3840, 2160))

    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"old-low"
    assert not low.exists()
    assert high.read_bytes() == b"good-4k"


def test_stale_partial_is_left_to_ytdlp_without_an_outer_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda _path: (1280, 720))
    part = tmp_path / "demo.f399.mp4.part"
    part.write_bytes(b"stale-partial")

    class FakeYoutubeDL:
        def __init__(self, _options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url: str, download: bool):
            calls.append(download)
            raise RuntimeError("ERROR: unable to download video data: HTTP Error 416: Requested range not satisfiable")

    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": False, "browser_path": None},
    )
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    with pytest.raises(YouTubeAccessError, match="续传失败"):
        download_with_ytdlp("https://example.test/video", tmp_path)

    assert calls == [True]
    assert part.read_bytes() == b"stale-partial"


def test_stale_partial_errors_have_precise_codes() -> None:
    with pytest.raises(YouTubeAccessError) as captured:
        _raise_with_youtube_hint(
            RuntimeError("ERROR: unable to download video data: HTTP Error 416: Requested range not satisfiable"),
            None,
            "auto",
        )
    assert captured.value.code == "youtube_stale_partial"
    assert "续传失败" in str(captured.value)


def test_network_timeout_keeps_partial_without_an_outer_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[bool] = []
    timeouts: list[int] = []
    part = tmp_path / "demo.f401.webm.part"
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda _path: (1280, 720))

    class FakeYoutubeDL:
        def __init__(self, options):
            timeouts.append(int(options.get("socket_timeout") or 0))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url: str, download: bool):
            calls.append(download)
            part.write_bytes(b"downloaded-before-timeout")
            raise RuntimeError("ERROR: [download] Got error: [WinError 10060] timed out")

    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": False, "browser_path": None},
    )
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    with pytest.raises(YouTubeAccessError) as captured:
        download_with_ytdlp("https://example.test/video", tmp_path, download_quality="720p")

    assert captured.value.code == "youtube_network_timeout"
    assert calls == [True]
    assert timeouts == [15]
    assert part.read_bytes() == b"downloaded-before-timeout"


def test_ssl_eof_does_not_propose_lower_quality(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    attempted_formats: list[str] = []

    class FakeYoutubeDL:
        def __init__(self, options):
            attempted_formats.append(str(options.get("format")))

        def __enter__(self): return self
        def __exit__(self, *_args): return False

        def extract_info(self, _url: str, download: bool):
            assert download is True
            raise RuntimeError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")

    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": False, "browser_path": None},
    )
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    with pytest.raises(YouTubeAccessError) as captured:
        download_with_ytdlp("https://example.test/video", tmp_path, download_quality="original")

    assert captured.value.code == "youtube_network_timeout"
    assert attempted_formats == ["bestvideo+bestaudio/best"]


def test_network_timeout_errors_have_precise_codes() -> None:
    with pytest.raises(YouTubeAccessError) as captured:
        _raise_with_youtube_hint(
            RuntimeError("ERROR: [download] Got error: [WinError 10060] timed out"),
            None,
            "auto",
        )
    assert captured.value.code == "youtube_network_timeout"
    assert "超时" in str(captured.value)


@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        ("720p", "bestvideo[height<=720]+bestaudio/best[height<=720]"),
        ("1080p", "bestvideo[height<=1080]+bestaudio/best[height<=1080]"),
        ("original", "bestvideo+bestaudio/best"),
    ],
)
def test_download_quality_never_has_an_unbounded_fallback(quality: str, expected: str) -> None:
    value = _format_for_quality(quality)
    assert value == expected
    assert "/worst" not in value
    if quality == "original":
        assert value == "bestvideo+bestaudio/best"


def test_download_rejects_file_below_selected_original_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    captured: dict = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            captured.update(options)

        def __enter__(self): return self
        def __exit__(self, *_args): return False

        def extract_info(self, _url: str, download: bool):
            assert download is True
            (tmp_path / "demo.mp4").write_bytes(b"video")
            return {"id": "demo", "title": "Demo", "width": 3840, "height": 2160}

    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": False, "browser_path": None},
    )
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda _path: (640, 360))
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    with pytest.raises(DownloadResolutionMismatch, match="3840×2160.*640×360"):
        download_with_ytdlp("https://example.test/video", tmp_path, download_quality="original")

    assert captured["format"] == "bestvideo+bestaudio/best"


@pytest.mark.parametrize(
    ("current_formats", "expected_format"),
    [
        (
            [
                {"format_id": "new4k", "width": 3840, "height": 2160, "vcodec": "av01", "acodec": "none", "ext": "mp4"},
                {"format_id": "new_audio", "vcodec": "none", "acodec": "opus", "abr": 128, "ext": "webm"},
            ],
            "new4k+new_audio",
        ),
        (
            [{"format_id": "mux4k", "width": 3840, "height": 2160, "vcodec": "av01", "acodec": "opus", "ext": "mp4"}],
            "mux4k",
        ),
    ],
)
def test_original_uses_fresh_4k_format_ids_without_downgrade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, current_formats: list[dict], expected_format: str,
) -> None:
    chosen: list[str] = []
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda _path: (3840, 2160))

    class FakeYoutubeDL:
        def __init__(self, options): self.options = options
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def build_format_selector(self, spec):
            return real_yt_dlp.YoutubeDL({"quiet": True}).build_format_selector(spec)
        def extract_info(self, _url, download):
            assert download is True
            formats = [{**item, "url": f"https://example.test/{item['format_id']}"} for item in current_formats]
            ctx = {"formats": formats, "has_merged_format": any(
                item["vcodec"] != "none" and item["acodec"] != "none" for item in current_formats
            ), "incomplete_formats": False}
            selected = list(real_yt_dlp.YoutubeDL({"quiet": True}).build_format_selector(self.options["format"])(ctx))
            chosen.append(selected[0]["format_id"])
            (tmp_path / "demo.mp4").write_bytes(b"video")
            return {"id": "demo", "title": "Demo", "formats": formats, **selected[0]}

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    promised = {"video_format_id": "old4k", "audio_format_id": "old_audio", "width": 3840,
                "height": 2160, "has_audio": True}
    job = download_with_ytdlp("https://example.test/video", tmp_path, download_quality="original", quality_proposal=promised)
    assert chosen == [expected_format]
    assert job.raw_video == tmp_path / "demo.mp4"


def test_original_rejects_lower_result_without_committing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class FakeYoutubeDL:
        def __init__(self, options): self.options = options
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def extract_info(self, _url, download):
            assert download is True
            (tmp_path / "demo.mp4").write_bytes(b"new candidate")
            return {"id": "demo", "title": "Demo", "width": 1920, "height": 1080,
                    "format_id": "v1080", "vcodec": "avc1", "acodec": "mp4a"}

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda _path: (1920, 1080))
    promised = {"video_format_id": "old4k", "audio_format_id": "", "width": 3840,
                "height": 2160, "has_audio": True}
    with pytest.raises(DownloadProposalMismatch, match="3840×2160.*1920×1080"):
        download_with_ytdlp("https://example.test/video", tmp_path, download_quality="original", quality_proposal=promised)
    assert (tmp_path / "demo.mp4").read_bytes() == b"new candidate"


def test_ytdlp_external_ffmpeg_wait_is_cancellable() -> None:
    from yt_dlp.downloader import external

    original = external.Popen
    with pytest.raises(CancellationRequested):
        with _cancellable_external_ffmpeg(
            lambda: (_ for _ in ()).throw(CancellationRequested("stop"))
        ):
            process = external.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"]
            )
            process.wait()
    assert process.poll() is not None
    assert external.Popen is original


def test_cc_gate_keeps_metadata_preflight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[bool] = []
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda _path: (1280, 720))

    class FakeYoutubeDL:
        def __init__(self, _options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url: str, download: bool):
            calls.append(download)
            if download:
                (tmp_path / "demo.mp4").write_bytes(b"video")
            return {"id": "demo", "title": "Demo", "license": "reuse allowed"}

    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": False, "browser_path": None},
    )
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    download_with_ytdlp(
        "https://example.test/video", tmp_path, require_reuse_allowed=True
    )

    assert calls == [False, True]


def test_youtube_failures_have_precise_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": True, "browser_path": "chrome.exe"},
    )
    with pytest.raises(YouTubeAccessError) as captured:
        _raise_with_youtube_hint(RuntimeError("HTTP Error 403: Forbidden"), None, "auto")
    assert captured.value.code == "youtube_forbidden"
    assert "不能仅凭 403 判定为 Cookies" in str(captured.value)
    assert _classify_job_error(captured.value) == (
        "youtube_forbidden",
        captured.value.suggested_action,
    )


def test_bot_challenge_is_not_misreported_as_login_cookie_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "yblocalizer.download.po_token_provider_status",
        lambda: {"available": True, "browser_path": "chrome.exe"},
    )
    with pytest.raises(YouTubeAccessError) as captured:
        _raise_with_youtube_hint(
            RuntimeError("Sign in to confirm you’re not a bot"), None, "auto"
        )
    assert captured.value.code == "youtube_po_token_failed"
    assert "cookies" not in str(captured.value).lower()


def test_job_snapshot_and_history_never_echo_proxy_credentials() -> None:
    options = normalize_options(
        {
            "translator": "none",
            "target_lang": "en",
            "youtube_proxy": "http://name:secret@127.0.0.1:7890",
        },
        "outputs",
    )
    public = _public_options(options)
    assert public["youtube_proxy"] == ""
    assert public["youtube_proxy_configured"] is True
    assert "secret" not in repr(public)
