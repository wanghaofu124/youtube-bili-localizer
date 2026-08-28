from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from yblocalizer.download import (
    YouTubeAccessError,
    _cancellable_external_ffmpeg,
    _format_for_quality,
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


@pytest.mark.parametrize(
    ("quality", "expected"),
    [("720p", "height<=720"), ("1080p", "height<=1080"), ("original", "bestvideo+bestaudio/best")],
)
def test_download_quality_has_an_enforced_cap(quality: str, expected: str) -> None:
    value = _format_for_quality(quality)
    assert expected in value
    if quality != "original":
        assert value.endswith("/worst")


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
