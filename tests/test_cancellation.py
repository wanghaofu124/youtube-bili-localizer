from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

from yblocalizer.cancellation import CancellationRequested
from yblocalizer.runtime import CancellationRequested as RuntimeCancellationRequested
from yblocalizer.download import download_with_ytdlp
from yblocalizer.pipeline import request_cancellation, reset_cancellation
from yblocalizer.util import run


@pytest.fixture(autouse=True)
def reset_cancel_state() -> None:
    reset_cancellation()
    yield
    reset_cancellation()


def test_ytdlp_progress_hook_stops_download_after_cancel(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url: str, download: bool):
            if not download:
                return {"id": "demo", "title": "demo", "license": "reuse allowed"}
            request_cancellation()
            self.options["progress_hooks"][0]({"status": "downloading"})
            raise AssertionError("The yt-dlp hook should have interrupted the download.")

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    with pytest.raises(CancellationRequested):
        download_with_ytdlp("https://example.test/video", tmp_path)


@pytest.mark.parametrize("url", ["", "youtube.com/watch?v=demo", "ftp://example.test/video"])
def test_metadata_rejects_incomplete_or_non_http_urls(url: str) -> None:
    from yblocalizer.download import get_video_metadata

    with pytest.raises(ValueError, match="完整"):
        get_video_metadata(url)


def test_run_terminates_known_child_process_when_cancelled() -> None:
    with pytest.raises(CancellationRequested):
        run([sys.executable, "-c", "import time; time.sleep(10)"], cancel_check=lambda: True)


def test_legacy_and_pipeline_cancellation_use_one_exception_type() -> None:
    assert CancellationRequested is RuntimeCancellationRequested
