from io import BytesIO
from pathlib import Path

import pytest

from yblocalizer.browser_media import _stream_media
from yblocalizer.public_posts import _write_segments


def test_stream_validation_reads_only_the_header(tmp_path, monkeypatch):
    media = b"\x00\x00\x00\x18ftypmp42" + b"v" * 2048
    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *_args: type("Opener", (), {"open": lambda *_args, **_kwargs: BytesIO(media)})(),
    )

    def no_whole_file_read(_path):
        raise AssertionError("Video validation must not load the whole file into memory")

    monkeypatch.setattr(Path, "read_bytes", no_whole_file_read)
    destination = tmp_path / "video.mp4"
    _stream_media("https://cdn.example/video.mp4", destination, "", "", "test", None)
    assert destination.stat().st_size == len(media)


def test_hls_validation_reads_only_the_header(tmp_path, monkeypatch):
    monkeypatch.setattr("yblocalizer.public_posts._http_bytes", lambda *_args: b"\x47" + b"v" * 2048)

    def no_whole_file_read(_path):
        raise AssertionError("HLS validation must not load every segment into memory again")

    monkeypatch.setattr(Path, "read_bytes", no_whole_file_read)
    assert _write_segments(["https://cdn.example/segment.ts"], tmp_path / "video.ts", None, "")
